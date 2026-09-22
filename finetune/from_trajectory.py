"""把教师模型（在线 8B）的成功轨迹转成训练样本 —— 蒸馏试点。

## 为什么做这件事

微调后的 3B 在离线指标上很好（格式合规 100%、坐标误差降 66%），端到端却是
0/25。已定位的原因之一是**训练集里没有一张客机截图**：ScreenAgent 的
「开始按钮」标在 Win10 左下角，客机是 Win11 底部居中；训练分辨率 1024×768，
客机曾长期跑 1920×1080。

而在线 8B 在同样的任务上做成过 24/25。**它做对的每一步，都是一份真实客机
界面上的示范**。把这些步骤转成训练样本喂给 3B，就是黑盒蒸馏——学生学的是
教师**做了什么**，不是教师内部的概率分布（走 API 拿不到那个）。

## 这个模块只做「轨迹 → 样本」

训练和评测都复用现成的入口，一行都不改：

    python -m finetune.from_trajectory --archive <存档.json> --trajectories <目录>
    python -m finetune.train_lora --data finetune/data-distill   # 超参一个不动
    python -m eval.action --adapter <新 adapter> --tag distill-pilot

输出目录自带 `train.jsonl` 与 `val.jsonl`：val **原样复制**基线的那一份，
这样微调前后的离线对比仍是同一把尺子。

## 筛选规则，以及每一条的理由

1. **只要程序化判定通过的轮次。** 失败轮里的步骤大多是错的，学它等于学错。
2. **只要 `execution_status == "ok"` 的步骤。** 被安全拦截、执行出错的不算示范。
3. **丢掉执行后屏幕没变的步骤。** M4 实测 38% 的动作「执行成功但屏幕纹丝不动」
   ——点空了。这类步骤在轨迹里和成功的点击长得一模一样，不按帧差过滤就会
   把「点空」当成正确答案教给学生。
4. **`done` 只收尾且限量。** 现有训练集里 `done` 占 41.6%，微调后误报率从
   7.8% 涨到 36.5%——这是已经踩过一次的坑。这里只保留成功轮最后一步的
   `done`，并把它在新样本里的占比压到 `DONE_RATIO_CAP` 以下。
5. **同图同动作去重。** 同一个任务跑 5 轮，前几步往往完全一样，不去重会让
   这几步的权重凭空翻几倍。

## 帧差为什么是离线算的

采集时**不开 Reflector**：它会否决动作，那就改变了教师的行为，采到的就不是
8B 原本的做法了。所以帧差在这里用落盘的 before/after 两张 PNG 现算。

## 坐标不需要换算

轨迹里的 `action_model_coords` 已经是模型坐标系（1000×1000）的值，而训练格式
要的正是这个空间（`finetune/dataset.COORD_SPACE`）。**两边本来就一致，
这里不做任何缩放**——多一次换算就多一个错位的来源。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from collections import Counter
from pathlib import Path

from finetune.dataset import COORD_SPACE, build_answer

logger = logging.getLogger(__name__)

#: 基线数据目录。train.jsonl 会与蒸馏样本合并，val.jsonl 原样复制。
BASE_DATA = Path("finetune/data")

#: 输出目录。**与基线分开**，基线那份要留着做对照。
OUT_DIR = Path("finetune/data-distill")

#: `done` 样本在蒸馏样本里的占比上限。理由见模块文档第 4 条。
DONE_RATIO_CAP = 0.20

#: 帧差阈值。与 `perception.change.DEFAULT_THRESHOLD` 同口径。
CHANGE_THRESHOLD = 0.01

#: 这些动作要坐标。与 `finetune.dataset.NEEDS_POINT` 一致。
NEEDS_POINT = frozenset({"left_click", "double_click", "right_click", "mouse_move"})

#: 不进训练目标的字段。`reasoning` 是我们自己塞进去的思考文本，
#: 不是模型该输出的动作参数。
DROP_PARAMS = frozenset({"action", "x", "y", "reasoning"})


def _console() -> None:
    """同 `scripts/export_trajectory.py`：GBK 控制台下打印 ⚠ 会抛异常。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def verified_trajectories(archive: Path) -> set[str]:
    """从一份运行存档里挑出**判定通过**的轨迹 id。

    三个条件缺一不可：起点建立了、没被事后剔除、程序化判定通过。
    **不看 `model_said_done`**——实测模型自报完成与任务实际状态无关
    （一轮 25/25 自报完成而实测 0/25），拿它筛样本等于没筛。
    """
    payload = json.loads(archive.read_text(encoding="utf-8"))
    ids = set()
    for record in payload.get("records", []):
        if not record.get("precondition_ok", True) or record.get("excluded"):
            continue
        if not record.get("verified"):
            continue
        traj = record.get("trajectory_id") or ""
        if traj:
            ids.add(traj)
    return ids


def screen_changed(before: Path, after: Path) -> bool | None:
    """这一步执行后屏幕动了没有。缺图返回 None（**判定不了，不是没变**）。

    调用方要把 None 和 False 分开处理：前者是数据缺失，后者是确认点空了。
    混为一谈的话，导出时漏掉几张图就会让一批正确的步骤被当成点空丢掉。
    """
    if not before.exists() or not after.exists():
        return None
    try:
        import numpy as np
        from PIL import Image

        from perception.change import compare

        left = np.array(Image.open(before).convert("RGB"))
        right = np.array(Image.open(after).convert("RGB"))
    except Exception as exc:  # noqa: BLE001 —— 读图失败按判定不了处理
        logger.warning("读帧失败 %s：%s", before.name, exc)
        return None
    return compare(left, right, threshold=CHANGE_THRESHOLD).changed


def image_ref(path: Path) -> str:
    """样本里写的图片路径。

    **必须是训练进程打得开的那一个。** `train_lora` 把 `record["image"]`
    原样交给 `process_vision_info`，它按**当前工作目录**解析——而基线数据里
    写的是 `data/raw/screenagent_repo/...` 这种仓库根相对路径，训练正是从
    仓库根跑的。

    所以这里也写成仓库根相对路径（跑不通时退回绝对路径），并一律用正斜杠：
    反斜杠路径拿到 Linux 服务器上会被当成一个含反斜杠的文件名，一张图都打不开，
    而报错指向「文件不存在」，离根因很远。
    """
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def step_params(action_dict: dict) -> dict:
    """从轨迹里的动作字典取出训练目标要用的参数。"""
    return {k: v for k, v in action_dict.items() if k not in DROP_PARAMS}


def build_record(step: dict, traj_id: str, image_rel: str) -> dict | None:
    """一条训练样本。字段与 `finetune.dataset.build_record` 对齐。

    对齐是硬要求：`train_lora` 只认 `image` / `instruction` / `answer`
    三个键，而 `eval.action` 要按 `action` 分桶。多一份格式就多一处会漂的地方。
    """
    subtask = (step.get("subtask") or "").strip()
    if not subtask:
        return None

    action_dict = step.get("action_model_coords") or {}
    action = action_dict.get("action") or ""
    if not action:
        return None

    point_norm = None
    if action in NEEDS_POINT:
        x, y = action_dict.get("x"), action_dict.get("y")
        if x is None or y is None:
            return None
        if not (0 <= x < COORD_SPACE[0] and 0 <= y < COORD_SPACE[1]):
            return None
        point_norm = (int(x), int(y))

    record = {
        "sample_id": f"{traj_id}-step{step.get('step', 0):03d}",
        "image": image_rel,
        "instruction": subtask,
        "answer": build_answer(action, step_params(action_dict), point_norm),
        "action": action,
        # 按轨迹分组。划分时要整条轨迹一起走，**不能按样本随机切**：
        # 同一轮的步骤高度相关，切开就是泄漏。
        "session_id": traj_id,
        "action_index": step.get("step"),
        "source": "distill-8b",
        "subtask_id": step.get("subtask_id"),
    }
    if point_norm is not None:
        record["point_norm"] = list(point_norm)
    return record


def done_record(step: dict, traj_id: str, image_rel: str) -> dict | None:
    """成功轮最后一步的 `done`。轨迹里它的 `execution_status` 是 `no_action`。"""
    if (step.get("action_intent") or {}).get("done") is not True:
        return None
    subtask = (step.get("subtask") or "").strip()
    if not subtask:
        return None
    return {
        "sample_id": f"{traj_id}-step{step.get('step', 0):03d}",
        "image": image_rel,
        "instruction": subtask,
        "answer": build_answer("done", {}, None),
        "action": "done",
        "session_id": traj_id,
        "action_index": step.get("step"),
        "source": "distill-8b",
        "subtask_id": step.get("subtask_id"),
    }


def cap_done(records: list[dict], cap: float = DONE_RATIO_CAP) -> list[dict]:
    """把 `done` 的占比压到 `cap` 以下，多出来的丢掉（保留靠前的）。

    **不是可选的清洗步骤。** 上一轮重训就是因为 `done` 占 41.6%，把误报率
    从 7.8% 推到 36.5%，端到端 25 轮里 24 轮以「自报完成」收尾而实测只成 3 次。
    """
    others = [r for r in records if r["action"] != "done"]
    dones = [r for r in records if r["action"] == "done"]
    if not dones:
        return records
    # n / (n + len(others)) <= cap  =>  n <= cap * len(others) / (1 - cap)
    allowed = int(cap * len(others) / (1 - cap)) if cap < 1 else len(dones)
    kept = dones[:allowed]
    merged = others + kept
    merged.sort(key=lambda r: (r["session_id"], r["action_index"] or 0))
    return merged


def dedupe(records: list[dict], frames_root: Path) -> tuple[list[dict], int]:
    """同一张图 + 同一个答案只留一条。

    同一个任务跑 5 轮，开头几步常常逐像素相同；不去重的话这几步的权重会
    凭空翻几倍，而它们恰恰是最简单的那几步。
    """
    seen: set[tuple[str, str]] = set()
    kept = []
    dropped = 0
    for record in records:
        path = frames_root / record["image"]
        try:
            digest = hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324 —— 只用来判重
        except OSError:
            digest = record["image"]
        key = (digest, record["answer"])
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        kept.append(record)
    return kept, dropped


def collect(
    trajectory_dirs: list[Path],
    verified: set[str],
    out_dir: Path,
) -> tuple[list[dict], Counter]:
    """遍历轨迹目录，产出样本，同时把用到的帧复制进输出目录。

    **复制而不是引用原路径**：训练可能在另一台机器上跑，输出目录必须自带
    全部素材。这是 `finetune/data/meta.json` 那次教训的同一类问题——
    数据和它依赖的东西分家，过几天就对不上了。
    """
    stats: Counter = Counter()
    records: list[dict] = []
    images_root = out_dir / "images"

    for traj_dir in trajectory_dirs:
        traj_id = traj_dir.name
        steps_file = traj_dir / "steps.jsonl"
        if not steps_file.exists():
            stats["轨迹缺 steps.jsonl"] += 1
            continue
        if verified and traj_id not in verified:
            stats["轮次判定未通过"] += 1
            continue

        steps = []
        for line in steps_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                steps.append(json.loads(line))
            except json.JSONDecodeError:
                stats["步骤解析失败"] += 1

        for index, step in enumerate(steps):
            status = step.get("execution_status") or ""
            before_rel = step.get("screenshot_before") or ""
            if not before_rel:
                stats["缺前置截图"] += 1
                continue

            is_last = index == len(steps) - 1
            if status == "no_action":
                if not is_last:
                    stats["非收尾的 done"] += 1
                    continue
                record = done_record(step, traj_id, "")
                if record is None:
                    stats["收尾步不是 done"] += 1
                    continue
            elif status == "ok":
                changed = screen_changed(
                    traj_dir / before_rel, traj_dir / (step.get("screenshot_after") or "")
                )
                if changed is False:
                    stats["执行后屏幕没变（点空）"] += 1
                    continue
                if changed is None:
                    stats["帧差判定不了（仍保留）"] += 1
                record = build_record(step, traj_id, "")
                if record is None:
                    stats["动作字段不完整"] += 1
                    continue
            else:
                stats[f"执行状态 {status or '空'}"] += 1
                continue

            # 帧落到 images/<轨迹>/<文件名>，样本里写相对输出目录的路径
            src = traj_dir / before_rel
            dst = images_root / traj_id / Path(before_rel).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)
            record["image"] = image_ref(dst)
            records.append(record)
            stats[f"收下 {record['action']}"] += 1

    return records, stats


def write_dataset(
    records: list[dict],
    out_dir: Path,
    base_data: Path,
    mix_base: bool = True,
) -> dict:
    """写出 train.jsonl / val.jsonl / meta.json。

    `val.jsonl` **原样复制基线那份**：微调前后要在同一把尺子上比，
    验证集一旦跟着变，前后两个数就不再可比（§11.5.2 已经栽过一次）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    base_train = base_data / "train.jsonl"
    base_val = base_data / "val.jsonl"

    merged: list[dict] = []
    base_count = 0
    if mix_base and base_train.exists():
        for line in base_train.read_text(encoding="utf-8").splitlines():
            if line.strip():
                merged.append(json.loads(line))
        base_count = len(merged)
    merged.extend(records)

    (out_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in merged) + "\n", encoding="utf-8"
    )
    if base_val.exists():
        shutil.copy2(base_val, out_dir / "val.jsonl")

    actions = Counter(r["action"] for r in merged)
    meta = {
        "objective": "action_generation",
        "built_from": "教师模型成功轨迹（黑盒蒸馏）",
        "base_data": str(base_data),
        "base_train": base_count,
        "distilled": len(records),
        "train_total": len(merged),
        "val_copied_from": str(base_val),
        "coord_space": list(COORD_SPACE),
        # 样本里的 image 路径相对谁。训练必须从这个目录启动，否则图打不开。
        "image_paths_relative_to": str(Path.cwd()),
        "done_ratio": round(actions.get("done", 0) / len(merged), 4) if merged else 0.0,
        "action_dist": dict(actions),
        "distill_sessions": sorted({r["session_id"] for r in records}),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


def main() -> int:
    _console()
    parser = argparse.ArgumentParser(description="教师轨迹 → 学生训练样本")
    parser.add_argument("--archive", action="append", default=[], help="运行存档 JSON，可多次给")
    parser.add_argument("--trajectories", required=True, help="轨迹根目录（内含若干轨迹子目录）")
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--base-data", default=str(BASE_DATA))
    parser.add_argument(
        "--no-mix-base", action="store_true", help="只写蒸馏样本，不与基线训练集合并"
    )
    parser.add_argument("--done-cap", type=float, default=DONE_RATIO_CAP)
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写文件")
    args = parser.parse_args()

    root = Path(args.trajectories)
    if not root.is_dir():
        print(f"轨迹目录不存在：{root}")
        return 1

    verified: set[str] = set()
    for archive in args.archive:
        verified |= verified_trajectories(Path(archive))
    if not args.archive:
        print("⚠ 没给 --archive，**不做判定过滤**，所有轨迹都会被收下。")
        print("  这只适合调试；正式采集必须给存档，否则失败轮里的错误动作会进训练集。")

    traj_dirs = sorted(d for d in root.iterdir() if d.is_dir() and (d / "steps.jsonl").exists())
    out_dir = Path(args.out)

    print("=" * 62)
    print("教师轨迹蒸馏")
    print("=" * 62)
    print(f"  轨迹目录   {root}（{len(traj_dirs)} 条）")
    print(f"  判定通过   {len(verified)} 条" if verified else "  判定过滤   **未启用**")

    records, stats = collect(traj_dirs, verified, out_dir)
    records, deduped = dedupe(records, out_dir)
    before_cap = len(records)
    records = cap_done(records, args.done_cap)

    print(
        f"\n  收下样本   {len(records)}（去重丢 {deduped}，done 限量丢 {before_cap - len(records)}）"
    )
    print("\n  明细：")
    for reason, count in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f"    {reason:<28}{count:>5}")

    if args.dry_run:
        print("\n  --dry-run，未写文件。")
        return 0

    meta = write_dataset(records, out_dir, Path(args.base_data), mix_base=not args.no_mix_base)
    print(f"\n  写出 {out_dir}")
    print(f"    基线 {meta['base_train']} + 蒸馏 {meta['distilled']} = {meta['train_total']}")
    print(f"    done 占比 {meta['done_ratio']:.1%}")
    print("\n  下一步（超参一个都不要动，这是单变量实验）：")
    print(f"    python -m finetune.train_lora --data {out_dir} --epochs 2 --lr 1e-4")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())
