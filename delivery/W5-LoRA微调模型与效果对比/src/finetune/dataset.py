"""把统一 schema 的样本转成**动作生成**训练格式 —— 对应大纲 W5。

## 训练目标是「动作生成」，不是「元素定位」

大纲 W5 任务 3 的原文：

> 对比微调前后模型在 **GUI 任务理解与动作生成** 上的效果

一条样本是「截图 + 任务目标 → 下一个动作（类型 + 坐标）」。

### 曾经走错过一次，记在这里

第一版按 grounding（元素描述 → 坐标）来构建，结果生成出这样的样本：

    在这张界面截图中找到：Find relevant information about von Neumann
    on the Internet。返回它的中心坐标。

「在互联网上查找冯·诺依曼的相关信息」是**任务目标**，不是元素描述——
让模型「找到这个」无从找起。569 条样本只有 135 个不重复描述，
「Insert a circle」出现 17 次对应 17 个不同坐标，**对 grounding 是矛盾监督**。

根因不是数据不好，是**我们把目标窄化错了**。ScreenAgent 提供的
（截图 + 任务 + 动作）三元组，正是大纲要的「动作生成」；
是我们改成 grounding 之后，数据才变得不匹配。改回大纲口径，数据立刻够用。

> 教训：**数据与指标不匹配时，先怀疑指标是不是自己改的**，
> 而不是先去找更多数据。

## 输出格式与推理时完全一致

    {"action": "left_click", "x": 500, "y": 251}

这正是 `ActionIntent` 解析的格式，也是 `executor_v1` 提示词要求模型输出的
格式。**训练与推理同构**，模型不必学两套。

## 坐标归一化到 [0,1000)

与 `SessionConfig.coordinate_space` 一致。实测 `qwen3-vl` 的原生输出就是
这个空间。训练与推理坐标系不一致时 loss 会正常下降而评测分数莫名其妙地低，
所以 `COORD_SPACE` 写死并被单测钉住。

## 只保留能执行的动作

ScreenAgent 的 934 条带坐标样本全是鼠标动作：

| 原始 | 条数 | 映射为 | 处理 |
|---|---:|---|---|
| click | 709 | `left_click` | 保留 |
| double_click | 105 | `double_click` | 保留 |
| move | 83 | `mouse_move` | 保留 |
| drag | 31 | — | **丢弃**：只有起点没有终点，构不成可执行的拖拽 |
| down / up | 6 | — | 丢弃：半个动作 |

**丢弃而不是硬凑。** 拿一个只有起点的拖拽去训练，教出来的是「拖拽=点一下」。

## 用法

    python -m finetune.dataset                    # 用默认冻结划分
    python -m finetune.dataset --check            # 只核对，不生成
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from data.schema import ACTION_REQUIREMENTS, training_action

logger = logging.getLogger(__name__)

#: 冻结划分。M3 开工第一件事是核对这个文件的指纹。
SPLIT_FILE = Path("docs/m3-prereq/split-screenagent-desktop.json")

#: 输出目录。
OUT_DIR = Path("finetune/data")

#: **训练坐标空间，与推理时必须一致。** 见模块 docstring。
COORD_SPACE = (1000, 1000)

#: 训练动作 → 输出 JSON 里除 `action` 外要带的键，以及从 `params` 的哪个键取值。
#:
#: **左边是推理时解析器认的名字，右边是装载器存的名字，两边不一样。**
#: `llm.parsing.PARAM_ALIASES` 会把 `key` 归一成 `keys`、`seconds` 归一成
#: `duration`——训练时就该直接输出归一后的名字，否则模型学的是别名，
#: 而**别名将来可能被改掉**。`test_训练输出能被真解析器读回来` 钉住这件事。
ANSWER_FIELDS: dict[str, tuple[tuple[str, str], ...]] = {
    "left_click": (),
    "double_click": (),
    "right_click": (),
    "mouse_move": (),
    "type": (("text", "text"),),
    "key": (("keys", "key"),),
    "wait": (("duration", "seconds"),),
    "scroll": (("direction", "direction"), ("amount", "repeat")),
}

#: 需要坐标的动作。与 `data.schema.ACTION_REQUIREMENTS` 里标了 `"point"` 的一致。
NEEDS_POINT = frozenset(name for name, needs in ACTION_REQUIREMENTS.items() if "point" in needs)


def normalize_point(x: int, y: int, size: tuple[int, int]) -> tuple[int, int]:
    """图像像素 → 归一化 [0,1000) 整数。

    **不做四舍五入到 1000。** 归一化空间是左闭右开的 [0,1000)，
    坐标恰好落在图像最右一列时 `x/width*1000` 会等于 1000，越界。
    夹到 999 而不是让它溢出——这类边界值在几千条样本里必然出现。
    """
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError(f"分辨率非法：{size}")
    nx = min(int(x / width * COORD_SPACE[0]), COORD_SPACE[0] - 1)
    ny = min(int(y / height * COORD_SPACE[1]), COORD_SPACE[1] - 1)
    return max(nx, 0), max(ny, 0)


def executable_action(sample) -> str:
    """样本对应哪个可执行动作。不可执行返回空串。

    **判定口径来自 `data.schema.training_action`，这里不再自己维护一份表。**
    2026-08-26 之前这里有 `ACTION_MAP` / `RAW_SUBTYPE_MAP` 两张表，而
    `data.split` 那边只看有没有坐标——两处口径不一致，池子里有的样本在这里
    被静默丢掉，两边的计数却都「看起来对」。

    `plan` 不在这里返回：它是规划器的训练目标，不是执行器能执行的动作。
    """
    action = training_action(sample)
    return action if action in ANSWER_FIELDS or action == "done" else ""


def task_of(sample) -> str:
    """这条样本的**指令**——即「这一步要干什么」。

    ## 2026-08-26：这里以前用的是会话级总目标

    原来返回 `sample.instruction`，而 ScreenAgent 的 `instruction` 恒为
    `task_prompt`（会话级总目标）。一个会话十几步共用同一句话，于是
    **716 条样本只对应 169 个不重复指令**，其中「Insert a circle」出现 17 次
    却对应 17 个不同坐标——对模型是矛盾监督。

    抽取在装载层完成（`data.loaders.screenagent._subtask_of`，那里才拿得到
    全部语言变体）；这里只负责取用。

    **取不到时返回空串，样本被丢弃**——不退回总目标。退回去等于把矛盾监督
    又放回训练集，而且从计数上完全看不出来。
    """
    return str((sample.meta or {}).get("subtask") or "").strip()


def build_answer(action: str, params: dict, point_norm: tuple[int, int] | None) -> str:
    """这条样本的训练目标（模型该输出的那串 JSON）。

    **格式与推理时解析器认的完全一致**，模型不必学两套。见 `ANSWER_FIELDS`
    关于「输出归一后的键名而不是别名」的说明。

    `done` 是唯一不带 `action` 的：它是完成信号不是动作，
    `llm.parsing` 走的也是 `DONE_KEYS` 那条分支。
    """
    if action == "done":
        return json.dumps({"done": True}, ensure_ascii=False)

    payload: dict = {"action": action}
    if point_norm is not None:
        payload["x"], payload["y"] = point_norm
    for out_key, src_key in ANSWER_FIELDS.get(action, ()):
        value = params.get(src_key)
        if value is not None:
            payload[out_key] = value
    return json.dumps(payload, ensure_ascii=False)


def build_record(sample) -> dict | None:
    """一条动作生成样本。缺关键字段或动作不可执行时返回 None。"""
    task = task_of(sample)
    if not task:
        return None

    action = executable_action(sample)
    if not action:
        return None

    if not sample.resolution or sample.resolution[0] <= 0:
        return None

    # 只有涉及坐标的动作才要坐标。**这里以前无条件要求 point 不为 None**，
    # 于是 `type` / `key` / `wait` / `done` 即使通过了上游的池子筛选，
    # 也会在这一行被第二次挡掉——两道门，改一道等于没改。
    point_norm: tuple[int, int] | None = None
    point = None
    if action in NEEDS_POINT:
        point = sample.point
        if point is None and sample.bbox is not None:
            point = sample.bbox.center
        if point is None:
            return None
        point_norm = normalize_point(point.x, point.y, sample.resolution)

    params = sample.params or {}
    answer = build_answer(action, params, point_norm)

    record = {
        "sample_id": sample.sample_id,
        # **一律写正斜杠。** Windows 上生成的 `data\raw\...` 拿到 Linux 服务器
        # 会被当成一个含反斜杠的文件名——图片一张都打不开，而报错指向
        # 「文件不存在」，离根因很远。正斜杠两边都认。
        "image": str(sample.screenshot_path).replace("\\", "/"),
        "resolution": list(sample.resolution),
        # 指令原样进提示词。训练时的问法与 executor 提示词一致
        "instruction": task,
        # 与 ActionIntent 的解析格式完全一致，模型不必学两套
        "answer": answer,
        "action": action,
        "session_id": str((sample.meta or {}).get("session_id", "")),
        "action_index": (sample.meta or {}).get("action_index"),
        "source": sample.source_dataset,
    }
    if point_norm is not None:
        record["point_norm"] = list(point_norm)
        record["point_px"] = [point.x, point.y]
    if params:
        record["params"] = params
    if sample.bbox is not None:
        record["bbox_px"] = [
            sample.bbox.left,
            sample.bbox.top,
            sample.bbox.right,
            sample.bbox.bottom,
        ]
    return record


def convert(
    split_file: Path = SPLIT_FILE,
    out_dir: Path = OUT_DIR,
    dataset: str = "screenagent",
    check_only: bool = False,
) -> dict:
    """按冻结划分生成 train/val 两个 JSONL。返回统计。"""
    from data.loaders import LOADERS
    from data.split import FrozenSplit

    if not split_file.exists():
        raise SystemExit(f"找不到冻结划分 {split_file}。\n先跑 python scripts/prepare_datasets.py")

    # 指纹不符会在这里抛，**不要吞掉它** —— 拿一份被改过的划分去训练，
    # 后面所有结论都不可信，而且事后查不出来
    split = FrozenSplit.read(split_file)
    leak = split.leakage()
    if leak:
        raise SystemExit(f"划分有泄漏：{len(leak)} 个样本同时在 train 与 val。拒绝生成。")

    loader_cls = LOADERS.get(dataset)
    if loader_cls is None:
        raise SystemExit(f"没有名为 {dataset!r} 的装载器。可用：{'、'.join(LOADERS)}")
    loader = loader_cls()
    ready, why = loader.available()
    if not ready:
        raise SystemExit(f"{dataset} 数据未就绪：{why}")

    wanted = {"train": set(split.train_ids), "val": set(split.val_ids)}
    buckets: dict[str, list] = {"train": [], "val": []}
    dropped: Counter = Counter()
    actions: Counter = Counter()

    for sample in loader.load():
        target = None
        for name, ids in wanted.items():
            if sample.sample_id in ids:
                target = name
                break
        if target is None:
            continue
        record = build_record(sample)
        if record is None:
            dropped[target] += 1
            continue
        actions[record["action"]] += 1
        buckets[target].append(record)

    stats = {
        "objective": "action_generation",
        "split_file": str(split_file),
        "fingerprint": split.fingerprint(),
        "seed": split.seed,
        "group_by": split.group_by,
        "expected": {"train": split.train_size, "val": split.val_size},
        "built": {k: len(v) for k, v in buckets.items()},
        "dropped": dict(dropped),
        "action_dist": dict(actions),
        "coord_space": list(COORD_SPACE),
    }

    if check_only:
        return stats

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, records in buckets.items():
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        stats.setdefault("files", {})[name] = str(path)

    (out_dir / "meta.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return stats


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="生成动作生成微调数据（大纲 W5）")
    parser.add_argument("--split", default=str(SPLIT_FILE))
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--dataset", default="screenagent")
    parser.add_argument("--check", action="store_true", help="只核对划分与条数，不写文件")
    args = parser.parse_args()

    stats = convert(
        split_file=Path(args.split),
        out_dir=Path(args.out),
        dataset=args.dataset,
        check_only=args.check,
    )

    print("=" * 70)
    print("动作生成微调数据构建")
    print("=" * 70)
    print("  目标      动作生成（截图 + 任务 → 动作类型 + 坐标），对应大纲 W5")
    print(f"  划分文件  {stats['split_file']}")
    print(f"  指纹      {stats['fingerprint']}    种子 {stats['seed']}")
    print(f"  划分单位  {stats['group_by']}")
    print(f"  坐标空间  {stats['coord_space'][0]}×{stats['coord_space'][1]}")
    print()
    for name in ("train", "val"):
        expected = stats["expected"][name]
        built = stats["built"][name]
        drop = stats["dropped"].get(name, 0)
        mark = "" if built + drop == expected else "  **与划分条数不符**"
        print(f"  {name:<6}期望 {expected:>4}    生成 {built:>4}    丢弃 {drop:>3}{mark}")

    if stats["action_dist"]:
        print("\n  动作分布：")
        total = sum(stats["action_dist"].values())
        for name, count in sorted(stats["action_dist"].items(), key=lambda kv: -kv[1]):
            print(f"    {name:<16}{count:>5}  {count / total:>6.1%}")
        print("\n  **类别极不均衡是可预期的**——真实 GUI 轨迹里点击本来就占绝大多数。")
        print("  报告里要按动作类型分别给准确率，只报总体会被 click 一类主导。")

    if stats["dropped"]:
        dropped = stats["dropped"]
        total = sum(dropped.values()) if isinstance(dropped, dict) else dropped
        print(f"\n  丢弃 {total} 条，绝大部分是 `plan` 样本——")
        print("  它们不是执行器能执行的动作，是**规划器**的训练目标（「总目标 → 子任务列表」），")
        print("  记录形状不同，由单独的构建流程处理。其余是拖拽（只有起点没有终点）与 down/up。")
        print("  **丢弃而不是硬凑**：拿只有起点的拖拽去训练，教出来的是「拖拽=点一下」。")

    if not args.check:
        print(f"\n  已写入 {args.out}/train.jsonl、val.jsonl、meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
