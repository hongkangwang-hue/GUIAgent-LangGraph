"""错误恢复率与 Reflector 判定的分项统计 —— M4 验收标准 3 与 5。

## 为什么这两条能靠算，不用再跑

M4 清单把它们标成「未开始」，但**数据早就有了**：138 次 Reflector 判定、
57 次否决及其后续步骤，全都在轨迹的 `meta["reflector"]` 与 `meta["change"]` 里。
缺的只是把它们算出来。

跑一轮客机要半天，算一遍几秒钟。**先把已有数据榨干，再决定要不要跑新的**。

## 标准 5：恢复率必须按错误类型分项

一个笼统的「恢复率 X%」没有用。M4 的错误分类把失败分成了 A/B 两类：

    A 类  动作执行了、屏幕变了，但方向不对（走错路）
    B 类  动作执行成功、屏幕纹丝不动（点空了）

**两类的可恢复性完全不同**：B 类换个动作类型（单击→双击）就可能救回来，
A 类需要模型自己改主意，重试多少次都一样。合成一个数会把这个区别抹掉——
而这个区别正是决定下一步该改什么的依据。

## 一个必须避开的口径陷阱

「Reflector 否决之后这个子任务最终成功了」**不等于「Reflector 救了它」**。
模型可能自己就改对了，与否决无关。所以这里报的是**相关性**，措辞是
「否决后最终成功的比例」，不是「恢复率」——真正的因果要靠开关 A/B 对照。

本项目已经在这个坑里栽过一次：Reflector 的「恢复率 90%」把安全阀兜底
算成了恢复，真实值是 0/30。所以这个脚本**只报能算的，不推断因果**。

## 标准 3：一致率没法纯算，这里只备料

标准 3 要的是「Reflector 判定与人工判定的一致率」——人工那一半得人来标。
本脚本导出待标注清单（`--export-labels`），把 `done` 时刻的截图路径、
子任务、Reflector 结论列成表，人填第四列。

用法：

    python scripts/analyze_recovery.py --trajectories outputs/trajectories
    python scripts/analyze_recovery.py --trajectories outputs/trajectories --export-labels 待标注.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

#: 错误类型，与 `docs/m4-错误分类体系.md` 的 A/B 分类一致。
ERROR_CLASSES = {
    "A": "走错路（执行了、屏幕变了、方向不对）",
    "B": "点空了（执行成功、屏幕没变化）",
    "unknown": "判定不了（没有帧差记录）",
}


def _console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


@dataclass
class Rejection:
    """一次 Reflector 否决，以及它之后发生了什么。"""

    trajectory: str
    subtask_id: int
    step: int
    subtask: str
    #: 这一步属于哪类错误。按帧差分：屏幕没变是 B，变了是 A
    error_class: str
    #: 同一子任务里，这次否决之后还有没有步骤
    had_following_steps: bool
    #: 该子任务最终是怎么结束的（成功 / 步数用尽 / 其他）
    subtask_outcome: str
    screenshot: str = ""


@dataclass
class Analysis:
    rejections: list[Rejection] = field(default_factory=list)
    judgements: Counter = field(default_factory=Counter)
    trajectories: int = 0
    steps: int = 0

    def by_class(self) -> dict:
        """按错误类型分项统计「否决后最终成功」的比例。"""
        buckets: dict[str, dict] = {}
        for name in ERROR_CLASSES:
            subset = [r for r in self.rejections if r.error_class == name]
            if not subset:
                continue
            succeeded = sum(1 for r in subset if r.subtask_outcome == "success")
            buckets[name] = {
                "n": len(subset),
                "后续成功": succeeded,
                "比例": round(succeeded / len(subset), 4),
                "说明": ERROR_CLASSES[name],
            }
        return buckets


def classify(step: dict) -> str:
    """这一步属于哪类错误。**没有帧差记录时返回 unknown，不猜。**

    把「判定不了」算进 A 或 B，都会让那一类的数字虚高，而这个脚本的
    全部意义就是把 A/B 分开看。
    """
    change = (step.get("meta") or {}).get("change")
    if not change:
        return "unknown"
    return "A" if change.get("changed") else "B"


def load_steps(traj_dir: Path) -> list[dict]:
    path = traj_dir / "steps.jsonl"
    if not path.exists():
        return []
    steps = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                steps.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return steps


def subtask_outcome(steps: list[dict], subtask_id: int) -> str:
    """这个子任务最终怎么结束的。

    判据是子任务最后一步的 `stop_reason` / `execution_status`——
    轨迹里没有「子任务成功」这个字段，只能从收尾形态推。
    **推不出来时返回 unknown**，不默认算失败：默认值会悄悄改变分母。
    """
    same = [s for s in steps if s.get("subtask_id") == subtask_id]
    if not same:
        return "unknown"
    last = same[-1]
    if (last.get("action_intent") or {}).get("done") is True:
        return "success"
    status = last.get("execution_status") or ""
    if status in ("blocked", "error"):
        return status
    return "exhausted"


def analyze(traj_dirs: list[Path]) -> Analysis:
    result = Analysis()
    for traj_dir in traj_dirs:
        steps = load_steps(traj_dir)
        if not steps:
            continue
        result.trajectories += 1
        result.steps += len(steps)

        for index, step in enumerate(steps):
            verdict = (step.get("meta") or {}).get("reflector") or {}
            decision = verdict.get("decision") or verdict.get("verdict") or ""
            if decision:
                result.judgements[decision] += 1
            if decision != "reject":
                continue

            subtask_id = step.get("subtask_id", 0)
            result.rejections.append(
                Rejection(
                    trajectory=traj_dir.name,
                    subtask_id=subtask_id,
                    step=step.get("step", 0),
                    subtask=(step.get("subtask") or "")[:40],
                    error_class=classify(step),
                    had_following_steps=index < len(steps) - 1,
                    subtask_outcome=subtask_outcome(steps, subtask_id),
                    screenshot=step.get("screenshot_before") or "",
                )
            )
    return result


def export_labels(traj_dirs: list[Path], out: Path) -> int:
    """导出待人工标注的 `done` 时刻清单 —— 标准 3 的备料。

    人要填的只有一列：这一刻子任务**真的**完成了吗。
    其余列都从轨迹里带出来，减少来回翻。
    """
    rows = []
    for traj_dir in traj_dirs:
        for step in load_steps(traj_dir):
            if (step.get("action_intent") or {}).get("done") is not True:
                continue
            verdict = (step.get("meta") or {}).get("reflector") or {}
            rows.append(
                {
                    "轨迹": traj_dir.name,
                    "步": step.get("step", 0),
                    "子任务": step.get("subtask", ""),
                    "截图": step.get("screenshot_before", ""),
                    "Reflector 结论": verdict.get("decision") or verdict.get("verdict") or "未启用",
                    "人工判定（请填 是/否）": "",
                }
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["轨迹", "步", "子任务", "截图", "Reflector 结论", "人工判定（请填 是/否）"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    _console()
    parser = argparse.ArgumentParser(description="恢复率与 Reflector 判定统计")
    parser.add_argument("--trajectories", default="outputs/trajectories")
    parser.add_argument("--export-labels", default="", help="导出待人工标注的 done 清单")
    args = parser.parse_args()

    root = Path(args.trajectories)
    if not root.is_dir():
        print(f"轨迹目录不存在：{root}")
        return 1
    traj_dirs = sorted(d for d in root.iterdir() if d.is_dir())

    result = analyze(traj_dirs)
    print("=" * 62)
    print("M4 标准 5：否决之后最终成功的比例（按错误类型分项）")
    print("=" * 62)
    print(f"  轨迹 {result.trajectories} 条    步骤 {result.steps} 步")
    print(f"  Reflector 判定 {sum(result.judgements.values())} 次：", dict(result.judgements))

    buckets = result.by_class()
    if not buckets:
        print("\n  没有否决记录。这一轮可能没开 --reflector。")
    else:
        print(f"\n  {'类型':<10}{'否决数':>8}{'后续成功':>10}{'比例':>10}  说明")
        for name, stats in buckets.items():
            print(
                f"  {name:<10}{stats['n']:>8}{stats['后续成功']:>10}"
                f"{stats['比例']:>9.1%}  {stats['说明']}"
            )
        print("\n  ⚠ 这是**相关性不是因果**：模型可能自己改对了，与否决无关。")
        print("    要因果得做开关 A/B 对照。本项目在这里栽过一次——")
        print("    「恢复率 90%」把安全阀兜底算成了恢复，真实值是 0/30。")

    if args.export_labels:
        count = export_labels(traj_dirs, Path(args.export_labels))
        print(f"\n  标准 3 备料：导出 {count} 条待标注记录 → {args.export_labels}")
        print("  人工只需填最后一列：这一刻子任务真的完成了吗。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
