"""把某一轮标记为「人为干预，不计入成功率」。

## 为什么要有这个

评测期间碰了鼠标、误按了键、或者中途接了个视频电话——那一轮的起点被
改过了，它既不是成功也不是失败。三种处理里只有一种是对的：

| 做法 | 后果 |
|---|---|
| 当成失败 | 低估能力 |
| 当成成功 | 高估能力 |
| **删掉** | **没人知道这批从 25 轮变成了 24 轮** |
| **标记** | ✅ 分子分母都不计入，但记录还在，理由也在 |

`RunRecord` 早就有 `excluded` / `exclusion_reason` 两个字段，
`run_basic_tasks.py` 也早就在统计时排除它们——**但一直没有工具把它们设上**，
只能手改 JSON。手改一个几百行的 JSON，改错了不会有任何提示。

2026-08-24 的实测期间用户碰了一次鼠标，才发现这个缺口。

## 用法

    # 先看这份存档里有哪些轮次
    python scripts/exclude_round.py docs/m2-runs/xxx.json --list

    # 标记「打开指定文件」第 3 次
    python scripts/exclude_round.py docs/m2-runs/xxx.json \\
        --task open_file --attempt 3 --reason "评测期间误触鼠标"

    # 反悔
    python scripts/exclude_round.py docs/m2-runs/xxx.json \\
        --task open_file --attempt 3 --undo

## 不做的事

**不删记录，不改 verified。** 这个脚本只动 `excluded` 与
`exclusion_reason` 两个字段，别的一个字节都不碰——一个会改评测数据的
工具，能改的范围越窄越好。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"找不到 {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def find(records: list[dict], task: str, attempt: int) -> dict | None:
    for record in records:
        if record.get("task") == task and record.get("attempt") == attempt:
            return record
    return None


def show(records: list[dict]) -> None:
    """列出所有轮次，标出已排除的。"""
    print(f"{'任务':<16}{'次':>3}  {'判定':<6}{'步数':>5}{'耗时':>8}  状态")
    print("-" * 74)
    for record in records:
        flags = []
        if not record.get("precondition_ok", True):
            flags.append("无效轮（起点未建立）")
        if record.get("excluded"):
            flags.append(f"已剔除：{record.get('exclusion_reason', '')}")
        print(
            f"{record.get('task', '?'):<16}"
            f"{record.get('attempt', 0):>3}  "
            f"{'✓' if record.get('verified') else '✗':<6}"
            f"{record.get('steps', 0):>5}"
            f"{record.get('duration_s', 0):>7.1f}s  " + "；".join(flags)
        )


def summarize(records: list[dict]) -> str:
    """按 `run_basic_tasks.py` 的同一口径重算成功率。

    **口径必须一致。** 这里算出来的数和那边不一样的话，报告里引用哪个都是错的。
    """
    valid = [r for r in records if r.get("precondition_ok", True) and not r.get("excluded")]
    if not valid:
        return "没有有效轮"
    ok = sum(1 for r in valid if r.get("verified"))
    return f"{ok}/{len(valid)} = {ok / len(valid):.0%}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="把某一轮标记为人为干预，不计入成功率")
    parser.add_argument("archive", help="存档 JSON，docs/m2-runs/ 下")
    parser.add_argument("--list", action="store_true", help="只列出轮次，不改动")
    parser.add_argument("--task", default="", help="任务名，如 open_file")
    parser.add_argument("--attempt", type=int, default=0, help="第几次")
    parser.add_argument("--reason", default="", help="为什么剔除。**必填**，见下")
    parser.add_argument("--undo", action="store_true", help="取消剔除标记")
    return parser


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args()
    path = Path(args.archive)
    payload = load(path)
    records = payload.get("records", [])

    if args.list or not args.task:
        show(records)
        print(f"\n当前成功率：{summarize(records)}")
        if not args.list:
            print("\n要剔除某一轮：--task <任务名> --attempt <次数> --reason <理由>")
        return 0

    if not args.attempt:
        raise SystemExit("要指定 --attempt（第几次）")

    record = find(records, args.task, args.attempt)
    if record is None:
        available = sorted({r.get("task", "") for r in records})
        raise SystemExit(f"这份存档里没有 {args.task} 第 {args.attempt} 次。有的任务：{available}")

    before = summarize(records)

    if args.undo:
        record["excluded"] = False
        record["exclusion_reason"] = ""
        action = "已取消剔除"
    else:
        # **理由是必填的。** 一个没有理由的剔除，三个月后没人说得清当时
        # 发生了什么——而"为什么少了一轮"恰恰是审稿人第一个会问的。
        if not args.reason.strip():
            raise SystemExit("必须给 --reason。没有理由的剔除等于偷偷改数据。")
        record["excluded"] = True
        record["exclusion_reason"] = args.reason.strip()
        action = "已剔除"

    # 剔除 = 事后确认「有人碰了鼠标」，是人工干预率的分子之一。
    # 存档里若带着安全指标（2026-09-17 之后的存档才有），一并刷新，否则它停在剔除之前的值。
    if "safety_metrics" in payload:
        from scripts.run_basic_tasks import safety_metrics

        payload["safety_metrics"] = safety_metrics(records)

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{action}：{record.get('title', args.task)} 第 {args.attempt} 次")
    if not args.undo:
        print(f"  理由：{record['exclusion_reason']}")
    print(f"  成功率 {before} → {summarize(records)}")
    print(f"  已写回 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
