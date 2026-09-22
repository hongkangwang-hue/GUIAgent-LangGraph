"""提示词消融 —— 大纲 W5 任务 4「优化提示词工程，提升模型的任务执行准确率」。

## 三档比什么

| 档 | 模板 | 相对上一档多了什么 |
|---|---|---|
| P1 | `executor_v0` | —— 零样本基线 |
| P2 | `executor_v1` | **3 条 few-shot 示例** |
| P3 | `executor_v2` | **显式思维链**（在 `thinking` 里按三步推理） |

**三份模板的系统提示逐字相同**，差异只在 few-shot 段与 CoT 段。所以
P1→P2 的差就是 few-shot 的净效果，P2→P3 的差就是 CoT 的净效果。
这一点由 `tests/test_prompt_ablation.py` 钉住——**模板一改就会红**。

## 为什么在验证集上跑，而不是跑真实任务

跑真实任务要开虚拟机、每档 25 轮、每轮 5-10 分钟，三档就是一整天；
而验证集 142 条、一档十几分钟，不碰键鼠。

代价是它测的是**单步动作生成**，不是端到端成功率。但 §7 的端到端实测
已经证明这两者是连着的：验证集的联合命中率 30.3% 直接决定了
「单步任务做得成、多步做不成」。**单步指标改善，端到端才有机会改善。**

## 必须单独量的一项：照抄

2026-08-24 的探针里撞见 3B 会**逐字吐出 few-shot 示例的答案**，坐标是
示例里的常数 `(470, 750)`。这种输出：

- 格式 100% 合规 ✓
- 动作类型也对 ✓
- **坐标全错** ✗

`format_acc` 与 `action_acc` 一个都拦不住它。不单独报，few-shot 那一档
看起来会只有好处。所以 `eval.action` 里加了 `parrot_rate`。

## 用法

    # 基座模型跑三档（回答「不微调，光靠提示词能到多少」）
    python scripts/ablate_prompts.py --local Qwen/Qwen2.5-VL-3B-Instruct --load-4bit

    # 微调模型跑三档（回答「微调之后提示词还有多少空间」）
    python scripts/ablate_prompts.py --local Qwen/Qwen2.5-VL-3B-Instruct \\
        --adapter finetune/outputs/20260824-170900/adapter --load-4bit --prefix lora

    # 只汇总已有结果
    python scripts/ablate_prompts.py --report

## 不做的事

**不自动挑「最好的一档」。** 哪一档更好取决于看哪个指标——CoT 可能让
坐标更准但输出更长更慢，few-shot 可能让格式更好但引入照抄。
挑哪个是人的判断，脚本只负责把三列数摆齐。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "src")
)  # 交付包把各个包放在 src/ 下

#: 三档模板。顺序即阶梯，报告里按这个顺序排。
LADDER = [
    ("P1 零样本", "executor_v0"),
    ("P2 +few-shot", "executor_v1"),
    ("P3 +思维链", "executor_v2"),
]


def tag_of(prefix: str, template: str) -> str:
    """结果文件名。带 prefix 区分基座与微调两轮消融。"""
    return f"{prefix}-{template}" if prefix else template


def run_one(args, template: str, tag: str) -> Path:
    from eval.action import evaluate

    return evaluate(
        Path(args.val),
        tag=tag,
        provider=args.provider,
        model=args.model,
        local_model=args.local,
        adapter=args.adapter,
        limit=args.limit,
        resume=not args.no_resume,
        load_4bit=args.load_4bit,
        prompt=template,
    )


def report(prefix: str) -> None:
    """把三档并排摆出来。缺哪一档就说缺哪一档，不用 0 填。"""
    from eval.action import RESULT_DIR, summarize

    rows = []
    for label, template in LADDER:
        path = RESULT_DIR / f"{tag_of(prefix, template)}.jsonl"
        rows.append((label, template, summarize(path) if path.exists() else None))

    print()
    print("=" * 78)
    print(f"提示词消融{('（' + prefix + '）') if prefix else ''}")
    print("=" * 78)

    missing = [label for label, _, stats in rows if stats is None]
    if missing:
        print(f"  缺这几档：{', '.join(missing)} —— 下面的表不完整\n")

    header = f"{'指标':<22}" + "".join(f"{label:>18}" for label, _, _ in rows)
    print(header)
    print("-" * len(header))

    def line(name: str, fmt, key):
        cells = ""
        for _, _, stats in rows:
            cells += f"{'—':>18}" if stats is None else f"{fmt(key(stats)):>18}"
        print(f"{name:<22}{cells}")

    pct = lambda v: f"{v:.1%}"  # noqa: E731
    num = lambda v: f"{v:.1f}"  # noqa: E731

    line("样本数", lambda v: str(int(v)), lambda s: s["total"])
    line("格式合规率", pct, lambda s: s["format_acc"])
    line("动作类型准确率", pct, lambda s: s["action_acc"])
    line("坐标误差中位数", num, lambda s: s["median_distance"])
    line("联合命中 @25", pct, lambda s: s["joint"]["25"])
    line("联合命中 @50", pct, lambda s: s["joint"]["50"])
    line("联合命中 @100", pct, lambda s: s["joint"]["100"])
    line("**逐字照抄示例**", pct, lambda s: s.get("parrot_rate", 0.0))
    line("中位延迟(ms)", lambda v: f"{v:.0f}", lambda s: s["median_latency_ms"])

    print()
    print("  **P1→P2 的差 = few-shot 的净效果；P2→P3 的差 = 思维链的净效果。**")
    print("  三份模板的系统提示逐字相同，差异只在 few-shot 段与 CoT 段。")
    print()
    print("  「逐字照抄示例」那一行要和「格式合规率」一起看：照抄的输出格式")
    print("  100% 合规、动作类型也可能对，只有坐标是示例里的常数。")
    print("  这一项不单独报，few-shot 会看起来只有好处。")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="提示词三档消融（大纲 W5 任务 4）")
    parser.add_argument("--val", default="finetune/data/val.jsonl")
    parser.add_argument("--provider", default="", help="走 API 时的平台名")
    parser.add_argument("--model", default="", help="覆盖平台默认模型")
    parser.add_argument("--local", default="", help="走本地模型，给 HF 模型名或路径")
    parser.add_argument("--adapter", default="", help="LoRA adapter 目录")
    parser.add_argument("--load-4bit", action="store_true", help="与训练时一致的加载方式")
    parser.add_argument("--limit", type=int, default=0, help="每档只跑前 N 条，先探路用")
    parser.add_argument("--no-resume", action="store_true", help="不续跑，重测全部")
    parser.add_argument(
        "--prefix",
        default="",
        help="结果文件前缀，区分基座与微调两轮消融（如 --prefix lora）",
    )
    parser.add_argument("--report", action="store_true", help="只汇总已有结果，不跑")
    return parser


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args()

    if not args.report:
        if not (args.local or args.provider):
            raise SystemExit("要给 --local（本地模型）或 --provider（API 平台）之一")
        for index, (label, template) in enumerate(LADDER, start=1):
            print(f"\n{'#' * 78}\n# {index}/{len(LADDER)}  {label}  （{template}）\n{'#' * 78}")
            run_one(args, template, tag_of(args.prefix, template))

    report(args.prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
