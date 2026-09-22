"""分辨率消融 —— 「送图缩太狠」这个假设的**便宜筛选**。

## 假设从哪来

报告 §7.3 把离线版的端到端瓶颈定位到**坐标精度**：单次点击的联合命中率
只有 27.5%，需要连点 3-4 处的任务在概率上做不成。

而客机屏幕是 **2560×1600**，送模型前缩到 1024×768（`SessionConfig.image_size`），
**缩了 2.5 倍**——一个 20 像素的按钮到模型眼里只剩 8 像素。分辨率是影响
坐标精度的一个**我们能控**的变量（模型体量不能）。

## 但原来的实验设计是错的

§10 遗留项 5 写的是「1024×768 → 1536×1152，用历史帧数换分辨率」。
这个设计有两个问题：

1. **验证集测不了。** ScreenAgent 的截图**原生就是 1024×768**（训练 543 条、
   验证 142 条无一例外）。没有更高分辨率的源可喂，放大不产生细节。
2. **会把分辨率效果和分布偏移混在一起。** `image_size = (1024, 768)` 精确
   匹配训练分布；改成 1536×1152 等于把微调模型喂进一个**它从没见过的
   分辨率**，违反本项目一直在守的「训练与推理同构」。

所以那个设计只能在虚拟机里端到端跑（每档 25 轮、几小时），而且结果不可归因。

## 这个脚本做的是筛选，不是证明

反过来做：把验证集的图**降**到 0.75 / 0.5 / 0.375 倍，看坐标精度掉多少。

    ×1.00   1024×768   原生，与训练同构
    ×0.75    768×576
    ×0.50    512×384
    ×0.375   384×288

| 曲线形状 | 结论 |
|---|---|
| 平 | **假设当场死掉**——这个模型的坐标精度不吃分辨率，20 分钟省下几小时 |
| 陡 | 假设活着，值得去做端到端那个贵的实验 |

**必须说清楚它证不了什么**：它量的是**低于** 1024 的衰减，而客机的问题是
**高于** 1024 的细节被丢掉。两者方向相反。曲线陡只能说明「这个模型对
分辨率敏感」，不能直接推出「客机升到 1536 就会变好」。

便宜的筛选排在昂贵的验证前面——这是它全部的用途。

## 为什么真值不用跟着缩

真值坐标存的是**归一化 1000 空间**（`point_norm`），模型也被要求在 1000
空间作答。缩图**不动真值、不动输出口径**，差值干净地归给分辨率。

## 用法

    python scripts/ablate_resolution.py --local Qwen/Qwen2.5-VL-3B-Instruct \
        --adapter finetune/outputs/20260824-170900/adapter --load-4bit --prefix lora

    python scripts/ablate_resolution.py --report --prefix lora     # 只汇总
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "src")
)  # 交付包把各个包放在 src/ 下

#: (标签, 缩放倍数)。1.0 是原生分辨率，即与训练同构的那一档。
LADDER = [
    ("1024×768 原生", 1.0),
    ("768×576", 0.75),
    ("512×384", 0.5),
    ("384×288", 0.375),
]


def tag_of(prefix: str, scale: float) -> str:
    """结果文件名。`res100` / `res75` / `res50` / `res38`。"""
    stem = f"res{round(scale * 100):d}"
    return f"{prefix}-{stem}" if prefix else stem


def run_one(args, scale: float, tag: str) -> Path:
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
        prompt=args.prompt,
        image_scale=scale,
    )


def report(prefix: str) -> None:
    from eval.action import RESULT_DIR, summarize

    rows = []
    for label, scale in LADDER:
        path = RESULT_DIR / f"{tag_of(prefix, scale)}.jsonl"
        rows.append((label, scale, summarize(path) if path.exists() else None))

    print()
    print("=" * 82)
    print(f"分辨率消融{('（' + prefix + '）') if prefix else ''}")
    print("=" * 82)

    missing = [label for label, _, stats in rows if stats is None]
    if missing:
        print(f"  缺这几档：{', '.join(missing)} —— 下面的表不完整\n")

    header = f"{'指标':<22}" + "".join(f"{label:>16}" for label, _, _ in rows)
    print(header)
    print("-" * len(header))

    def line(name: str, fmt, key):
        cells = ""
        for _, _, stats in rows:
            cells += f"{'—':>16}" if stats is None else f"{fmt(key(stats)):>16}"
        print(f"{name:<22}{cells}")

    pct = lambda v: f"{v:.1%}"  # noqa: E731

    line("样本数", lambda v: str(int(v)), lambda s: s["total"])
    line("格式合规率", pct, lambda s: s["format_acc"])
    line("动作类型准确率", pct, lambda s: s["action_acc"])
    line("坐标误差中位数", lambda v: f"{v:.1f}", lambda s: s["median_distance"])
    line("联合命中 @25", pct, lambda s: s["joint"]["25"])
    line("联合命中 @50", pct, lambda s: s["joint"]["50"])
    line("联合命中 @100", pct, lambda s: s["joint"]["100"])
    line("中位延迟(ms)", lambda v: f"{v:.0f}", lambda s: s["median_latency_ms"])

    print()
    print("  **看「坐标误差中位数」和「联合命中 @50」这两行的斜率。**")
    print()
    print("  平  → 这个模型的坐标精度不吃分辨率，「送图缩太狠」的假设死掉。")
    print("  陡  → 假设活着，值得去做端到端那个贵的实验。")
    print()
    print("  **这一步证不了「客机升到 1536 会变好」。** 它量的是低于 1024 的")
    print("  衰减，客机的问题是高于 1024 的细节被丢掉——方向相反。")
    print("  它的用途只有一个：让便宜的筛选排在昂贵的验证前面。")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="分辨率消融（送图缩放对坐标精度的影响）")
    parser.add_argument("--val", default="finetune/data/val.jsonl")
    parser.add_argument("--provider", default="", help="走 API 时的平台名")
    parser.add_argument("--model", default="", help="覆盖平台默认模型")
    parser.add_argument("--local", default="", help="走本地模型，给 HF 模型名或路径")
    parser.add_argument("--adapter", default="", help="LoRA adapter 目录")
    parser.add_argument("--load-4bit", action="store_true", help="与训练时一致的加载方式")
    parser.add_argument("--prompt", default="", help="提示词模板名，留空用训练时的 SYSTEM_PROMPT")
    parser.add_argument("--limit", type=int, default=0, help="每档只跑 N 条（按 session 轮流取）")
    parser.add_argument("--no-resume", action="store_true", help="不续跑，重测全部")
    parser.add_argument("--prefix", default="", help="结果文件前缀，区分基座与微调两轮")
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
        for index, (label, scale) in enumerate(LADDER, start=1):
            print(f"\n{'#' * 82}\n# {index}/{len(LADDER)}  {label}  （×{scale}）\n{'#' * 82}")
            run_one(args, scale, tag_of(args.prefix, scale))

    report(args.prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
