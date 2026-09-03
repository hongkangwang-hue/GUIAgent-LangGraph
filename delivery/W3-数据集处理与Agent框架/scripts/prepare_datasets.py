"""数据集预处理流水线 —— M2 交付物 1、2 与 M3 前置件①。

一条命令从原始数据走到统一 JSONL、四张图表、统计报告和冻结的划分文件。

## 用法

    python scripts/prepare_datasets.py --download        # 下载 / clone 三个数据集
    python scripts/prepare_datasets.py --extract         # 解压两个 zip（1.3GB + 50MB）
    python scripts/prepare_datasets.py                   # 装载 → 清洗 → 统计 → 出图 → 冻结划分
    python scripts/prepare_datasets.py --no-charts       # 只出数字，不画图

`--download` 与 `--extract` 分开，是因为解压 ScreenSpot-v2 的图片包要写
1.3GB 到磁盘，这种动作不该混在别的步骤里悄悄发生。

## 三个数据集的获取方式各不相同

| 数据集 | 来源 | 备注 |
|---|---|---|
| ScreenSpot | HF `rootsautomation/ScreenSpot` | parquet，截图内嵌 |
| ScreenSpot-v2 | HF `OS-Copilot/ScreenSpot-v2` | JSON + 1.3GB 图片 zip |
| ScreenAgent | **GitHub** `niuzaisheng/ScreenAgent` | **HF 上只有权重没有数据** |

最后一条容易踩：按 "ScreenAgent" 在 HF 上搜数据集是搜不到的。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "src")
)  # 交付包把各个包放在 src/ 下

from data.clean import DEFAULT_RULES, clean  # noqa: E402
from data.loaders import build_all  # noqa: E402
from data.schema import Platform, training_action, write_jsonl  # noqa: E402
from data.split import freeze_split, training_pool  # noqa: E402
from data.stats import collect, collect_by_dataset, overlap  # noqa: E402

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
FIGURES = Path("docs/figures")
PREREQ = Path("docs/m3-prereq")
REPORT = Path("docs/数据集统计分析报告.md")

HF_DATASETS = {
    "rootsautomation/ScreenSpot": RAW / "screenspot",
    "OS-Copilot/ScreenSpot-v2": RAW / "screenspot_v2",
}
SCREENAGENT_GIT = "https://github.com/niuzaisheng/ScreenAgent.git"


def _console() -> None:
    """Windows 控制台默认 GBK，✓ 与中文之外的符号会直接抛 UnicodeEncodeError。

    本项目在 M1 因此崩过三次（`StepRecord.summary()`、模型回复里的表情、
    `verify_control.py` 的汇总行），一律改成先把 stdout 转成 UTF-8。
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


# ===================================================================== #
# 下载与解压
# ===================================================================== #


def download() -> None:
    from huggingface_hub import snapshot_download

    for repo_id, target in HF_DATASETS.items():
        if target.exists() and any(target.iterdir()):
            print(f"[跳过] {repo_id} 已存在于 {target}")
            continue
        print(f"[下载] {repo_id} → {target}")
        snapshot_download(repo_id, repo_type="dataset", local_dir=str(target))

    repo = RAW / "screenagent_repo"
    if repo.exists():
        print(f"[跳过] ScreenAgent 已 clone 到 {repo}")
    else:
        print(f"[clone] {SCREENAGENT_GIT} → {repo}")
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", SCREENAGENT_GIT, str(repo)], check=True)


def extract() -> None:
    jobs = [
        (
            RAW / "screenspot_v2" / "screenspotv2_image.zip",
            RAW / "screenspot_v2",
            RAW / "screenspot_v2" / "screenspotv2_image",
        ),
        (
            RAW / "screenagent_repo/data/ScreenAgent/test.zip",
            RAW / "screenagent_repo/data/ScreenAgent",
            RAW / "screenagent_repo/data/ScreenAgent/test",
        ),
    ]
    for archive, target, marker in jobs:
        if marker.exists():
            print(f"[跳过] 已解压 {marker}")
            continue
        if not archive.exists():
            print(f"[缺失] {archive}，先跑 --download")
            continue
        size_mb = archive.stat().st_size / 1e6
        print(f"[解压] {archive.name}（{size_mb:.0f}MB）→ {target}")
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(target)


# ===================================================================== #
# 主流程
# ===================================================================== #


def build(with_charts: bool = True) -> int:
    print("=" * 66)
    print("装载")
    print("=" * 66)
    results = build_all(root=str(RAW))
    samples = []
    for result in results:
        if not result.available:
            print(f"  {result.dataset:<15} 未就绪：{result.reason}")
            continue
        skipped = "，".join(f"{k} {v}" for k, v in result.skipped.items()) or "无"
        print(f"  {result.dataset:<15} {len(result.samples):>6} 条   跳过：{skipped}")
        samples.extend(result.samples)

    if not samples:
        print("\n没有任何可用样本。先跑 --download 与 --extract。")
        return 1

    print(f"\n合计 {len(samples)} 条")

    print("\n" + "=" * 66)
    print("清洗")
    print("=" * 66)
    kept, report = clean(samples, DEFAULT_RULES)
    for line in report.summary_lines():
        print("  " + line)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    written = write_jsonl(kept, PROCESSED / "unified.jsonl")
    print(f"\n统一样本已写入 {PROCESSED / 'unified.jsonl'}（{written} 条）")

    print("\n" + "=" * 66)
    print("统计")
    print("=" * 66)
    by_dataset = collect_by_dataset(kept)
    overall = collect(kept, name="all")
    for name, stats in by_dataset.items():
        print(
            f"  {name:<15} {stats.total:>6} 条  截图 {stats.image_count:>5} 张  "
            f"带框 {stats.with_bbox:>5}  带点 {stats.with_point:>5}"
        )

    by_platform = {
        p.value: collect([s for s in kept if s.platform is p], name=p.value) for p in Platform
    }

    # 泄漏检查：训练来源 vs 评测集
    overlaps = []
    train_source = [s for s in kept if s.source_dataset == "screenagent"]
    for evalset in ("screenspot", "screenspot_v2"):
        target = [s for s in kept if s.source_dataset == evalset]
        if train_source and target:
            overlaps.append(overlap("screenagent", train_source, evalset, target))
    v1 = [s for s in kept if s.source_dataset == "screenspot"]
    v2 = [s for s in kept if s.source_dataset == "screenspot_v2"]
    if v1 and v2:
        overlaps.append(overlap("screenspot", v1, "screenspot_v2", v2))

    print("\n  截图级重叠：")
    for item in overlaps:
        print(
            f"    {item.left} ∩ {item.right}: {item.shared_images} 张"
            f"（占 {item.left} 的 {item.left_leaked_ratio:.1%}，"
            f"占 {item.right} 的 {item.right_leaked_ratio:.1%}）"
        )

    print("\n" + "=" * 66)
    print("M3 前置件①：三分划定与冻结")
    print("=" * 66)
    pool = training_pool(kept)
    print(f"  动作生成可用池：{len(pool)} 条（ScreenAgent train 划分中字段齐全的桌面样本）")
    breakdown = Counter(training_action(s) for s in pool)
    print("    " + "  ".join(f"{k} {v}" for k, v in breakdown.most_common()))
    split = freeze_split(pool)
    print(
        f"  训练集 {split.train_size} 条 / {len(split.train_groups)} 个会话；"
        f"验证集 {split.val_size} 条 / {len(split.val_groups)} 个会话"
    )
    print(f"  种子 {split.seed}，指纹 {split.fingerprint()}")
    for note in split.notes:
        print(f"  [注意] {note}")
    split_path = split.write(PREREQ / "split-screenagent-desktop.json")
    print(f"  划分已冻结到 {split_path}")

    figures = []
    if with_charts:
        print("\n" + "=" * 66)
        print("图表")
        print("=" * 66)
        from data.charts import render_all

        figures = render_all(by_dataset, by_platform, FIGURES)
        for path in figures:
            print(f"  {path}")

    write_report(overall, by_dataset, by_platform, report, overlaps, split, figures)
    print(f"\n报告已写入 {REPORT}")
    return 0


# ===================================================================== #
# 报告
# ===================================================================== #


def write_report(overall, by_dataset, by_platform, cleaning, overlaps, split, figures) -> None:
    from data.charts import DISPLAY_NAMES, PLATFORM_NAMES
    from data.split import TARGET_TRAIN_MIN, TARGET_VAL

    lines: list[str] = []
    add = lines.append

    add("# 数据集统计分析报告")
    add("")
    add("> 由 `python scripts/prepare_datasets.py` 生成。对应 M2 交付物 1、2")
    add("> 与验收标准 11。**本文件是生成物，改它没有意义，改脚本。**")
    add("")

    add("## 1. 总览")
    add("")
    add("| 数据集 | 样本 | 不重复截图 | 带边界框 | 带坐标点 | 划分 |")
    add("|---|---:|---:|---:|---:|---|")
    for name, stats in by_dataset.items():
        splits = "、".join(f"{k} {v}" for k, v in sorted(stats.splits.items())) or "—"
        add(
            f"| {DISPLAY_NAMES.get(name, name)} | {stats.total:,} | {stats.image_count:,} "
            f"| {stats.with_bbox:,} | {stats.with_point:,} | {splits} |"
        )
    add(
        f"| **合计** | **{overall.total:,}** | **{overall.image_count:,}** "
        f"| **{overall.with_bbox:,}** | **{overall.with_point:,}** | |"
    )
    add("")

    add("## 2. 清洗")
    add("")
    for line in cleaning.summary_lines():
        add(f"- {line.strip()}" if not line.startswith("  ") else line)
    add("")
    add("清洗规则与理由：")
    add("")
    add("| 规则 | 理由 |")
    add("|---|---|")
    for rule in DEFAULT_RULES:
        add(f"| `{rule.name}` | {rule.why} |")
    add("")

    add("## 3. 平台分布")
    add("")
    add("| 平台 | 样本 | 占比 |")
    add("|---|---:|---:|")
    for key in ("desktop", "web", "mobile", "unknown"):
        stats = by_platform.get(key)
        if not stats or not stats.total:
            continue
        add(
            f"| {PLATFORM_NAMES.get(key, key)} | {stats.total:,} "
            f"| {stats.total / overall.total:.1%} |"
        )
    add("")

    desktop = by_platform.get("desktop")
    if desktop and desktop.area_ratios:
        pcts = desktop.percentiles()
        add("## 4. 元素尺寸（桌面样本，面积占屏幕比例）")
        add("")
        add("| 分位 | 面积占比 |")
        add("|---|---:|")
        for p, value in pcts.items():
            add(f"| p{int(p)} | {value:.4%} |")
        add("")
        add(
            f"桌面样本中 **{desktop.tiny_ratio:.1%}** 的目标元素小于屏幕面积的 0.05%"
            "（约合 1920×1080 上的 32×32 图标）。这是 grounding 精度要求的直接来源。"
        )
        add("")

    add("## 5. 截图级重叠（M3 泄漏风险）")
    add("")
    add("| 左 | 右 | 共有截图 | 占左 | 占右 |")
    add("|---|---|---:|---:|---:|")
    for item in overlaps:
        add(
            f"| {DISPLAY_NAMES.get(item.left, item.left)} "
            f"| {DISPLAY_NAMES.get(item.right, item.right)} | {item.shared_images:,} "
            f"| {item.left_leaked_ratio:.1%} | {item.right_leaked_ratio:.1%} |"
        )
    add("")

    add("## 6. M3 前置件①：三分划定")
    add("")
    add(f"- grounding 可用池：**{split.pool_size:,} 条**")
    add(f"- 训练集 **{split.train_size:,} 条** / {len(split.train_groups)} 个会话")
    add(f"- 验证集 **{split.val_size:,} 条** / {len(split.val_groups)} 个会话（请求 {TARGET_VAL}）")
    add(f"- 随机种子 `{split.seed}`，划分指纹 `{split.fingerprint()}`")
    add("- 划分单位：**会话**，不是样本（同一截图不跨集合）")
    add(f"- 大纲目标训练集下限 {TARGET_TRAIN_MIN:,} 条")
    add("")
    for note in split.notes:
        add(f"> ⚠ {note}")
        add("")

    if figures:
        add("## 7. 图表")
        add("")
        for path in figures:
            rel = Path(path).as_posix().replace("docs/", "")
            add(f"![{Path(path).stem}]({rel})")
            add("")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    _console()
    parser = argparse.ArgumentParser(description="数据集预处理流水线")
    parser.add_argument("--download", action="store_true", help="下载 / clone 三个数据集")
    parser.add_argument("--extract", action="store_true", help="解压图片包与 test.zip")
    parser.add_argument("--no-charts", action="store_true", help="跳过绘图")
    args = parser.parse_args()

    if args.download:
        download()
    if args.extract:
        extract()
    if args.download or args.extract:
        print()

    return build(with_charts=not args.no_charts)


if __name__ == "__main__":
    raise SystemExit(main())
