"""4 张统计图表（M2 交付物 2）。

## 配色不是随手选的

三个数据集用同一套定序分类色的前三槽——蓝 `#2a78d6`、橙 `#eb6834`、
青 `#1baf7a`。这三槽是经色觉障碍模拟校验过的组合（全配对 CVD ΔE ≥ 9.2，
常视觉 ΔE ≥ 24.0），三条序列同屏时任意两两都分得开。

**颜色按数据集固定分配，不按排名循环。** 图 1 里 screenspot 排第一、
图 2 里可能排第三，但它在四张图里始终是蓝色，读者不用每张图重新认一遍图例。

青色在浅色底上对比度低于 3:1，所以每根条形都直接标了数值——不靠颜色
本身传递量值。

## 为什么面积用相对值和对数轴

数据集分辨率从 1024×768 跨到 2560×1600，同样 32×32 的图标在两端占屏比例
差 4 倍。直接画绝对像素面积，得到的是"分辨率分布"的影子而不是元素大小。
面积本身又是长尾——从一个复选框到整块内容区跨了四五个数量级，线性轴上
99% 的样本会挤在最左边一格。
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from pathlib import Path

from data.stats import TINY_ELEMENT_RATIO, DatasetStats

logger = logging.getLogger(__name__)

#: 定序分类色，按数据集固定分配。见模块文档。
SERIES_COLORS: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a")

#: 中性色，用于网格、坐标轴与文字。数值文字一律用墨色而非序列色。
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

#: 数据集的展示名。原始注册名对读者没有意义
DISPLAY_NAMES = {
    "screenspot": "ScreenSpot",
    "screenspot_v2": "ScreenSpot-v2",
    "screenagent": "ScreenAgent",
}

PLATFORM_NAMES = {
    "desktop": "桌面",
    "mobile": "移动",
    "web": "网页",
    "unknown": "未标注",
}

ACTION_NAMES = {
    "locate": "定位（元素标注）",
    "click": "单击",
    "double_click": "双击",
    "right_click": "右键",
    "drag": "拖拽",
    "scroll": "滚动",
    "type": "输入文本",
    "key": "按键",
    "wait": "等待",
    "other": "其他（规划 / 评估等）",
}


def apply_style() -> None:
    """全局样式。中文字体必须显式设置，否则所有中文标签变成方框。"""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            # Windows 自带，M0 的环境文档已限定开发机为 Windows
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            # 中文字体的 minus 字形缺失，不关会让负号显示成方框
            "axes.unicode_minus": False,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.grid": False,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "figure.autolayout": False,
        }
    )


def _color_for(dataset: str, order: Sequence[str]) -> str:
    """颜色随数据集走，与它在当前图里的排名无关。"""
    try:
        return SERIES_COLORS[order.index(dataset) % len(SERIES_COLORS)]
    except ValueError:
        return INK_MUTED


def _tidy(ax, *, xgrid: bool = True) -> None:
    """去掉顶右边框，网格退到数据后面。"""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    if xgrid:
        ax.xaxis.grid(True, linewidth=0.8, color=GRID)
        ax.set_axisbelow(True)


def _stacked_barh(
    ax,
    categories: Sequence[str],
    per_dataset: dict[str, list[int]],
    order: Sequence[str],
    *,
    label_threshold: float = 0.04,
) -> None:
    """横向堆叠条形。类别多、名字长时横向比纵向好读。

    段间留 2px 表面色间隙，让相邻两段即便颜色相近也不会糊在一起。
    """
    positions = range(len(categories))
    left = [0.0] * len(categories)
    total = [sum(per_dataset[d][i] for d in order) for i in range(len(categories))]
    grand = max(total) if total else 1

    for dataset in order:
        values = per_dataset[dataset]
        ax.barh(
            list(positions),
            values,
            left=left,
            height=0.62,
            color=_color_for(dataset, order),
            label=DISPLAY_NAMES.get(dataset, dataset),
            edgecolor=SURFACE,
            linewidth=1.6,
        )
        # 段内直接标数值：青色槽在浅底上对比度不足 3:1，量值不能只靠颜色
        for index, value in enumerate(values):
            if value and value / grand >= label_threshold:
                ax.text(
                    left[index] + value / 2,
                    index,
                    f"{value:,}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=SURFACE,
                    fontweight="bold",
                )
            left[index] += value

    # 行尾标总计
    for index, value in enumerate(total):
        ax.text(
            value + grand * 0.012,
            index,
            f"{value:,}",
            ha="left",
            va="center",
            fontsize=9,
            color=INK_SECONDARY,
        )

    ax.set_yticks(list(positions))
    ax.set_yticklabels(categories, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, grand * 1.14)
    _tidy(ax)


def _save(fig, out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.png"
    fig.savefig(path)
    import matplotlib.pyplot as plt

    plt.close(fig)
    logger.info("图表已保存 %s", path)
    return path


# ===================================================================== #
# 四张图
# ===================================================================== #


def chart_resolutions(by_dataset: dict[str, DatasetStats], out_dir: Path, top_n: int = 12) -> Path:
    """图 1：截图分辨率分布。

    只画出现最多的 top_n 个。分辨率是离散标签而不是连续量，所以用条形
    而不是直方图——"1024×768 与 1920×1080 之间"没有意义。
    """
    import matplotlib.pyplot as plt

    order = [name for name in DISPLAY_NAMES if name in by_dataset] or list(by_dataset)
    totals: dict[str, int] = {}
    for stats in by_dataset.values():
        for res, count in stats.resolutions.items():
            totals[res] = totals.get(res, 0) + count
    top = [res for res, _ in sorted(totals.items(), key=lambda kv: -kv[1])[:top_n]]

    per_dataset = {
        name: [by_dataset[name].resolutions.get(res, 0) for res in top] for name in order
    }

    fig, ax = plt.subplots(figsize=(9.2, 0.46 * len(top) + 2.1))
    _stacked_barh(ax, [r.replace("x", " × ") for r in top], per_dataset, order)
    ax.set_xlabel("样本条数")
    ax.set_title(
        f"截图分辨率分布（前 {len(top)} 种，共 {len(totals)} 种）",
        fontsize=13,
        pad=34,
        loc="left",
    )
    covered = sum(totals[r] for r in top)
    grand = sum(totals.values())
    ax.text(
        0,
        1.015,
        f"前 {len(top)} 种覆盖 {covered / grand:.1%} 的样本",
        transform=ax.transAxes,
        fontsize=9.5,
        color=INK_SECONDARY,
    )
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    return _save(fig, out_dir, "01-分辨率分布")


def chart_action_types(by_dataset: dict[str, DatasetStats], out_dir: Path) -> Path:
    """图 2：动作类型分布。"""
    import matplotlib.pyplot as plt

    order = [name for name in DISPLAY_NAMES if name in by_dataset] or list(by_dataset)
    totals: dict[str, int] = {}
    for stats in by_dataset.values():
        for action, count in stats.action_types.items():
            totals[action] = totals.get(action, 0) + count
    keys = [k for k, _ in sorted(totals.items(), key=lambda kv: -kv[1])]

    per_dataset = {name: [by_dataset[name].action_types.get(k, 0) for k in keys] for name in order}

    fig, ax = plt.subplots(figsize=(9.2, 0.46 * len(keys) + 2.3))
    _stacked_barh(ax, [ACTION_NAMES.get(k, k) for k in keys], per_dataset, order)
    ax.set_xlabel("样本条数")
    ax.set_title("动作类型分布（归一化后）", fontsize=13, pad=34, loc="left")
    ax.text(
        0,
        1.015,
        "ScreenSpot 标注的是元素类别而非动作，统一归入「定位」，不臆断为单击",
        transform=ax.transAxes,
        fontsize=9.5,
        color=INK_SECONDARY,
    )
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    return _save(fig, out_dir, "02-动作类型分布")


def chart_platforms(
    by_dataset: dict[str, DatasetStats],
    out_dir: Path,
    desktop_stats: DatasetStats | None = None,
) -> Path:
    """图 3：平台分布。

    这张图要回答的问题很具体：**公开数据里有多少是我们真正要做的桌面场景。**
    """
    import matplotlib.pyplot as plt

    order = [name for name in DISPLAY_NAMES if name in by_dataset] or list(by_dataset)
    keys = ["desktop", "web", "mobile", "unknown"]
    totals = {k: sum(s.platforms.get(k, 0) for s in by_dataset.values()) for k in keys}
    keys = [k for k in keys if totals[k]]

    per_dataset = {name: [by_dataset[name].platforms.get(k, 0) for k in keys] for name in order}

    fig, ax = plt.subplots(figsize=(9.2, 0.56 * len(keys) + 2.5))
    _stacked_barh(ax, [PLATFORM_NAMES.get(k, k) for k in keys], per_dataset, order)
    ax.set_xlabel("样本条数")
    ax.set_title("平台分布", fontsize=13, pad=34, loc="left")
    grand = sum(totals.values())
    desktop = totals.get("desktop", 0)
    # 只报"桌面 4,680 条"会被读成"有 4,680 条可训练的桌面 grounding 样本"，
    # 而这正是 M3 前置件①要防的误判——其中大多数是规划 / 评估 / 键盘动作，
    # 根本没有坐标。带坐标的条数必须和总数并列出现。
    subtitle = f"桌面样本 {desktop:,} 条（占全部 {desktop / grand:.1%}）"
    if desktop_stats is not None:
        subtitle += f"，其中带坐标、可作 grounding 监督的 {desktop_stats.with_point:,} 条"
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.5, color=INK_SECONDARY)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    return _save(fig, out_dir, "03-平台分布")


def chart_element_sizes(by_platform: dict[str, DatasetStats], out_dir: Path) -> Path:
    """图 4：元素尺寸分布，按平台分。

    横轴是**元素面积占屏幕面积的比例**，对数刻度。理由见模块文档。
    只画有 bbox 的样本——ScreenAgent 训练集只有落点没有框，强行给它编一个
    框会污染这张图。
    """
    import matplotlib.pyplot as plt

    order = [
        p for p in ("desktop", "web", "mobile") if by_platform.get(p, DatasetStats(p)).area_ratios
    ]
    if not order:
        raise ValueError("没有任何带 bbox 的样本，无法绘制元素尺寸分布")

    lo = min(min(by_platform[p].area_ratios) for p in order)
    hi = max(max(by_platform[p].area_ratios) for p in order)
    edges = [10**x for x in _linspace(math.log10(max(lo, 1e-7)), math.log10(min(hi, 1.0)), 31)]

    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    for index, platform in enumerate(order):
        ratios = by_platform[platform].area_ratios
        # 纵轴用**该平台样本的百分比**而不是概率密度。密度值（"215"）读者
        # 无从解释，而三个平台样本量不同又不能直接画计数——百分比两头都占。
        ax.hist(
            ratios,
            bins=edges,
            histtype="step",
            linewidth=2.0,
            color=SERIES_COLORS[index % len(SERIES_COLORS)],
            label=f"{PLATFORM_NAMES.get(platform, platform)}（{len(ratios):,} 条）",
            weights=[100.0 / len(ratios)] * len(ratios),
        )

    ax.axvline(TINY_ELEMENT_RATIO, color=INK_MUTED, linewidth=1.4, linestyle=(0, (4, 3)))
    # 参考线标签压在折线上会看不清，垫一层表面色底
    ax.text(
        TINY_ELEMENT_RATIO,
        ax.get_ylim()[1] * 0.99,
        f" {TINY_ELEMENT_RATIO:.2%} 屏幕面积 ",
        fontsize=8.8,
        color=INK_SECONDARY,
        va="top",
        ha="left",
        bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5},
    )

    ax.set_xscale("log")
    ax.set_xlabel("元素面积 ÷ 屏幕面积（对数刻度）")
    ax.set_ylabel("占该平台样本的百分比")
    ax.set_title("元素尺寸分布（仅含带边界框的样本）", fontsize=13, pad=34, loc="left")

    desktop = by_platform.get("desktop")
    if desktop and desktop.area_ratios:
        ax.text(
            0,
            1.015,
            f"桌面样本中 {desktop.tiny_ratio:.1%} 的目标元素小于 {TINY_ELEMENT_RATIO:.2%} 屏幕面积",
            transform=ax.transAxes,
            fontsize=9.5,
            color=INK_SECONDARY,
        )
    ax.legend(frameon=False, fontsize=9.5)
    _tidy(ax, xgrid=False)
    ax.xaxis.grid(True, linewidth=0.8, color=GRID, which="major")
    ax.set_axisbelow(True)
    return _save(fig, out_dir, "04-元素尺寸分布")


def _linspace(start: float, stop: float, count: int) -> list[float]:
    if count < 2:
        return [start]
    step = (stop - start) / (count - 1)
    return [start + step * i for i in range(count)]


def render_all(
    by_dataset: dict[str, DatasetStats],
    by_platform: dict[str, DatasetStats],
    out_dir: str | Path = "docs/figures",
) -> list[Path]:
    """画全部四张。"""
    apply_style()
    out = Path(out_dir)
    return [
        chart_resolutions(by_dataset, out),
        chart_action_types(by_dataset, out),
        chart_platforms(by_dataset, out, by_platform.get("desktop")),
        chart_element_sizes(by_platform, out),
    ]


__all__ = [
    "SERIES_COLORS",
    "apply_style",
    "chart_action_types",
    "chart_element_sizes",
    "chart_platforms",
    "chart_resolutions",
    "render_all",
]
