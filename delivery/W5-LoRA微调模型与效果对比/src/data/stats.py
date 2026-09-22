"""统计聚合 —— 图表与文字报告共用同一份数字。

图和文字如果各算各的，早晚会出现"图上写 716、正文写 720"这种没法解释的
不一致。所以本模块产出一个 `DatasetStats`，`data.charts` 画它，
`scripts/prepare_datasets.py` 写它，两边不各自 groupby。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from data.schema import ActionType, Platform, UnifiedSample

#: 元素"很小"的判据（相对面积）。0.05% 在 1920×1080 上约 1000px²，
#: 大致是一个 32×32 的工具栏图标。用相对值而不是绝对像素，因为数据集
#: 分辨率从 1024×768 到 2560×1600 都有，绝对阈值在两端含义完全不同。
TINY_ELEMENT_RATIO = 0.0005


@dataclass
class DatasetStats:
    """一个数据集（或全体）的统计切面。"""

    name: str
    total: int = 0
    #: "1920x1080" → 条数
    resolutions: Counter = field(default_factory=Counter)
    #: 平台 → 条数
    platforms: Counter = field(default_factory=Counter)
    #: 统一动作类型 → 条数
    action_types: Counter = field(default_factory=Counter)
    #: 原始动作类型 → 条数。归一化会合并类别，报告里要能追回去
    raw_action_types: Counter = field(default_factory=Counter)
    #: 应用 / 站点 → 条数
    apps: Counter = field(default_factory=Counter)
    #: 数据集自带划分 → 条数
    splits: Counter = field(default_factory=Counter)
    #: 每个带框样本的 bbox 面积占屏幕面积的比例
    area_ratios: list[float] = field(default_factory=list)
    #: 带框 / 带点的条数。两者可以同时成立
    with_bbox: int = 0
    with_point: int = 0
    #: 不重复截图数。ScreenSpot 一张图对应多条标注，样本数不等于图片数
    unique_images: set[str] = field(default_factory=set)

    @property
    def image_count(self) -> int:
        return len(self.unique_images)

    @property
    def groundable(self) -> int:
        """有坐标（框或点）可作为 grounding 监督信号的条数。"""
        return self.with_point

    @property
    def tiny_ratio(self) -> float:
        """相对面积小于 `TINY_ELEMENT_RATIO` 的元素占比。

        这个数字直接对应"我们的 agent 要点中多小的东西"，比平均面积有用——
        平均值会被少数整窗口大小的框拉高。
        """
        if not self.area_ratios:
            return 0.0
        tiny = sum(1 for r in self.area_ratios if r < TINY_ELEMENT_RATIO)
        return tiny / len(self.area_ratios)

    def percentiles(self, ps: Sequence[float] = (5, 25, 50, 75, 95)) -> dict[float, float]:
        """相对面积的分位数。手写而不用 numpy，让本模块保持可无依赖测试。"""
        if not self.area_ratios:
            return {p: 0.0 for p in ps}
        ordered = sorted(self.area_ratios)
        out = {}
        for p in ps:
            index = min(int(p / 100 * len(ordered)), len(ordered) - 1)
            out[p] = ordered[index]
        return out


def collect(samples: Iterable[UnifiedSample], name: str = "all") -> DatasetStats:
    stats = DatasetStats(name=name)
    for sample in samples:
        stats.total += 1
        stats.resolutions[f"{sample.width}x{sample.height}"] += 1
        stats.platforms[sample.platform.value] += 1
        stats.action_types[sample.action_type.value] += 1
        raw = sample.meta.get("raw_action_type") or sample.meta.get("element_kind") or ""
        subtype = sample.meta.get("raw_action_subtype") or ""
        if raw:
            stats.raw_action_types[f"{raw}.{subtype}" if subtype else str(raw)] += 1
        if sample.app:
            stats.apps[sample.app] += 1
        if sample.split:
            stats.splits[sample.split] += 1
        stats.unique_images.add(sample.screenshot_path)

        if sample.bbox is not None:
            stats.with_bbox += 1
            screen_area = sample.width * sample.height
            if screen_area > 0:
                stats.area_ratios.append(sample.bbox.area / screen_area)
        if sample.resolve_point() is not None:
            stats.with_point += 1
    return stats


def collect_by_dataset(samples: Iterable[UnifiedSample]) -> dict[str, DatasetStats]:
    """按来源数据集分组统计。"""
    buckets: dict[str, list[UnifiedSample]] = {}
    for sample in samples:
        buckets.setdefault(sample.source_dataset, []).append(sample)
    return {name: collect(group, name=name) for name, group in buckets.items()}


# ===================================================================== #
# 跨数据集的重叠检查（M3 泄漏风险）
# ===================================================================== #


@dataclass
class OverlapReport:
    """两个数据集之间的截图级重叠。

    M2 任务 1 明确要求"检查训练来源与 ScreenSpot 测试集是否存在截图或
    站点级重叠，避免测试泄漏"。比对的是**截图文件名**而不是路径——同一张
    截图在两个数据集里的存放路径必然不同。
    """

    left: str
    right: str
    left_images: int
    right_images: int
    shared_images: int

    @property
    def left_leaked_ratio(self) -> float:
        return self.shared_images / self.left_images if self.left_images else 0.0

    @property
    def right_leaked_ratio(self) -> float:
        return self.shared_images / self.right_images if self.right_images else 0.0


def _image_keys(samples: Iterable[UnifiedSample]) -> set[str]:
    """截图的比对键：优先用数据集自带的文件名，回落到路径的最后一段。"""
    keys = set()
    for sample in samples:
        name = sample.meta.get("file_name")
        keys.add(
            str(name) if name else sample.screenshot_path.replace("\\", "/").rsplit("/", 1)[-1]
        )
    return keys


def overlap(
    left_name: str,
    left: Iterable[UnifiedSample],
    right_name: str,
    right: Iterable[UnifiedSample],
) -> OverlapReport:
    left_keys, right_keys = _image_keys(left), _image_keys(right)
    return OverlapReport(
        left=left_name,
        right=right_name,
        left_images=len(left_keys),
        right_images=len(right_keys),
        shared_images=len(left_keys & right_keys),
    )


__all__ = [
    "TINY_ELEMENT_RATIO",
    "ActionType",
    "DatasetStats",
    "OverlapReport",
    "Platform",
    "collect",
    "collect_by_dataset",
    "overlap",
]
