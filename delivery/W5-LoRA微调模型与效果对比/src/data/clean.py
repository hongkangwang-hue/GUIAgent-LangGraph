"""清洗规则。

大纲指定三条：**剔除 bbox 越界、面积小于 100px²、无文本且无类型的样本**。
本模块把这三条实现出来，另外补两条（bbox 退化、落点越界），理由见各规则
的 `why`。

## 为什么每条规则要单独计数

"清洗后剩 N 条"是个没法复核的数字。报告里要能写清"越界剔除 12 条、
过小剔除 47 条"，才谈得上说明数据质量，也才能在剔除量异常时发现是自己
的解析写错了——比如把归一化 bbox 当绝对像素解析，越界计数会瞬间飙到
接近全量，这比"清洗后剩 3 条"更容易看出问题出在哪。

## 越界为什么不裁剪而是剔除

一个越界的框有两种可能：标注本身有轻微溢出（裁一下就好），或者我们的
格式解析错了（裁完会得到一个完全错误的框，而且看不出来）。M3 的评测集
容不下第二种。剔除是保守选择，代价是丢掉少量本可挽救的样本，换来的是
剔除量本身成为格式解析的校验信号。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from data.schema import UnifiedSample

#: 大纲指定的最小元素面积。小于此值的框在 1024×768 上不到 10×10 像素，
#: 既难以点中，也多半是标注噪声。
MIN_AREA_PX2 = 100


@dataclass(frozen=True)
class CleaningRule:
    """一条清洗规则。

    `reject` 返回 True 表示该样本应被剔除。
    """

    name: str
    why: str
    reject: Callable[[UnifiedSample], bool]


def _bbox_out_of_bounds(sample: UnifiedSample) -> bool:
    box = sample.bbox
    if box is None:
        return False
    return box.left < 0 or box.top < 0 or box.right > sample.width or box.bottom > sample.height


def _bbox_too_small(sample: UnifiedSample) -> bool:
    return sample.bbox is not None and sample.bbox.area < MIN_AREA_PX2


def _bbox_degenerate(sample: UnifiedSample) -> bool:
    return sample.bbox is not None and (sample.bbox.width <= 0 or sample.bbox.height <= 0)


def _point_out_of_bounds(sample: UnifiedSample) -> bool:
    point = sample.point
    if point is None:
        return False
    return not (0 <= point.x < sample.width and 0 <= point.y < sample.height)


def _no_text_no_type(sample: UnifiedSample) -> bool:
    """既没有指令文本、也没有元素类型信息——这种样本无法构成监督信号。

    注意判的是"两者都没有"，不是"没有指令"。ScreenSpot 的样本有指令没
    类型，ScreenAgent 的 PlanAction 有描述没元素类别，都应保留。
    """
    has_text = bool(sample.instruction.strip())
    has_type = bool(str(sample.meta.get("element_kind", "")).strip()) or bool(
        str(sample.meta.get("raw_action_type", "")).strip()
    )
    return not has_text and not has_type


def _no_location(sample: UnifiedSample) -> bool:
    """既无框也无点。

    **默认不启用。** 这不是"数据坏了"，而是"这条样本不适合做 grounding"——
    ScreenAgent 的 PlanAction / EvaluateSubTaskAction / 键盘动作本来就没有
    坐标，它们是数据集结构的一部分。把它们当脏数据剔掉，"动作类型分布"
    那张图就只剩鼠标动作，等于抹掉了这个数据集最有特点的地方。

    需要挑选动作生成训练样本时用 `data.split.training_pool()`，那是
    "选取"不是"清洗"，两件事分开。
    """
    return sample.resolve_point() is None


#: 大纲指定的三条 + 两条格式健壮性补充。顺序即报告中的呈现顺序。
DEFAULT_RULES: tuple[CleaningRule, ...] = (
    CleaningRule(
        "bbox_out_of_bounds",
        "元素框超出截图范围。剔除而不裁剪——剔除量异常时正好暴露格式解析错误",
        _bbox_out_of_bounds,
    ),
    CleaningRule(
        "bbox_too_small",
        f"元素框面积小于 {MIN_AREA_PX2}px²，既难点中也多半是标注噪声",
        _bbox_too_small,
    ),
    CleaningRule(
        "bbox_degenerate",
        "元素框宽或高为 0，通常是 xywh / xyxy 混淆的产物",
        _bbox_degenerate,
    ),
    CleaningRule(
        "point_out_of_bounds",
        "落点超出截图范围",
        _point_out_of_bounds,
    ),
    CleaningRule(
        "no_text_no_type",
        "既无指令文本也无类型信息，无法构成监督信号",
        _no_text_no_type,
    ),
)

#: 可选规则，需要时显式加进规则表
OPTIONAL_RULES: dict[str, CleaningRule] = {
    "no_location": CleaningRule(
        "no_location", "既无元素框也无落点（默认不启用，见 `_no_location` 说明）", _no_location
    ),
}


@dataclass
class CleaningReport:
    """清洗结果。`rejected` 按规则名分组，供报告直接引用。"""

    total: int = 0
    kept: int = 0
    #: 规则名 → 被该规则剔除的条数。**一条样本只计入第一条命中的规则**，
    #: 否则各项相加会大于剔除总数，报告里没法交代。
    rejected: dict[str, int] = field(default_factory=dict)
    #: 规则名 → 数据集名 → 条数。用来回答"越界的都来自哪个数据集"，
    #: 这是判断格式解析是否写错的最直接线索。
    rejected_by_dataset: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def rejected_total(self) -> int:
        return self.total - self.kept

    @property
    def kept_ratio(self) -> float:
        return self.kept / self.total if self.total else 0.0

    def summary_lines(self) -> list[str]:
        lines = [
            f"总计 {self.total} 条，保留 {self.kept} 条"
            f"（{self.kept_ratio:.1%}），剔除 {self.rejected_total} 条"
        ]
        for name, count in sorted(self.rejected.items(), key=lambda kv: -kv[1]):
            by_ds = self.rejected_by_dataset.get(name, {})
            detail = "，".join(f"{k} {v}" for k, v in sorted(by_ds.items()))
            lines.append(f"  - {name}: {count} 条（{detail}）")
        return lines


def clean(
    samples: Iterable[UnifiedSample],
    rules: Sequence[CleaningRule] = DEFAULT_RULES,
) -> tuple[list[UnifiedSample], CleaningReport]:
    """按规则表清洗，返回 (保留的样本, 报告)。"""
    report = CleaningReport()
    kept: list[UnifiedSample] = []

    for sample in samples:
        report.total += 1
        hit = next((rule for rule in rules if rule.reject(sample)), None)
        if hit is None:
            kept.append(sample)
            report.kept += 1
            continue
        report.rejected[hit.name] = report.rejected.get(hit.name, 0) + 1
        per_dataset = report.rejected_by_dataset.setdefault(hit.name, {})
        per_dataset[sample.source_dataset] = per_dataset.get(sample.source_dataset, 0) + 1

    return kept, report


__all__ = [
    "DEFAULT_RULES",
    "MIN_AREA_PX2",
    "OPTIONAL_RULES",
    "CleaningReport",
    "CleaningRule",
    "clean",
]
