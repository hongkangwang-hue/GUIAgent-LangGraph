"""感知层的基础数据类型。

本模块**只依赖标准库**，因此坐标运算、去重逻辑等核心算法可以在没有安装
mss / OpenCV / PaddleOCR 的机器上直接单元测试。这是有意为之：M1 的关键
不是识别准确率，而是"坐标不会错"，而坐标逻辑必须能被独立验证。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum


class ElementSource(str, Enum):
    """UI 元素的来源通道。

    双通道融合时 UIA 优先于 OCR：UIA 给出的是控件的真实边界，OCR 给出的
    只是文字外接框，后者往往比可点击区域小。
    """

    UIA = "uia"
    OCR = "ocr"


@dataclass(frozen=True)
class Point:
    """一个坐标点。

    x / y 保持为 int：像素是离散的，浮点会在多次转换中累积误差，也让
    "往返误差不超过 2px" 这类验收标准难以判定。
    """

    x: int
    y: int

    def __post_init__(self) -> None:
        if not isinstance(self.x, int) or not isinstance(self.y, int):
            raise TypeError(f"Point 坐标必须是 int，收到 ({type(self.x)}, {type(self.y)})")

    def as_tuple(self) -> tuple[int, int]:
        return (self.x, self.y)


@dataclass(frozen=True)
class BBox:
    """轴对齐边界框，采用 (left, top, right, bottom) 且 right / bottom 为**开区间**。

    开区间的理由：`width = right - left` 直接成立，不需要 +1，与 OpenCV /
    PIL 的切片语义一致，避免了 off-by-one 这类最难查的坐标 bug。
    """

    left: int
    top: int
    right: int
    bottom: int

    def __post_init__(self) -> None:
        if self.right < self.left or self.bottom < self.top:
            raise ValueError(f"BBox 边界非法：{self!r}")

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Point:
        """框中心。点击目标默认取这里。"""
        return Point(self.left + self.width // 2, self.top + self.height // 2)

    def contains(self, point: Point) -> bool:
        return self.left <= point.x < self.right and self.top <= point.y < self.bottom

    def intersection_area(self, other: BBox) -> int:
        w = min(self.right, other.right) - max(self.left, other.left)
        h = min(self.bottom, other.bottom) - max(self.top, other.top)
        return w * h if w > 0 and h > 0 else 0

    def iou(self, other: BBox) -> float:
        """交并比。双通道融合去重的判据。"""
        inter = self.intersection_area(other)
        if inter == 0:
            return 0.0
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)

    @classmethod
    def from_xywh(cls, x: int, y: int, w: int, h: int) -> BBox:
        """从 (左上角 + 宽高) 构造。UIA 的 BoundingRectangle 是这种格式。"""
        return cls(x, y, x + w, y + h)


@dataclass
class UIElement:
    """一个被识别出的 UI 元素。

    这是感知层交给上层的唯一元素表示。M4 的 Reflector 用它做控件存在性
    校验，M3 的 grounding 用它做兜底纠偏，visualizer 用它画调试图。
    """

    bbox: BBox
    source: ElementSource
    text: str = ""
    control_type: str = ""
    confidence: float = 1.0
    #: 融合后由 ElementDetector 统一分配，从 1 开始，用于调试图上的编号标注
    index: int | None = None
    #: 各通道的原始信息，便于排查（如 UIA 的 AutomationId、OCR 的原始多边形）
    meta: dict = field(default_factory=dict)

    @property
    def click_point(self) -> Point:
        """默认点击位置。"""
        return self.bbox.center

    def label(self) -> str:
        """调试图上显示的短标签。"""
        prefix = f"{self.index}." if self.index is not None else ""
        body = self.text.strip() or self.control_type or self.source.value
        if len(body) > 20:
            body = body[:19] + "…"
        return f"{prefix}{body}"


def dedupe_by_iou(
    elements: Iterable[UIElement],
    iou_threshold: float = 0.5,
) -> list[UIElement]:
    """按 IoU 去重，来源优先级高的胜出。

    保留策略：先按 (来源优先级, 面积) 排序，UIA 排在 OCR 前面；再逐个尝试
    加入结果集，与已保留元素 IoU 超过阈值的丢弃。

    为什么 UIA 优先：OCR 只框住文字，UIA 框住整个可点击控件。一个按钮上
    的文字被 OCR 框成小框、被 UIA 框成大框时，**点击大框的中心才对**。
    """
    source_rank = {ElementSource.UIA: 0, ElementSource.OCR: 1}
    ordered = sorted(
        elements,
        key=lambda e: (source_rank.get(e.source, 99), -e.bbox.area),
    )

    kept: list[UIElement] = []
    for candidate in ordered:
        if any(candidate.bbox.iou(k.bbox) > iou_threshold for k in kept):
            continue
        kept.append(candidate)

    # 稳定的阅读顺序：从上到下、从左到右。编号在调试图上才有意义。
    kept.sort(key=lambda e: (e.bbox.top, e.bbox.left))
    for i, element in enumerate(kept, start=1):
        element.index = i
    return kept
