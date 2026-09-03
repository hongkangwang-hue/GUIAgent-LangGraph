"""感知层：截图、坐标、元素识别、调试可视化。"""

from perception.coordinate import CoordinateScaler, CoordinateSpace, ScaleMode
from perception.types import BBox, ElementSource, Point, UIElement, dedupe_by_iou

__all__ = [
    "BBox",
    "CoordinateScaler",
    "CoordinateSpace",
    "ElementSource",
    "Point",
    "ScaleMode",
    "UIElement",
    "dedupe_by_iou",
]
