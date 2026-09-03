"""坐标系统——本项目的地基。

**真实屏幕坐标与模型坐标之间的转换只允许经过 `CoordinateScaler`。**
任何模块自行换算（哪怕只是一次 `x * 1024 // 1920`）都会在某个 DPI 缩放
或某台显示器上错位，而且这类 bug 表现为"点偏了一点点"，极难定位。

## 为什么需要多个命名坐标系

模式 B 下，送规划模型的截图与送本地 grounding 模型的截图**分辨率不同**：
规划模型只需语义可辨，可以降分辨率；grounding 模型要出精确坐标，需要更
高分辨率。此外 Qwen2.5-VL 一类模型输出的是 0-1000 归一化坐标。三者必须
同时存在且互不干扰，所以坐标系按名字注册，而不是硬编码单一目标。

## 本模块只依赖标准库

因此坐标算法可以在没装 mss / OpenCV 的机器上直接单元测试。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from perception.types import BBox, Point

logger = logging.getLogger(__name__)


class ScaleMode(str, Enum):
    """真实区域映射到模型画布的方式。"""

    #: 独立缩放 x / y 铺满画布。会改变宽高比，但与 `PIL.Image.resize((w, h))`
    #: 的行为一致——多数视觉模型的预处理就是这么做的，所以这是默认值。
    STRETCH = "stretch"
    #: 保持宽高比，居中留边（letterbox）。画面不变形，但画布上有黑边，
    #: 且模型可能把黑边当成界面的一部分。仅在明确需要时使用。
    FIT = "fit"


@dataclass(frozen=True)
class CoordinateSpace:
    """一个命名的模型坐标系。"""

    name: str
    width: int
    height: int
    mode: ScaleMode = ScaleMode.STRETCH

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"坐标系 {self.name!r} 尺寸非法：{self.width}×{self.height}")


class CoordinateScaler:
    """真实屏幕坐标 ↔ 多个命名模型坐标系之间的唯一转换入口。

    `screen_region` 是**被截图的那块真实区域**。全屏截图时它是显示器的完整
    矩形；截取某个显示器时它带有该显示器在虚拟桌面上的偏移；截取局部区域
    时它就是那块区域。偏移由本类统一处理，上层不需要关心多显示器布局。

    典型用法::

        scaler = CoordinateScaler(BBox(0, 0, 2560, 1600))
        scaler.register("planner", 1024, 768)       # 送规划模型的截图
        scaler.register("grounding", 1280, 800)     # 送 grounding 模型的截图
        scaler.register("normalized", 1000, 1000)   # Qwen2.5-VL 归一化输出

        # 模型说"点 (512, 384)"，换算成真实屏幕坐标去点
        real = scaler.to_real(Point(512, 384), "planner")
    """

    def __init__(self, screen_region: BBox) -> None:
        if screen_region.width <= 0 or screen_region.height <= 0:
            raise ValueError(f"截图区域非法：{screen_region!r}")
        self._region = screen_region
        self._spaces: dict[str, CoordinateSpace] = {}

    # ------------------------------------------------------------------ #
    # 构造与注册
    # ------------------------------------------------------------------ #

    @classmethod
    def from_monitor(cls, monitor: dict) -> CoordinateScaler:
        """从 mss 的 monitor 字典构造（含 left/top/width/height）。"""
        return cls(
            BBox.from_xywh(monitor["left"], monitor["top"], monitor["width"], monitor["height"])
        )

    @property
    def region(self) -> BBox:
        """当前的真实截图区域。"""
        return self._region

    @property
    def spaces(self) -> dict[str, CoordinateSpace]:
        return dict(self._spaces)

    def register(
        self,
        name: str,
        width: int,
        height: int,
        mode: ScaleMode = ScaleMode.STRETCH,
    ) -> CoordinateSpace:
        """注册一个命名坐标系。同名注册会覆盖并告警。"""
        if name in self._spaces:
            logger.warning("坐标系 %r 被重新注册，旧定义 %r 已覆盖", name, self._spaces[name])
        space = CoordinateSpace(name=name, width=width, height=height, mode=mode)
        self._spaces[name] = space

        bound = self.roundtrip_error_bound(name)
        if bound > 2.0:
            logger.warning(
                "坐标系 %r (%d×%d) 在 %d×%d 区域上的往返误差上界为 %.2f px，超过 2px 验收标准。"
                "降分辨率过于激进时定位精度会被舍入吃掉。",
                name,
                width,
                height,
                self._region.width,
                self._region.height,
                bound,
            )
        return space

    def get(self, name: str) -> CoordinateSpace:
        try:
            return self._spaces[name]
        except KeyError:
            known = ", ".join(sorted(self._spaces)) or "（无）"
            raise KeyError(f"未注册的坐标系 {name!r}。已注册：{known}") from None

    # ------------------------------------------------------------------ #
    # 缩放因子
    # ------------------------------------------------------------------ #

    def _factors(self, space: CoordinateSpace) -> tuple[float, float, float, float]:
        """返回 (sx, sy, pad_x, pad_y)。

        STRETCH 下 pad 恒为 0；FIT 下 sx == sy，pad 为居中留边。
        """
        rw, rh = self._region.width, self._region.height
        if space.mode is ScaleMode.FIT:
            scale = min(space.width / rw, space.height / rh)
            pad_x = (space.width - rw * scale) / 2.0
            pad_y = (space.height - rh * scale) / 2.0
            return scale, scale, pad_x, pad_y
        return space.width / rw, space.height / rh, 0.0, 0.0

    def roundtrip_error_bound(self, space_name: str) -> float:
        """真实 → 模型 → 真实 往返的最大误差（真实像素）。

        一个模型像素覆盖 ``1/sx`` 个真实像素，四舍五入最多丢掉其中一半，
        因此上界是 ``0.5 / sx``。**这个值随降分辨率变激进而线性增长**：
        2560 宽的屏幕降到 1024 是 1.25px（可接受），降到 640 就是 2px 以上。

        验收标准"往返误差不超过 2px"应当对照本方法的返回值检查，而不是
        写死一个常数——常数在换分辨率时会悄悄失效。
        """
        space = self.get(space_name)
        sx, sy, _, _ = self._factors(space)
        return max(0.5 / sx, 0.5 / sy)

    # ------------------------------------------------------------------ #
    # 点转换
    # ------------------------------------------------------------------ #

    def to_model(self, point: Point, space_name: str) -> Point:
        """真实屏幕坐标 → 模型坐标。超出画布的结果会被夹到边界内。"""
        space = self.get(space_name)
        sx, sy, pad_x, pad_y = self._factors(space)
        mx = round((point.x - self._region.left) * sx + pad_x)
        my = round((point.y - self._region.top) * sy + pad_y)
        return Point(
            _clamp(mx, 0, space.width - 1),
            _clamp(my, 0, space.height - 1),
        )

    def to_real(self, point: Point, space_name: str) -> Point:
        """模型坐标 → 真实屏幕坐标。超出截图区域的结果会被夹到区域内。"""
        space = self.get(space_name)
        sx, sy, pad_x, pad_y = self._factors(space)
        rx = round((point.x - pad_x) / sx) + self._region.left
        ry = round((point.y - pad_y) / sy) + self._region.top
        return Point(
            _clamp(rx, self._region.left, self._region.right - 1),
            _clamp(ry, self._region.top, self._region.bottom - 1),
        )

    def convert(self, point: Point, from_space: str, to_space: str) -> Point:
        """在两个模型坐标系之间直接转换，内部经真实坐标中转。

        用于模式 B：规划模型在 ``planner`` 空间里说"点这个区域"，
        grounding 模型工作在 ``grounding`` 空间里。
        """
        if from_space == to_space:
            return point
        return self.to_model(self.to_real(point, from_space), to_space)

    # ------------------------------------------------------------------ #
    # 边界框转换
    # ------------------------------------------------------------------ #

    def bbox_to_model(self, bbox: BBox, space_name: str) -> BBox:
        top_left = self.to_model(Point(bbox.left, bbox.top), space_name)
        # right / bottom 是开区间，转换时先退一个像素取实际最后一个像素，
        # 转完再加回去，避免框在缩放后系统性地变大一个像素。
        bottom_right = self.to_model(Point(bbox.right - 1, bbox.bottom - 1), space_name)
        return BBox(
            top_left.x,
            top_left.y,
            max(bottom_right.x + 1, top_left.x),
            max(bottom_right.y + 1, top_left.y),
        )

    def bbox_to_real(self, bbox: BBox, space_name: str) -> BBox:
        top_left = self.to_real(Point(bbox.left, bbox.top), space_name)
        bottom_right = self.to_real(Point(bbox.right - 1, bbox.bottom - 1), space_name)
        return BBox(
            top_left.x,
            top_left.y,
            max(bottom_right.x + 1, top_left.x),
            max(bottom_right.y + 1, top_left.y),
        )

    # ------------------------------------------------------------------ #
    # 校验
    # ------------------------------------------------------------------ #

    def is_in_region(self, point: Point) -> bool:
        """真实坐标是否落在截图区域内。动作执行前的越界检查用这个。"""
        return self._region.contains(point)

    def region_matches(self, width: int, height: int) -> bool:
        """截图的实际尺寸是否与建立映射时的区域一致。

        **这是防"分辨率被人偷偷改掉"的闸门。** 虚拟机上尤其要防：
        VMware Tools 装好后默认会让客机分辨率自动适配窗口大小——拖一下
        VMware 的窗口边框，客机就可能从 1920×1080 变成 1760×990，
        而且**没有任何提示**。

        这类错配不会崩，只会让每一次点击都系统性偏移，看起来像"模型定位
        不准"。M1 已经吃过一次同形态的亏（模型坐标系被当成图像尺寸），
        那次查了三组实验才定位到。所以这里改成开机即查、每帧都查。
        """
        return (width, height) == (self._region.width, self._region.height)

    def __repr__(self) -> str:
        names = ", ".join(f"{s.name}({s.width}×{s.height})" for s in self._spaces.values())
        return f"CoordinateScaler(region={self._region.as_tuple()}, spaces=[{names}])"


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))
