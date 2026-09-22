"""模式 A 的定位实现：原样透传规划模型给出的坐标。

## 一个"什么都不做"的类为什么值得单独存在

M2 的里程碑文档把它描述为"``locate()`` 为空实现，仅用于满足接口"。真写
成空实现会漏掉一件事：**模型给的坐标不一定合法**。

规划模型输出坐标时会犯两类错：
- 超出模型坐标系范围（比如在 1024×768 的坐标系里给出 x=1500）
- 干脆漏掉 x 或 y

这两类在模式 A 下**必然出现**，开源模型尤其如此。放任它们流到
`ActionExecutor`，会在安全白名单那里被拒绝执行——那时报的是"坐标越界"，
看不出根因是模型输出不合规。在这里拦住并标注来源，轨迹日志里就能直接
分辨"模型没给坐标"和"给了但越界"，这是 M3 统计模式 A 失败原因的依据。

所以这个类不是空壳，它是模式 A 的**输入校验点**。
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from grounding.base import GroundingBackend, GroundingResult
from perception.types import Point

if TYPE_CHECKING:  # pragma: no cover - 仅供类型检查
    from llm.base import ActionIntent
    from perception.capture import Screenshot

logger = logging.getLogger(__name__)


class NativeGrounding(GroundingBackend):
    """模式 A：规划模型原生 grounding。

    坐标来自模型自己，本类只做范围校验与来源标注。
    """

    name = "native"

    def __init__(self, space_width: int, space_height: int) -> None:
        """``space_width/height`` 是模型坐标系的尺寸，用来判越界。

        取模型坐标系而非屏幕尺寸：模型是在这个坐标系里作答的，越界与否
        要按它作答的那把尺子量。
        """
        self.space_width = space_width
        self.space_height = space_height

    def locate(
        self,
        screenshot: Screenshot,
        target_description: str,
        intent: ActionIntent | None = None,
    ) -> GroundingResult:
        start = time.perf_counter()

        def finish(result: GroundingResult) -> GroundingResult:
            result.latency_ms = (time.perf_counter() - start) * 1000.0
            return result

        if intent is None:
            return finish(
                GroundingResult(
                    source=self.name,
                    error="模式 A 需要模型给出的坐标，但未传入 ActionIntent",
                )
            )

        x, y = intent.params.get("x"), intent.params.get("y")
        if x is None or y is None:
            # 模式 A 下模型该给坐标却没给。这不是异常路径，是开源模型的
            # 常见失误，要当数据记下来而不是当 bug 崩掉
            logger.warning(
                "模型未给出坐标（动作 %s，描述 %r），模式 A 无法定位",
                intent.action_type,
                target_description,
            )
            return finish(
                GroundingResult(
                    source=self.name,
                    error="模型输出缺少 x / y 坐标",
                    meta={"reason": "missing_coordinates"},
                )
            )

        try:
            point = Point(int(x), int(y))
        except (TypeError, ValueError):
            return finish(
                GroundingResult(
                    source=self.name,
                    error=f"模型给出的坐标不是整数：x={x!r}, y={y!r}",
                    meta={"reason": "non_integer_coordinates"},
                )
            )

        if not self._in_space(point):
            logger.warning(
                "模型坐标 %s 超出 %d×%d 坐标系范围",
                point,
                self.space_width,
                self.space_height,
            )
            return finish(
                GroundingResult(
                    point=None,
                    source=self.name,
                    error=(
                        f"模型坐标 ({point.x}, {point.y}) 超出模型坐标系 "
                        f"{self.space_width}×{self.space_height}"
                    ),
                    meta={
                        "reason": "out_of_space",
                        "raw_point": (point.x, point.y),
                    },
                )
            )

        return finish(GroundingResult(point=point, source=self.name, confidence=1.0))

    def _in_space(self, point: Point) -> bool:
        return 0 <= point.x < self.space_width and 0 <= point.y < self.space_height

    def __repr__(self) -> str:
        return f"<NativeGrounding space={self.space_width}x{self.space_height}>"
