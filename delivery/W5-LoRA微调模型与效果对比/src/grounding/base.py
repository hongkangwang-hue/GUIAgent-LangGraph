"""Grounding 后端抽象 —— "元素描述 → 像素坐标"。

## 这一层是模式 A/B 消融的结构前提

把定位从模型调用里拆出来，是为了让下面这两条路可以**在配置里切换**，
而不是在代码里分叉：

- **模式 A**：规划模型原生 grounding，直接输出坐标。`NativeGrounding`
  原样透传。M2 的唯一实现。
- **模式 B**：规划模型只说"点地址栏"，本地 grounding 模型（M3 的
  `LocalVLMGrounding`）负责把这句话变成 (x, y)。

M5 的消融实验要对比两种模式的定位精度与成本。**如果两种模式散落在
Loop 的 if/else 里，消融就退化成了改代码重跑**——那样的对比说明不了
问题，因为你没法保证除了模式之外别的都没变。

## 坐标是哪个坐标系的

`locate` 返回的是**模型坐标系**的点，不是屏幕真实坐标。转换统一由
`CoordinateScaler` 负责，这是 M1 定死的规矩：真实坐标与模型坐标之间的
转换只允许经过那一个类，任何模块不得自行换算。

grounding 后端只需回答"在这张图的什么位置"，图多大它自己看得见。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from perception.types import Point

if TYPE_CHECKING:  # pragma: no cover - 仅供类型检查
    from llm.base import ActionIntent
    from perception.capture import Screenshot

logger = logging.getLogger(__name__)


@dataclass
class GroundingResult:
    """一次定位的结果与来源。

    ## 为什么要记 source 和 confidence

    M1 的里程碑文档写明，UIA + OCR 的识别结果不是主链路定位来源，而是
    服务于"grounding 结果的兜底与纠偏"。一旦有兜底，**同一个点可能来自
    不同渠道**——模型原生给的、本地模型算的、UIA 纠偏后的。

    轨迹日志里必须能分清是哪一种，否则 M3 分析"grounding 到底准不准"时，
    会把兜底救回来的那些算进模型的功劳里，把模型的能力估高。
    """

    point: Point | None = None
    #: 定位来源：native / local_vlm / uia_fallback / ocr_fallback / none
    source: str = "none"
    confidence: float = 1.0
    #: 纠偏前的原始点。没纠偏时为 None
    original_point: Point | None = None
    latency_ms: float = 0.0
    error: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def found(self) -> bool:
        return self.point is not None

    @property
    def corrected(self) -> bool:
        """是否被纠偏过。"""
        return self.original_point is not None and self.original_point != self.point

    def as_dict(self) -> dict:
        payload: dict = {
            "point": self.point.as_tuple() if self.point else None,
            "source": self.source,
            "confidence": round(self.confidence, 4),
            "latency_ms": round(self.latency_ms, 2),
        }
        if self.original_point is not None:
            payload["original_point"] = self.original_point.as_tuple()
            payload["corrected"] = self.corrected
        if self.error:
            payload["error"] = self.error
        if self.meta:
            payload["meta"] = self.meta
        return payload


class GroundingBackend(ABC):
    """定位后端。

    实现类只需回答一个问题：在这张截图上，这个描述指的是哪个点。
    """

    #: 进轨迹日志的来源标识
    name: str = "base"

    @abstractmethod
    def locate(
        self,
        screenshot: Screenshot,
        target_description: str,
        intent: ActionIntent | None = None,
    ) -> GroundingResult:
        """把元素描述解析成模型坐标系下的点。

        ``intent`` 可选，给实现类一个读取模型已有输出的机会——模式 A 的
        透传实现就是从这里取坐标的。定位不到时返回 ``found=False`` 的
        结果而**不是抛异常**：定位失败是常态，Loop 要能按业务逻辑处理
        （记录、重试、终止），而不是被异常打断。
        """

    def close(self) -> None:  # noqa: B027 - 可选钩子，理由同 LLMBackend.close
        """释放资源。默认空实现，M3 的本地 grounding 模型会用到。"""

    def __enter__(self) -> GroundingBackend:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"
