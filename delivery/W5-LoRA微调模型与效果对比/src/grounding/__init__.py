"""Grounding 后端抽象层。

M2 只有模式 A 的 `NativeGrounding`；M3 新增 `LocalVLMGrounding`（模式 B）。
两者的切换是配置项，Agent 层与 Loop 层对此无感知。
"""

from grounding.base import GroundingBackend, GroundingResult
from grounding.native import NativeGrounding

__all__ = ["GroundingBackend", "GroundingResult", "NativeGrounding"]
