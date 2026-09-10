"""Agent Loop 与轨迹日志 —— 系统的心脏与它的黑匣子。

`loop` 只负责编排：发请求 → 拿动作意图 → 交给 grounding 定位 → 交给
执行器 → 回传截图。`trajectory` 负责把这一路上发生的一切记下来。
"""

from core.loop import AgentLoop, LoopConfig, LoopResult
from core.trajectory import (
    ERROR_LABELS,
    LatencyBreakdown,
    StepRecord,
    TrajectoryMeta,
    TrajectoryReader,
    TrajectoryWriter,
    list_trajectories,
    new_trajectory_id,
)

__all__ = [
    "ERROR_LABELS",
    "AgentLoop",
    "LoopConfig",
    "LoopResult",
    "LatencyBreakdown",
    "StepRecord",
    "TrajectoryMeta",
    "TrajectoryReader",
    "TrajectoryWriter",
    "list_trajectories",
    "new_trajectory_id",
]
