"""重试策略 —— 大纲 W6 任务 2「开发错误检测与自动重试机制」。

## 这个模块要解决的具体问题

2026-08-25 的实测（`open_browser`，微调 3B，5 轮 × 2 组配置）：

    出现过 double_click 的轮次    2/2 成功
    全程只有 left_click 的轮次    0/8 成功        Fisher p = 0.0222

失败的形态是同一个动作重复到步数耗尽：

    子任务「点击任务栏上的Microsoft Edge图标」
       left_click (15, 190) ×6      execution_status 全是 ok

单击桌面图标只是选中它，不会打开。而"选中"和"打开"在
`execution_status` 里长得一模一样——**系统没有任何一处能发现
"点了六次，什么都没发生"。**

## 为什么不是"多试几次"

轨迹探针量过：离线组的重复率 **75.3%**，一半以上有重试的子任务里
**每一次尝试都完全相同**。在这种形态下，把 `max_iterations` 从 6 调到 12
只是把同一个错误做十二遍。

    重试要涨，前提是每次尝试**真的不一样**。

所以这里做的是**升级**，不是重复：同一个位置单击没反应，就改成双击。

## 升级阶梯

    left_click  →  double_click  →  （不再改写，交回上层）

只有这一级。更长的阶梯（右键、拖拽、换坐标）需要证据支撑，而现在只有
"单击变双击"这一条有实测（p=0.0222）。**没有证据的阶梯不写**——它们
会让失败的原因更难查，而不是更少。

## 触发条件：重复 **且** 屏幕没动

两个条件缺一不可：

- **只看重复**：模型第二次给出同一个坐标，可能是它确认了目标位置，
  而第一次点击其实已经生效（比如打开了一个还在渲染的窗口）。
- **只看屏幕没动**：动作本来就可能不改变画面（`mouse_move` 到空白处），
  那不该触发升级。

两个都满足才升级，误报的代价（把一次正常的单击变成双击）就低得多。

## 同一次尝试的判据

坐标距离 ≤ `SAME_SPOT_RADIUS`（归一化 1000 空间里 10 个单位）。不按逐字
相等算——GUI 按钮几十像素宽，模型抖动两三个像素点的是同一个东西，
而逐字比对会把它当成"改主意了"从而不升级。这个口径与
`scripts/probe_retries.py` 一致。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

#: 坐标差在这个半径内算同一次尝试。与 `scripts/probe_retries.py` 同口径。
SAME_SPOT_RADIUS = 10.0

#: 升级阶梯。只有一级，理由见模块文档。
ESCALATION = {"left_click": "double_click"}


@dataclass(frozen=True)
class Attempt:
    """一次尝试的指纹。"""

    action_type: str
    x: float | None = None
    y: float | None = None

    def same_as(self, other: Attempt) -> bool:
        if self.action_type != other.action_type:
            return False
        if self.x is None or other.x is None:
            return self.x is None and other.x is None
        return math.dist((self.x, self.y), (other.x, other.y)) <= SAME_SPOT_RADIUS


@dataclass
class Revision:
    """策略对一个待执行动作的处理结果。"""

    #: 改写后的动作类型；没改写时与原来相同
    action_type: str
    #: 有没有改写
    escalated: bool = False
    #: 为什么。**进轨迹日志**——事后要能分清"模型自己选了双击"和"策略改的"
    reason: str = ""
    #: 这个位置已经试过几次（含本次）
    repeat_count: int = 1

    def as_dict(self) -> dict:
        return {
            "action_type": self.action_type,
            "escalated": self.escalated,
            "reason": self.reason,
            "repeat_count": self.repeat_count,
        }


@dataclass
class RetryPolicy:
    """按子任务维护尝试历史，决定要不要升级。

    **每个子任务开始时必须 `reset()`。** 不重置的话，上一个子任务在
    同一个坐标上的尝试会让这个子任务的第一次点击就被升级成双击——
    而那时屏幕状态已经完全不同了。
    """

    #: 触发升级前，同一个位置要无效地重复几次。1 表示第二次就升级。
    patience: int = 1

    #: (尝试, 屏幕有没有动)。屏幕状态未知时记 None。
    _history: list[tuple[Attempt, bool | None]] = field(default_factory=list)

    def reset(self) -> None:
        self._history.clear()

    def observe(self, action_type: str, x=None, y=None, changed: bool | None = None) -> None:
        """记下一次已经执行完的尝试及其效果。"""
        self._history.append((Attempt(action_type, x, y), changed))

    def revise(self, action_type: str, x=None, y=None) -> Revision:
        """决定这个待执行的动作要不要改写。**不改状态**，可安全重复调用。"""
        candidate = Attempt(action_type, x, y)
        # 同一位置、同一动作、且执行后屏幕明确没动过的次数
        ineffective = sum(
            1 for past, changed in self._history if past.same_as(candidate) and changed is False
        )
        repeat_count = sum(1 for past, _ in self._history if past.same_as(candidate)) + 1

        if ineffective < self.patience:
            return Revision(action_type=action_type, repeat_count=repeat_count)

        upgraded = ESCALATION.get(action_type)
        if upgraded is None:
            # 阶梯到头了。**不静默重复**——说清楚，让上层决定是换目标还是放弃。
            return Revision(
                action_type=action_type,
                reason=f"在同一位置重复 {ineffective} 次且屏幕未变化，但 {action_type} 没有可升级的动作",
                repeat_count=repeat_count,
            )

        logger.info("同一位置 %s 无效 %d 次，升级为 %s", action_type, ineffective, upgraded)
        return Revision(
            action_type=upgraded,
            escalated=True,
            reason=f"在同一位置 {action_type} 重复 {ineffective} 次且屏幕未变化，升级为 {upgraded}",
            repeat_count=repeat_count,
        )

    def hint(self) -> str:
        """给模型的一句反馈。空字符串表示没什么要说的。

        **升级只改这一步的动作，改不了模型下一步的判断。** 不把"你刚才
        点的地方没反应"告诉它，它下一步多半还给同一个坐标——探针量到的
        重复率 75.3% 就是这么来的。
        """
        dead = [past for past, changed in self._history if changed is False]
        if not dead:
            return ""
        last = dead[-1]
        where = f"({last.x:.0f}, {last.y:.0f})" if last.x is not None else "刚才那个位置"
        return (
            f"上一次在 {where} 执行 {last.action_type} 之后**屏幕没有任何变化**，"
            f"说明那里可能不是有效的目标。换一个位置或换一种动作。"
        )
