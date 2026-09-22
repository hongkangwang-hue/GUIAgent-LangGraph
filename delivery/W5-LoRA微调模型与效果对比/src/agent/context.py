"""上下文窗口管理 —— 决定每一步给模型看多少历史、看多清楚。

## 为什么不是简单地"留最近 k 步"

M2 任务 3 要求「历史截图窗口 `k` 可配置（默认 3），**旧帧降采样**」。

"降采样"这三个字容易被做成"丢掉"，但两者不是一回事：

- **丢掉**：三步之前发生了什么，模型完全不知道
- **降采样**：三步之前的界面模型还看得见轮廓，只是看不清小字

对 GUI 任务，前者会造成一类具体的失败：模型点开了一个菜单、下一步被
菜单遮挡搞混，却不记得菜单是自己点开的，于是反复点同一个地方——也就是
`label` 词表里的 ``stuck_loop``。留一张模糊的旧帧就能避免。

## 但也不能全留全清晰

每张图都要按 token 计费，而且**视觉 token 占比极高**：一张 1024×768 的
截图在多数 VLM 上折合上千 token，比整段提示词还贵。全留高清等于每步
成本翻几倍，而 M2 交付物之一就是"单任务真实成本实测数据"。

因此策略是三档递减：最近几步全分辨率 → 更早的降采样 → 再早的只留文字
摘要。文字摘要（"left_click(x=100, y=200) → 失败：坐标越界"）几乎不花钱，
却保住了"做过什么、成没成"这条最关键的信息。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from llm.base import HistoryStep

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContextPolicy:
    """上下文窗口策略。

    三个参数构成从近到远的三档：``full_res_steps`` 步全分辨率，
    再往前到 ``k`` 步降采样，更早的只留文字摘要。
    """

    #: 保留多少步历史（含带图与不带图的）。M2 规定默认 3
    k: int = 3

    #: 最近几步用**全分辨率**。默认 1——当前这一步的截图是另外传的，
    #: 这里说的是"上一步执行完的样子"，它对判断"我刚才那下点成没点成"
    #: 最关键，值得给全清晰度
    full_res_steps: int = 1

    #: 更早的帧缩到原尺寸的多少倍。0.5 是经验取值：面积变 1/4，视觉
    #: token 大致同比例下降，而窗口布局、弹窗位置这类**结构信息**在
    #: 半分辨率下仍然认得出来
    downscale: float = 0.5

    #: 完全不带图、只留文字摘要的门槛。超过 k 步的自然全是摘要，
    #: 这个值用于在 k 以内再收紧（比如 k=3 但只想带 2 张图）
    image_steps: int = 2

    def __post_init__(self) -> None:
        if self.k < 0:
            raise ValueError("k 不能为负")
        if not 0.0 < self.downscale <= 1.0:
            raise ValueError(f"downscale 必须在 (0, 1] 之间，收到 {self.downscale}")
        if self.full_res_steps < 0:
            raise ValueError("full_res_steps 不能为负")
        if self.image_steps < 0:
            raise ValueError("image_steps 不能为负")

    def as_dict(self) -> dict:
        """进轨迹日志。M3 消融要能追溯当时用的是哪档策略。"""
        return {
            "k": self.k,
            "full_res_steps": self.full_res_steps,
            "downscale": self.downscale,
            "image_steps": self.image_steps,
        }


#: 省钱档：只带一张全分辨率的上一帧，更早的全靠文字。
#: 成本消融时作为对照臂
FRUGAL = ContextPolicy(k=3, full_res_steps=1, image_steps=1)

#: 默认档
DEFAULT_POLICY = ContextPolicy()

#: 富上下文档：带更多帧。用于排查"模型是不是因为看不到历史才犯错"
RICH = ContextPolicy(k=5, full_res_steps=2, image_steps=4, downscale=0.6)


@dataclass
class ContextFrame:
    """一个历史步骤在本轮该以什么形式出现。"""

    step: HistoryStep
    #: 距离当前有多少步。0 = 最近的一步
    age: int
    #: 要不要带图
    with_image: bool = False
    #: 带图时的缩放比例。1.0 表示全分辨率
    scale: float = 1.0

    @property
    def summary(self) -> str:
        return self.step.summary()


class ContextWindow:
    """按策略挑选历史步骤。

    只做**决策**，不碰图片——实际的缩放与编码在后端里做（那里已经有
    `encode_screenshot`，且知道平台的字节上限）。分开的好处是这个类
    纯逻辑、好测，且换后端时策略不用重写。
    """

    def __init__(self, policy: ContextPolicy | None = None) -> None:
        self.policy = policy or DEFAULT_POLICY

    def select(self, history: list[HistoryStep]) -> list[ContextFrame]:
        """挑出本轮要回传的历史，按时间从旧到新排列。

        返回的顺序与 ``history`` 一致（旧在前），因为消息要按时间顺序
        拼进对话——倒序会让模型以为时间是反的。
        """
        if not history or self.policy.k <= 0:
            return []

        window = history[-self.policy.k :]
        total = len(window)
        frames: list[ContextFrame] = []

        for index, step in enumerate(window):
            age = total - 1 - index  # 0 = 最近
            with_image = step.screenshot is not None and age < self.policy.image_steps
            scale = 1.0 if age < self.policy.full_res_steps else self.policy.downscale
            frames.append(
                ContextFrame(
                    step=step,
                    age=age,
                    with_image=with_image,
                    scale=scale if with_image else 1.0,
                )
            )
        return frames

    def select_steps(self, history: list[HistoryStep]) -> list[HistoryStep]:
        """挑选并**标注**历史步骤，直接可以交给后端。

        返回的是副本：``with_image`` 与 ``image_scale`` 是"本轮怎么回传"
        的决策，不是这一步的固有属性。写回原对象的话，同一段历史在不同
        策略下会互相污染——而 M3 的上下文消融正是要拿同一段历史跑不同
        策略，那时结果就不可比了。
        """
        selected: list[HistoryStep] = []
        for frame in self.select(history):
            step = frame.step
            selected.append(
                HistoryStep(
                    action=step.action,
                    thinking=step.thinking,
                    screenshot=step.screenshot,
                    success=step.success,
                    error=step.error,
                    with_image=frame.with_image,
                    image_scale=frame.scale,
                )
            )
        return selected

    def stats(self, history: list[HistoryStep]) -> dict:
        """本轮上下文的构成，进轨迹日志。

        排查"这步为什么决策错了"时，第一个要问的就是"当时模型看到了什么"。
        没有这个记录，那个问题回答不了。
        """
        frames = self.select(history)
        return {
            "history_total": len(history),
            "selected": len(frames),
            "with_image": sum(1 for f in frames if f.with_image),
            "full_res": sum(1 for f in frames if f.with_image and f.scale == 1.0),
            "downscaled": sum(1 for f in frames if f.with_image and f.scale < 1.0),
            "policy": self.policy.as_dict(),
        }

    def __repr__(self) -> str:
        return f"<ContextWindow {self.policy.as_dict()}>"


@dataclass
class Conversation:
    """一次任务执行期间的历史累积。

    与 `AgentLoop.history` 的关系：Loop 持有原始列表，本类负责按策略
    裁剪。拆开是因为**裁剪策略是要做消融的变量**（M3 提示词与上下文
    消融），而原始历史是不该被策略影响的事实记录。
    """

    steps: list[HistoryStep] = field(default_factory=list)
    window: ContextWindow = field(default_factory=ContextWindow)

    def append(self, step: HistoryStep) -> None:
        self.steps.append(step)

    def recent(self) -> list[HistoryStep]:
        """按策略裁剪并标注后的历史，直接喂给后端。"""
        return self.window.select_steps(self.steps)

    def frames(self) -> list[ContextFrame]:
        return self.window.select(self.steps)

    def stats(self) -> dict:
        return self.window.stats(self.steps)

    def clear(self) -> None:
        """子任务切换时清空。

        每个子任务独立带历史，是有意的：上一个子任务的操作对下一个子任务
        的决策价值很低（"点了开始菜单"对"在记事本里打字"没有帮助），却要
        一直付 token。M2 设计思路里"一次只带一个子目标进 Loop"说的就是
        这种隔离。
        """
        self.steps.clear()
