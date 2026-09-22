"""脚本化的假后端 —— 让 Agent Loop 在没有 API key 时也能被完整测试。

## 它不是"测试用的玩具"，是 Loop 的对照组

Agent Loop 里真正容易出错的是编排逻辑：迭代上限有没有生效、成本熔断有
没有触发、坐标有没有在正确的时机转换、失败时轨迹落没落盘。**这些和模型
聪不聪明毫无关系**，却每一条都会毁掉一次任务。

用真实 API 测这些，每跑一次要花钱、要等网络、还不可复现——模型这次给
坐标下次给描述，同一段代码两次跑出不同分支，测试就成了掷骰子。

`ScriptedBackend` 把模型的输出固定下来，于是：

- 迭代上限、熔断、异常分支都能精确构造
- CI 上不需要任何凭据就能跑完整条 Loop
- M3 换真实后端时，Loop 的行为已经被锁死，出问题必定在后端里

M2 交付物"5 个基础任务测试报告"当然要用真实模型跑；但**在那之前，
Loop 本身的正确性应该已经被这个假后端证明过了**。
"""

from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING

from llm.base import (
    ActionIntent,
    HistoryStep,
    LLMBackend,
    LLMBackendError,
    RawResponse,
    TokenUsage,
)

if TYPE_CHECKING:  # pragma: no cover - 仅供类型检查
    from perception.capture import Screenshot

logger = logging.getLogger(__name__)


class ScriptedBackend(LLMBackend):
    """按预设脚本逐条返回动作意图。

    脚本里可以放三种东西：

    - `ActionIntent` —— 原样返回
    - ``dict`` —— 按 ``{action, x, y, thinking, done, ...}`` 构造意图
    - ``Exception`` 实例 —— 抛出。用来测 Loop 的异常处理与重试

    脚本走完后的行为由 ``on_exhausted`` 决定：``done`` 返回完成意图
    （最常见），``repeat`` 重复最后一条（测迭代上限用），``raise`` 抛错。
    """

    name = "scripted"

    def __init__(
        self,
        script: list | None = None,
        model: str = "scripted-v0",
        on_exhausted: str = "done",
        usage_per_call: TokenUsage | None = None,
        price=None,
    ) -> None:
        super().__init__(model=model, price=price)
        self.script = list(script or [])
        if on_exhausted not in ("done", "repeat", "raise"):
            raise ValueError(f"on_exhausted 只能是 done / repeat / raise，收到 {on_exhausted!r}")
        self.on_exhausted = on_exhausted
        #: 每次调用记多少 token。默认给个非零值，否则成本熔断测不出来
        self.usage_per_call = usage_per_call or TokenUsage(prompt_tokens=1000, completion_tokens=50)
        self._cursor = itertools.count()
        #: 收到过的调用参数，断言用
        self.calls: list[dict] = []

    # ------------------------------------------------------------------ #

    def complete(
        self,
        prompt: str,
        screenshot: Screenshot | None = None,
        history: list[HistoryStep] | None = None,
    ) -> RawResponse:
        """返回脚本里的 ``raw_text``。

        Planner 走的是这条路（它要的是原始文本，不是动作），因此脚本条目
        里可以直接放 ``{"raw_text": '{"subtasks": [...]}'}``。
        """
        intent = self.predict_action(prompt, screenshot, history)
        return RawResponse(
            text=intent.raw_text,
            usage=intent.usage,
            cost_cny=intent.cost_cny,
            request_id=intent.request_id,
            latency_ms=intent.latency_ms,
        )

    def predict_action(
        self,
        instruction: str,
        screenshot: Screenshot | None = None,
        history: list[HistoryStep] | None = None,
    ) -> ActionIntent:
        index = next(self._cursor)
        self.calls.append(
            {
                "instruction": instruction,
                "screenshot": screenshot,
                "history_len": len(history or []),
            }
        )

        item = self._pick(index)
        if isinstance(item, Exception):
            raise item

        intent = self._as_intent(item)
        # 用量与成本照样走基类，这样 Loop 里的熔断逻辑面对的是真实形状的数据
        intent.usage = self.usage_per_call
        intent.cost_cny = self.record_usage(self.usage_per_call)
        intent.request_id = f"scripted-{index}"
        return intent

    # ------------------------------------------------------------------ #

    def _pick(self, index: int):
        if index < len(self.script):
            return self.script[index]

        if self.on_exhausted == "raise":
            raise LLMBackendError(
                f"脚本只有 {len(self.script)} 条，第 {index + 1} 次调用无内容",
                kind="script_exhausted",
            )
        if self.on_exhausted == "repeat" and self.script:
            return self.script[-1]
        return ActionIntent(done=True, thinking="脚本已执行完毕")

    @staticmethod
    def _as_intent(item) -> ActionIntent:
        if isinstance(item, ActionIntent):
            # 复制一份，避免同一条脚本被多次返回时共享可变的 params
            return ActionIntent(
                action_type=item.action_type,
                params=dict(item.params),
                target_description=item.target_description,
                thinking=item.thinking,
                done=item.done,
                raw_text=item.raw_text,
            )
        if isinstance(item, dict):
            payload = dict(item)
            action_type = payload.pop("action", None) or payload.pop("action_type", "") or ""
            return ActionIntent(
                action_type=action_type,
                thinking=payload.pop("thinking", ""),
                done=bool(payload.pop("done", False)),
                target_description=payload.pop("target_description", ""),
                raw_text=payload.pop("raw_text", ""),
                params=payload,
            )
        raise TypeError(f"脚本条目必须是 ActionIntent / dict / Exception，收到 {type(item)}")
