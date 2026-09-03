"""LLM 后端抽象 —— "看截图 + 历史 → 决定下一步动作"。

## 这一层为什么必须存在

M2 只接一个规划模型（Qwen-VL），单看本阶段，抽象是多余的。它的价值在
M3：六档横评要在 GLM-4V、Llama 3.2 Vision、微调前后的本地模型之间来回
切，**如果切换要改 Agent 层代码，横评就做不成对照实验**——每次切换都
可能顺手改掉别的东西，跑出来的差异分不清是模型带来的还是代码带来的。

因此 M2 的验收标准第 8 条写的是"新增一个模型后端不需要改动 Agent 层与
Loop 层"。这条现在就要满足，不能等 M3 再重构。

## 与 GroundingBackend 的分工

两层抽象各管一件事，边界在"坐标从哪来"：

- `LLMBackend`：决定**做什么**（点击？输入？滚动？），以及目标是什么
- `GroundingBackend`：决定目标**在哪**（像素坐标）

模式 A 下规划模型直接给坐标，`NativeGrounding` 原样透传；模式 B 下规划
模型只给元素描述，本地 grounding 模型负责转成坐标。**切换发生在单步
内部**——在"拿到动作意图"与"执行动作"之间，不是工具层面的切换。这正是
不用 LangChain ``AgentExecutor`` 的原因之一。

`ActionIntent` 因此要同时容纳两种形态：坐标齐全的（模式 A）与只有描述
的（模式 B）。谁来补坐标由 Loop 决定，后端自己不管。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from control.actions import Action

if TYPE_CHECKING:  # pragma: no cover - 仅供类型检查
    from perception.capture import Screenshot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# 用量与成本
# ---------------------------------------------------------------------- #


@dataclass(frozen=True)
class PriceSheet:
    """某个模型的计费单价，单位 **元 / 千 token**。

    单价写成数据而不是散在成本计算里，是因为 M2 交付物之一是"单任务真实
    成本实测数据，用于校准 M0 的成本估算"——单价会变，估算要能跟着改。

    ``cached_input_per_1k`` 是上下文缓存命中时的输入单价。GUI Agent 每步
    都要回传截图，系统提示词与 few-shot 示例完全重复，缓存能省的比例不小。
    平台不支持缓存时置 None，命中量按 0 计。
    """

    model: str
    input_per_1k: float = 0.0
    output_per_1k: float = 0.0
    cached_input_per_1k: float | None = None

    def cost_of(self, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> float:
        """算一次调用的费用。缓存命中的那部分按缓存单价计。"""
        fresh = max(prompt_tokens - cached_tokens, 0)
        cached_rate = (
            self.cached_input_per_1k if self.cached_input_per_1k is not None else self.input_per_1k
        )
        return (
            fresh / 1000.0 * self.input_per_1k
            + cached_tokens / 1000.0 * cached_rate
            + completion_tokens / 1000.0 * self.output_per_1k
        )


@dataclass
class TokenUsage:
    """一次调用的 token 用量。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    #: 上下文缓存命中的输入 token。平台不报此项时保持 0
    cached_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict:
        payload = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
        if self.cached_tokens:
            payload["cached_tokens"] = self.cached_tokens
        return payload


@dataclass
class CostInfo:
    """一个后端实例的累计用量与成本。

    累计而非单次：单任务成本是 M2 的交付物，而一个任务包含多次调用。
    """

    model: str = ""
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost_cny: float = 0.0
    #: 单价未知时置 False，此时 cost_cny 不可信，报告里要标注
    priced: bool = True

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cache_hit_rate(self) -> float:
        """缓存命中的输入 token 占比。接入缓存后用它量化省了多少。"""
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    def add(self, usage: TokenUsage, cost: float, priced: bool = True) -> None:
        self.requests += 1
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.cached_tokens += usage.cached_tokens
        self.cost_cny += cost
        if not priced:
            self.priced = False

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "requests": self.requests,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "total_tokens": self.total_tokens,
            "cost_cny": round(self.cost_cny, 6),
            "priced": self.priced,
        }


# ---------------------------------------------------------------------- #
# 异常
# ---------------------------------------------------------------------- #


class LLMBackendError(RuntimeError):
    """后端调用失败。

    携带 ``retryable`` 标志：网络超时和限流值得重试，余额不足和参数错误
    重试多少次都一样。M2 任务拆解要求"网络超时、限流、余额不足、输出不
    可解析，均返回可理解错误"——区分可重试与否是"可理解"的最低要求。
    """

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        kind: str = "unknown",
        raw: str = "",
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.kind = kind
        #: 解析失败时模型的原始输出。**必须带着**——没有原文就分不清是
        #: 模型的问题还是解析层的问题，而那正是 M3 比较各模型格式稳定性
        #: 的依据。2026-08-25 的 8B 评测上栽过：22 条解析失败全部 raw=''，
        #: 结果这 22 条既看不出模型说了什么，又被 `summarize()` 从分母里
        #: 剔掉了，把 8B 的格式合规率从 38.7% 抬成 45.8%。
        self.raw = raw


# ---------------------------------------------------------------------- #
# 动作意图
# ---------------------------------------------------------------------- #


@dataclass
class ActionIntent:
    """模型对下一步的决定。

    ## 为什么不直接返回 `Action`

    模式 B 下模型给的是 ``{action: "left_click", target_description: "地址栏"}``，
    **没有坐标**，构造不出合法的 `Action`（`Action.validate` 会因缺 x/y 报错）。
    坐标要等 `GroundingBackend` 解析完才有。

    因此这里存的是"意图"：动作类型确定了，坐标可能还没有。Loop 负责在
    grounding 之后调用 `with_point` 补全成可执行的 `Action`。

    ## done 与 action_type 的关系

    模型判定当前子任务已完成时置 ``done=True``，此时 ``action_type`` 可以
    为空。M2 的子任务完成判定用简单规则（不做 Reflector，见「明确不做」
    第 5 条），模型自报是其中一条。
    """

    #: 动作类型与已知参数。模式 B 下坐标为 None，待 grounding 补全
    action_type: str = ""
    params: dict = field(default_factory=dict)
    #: 模式 B：目标元素的自然语言描述。模式 A 下为空
    target_description: str = ""
    #: 模型的思考过程。只进日志，不影响执行，但复盘时最有价值
    thinking: str = ""
    #: 模型认为当前子任务已完成
    done: bool = False
    #: 模型原始输出。解析失败时的唯一线索，必须留
    raw_text: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    #: 本次调用的费用（元）
    cost_cny: float = 0.0
    #: API 请求 ID，向平台报障用
    request_id: str = ""
    latency_ms: float = 0.0

    @property
    def requires_point(self) -> bool:
        """这个**动作类型**是否涉及坐标——与坐标在不在无关。

        `left_click` 恒为 True（哪怕模型忘了给 x/y），`type` / `wait` 恒为
        False。Loop 用它决定要不要走 grounding 这一步：不涉及坐标的动作
        根本不该去打扰定位后端。
        """
        if self.done or not self.action_type:
            return False
        try:
            probe = Action(type=self.action_type, x=0, y=0)
        except Exception:  # noqa: BLE001 - 未知动作类型，留给 to_action 报错
            return False
        return probe.requires_coordinates()

    @property
    def needs_grounding(self) -> bool:
        """是否**缺**坐标、需要 grounding 补上。

        判据是"该动作要坐标但还没有坐标"，**不是"当前配的是模式 A 还是 B"**
        ——模式 A 的模型偶尔也会漏给坐标，那种情况同样该走 grounding 兜底，
        而不是让 `Action.validate` 崩掉。判据绑在数据上而非配置上，才能兜住
        这类模型不听话的情形。
        """
        if not self.requires_point:
            return False
        return self.params.get("x") is None or self.params.get("y") is None

    def to_action(self) -> Action:
        """转成可执行的 `Action`。参数不合法时抛 ``ActionValidationError``。"""
        payload = {"action": self.action_type, **self.params}
        if self.thinking:
            payload.setdefault("reasoning", self.thinking)
        return Action.from_dict(payload)

    def with_point(self, x: int, y: int) -> ActionIntent:
        """补上 grounding 解析出的坐标，返回**新实例**。

        不原地改是有意的：原始意图要原样进轨迹日志。模式 B 下"模型说了
        什么"与"grounding 定位到哪"是两条独立证据，M3 分析 grounding 误差
        时需要它们分开可查。就地覆盖会把前一条抹掉。
        """
        return ActionIntent(
            action_type=self.action_type,
            params={**self.params, "x": x, "y": y},
            target_description=self.target_description,
            thinking=self.thinking,
            done=self.done,
            raw_text=self.raw_text,
            usage=self.usage,
            cost_cny=self.cost_cny,
            request_id=self.request_id,
            latency_ms=self.latency_ms,
        )

    def as_dict(self) -> dict:
        payload: dict = {
            "action_type": self.action_type,
            "params": dict(self.params),
            "done": self.done,
        }
        if self.target_description:
            payload["target_description"] = self.target_description
        if self.thinking:
            payload["thinking"] = self.thinking
        return payload


@dataclass
class RawResponse:
    """一次调用的原始产物：还没被解释成任何东西的文本 + 记账信息。

    ## 为什么要有这一层

    最初 `LLMBackend` 只有 `predict_action`，它发请求并**顺手把回复解析
    成动作**。Planner 借它做任务拆解时立刻炸了：模型正确返回了
    ``{"subtasks": [...]}``，却因为没有 ``action`` 字段被判"无法解析"。

    根因是把两件事焊在了一起——"发请求"和"按动作解析"。规划、动作决策、
    M3 的提示词消融、grounding 的坐标提取，都是"发请求"，但各自的解析
    规则完全不同。

    因此 `complete` 是原语（每个后端只需实现它），`predict_action` 退化
    成"complete + 按动作解析"的组合。副作用是动作解析从各后端里收归基类，
    M3 加本地后端时不用再抄一遍。
    """

    text: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    cost_cny: float = 0.0
    request_id: str = ""
    latency_ms: float = 0.0

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "usage": self.usage.as_dict(),
            "cost_cny": round(self.cost_cny, 6),
            "request_id": self.request_id,
            "latency_ms": round(self.latency_ms, 2),
        }


@dataclass
class HistoryStep:
    """历史中的一步，喂回给模型看。

    只保留模型需要的部分。完整记录在轨迹日志里，两者用途不同：这个是
    **给模型的输入**，要精简（每个 token 都要付费）；那个是**给人和分析
    脚本的证据**，要完整。混为一谈会两头不讨好。
    """

    action: Action | None = None
    thinking: str = ""
    screenshot: Screenshot | None = None
    success: bool = True
    error: str = ""

    #: 本轮回传时要不要带图。由 `agent.context.ContextWindow` 按策略决定，
    #: 后端只负责照做——**策略与执行分开**，因为上下文策略是 M3 要做消融
    #: 的变量，而后端不该知道消融这回事
    with_image: bool = True
    #: 带图时缩到原尺寸的多少倍。1.0 为全分辨率。
    #:
    #: 这个字段是"旧帧降采样"落地的地方。没有它，ContextWindow 算出的
    #: 分辨率决策就传不到编码那一步，降采样只能退化成"带或不带"。
    image_scale: float = 1.0

    def summary(self) -> str:
        """一行文本摘要，塞进提示词。"""
        if self.action is None:
            return "（无动作）"
        line = str(self.action)
        if not self.success:
            line += f" → 失败：{self.error}"
        return line


# ---------------------------------------------------------------------- #
# 抽象基类
# ---------------------------------------------------------------------- #


class LLMBackend(ABC):
    """规划模型后端。

    子类只需实现 `predict_action`。用量累计与成本计算由基类的
    `record_usage` 统一处理，**保证横评时各后端的成本口径一致**——口径
    不一致的成本对比毫无意义。
    """

    #: 后端标识，进轨迹日志。横评报告按这个字段分组
    name: str = "base"

    #: 送模型的默认用户提示模板。``{instruction}`` 会被子任务目标替换，
    #: ``{width}`` / ``{height}`` 被 **坐标系** 尺寸替换（不是图片尺寸，
    #: 见 `coordinate_space`）。图片尺寸另有 ``{image_width}`` /
    #: ``{image_height}`` 可用
    DEFAULT_USER_TEMPLATE = """当前子任务目标：{instruction}

这是当前屏幕。请判断下一步动作，坐标按 {width}×{height} 的范围给出。
若该子任务已完成，输出 {"done": true}。"""

    def __init__(self, model: str = "", price: PriceSheet | None = None) -> None:
        self.model = model
        self.price = price
        self._cost = CostInfo(model=model)

        # --- 提示词配置放在基类上 ---
        #
        # 这三项是**每个后端都该认的通用配置**，不是某个实现的私有属性。
        # 放基类的理由是 M2 设计思路那条硬要求：Agent 层对后端切换无感知。
        # 若只有 OpenAICompatBackend 有 system_prompt，Planner 就得先判断
        # 面对的是哪种后端才敢赋值——那等于把切换成本转嫁给了 Agent 层，
        # M3 加本地后端时这段判断还要再改一次。
        #
        # 不认这些配置的实现（比如测试用的 ScriptedBackend）留着不用即可，
        # 有属性总比让调用方做类型判断便宜。
        self.system_prompt: str = ""
        self.few_shot: list[dict] = []
        self.user_template: str = self.DEFAULT_USER_TEMPLATE

        #: **模型作答所用的坐标系尺寸**，用来渲染用户模板里的
        #: ``{width}`` / ``{height}``。与送去的图片尺寸是两回事。
        #:
        #: 这个字段是补一个实测发现的缺陷。`agent.session._prepare_backend`
        #: 一直用坐标系尺寸渲染**系统提示**（它的注释写着「提示词里说的
        #: 尺寸必须是坐标系的尺寸」），却把 `image_size` 交给了**用户模板**。
        #: 于是 M2 的每一次请求都同时告诉模型两个范围：
        #:
        #:     系统提示：坐标范围是 x ∈ [0, 1000)，y ∈ [0, 1000)
        #:     用户消息：坐标按 1024×768 的范围给出
        #:
        #: 两句话矛盾，而矛盾的后果不是报错，是模型按哪把尺子作答不确定
        #: ——看起来就像「模型定位不稳」。默认与 `SessionConfig.coordinate_space`
        #: 一致；输出图像像素坐标的模型应改成与图片尺寸相同。
        self.coordinate_space: tuple[int, int] = (1000, 1000)

    # ------------------------------------------------------------------ #

    @abstractmethod
    def complete(
        self,
        prompt: str,
        screenshot: Screenshot | None = None,
        history: list[HistoryStep] | None = None,
    ) -> RawResponse:
        """发一次请求，返回**原始文本**，不做任何解释。

        这是每个后端唯一必须实现的方法。截图为 None 时走纯文本调用。
        """

    def predict_action(
        self,
        instruction: str,
        screenshot: Screenshot,
        history: list[HistoryStep] | None = None,
    ) -> ActionIntent:
        """看当前截图与历史，决定下一步。

        ``instruction`` 是**单个子任务**的目标，不是整个任务——子任务粒度
        切到原子动作级是开源模型能跑起来的前提，见 M2 设计思路。

        解析放在基类而不是各后端里：动作解析规则只该有一份，M3 新增本地
        后端时不用抄。后端只管把文本拿回来。
        """
        from llm.parsing import OutputParseError, parse_action_payload

        raw = self.complete(instruction, screenshot, history)

        try:
            payload = parse_action_payload(raw.text)
        except OutputParseError as exc:
            # **解析失败要带着原文抛。** 没有原文就没法判断是模型的问题
            # 还是解析层的问题，而这正是 M3 比较各模型格式稳定性的依据
            raise LLMBackendError(
                f"模型输出无法解析：{exc}",
                retryable=True,
                kind="parse_error",
                raw=raw.text,
            ) from exc

        return ActionIntent(
            **payload,
            raw_text=raw.text,
            usage=raw.usage,
            cost_cny=raw.cost_cny,
            request_id=raw.request_id,
            latency_ms=raw.latency_ms,
        )

    def get_cost(self) -> CostInfo:
        """累计用量与成本。"""
        return self._cost

    def reset_cost(self) -> None:
        """清零。批量任务里按任务分别统计时用。"""
        self._cost = CostInfo(model=self.model)

    # ------------------------------------------------------------------ #

    def record_usage(self, usage: TokenUsage) -> float:
        """累计一次调用的用量，返回本次费用（元）。

        单价未知时费用记 0 并把 `CostInfo.priced` 置 False——**不猜单价**。
        宁可标"未知"，也不能给一个看起来像真的假数字：M0 的成本估算要靠
        这些数来校准，掺了猜测就白测了。
        """
        if self.price is None:
            self._cost.add(usage, 0.0, priced=False)
            return 0.0
        cost = self.price.cost_of(usage.prompt_tokens, usage.completion_tokens, usage.cached_tokens)
        self._cost.add(usage, cost)
        return cost

    def close(self) -> None:  # noqa: B027 - 可选钩子，不该逼每个后端写空实现
        """释放资源。

        默认空实现：API 后端没有需要释放的东西，M3 的本地模型后端才会用到
        （卸载权重、清显存）。故意不设 @abstractmethod——否则每个只调 API
        的后端都要写一遍 ``pass``，纯噪音。
        """

    def __enter__(self) -> LLMBackend:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} model={self.model!r}>"
