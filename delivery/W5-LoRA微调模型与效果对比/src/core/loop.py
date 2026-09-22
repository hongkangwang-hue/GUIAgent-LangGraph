"""Agent Loop —— 系统的心脏。

一轮做五件事，顺序固定：

    截图 → 问模型要动作 → grounding 定位 → 执行 → 等界面稳定后再截图

## 为什么自己写而不用 LangChain 的 AgentExecutor

M2 文档给了三条理由，落到代码上是这样的：

1. **交互形态不匹配。** ``AgentExecutor`` 的循环是"文本进 → 工具调用 →
   文本结果回传"。这里每轮的反馈是**图像**，不是文本。
2. **轨迹日志需要循环内部的完全控制。** 下面 `run_subtask` 里，一个
   `StepRecord` 的字段来自四个不同位置：模型响应、grounding 结果、执行
   结果、后置截图。回调接口够不到这些中间态。
3. **模式 A/B 的切换发生在单步内部**——在"拿到意图"和"执行"之间，不是
   工具层面的切换。

代价是这个文件要自己写；收益是上面每一条都不用绕。

## 三道刹车

开源规划模型在多步任务上会犯两类要命的错：原地打转，和越走越偏。因此
循环必须能自己停下来：

- ``max_iterations`` —— 步数上限。调试期务必调到 8，别用默认的 25
- ``cost_limit_cny`` —— 单任务成本熔断。防的是"模型卡在一个循环里，
  每轮都在烧钱"
- 急停热键 —— 由 `ActionExecutor` 负责，这里只需保证它返回的
  ``emergency_stopped`` 会让循环立刻退出

三道都触发不了的情况下，循环靠模型自报 ``done`` 结束。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from control.actions import ActionValidationError
from control.executor import ActionExecutor
from core.reflector import Reflector
from core.retry import RetryPolicy
from core.trajectory import LatencyBreakdown, StepRecord, TrajectoryWriter
from grounding.base import GroundingBackend, GroundingResult
from llm.base import ActionIntent, CostInfo, HistoryStep, LLMBackend, LLMBackendError
from perception.capture import ScreenCapturer, Screenshot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# 配置与结果
# ---------------------------------------------------------------------- #


@dataclass
class LoopConfig:
    """一轮循环的行为参数。"""

    #: 单个子任务的步数上限。**调试期请设成 8**：默认 25 意味着一次跑飞
    #: 要烧 25 次调用，等你发现时钱已经花完了
    max_iterations: int = 25

    #: 单任务成本上限（元）。超过即熔断
    cost_limit_cny: float = 1.0

    #: 动作执行后等多久再截图。等的是界面动画、加载、焦点切换——
    #: 不等就会拍到中间态，模型看着一张"正在展开的菜单"做决策，
    #: 下一步必错。M2 任务拆解给的范围是 0.5-1.5 秒
    settle_seconds: float = 0.8

    #: 回传给模型的历史步数。旧帧对当前决策的价值衰减很快，而每帧都要
    #: 付 token 费，k 太大是纯浪费
    history_k: int = 3

    #: 输出解析失败后的重试次数。开源模型的结构化输出稳定性低于前沿
    #: 模型，一次解析失败不该直接判死；但也不能无限重试，那会把成本
    #: 花在一个明显不听话的模型上
    parse_retries: int = 1

    #: 执行器使用的坐标系名。与 `CoordinateScaler` 注册的名字对应
    space_name: str = "planner"

    #: 每步是否落盘前后两帧。关掉能省磁盘，但 replay 就没图可放了
    save_frames: bool = True

    #: 动作执行后屏幕没变化时，自动升级策略（单击 → 双击），
    #: 并把"上一次点了没反应"回传给模型。大纲 W6 任务 2。
    #:
    #: **默认关闭。** 打开会改变行为，而 M2/M3 的全部实测都是在关闭下跑的
    #: ——默认打开等于让那些复现命令悄悄跑出另一套结果。要用就显式开，
    #: 这样 A/B 才做得成。
    escalate_on_no_change: bool = False

    #: 变化像素占比超过它算"屏幕动了"。见 `perception.change` 的模块文档
    #: ——这个阈值是拍的，不是测的。
    change_threshold: float = 0.01

    #: 模型报 `done` 时先做级联判定，无依据就否决。M4 任务 1 / 大纲 W6。
    #: 见 `agent/reflector.py` 与 `docs/m4-错误分类体系.md`。
    #:
    #: **默认关闭，理由与 `escalate_on_no_change` 相同**：打开会改变行为，
    #: 而 M2/M3 的全部实测都是在关闭下跑的。默认打开等于让那些复现命令
    #: 悄悄跑出另一套结果。
    #:
    #: **打开它会自动启用帧差** —— 级联 1 就是帧差，没有帧差 Reflector
    #: 只能一路 `unknown`，等于没开。
    reflector: bool = False

    #: 同一子任务里 Reflector 最多连续否决几次。到上限接受，交程序化判定兜底。
    reflector_max_rejects: int = 2

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations 至少为 1")
        if self.history_k < 0:
            raise ValueError("history_k 不能为负")


#: 循环的结束原因。
#:
#: 分这么细是因为**它们的含义完全不同**：``done`` 是模型说完成了，
#: ``max_iterations`` 是它没完没了，``cost_limit`` 是它在烧钱，
#: ``grounding_failed`` 是它给不出能用的坐标。M3 统计失败原因时，把这
#: 四种混成一个"失败"会丢掉最有价值的信息。
STOP_DONE = "done"
STOP_MAX_ITERATIONS = "max_iterations"
STOP_COST_LIMIT = "cost_limit"
STOP_GROUNDING_FAILED = "grounding_failed"
STOP_ACTION_FAILED = "action_failed"
STOP_EMERGENCY = "emergency_stopped"
STOP_BACKEND_ERROR = "backend_error"
STOP_PARSE_ERROR = "parse_error"
#: 截图尺寸与坐标系建立时的区域不一致。**虚拟机上最常见的成因是分辨率被
#: 自动改掉**（VMware Tools 默认让客机分辨率跟随窗口大小）。继续跑只会
#: 产出一串系统性偏移的点击，且看起来像模型定位不准——立刻停，比事后
#: 对着一堆偏移量猜便宜得多。
STOP_RESOLUTION_CHANGED = "resolution_changed"

#: grounding 未被调用（动作不涉及坐标）。与"调用了但没找到"必须分开，
#: 否则 M3 的定位成功率会被一堆 type / wait 动作稀释
GROUNDING_SKIPPED = "skipped"


@dataclass
class LoopResult:
    """一个子任务跑完的结果。"""

    status: str = STOP_DONE
    steps: int = 0
    reason: str = ""
    cost_cny: float = 0.0
    records: list[StepRecord] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        """**只有模型自报完成才算成功。**

        注意这不是任务真的成功了——M2 明确要求"成功判定必须程序化：
        进程检查、窗口标题检查、文件内容检查，不用人工目测"。模型说
        自己做完了和它真做完了是两回事，后者由 CLI 层的校验器判定。
        这个属性只回答"循环是正常收尾还是被刹车拦下的"。
        """
        return self.status == STOP_DONE

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "steps": self.steps,
            "reason": self.reason,
            "cost_cny": round(self.cost_cny, 6),
        }


# ---------------------------------------------------------------------- #
# 循环
# ---------------------------------------------------------------------- #


class AgentLoop:
    """把感知、规划、定位、执行串起来。

    这个类**只负责编排**。它不知道模型是哪家的，不知道坐标是模型给的
    还是本地算的，也不知道动作最后是怎么变成键鼠事件的——那三件事分别
    由 `LLMBackend`、`GroundingBackend`、`ActionExecutor` 负责。

    换后端不改这个文件，是 M2 验收标准第 8 条的具体含义。
    """

    def __init__(
        self,
        llm: LLMBackend,
        grounding: GroundingBackend,
        executor: ActionExecutor,
        capturer: ScreenCapturer,
        writer: TrajectoryWriter | None = None,
        config: LoopConfig | None = None,
        history_selector: Callable[[list[HistoryStep]], list[HistoryStep]] | None = None,
        on_step: Callable[[StepRecord], None] | None = None,
    ) -> None:
        self.llm = llm
        self.grounding = grounding
        self.executor = executor
        self.capturer = capturer
        self.writer = writer
        self.config = config or LoopConfig()
        #: 跨子任务累积，供上层做整任务的成本熔断
        self.history: list[HistoryStep] = []

        #: 从完整历史里挑出本轮要回传的部分。
        #:
        #: 做成注入的钩子而不是直接 import `agent.context`，是为了**依赖
        #: 方向不反**：Loop 是被编排的一层，不该反过来依赖编排层。上下文
        #: 策略是 M3 要做消融的变量，让它从外面进来，Loop 自己只保留一个
        #: 够用的默认实现（简单切片）。
        self.history_selector = history_selector
        #: 重试策略。每个子任务开始时 reset——不重置的话，上个子任务在同一
        #: 坐标上的尝试会让这个子任务的第一次点击就被升级。
        self.retry_policy = RetryPolicy()
        self.reflector = Reflector(max_rejects=self.config.reflector_max_rejects)
        #: Reflector 否决后要回传给模型的话。只带一轮，用完即清。
        self._reflector_hint = ""

        #: 每步落盘后触发，CLI 用它刷新实时面板。
        #: 回调抛异常不影响任务——显示层的问题不该把正在执行的任务带崩
        self.on_step = on_step

    # ------------------------------------------------------------------ #

    def run_subtask(self, subtask: str, subtask_id: int = 1) -> LoopResult:
        """把一个子任务跑到收尾或撞上刹车。

        ``subtask`` 是**单个原子级子目标**，不是整个任务。粒度切细是开源
        模型能跑起来的前提，见 M2 设计思路"两条针对开源模型的必要设计"。
        """
        result = LoopResult()
        self.retry_policy.reset()
        self.reflector.reset()
        self._reflector_hint = ""
        logger.info("子任务 #%d 开始：%s", subtask_id, subtask)

        for iteration in range(1, self.config.max_iterations + 1):
            if self._over_budget():
                return self._stop(
                    result,
                    STOP_COST_LIMIT,
                    f"累计成本 {self.llm.get_cost().cost_cny:.4f} 元已超上限 "
                    f"{self.config.cost_limit_cny} 元",
                )

            record, stop = self._run_one_step(subtask, subtask_id, iteration)
            if record is not None:
                result.records.append(record)
                result.steps = iteration
                result.cost_cny += record.cost_cny

            if stop is not None:
                status, reason = stop
                return self._stop(result, status, reason)

        return self._stop(
            result,
            STOP_MAX_ITERATIONS,
            f"已达步数上限 {self.config.max_iterations} 步，子任务仍未完成",
        )

    def run_subtasks(self, subtasks: list[str]) -> list[LoopResult]:
        """顺序跑多个子任务。**一个失败即停。**

        GUI 操作有强顺序依赖：第 2 步"在地址栏输入网址"依赖第 1 步"点开
        浏览器"成功。前一步没成还往下走，后面全是对着错误界面的无效点击，
        既浪费钱又把轨迹搞脏。
        """
        results: list[LoopResult] = []
        for index, subtask in enumerate(subtasks, start=1):
            result = self.run_subtask(subtask, subtask_id=index)
            results.append(result)
            if not result.succeeded:
                logger.warning(
                    "子任务 #%d（%s）以 %s 结束，后续 %d 个子任务不再执行",
                    index,
                    subtask,
                    result.status,
                    len(subtasks) - index,
                )
                break
        return results

    # ------------------------------------------------------------------ #
    # 单步
    # ------------------------------------------------------------------ #

    def _run_one_step(
        self, subtask: str, subtask_id: int, iteration: int
    ) -> tuple[StepRecord | None, tuple[str, str] | None]:
        """跑一步。返回 (记录, 停止信号)；停止信号为 None 表示继续。"""
        latency = LatencyBreakdown()
        step_index = self.writer.next_step_index() if self.writer else iteration

        record = StepRecord(
            step=step_index,
            subtask_id=subtask_id,
            subtask=subtask,
            backend=self.llm.name,
        )

        # --- 1. 截图 ---
        start = time.perf_counter()
        before = self.capturer.capture(fresh=True)
        latency.screenshot_ms += (time.perf_counter() - start) * 1000.0
        record.screenshot_before = self._save_frame(before, step_index, "before")

        mismatch = self._resolution_mismatch(before)
        if mismatch:
            record.execution_status = "error"
            record.error, record.error_type = mismatch, "resolution_changed"
            record.latency = latency.as_dict()
            return self._commit(record), (STOP_RESOLUTION_CHANGED, mismatch)

        # --- 2. 问模型 ---
        #
        # **升级动作改不了模型下一步的判断。** 只把 left_click 换成
        # double_click 的话，模型下一轮多半还给同一个坐标——探针量到的
        # 重复率 75.3% 就是这么来的。所以把"上一次点了没反应"写进这一轮
        # 的子任务描述里，让它自己换目标。
        asked = subtask
        hints: list[str] = []
        if self.config.escalate_on_no_change:
            hint = self.retry_policy.hint()
            if hint:
                hints.append(hint)
        if self._reflector_hint:
            hints.append(self._reflector_hint)
            self._reflector_hint = ""  # 只回传一轮，不累积
        if hints:
            joined = "\n\n".join(hints)
            asked = subtask + "\n\n" + joined
            record.meta["hint"] = joined
        try:
            intent, retries = self._predict(asked, before)
        except LLMBackendError as exc:
            record.execution_status = "error"
            record.error, record.error_type = str(exc), exc.kind
            record.latency = latency.as_dict()
            return self._commit(record), (STOP_BACKEND_ERROR, str(exc))

        latency.api_ms = intent.latency_ms
        record.retry_count = retries
        record.model_thinking = intent.thinking
        record.raw_output = intent.raw_text
        record.action_intent = intent.as_dict()
        record.request_id = intent.request_id
        record.tokens = intent.usage.as_dict()
        record.cost_cny = intent.cost_cny

        # --- 3. 模型自报完成 ---
        # **不打开 Reflector 时，这里与 M2/M3 实测时一字不差。**
        if intent.done:
            record.execution_status = "no_action"
            if self.config.reflector:
                verdict = self.reflector.judge()
                record.meta["reflector"] = verdict.as_dict()
                if verdict.rejected:
                    # 否决：不结束子任务，让循环进下一轮。反馈通过
                    # `_reflector_hint` 写进下一轮的子任务描述 —— 只改动作
                    # 类型改不了模型的判断，探针实测坐标重复率 75.3%。
                    self._reflector_hint = verdict.hint
                    logger.info("步骤 %d：%s", step_index, verdict.reason)
                    record.latency = latency.as_dict()
                    return self._commit(record), None
            record.latency = latency.as_dict()
            return self._commit(record), (STOP_DONE, intent.thinking or "模型报告子任务完成")

        # --- 4. grounding 定位 ---
        grounding_result = self._locate(intent, before)
        latency.grounding_ms = grounding_result.latency_ms
        record.grounding = grounding_result.as_dict()

        if grounding_result.source == GROUNDING_SKIPPED:
            # 这个动作压根不需要坐标（type / wait / key），继续
            pass
        elif grounding_result.found:
            intent = intent.with_point(grounding_result.point.x, grounding_result.point.y)
        else:
            # **grounding 跑了但没给出可用的点，就是走不下去了。**
            #
            # 判据是"定位有没有成功"，不是"坐标缺不缺"——模型给了 x=5000
            # 这种越界坐标时，坐标是"有"的，`needs_grounding` 为 False，
            # 但 `NativeGrounding` 已经判它不可用。若按缺不缺来判，这里
            # 会带着一个已知非法的坐标继续执行，最后被安全白名单拦下、
            # 报成 action_failed —— 根因（模型输出不合规）就此丢失，而
            # 那正是 NativeGrounding 存在的全部意义。
            #
            # M2 不做自动重试与失败重规划（「明确不做」第 4 条），
            # 失败即终止并记录。
            record.execution_status = "failed"
            record.error = grounding_result.error or "grounding 未能定位目标"
            record.error_type = "grounding_failed"
            record.latency = latency.as_dict()
            return self._commit(record), (STOP_GROUNDING_FAILED, record.error)

        # --- 5. 构造动作 ---
        try:
            action = intent.to_action()
        except ActionValidationError as exc:
            record.execution_status = "failed"
            record.error, record.error_type = str(exc), "invalid_action"
            record.latency = latency.as_dict()
            return self._commit(record), (STOP_PARSE_ERROR, str(exc))

        record.action_model_coords = action.to_dict()

        # --- 5.5 重试策略：同一位置点了没反应就升级（大纲 W6 任务 2）---
        if self.config.escalate_on_no_change:
            revision = self.retry_policy.revise(
                action.type.value, getattr(action, "x", None), getattr(action, "y", None)
            )
            record.meta["retry"] = revision.as_dict()
            if revision.escalated:
                # **只改动作类型，不改坐标。** 坐标是模型对目标位置的判断,
                # 策略没有比它更好的信息;要改的是"用什么方式碰它"。
                action = action.with_type(revision.action_type)
                record.action_model_coords = action.to_dict()
                logger.info("步骤 %d：%s", step_index, revision.reason)

        # --- 6. 执行 ---
        start = time.perf_counter()
        outcome = self.executor.execute(action)
        latency.execute_ms = (time.perf_counter() - start) * 1000.0

        record.action_real_coords = self._real_coords(outcome)
        record.execution_status = self._status_of(outcome)
        record.error, record.error_type = outcome.error, outcome.error_type

        # --- 7. 等界面稳定，再拍一张 ---
        after = None
        if outcome.success:
            time.sleep(self.config.settle_seconds)
            start = time.perf_counter()
            after = self.capturer.capture(fresh=True)
            latency.screenshot_ms += (time.perf_counter() - start) * 1000.0
            record.screenshot_after = self._save_frame(after, step_index, "after")

        # --- 7.5 这一步到底有没有用 ---
        # **两个开关共用这一次帧差。** Reflector 的级联 1 就是帧差，
        # 重复算一次纯属浪费；分别算还可能因为取帧时机不同得出不同结论。
        if self.config.escalate_on_no_change or self.config.reflector:
            from perception.change import compare

            report = compare(before, after, self.config.change_threshold)
            record.meta["change"] = report.as_dict()
            changed = report.changed if after is not None else None
            if self.config.escalate_on_no_change:
                self.retry_policy.observe(
                    action.type.value,
                    getattr(action, "x", None),
                    getattr(action, "y", None),
                    changed=changed,
                )
            if self.config.reflector:
                self.reflector.observe(changed=changed)

        record.latency = latency.as_dict()
        self._push_history(action, intent.thinking, after or before, outcome)
        self._commit(record)

        if outcome.error_type == "emergency_stopped":
            return record, (STOP_EMERGENCY, outcome.error)
        if not outcome.success:
            return record, (STOP_ACTION_FAILED, outcome.error)
        return record, None

    # ------------------------------------------------------------------ #
    # 各步的细节
    # ------------------------------------------------------------------ #

    def _resolution_mismatch(self, screenshot: Screenshot) -> str:
        """截图尺寸与坐标系区域不符时返回说明，相符返回空串。

        执行器没挂 scaler 时跳过检查——单元测试里的假执行器就是这种情况。
        """
        scaler = getattr(self.executor, "scaler", None)
        if scaler is None or not hasattr(scaler, "region_matches"):
            return ""
        if scaler.region_matches(screenshot.width, screenshot.height):
            return ""
        region = scaler.region
        return (
            f"截图尺寸 {screenshot.width}×{screenshot.height} 与坐标系建立时的区域 "
            f"{region.width}×{region.height} 不一致。分辨率很可能在运行中被改掉了"
            "（虚拟机请检查是否开着「自动调整客机分辨率」）。继续执行会让每次点击"
            "都系统性偏移，因此在此终止。"
        )

    def _predict(self, subtask: str, screenshot: Screenshot) -> tuple[ActionIntent, int]:
        """调用模型，必要时重试。返回 (意图, 重试次数)。

        只重试**可重试**的错误：网络超时和限流值得再试一次，余额不足和
        参数错误重试多少次都一样，只会多烧一次时间。
        """
        attempts = self.config.parse_retries + 1
        last: LLMBackendError | None = None

        for attempt in range(attempts):
            start = time.perf_counter()
            try:
                intent = self.llm.predict_action(subtask, screenshot, self._recent_history())
            except LLMBackendError as exc:
                last = exc
                if not exc.retryable or attempt == attempts - 1:
                    raise
                logger.warning("第 %d 次调用失败（%s），重试：%s", attempt + 1, exc.kind, exc)
                continue

            if not intent.latency_ms:
                intent.latency_ms = (time.perf_counter() - start) * 1000.0
            return intent, attempt

        raise last or LLMBackendError("模型调用失败", kind="unknown")

    def _locate(self, intent: ActionIntent, screenshot: Screenshot) -> GroundingResult:
        """定位。

        `type` / `wait` / `key` 这类不涉及坐标的动作直接跳过——它们没有
        目标可定位，送进去只会让后端白跑一趟，还会在轨迹里留下一条无意义
        的 grounding 记录，污染 M3 的定位成功率统计。

        **截图必须传下去。** 模式 A 的 `NativeGrounding` 用不着它（坐标是
        模型给的），但模式 B 的本地 grounding 模型正是靠看这张图才能把
        "地址栏"变成 (x, y)。这里传 None 的话，M3 换上模式 B 会立刻炸，
        而那时排查成本比现在写对高得多。
        """
        if not intent.requires_point:
            return GroundingResult(source=GROUNDING_SKIPPED, confidence=1.0)

        return self.grounding.locate(
            screenshot=screenshot,
            target_description=intent.target_description,
            intent=intent,
        )

    def _push_history(self, action, thinking: str, screenshot, outcome) -> None:
        self.history.append(
            HistoryStep(
                action=action,
                thinking=thinking,
                screenshot=screenshot,
                success=outcome.success,
                error=outcome.error,
            )
        )

    def _recent_history(self) -> list[HistoryStep]:
        """本轮要回传给模型的历史。

        注入了 `history_selector` 就用它（`agent.context.ContextWindow`
        会做分档降采样）；没注入则退回简单切片——够用，且让 Loop 在不接
        编排层时也能独立跑起来，单元测试因此不必拖上整个 agent 层。
        """
        if self.history_selector is not None:
            return self.history_selector(self.history)
        if self.config.history_k <= 0:
            return []
        return self.history[-self.config.history_k :]

    # ------------------------------------------------------------------ #
    # 小工具
    # ------------------------------------------------------------------ #

    def _over_budget(self) -> bool:
        cost = self.llm.get_cost()
        # 单价未知时熔断无从判起。此时**不熔断**并已在 CostInfo.priced
        # 上标了不可信 —— 拿一个恒为 0 的成本去比上限，等于关掉了熔断，
        # 那不如老实承认这一路没有成本保护
        if not cost.priced:
            return False
        return cost.cost_cny >= self.config.cost_limit_cny

    def _save_frame(self, screenshot: Screenshot, step: int, phase: str) -> str:
        if not (self.config.save_frames and self.writer):
            return ""
        path = self.writer.frame_path(step, phase)
        try:
            screenshot.save(str(path))
        except Exception as exc:  # noqa: BLE001
            # 存图失败不该中断任务：图是给事后看的，动作还得接着做
            logger.warning("第 %d 步的 %s 帧存盘失败：%s", step, phase, exc)
            return ""
        return self.writer.relative(path)

    @staticmethod
    def _real_coords(outcome) -> dict:
        payload = {"action": outcome.action.type.value}
        if outcome.real_point is not None:
            payload["x"], payload["y"] = outcome.real_point.as_tuple()
        if outcome.real_point_to is not None:
            payload["to_x"], payload["to_y"] = outcome.real_point_to.as_tuple()
        return payload

    @staticmethod
    def _status_of(outcome) -> str:
        if outcome.success:
            return "ok"
        # 被安全白名单拦下和执行出错是两回事，分开记 —— 前者说明模型想
        # 干危险的事，后者说明环境有问题，M4 的应对策略完全不同
        return "rejected" if outcome.error_type == "blocked" else "failed"

    def _commit(self, record: StepRecord) -> StepRecord:
        if self.writer:
            self.writer.append(record)
        if self.on_step:
            try:
                self.on_step(record)
            except Exception:  # noqa: BLE001 - 显示层不该拖垮执行层
                logger.warning("on_step 回调出错，已忽略", exc_info=True)
        return record

    def _stop(self, result: LoopResult, status: str, reason: str) -> LoopResult:
        result.status = status
        result.reason = reason
        logger.info("子任务结束：%s —— %s", status, reason)
        return result

    # ------------------------------------------------------------------ #

    def cost(self) -> CostInfo:
        return self.llm.get_cost()

    def __repr__(self) -> str:
        return (
            f"<AgentLoop llm={self.llm.name!r} grounding={self.grounding.name!r} "
            f"max_iter={self.config.max_iterations}>"
        )
