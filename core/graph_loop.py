"""LangGraph 版 Agent Loop —— 同一套单步逻辑，改由状态图编排。

大纲第 3 周任务 2 要求「基于 LangChain / LlamaIndex 搭建基础多模态 Agent 框架」。
v1.0 的执行循环是 `core.loop.AgentLoop` 自写的，LangChain 只用在调模型那一层。
这个模块把**控制流**迁到 LangGraph（LangChain 官方的 Agent 编排层），
**领域操作**（截图、问模型、定位、执行、落盘）继续沿用 `AgentLoop` 的辅助方法。

## 当初不用 AgentExecutor 的三条理由，在 StateGraph 下都不成立了

1. 交互形态不匹配 —— 节点是自己定义的，每轮反馈是截图就把截图放进状态
2. 轨迹字段散在循环各处、回调够不着 —— 每个节点直接往同一条 `StepRecord` 上写
3. 模式 A/B 切换发生在单步内部 —— 就是「问模型」与「执行」之间的一条边

## 图的形状

    预算检查 ──超成本/超步数──────────────────────────────► 结束
       │
      截图 ──分辨率变了──────────────────────────► 收尾本步
       │
     问模型 ──后端报错──────────────────────────► 收尾本步
       │ └──模型说完成──► Reflector 判定 ─────────► 收尾本步
       │
      定位 ──定位失败────────────────────────────► 收尾本步
       │
    构造动作 ──动作非法──────────────────────────► 收尾本步
       │
      执行 ──► 拍后图、帧差 ──────────────────────► 收尾本步
                                                      │
                     有停止信号 ──► 结束               │
                     没有 ──► 回到预算检查 ◄───────────┘

`core.loop.AgentLoop._run_one_step` 里带编号的 1~7 段，逐段对应到下面的节点，
**每段的语句原样搬过来，只把 `return` 换成写状态**。那 7 处提前 `return`
在这里变成了图上看得见的边——后续阶段要建模的急停与人工干预，就挂在这些边上。

## 与 legacy 的一份临时重复

节点里的单步逻辑与 `AgentLoop._run_one_step` 是**两份**。这是「两个引擎并存、
端到端对照通过后删 legacy」这个决定的直接代价：legacy 必须保持原样，A/B 才是
真的在跟旧代码比。防止两份跑偏靠的是测试——`tests/test_loop.py` 整个文件在
两个引擎上各跑一遍，`tests/test_graph_loop.py` 逐字段比对两者落盘的轨迹。

## 状态里放的是 Python 对象

截图、意图、执行结果都直接放进状态，**没有挂检查点**，因此不需要序列化。
第 4 阶段用 `interrupt` 建模急停时要挂检查点，届时截图需改为按帧路径引用。
"""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Any, TypedDict

from control.actions import ActionValidationError
from core.loop import (
    GROUNDING_SKIPPED,
    STOP_ACTION_FAILED,
    STOP_BACKEND_ERROR,
    STOP_COST_LIMIT,
    STOP_DONE,
    STOP_EMERGENCY,
    STOP_GROUNDING_FAILED,
    STOP_MAX_ITERATIONS,
    STOP_PARSE_ERROR,
    STOP_RESOLUTION_CHANGED,
    AgentLoop,
    LoopConfig,
    LoopResult,
)
from core.trajectory import LatencyBreakdown, StepRecord
from llm.base import LLMBackendError

logger = logging.getLogger(__name__)

#: 一次迭代最多经过几个节点：预算检查、截图、问模型、定位、构造动作、执行、
#: 拍后图、收尾本步。LangGraph 默认 25 个超步就抛 `GraphRecursionError`，
#: 而一个子任务默认跑 25 次迭代——不按这个数放宽，第 4 步左右就会被框架掐断，
#: 且报错看起来像死循环。
NODES_PER_STEP_MAX = 8


class LoopState(TypedDict, total=False):
    """一个子任务在图里流转的状态。"""

    subtask: str
    subtask_id: int
    iteration: int
    result: LoopResult
    #: 子任务的结束原因 (status, reason)。由「预算检查」或「收尾本步」写入
    terminal: tuple[str, str] | None

    # ---- 单步暂存，每步开始时由「预算检查」清空 ----
    record: StepRecord | None
    latency: LatencyBreakdown | None
    step_index: int
    before: Any
    after: Any
    intent: Any
    action: Any
    outcome: Any
    #: 本步的停止信号 (status, reason)；None 表示继续下一步
    stop: tuple[str, str] | None
    #: 本步是否要把动作写进历史。只有真正执行过动作的步才写，与 legacy 一致
    push_history: bool


def _tracing_disabled():
    """在图的执行期间强制关闭 LangSmith 追踪。

    **本项目的硬约束：轨迹截图不出客机。** 而一旦机器上设了
    ``LANGSMITH_TRACING=true``，LangGraph 会为每个节点建一条追踪记录，把节点的
    输入输出——也就是状态里的截图、子任务原文、文件路径——送往 LangSmith 云端。
    实测过：启动时设了这个变量，节点内的追踪就是开着的。宿主机与仓库当前都
    没设，但这是一台机器改一个环境变量就会踩中的雷，不能指望每个人都记得。

    没装 langsmith 时它本来就追踪不了，直接放行。
    """
    try:
        from langsmith import tracing_context
    except ImportError:
        return contextlib.nullcontext()
    return tracing_context(enabled=False)


class GraphAgentLoop(AgentLoop):
    """`AgentLoop` 的 LangGraph 实现。对外接口与 `AgentLoop` 完全一致。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._graph = None

    # ------------------------------------------------------------------ #

    def run_subtask(self, subtask: str, subtask_id: int = 1) -> LoopResult:
        """把一个子任务跑到收尾或撞上刹车。行为与 `AgentLoop.run_subtask` 一致。"""
        result = LoopResult()
        self.retry_policy.reset()
        self.reflector.reset()
        self._reflector_hint = ""
        logger.info("子任务 #%d 开始：%s", subtask_id, subtask)

        with _tracing_disabled():
            final = self.graph.invoke(
                {
                    "subtask": subtask,
                    "subtask_id": subtask_id,
                    "iteration": 1,
                    "result": result,
                    "terminal": None,
                },
                config={"recursion_limit": self.recursion_limit},
            )
        status, reason = final["terminal"]
        return self._stop(final["result"], status, reason)

    @property
    def recursion_limit(self) -> int:
        # 最后一次「预算检查」判超步数也占一个超步，再留一点余量
        return self.config.max_iterations * NODES_PER_STEP_MAX + 10

    @property
    def graph(self):
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    # ------------------------------------------------------------------ #
    # 建图
    # ------------------------------------------------------------------ #

    def _build_graph(self):
        # **延迟导入。** `agent.session` 在模块顶部导入本模块以拿到工厂函数，
        # 若这里在模块顶部导入 langgraph，没装 langgraph 的机器（比如依赖较旧的
        # 客机）连 legacy 引擎都跑不起来——新代码把旧路径也拖下水。
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:
            raise RuntimeError(
                "engine=langgraph 需要 langgraph：pip install langgraph。"
                "不装的话可以继续用默认的 engine=legacy。"
            ) from exc

        g = StateGraph(LoopState)
        g.add_node("check_budget", self._node_check_budget)
        g.add_node("capture", self._node_capture)
        g.add_node("predict", self._node_predict)
        g.add_node("judge_done", self._node_judge_done)
        g.add_node("locate", self._node_locate)
        g.add_node("build_action", self._node_build_action)
        g.add_node("execute", self._node_execute)
        g.add_node("observe", self._node_observe)
        g.add_node("finish_step", self._node_finish_step)

        g.add_edge(START, "check_budget")
        g.add_conditional_edges(
            "check_budget", self._route_terminal_or, {"end": END, "next": "capture"}
        )
        g.add_conditional_edges(
            "capture", self._route_stop_or, {"finish": "finish_step", "next": "predict"}
        )
        g.add_conditional_edges(
            "predict",
            self._route_after_predict,
            {"finish": "finish_step", "done": "judge_done", "next": "locate"},
        )
        g.add_edge("judge_done", "finish_step")
        g.add_conditional_edges(
            "locate", self._route_stop_or, {"finish": "finish_step", "next": "build_action"}
        )
        g.add_conditional_edges(
            "build_action", self._route_stop_or, {"finish": "finish_step", "next": "execute"}
        )
        g.add_edge("execute", "observe")
        g.add_edge("observe", "finish_step")
        g.add_conditional_edges(
            "finish_step", self._route_terminal_or, {"end": END, "next": "check_budget"}
        )
        return g.compile()

    # ---- 路由 ----

    @staticmethod
    def _route_terminal_or(state: LoopState) -> str:
        return "end" if state.get("terminal") else "next"

    @staticmethod
    def _route_stop_or(state: LoopState) -> str:
        return "finish" if state.get("stop") else "next"

    @staticmethod
    def _route_after_predict(state: LoopState) -> str:
        if state.get("stop"):
            return "finish"
        return "done" if state["intent"].done else "next"

    # ------------------------------------------------------------------ #
    # 节点
    # ------------------------------------------------------------------ #

    def _node_check_budget(self, state: LoopState) -> dict:
        """对应 `AgentLoop.run_subtask` 的 for 循环头与成本熔断。"""
        iteration = state["iteration"]
        if iteration > self.config.max_iterations:
            return {
                "terminal": (
                    STOP_MAX_ITERATIONS,
                    f"已达步数上限 {self.config.max_iterations} 步，子任务仍未完成",
                )
            }
        if self._over_budget():
            return {
                "terminal": (
                    STOP_COST_LIMIT,
                    f"累计成本 {self.llm.get_cost().cost_cny:.4f} 元已超上限 "
                    f"{self.config.cost_limit_cny} 元",
                )
            }

        subtask, subtask_id = state["subtask"], state["subtask_id"]
        step_index = self.writer.next_step_index() if self.writer else iteration
        return {
            "record": StepRecord(
                step=step_index,
                subtask_id=subtask_id,
                subtask=subtask,
                backend=self.llm.name,
            ),
            "latency": LatencyBreakdown(),
            "step_index": step_index,
            "before": None,
            "after": None,
            "intent": None,
            "action": None,
            "outcome": None,
            "stop": None,
            "push_history": False,
        }

    def _node_capture(self, state: LoopState) -> dict:
        """--- 1. 截图 ---"""
        record, latency, step_index = state["record"], state["latency"], state["step_index"]

        start = time.perf_counter()
        before = self.capturer.capture(fresh=True)
        latency.screenshot_ms += (time.perf_counter() - start) * 1000.0
        record.screenshot_before = self._save_frame(before, step_index, "before")

        mismatch = self._resolution_mismatch(before)
        if mismatch:
            record.execution_status = "error"
            record.error, record.error_type = mismatch, "resolution_changed"
            return {"before": before, "stop": (STOP_RESOLUTION_CHANGED, mismatch)}
        return {"before": before}

    def _node_predict(self, state: LoopState) -> dict:
        """--- 2. 问模型 ---"""
        record, latency = state["record"], state["latency"]
        subtask, before = state["subtask"], state["before"]

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
            return {"stop": (STOP_BACKEND_ERROR, str(exc))}

        latency.api_ms = intent.latency_ms
        record.retry_count = retries
        record.model_thinking = intent.thinking
        record.raw_output = intent.raw_text
        record.action_intent = intent.as_dict()
        record.request_id = intent.request_id
        record.tokens = intent.usage.as_dict()
        record.cost_cny = intent.cost_cny
        return {"intent": intent}

    def _node_judge_done(self, state: LoopState) -> dict:
        """--- 3. 模型自报完成 ---

        **不打开 Reflector 时，这里与 M2/M3 实测时一字不差。**
        """
        record, intent, step_index = state["record"], state["intent"], state["step_index"]

        record.execution_status = "no_action"
        if self.config.reflector:
            verdict = self.reflector.judge()
            record.meta["reflector"] = verdict.as_dict()
            if verdict.rejected:
                # 否决：不结束子任务，进下一轮。反馈写进下一轮的子任务描述
                self._reflector_hint = verdict.hint
                logger.info("步骤 %d：%s", step_index, verdict.reason)
                return {"stop": None}
        return {"stop": (STOP_DONE, intent.thinking or "模型报告子任务完成")}

    def _node_locate(self, state: LoopState) -> dict:
        """--- 4. grounding 定位 ---"""
        record, latency = state["record"], state["latency"]
        intent, before = state["intent"], state["before"]

        grounding_result = self._locate(intent, before)
        latency.grounding_ms = grounding_result.latency_ms
        record.grounding = grounding_result.as_dict()

        if grounding_result.source == GROUNDING_SKIPPED:
            # 这个动作压根不需要坐标（type / wait / key），继续
            pass
        elif grounding_result.found:
            intent = intent.with_point(grounding_result.point.x, grounding_result.point.y)
        else:
            # grounding 跑了但没给出可用的点，就是走不下去了。
            # 判据是「定位有没有成功」而不是「坐标缺不缺」，理由见 legacy 同一段注释。
            record.execution_status = "failed"
            record.error = grounding_result.error or "grounding 未能定位目标"
            record.error_type = "grounding_failed"
            return {"stop": (STOP_GROUNDING_FAILED, record.error)}
        return {"intent": intent}

    def _node_build_action(self, state: LoopState) -> dict:
        """--- 5. 构造动作 --- 与 --- 5.5 重试策略 ---"""
        record, intent, step_index = state["record"], state["intent"], state["step_index"]

        try:
            action = intent.to_action()
        except ActionValidationError as exc:
            record.execution_status = "failed"
            record.error, record.error_type = str(exc), "invalid_action"
            return {"stop": (STOP_PARSE_ERROR, str(exc))}

        record.action_model_coords = action.to_dict()

        if self.config.escalate_on_no_change:
            revision = self.retry_policy.revise(
                action.type.value, getattr(action, "x", None), getattr(action, "y", None)
            )
            record.meta["retry"] = revision.as_dict()
            if revision.escalated:
                # 只改动作类型，不改坐标
                action = action.with_type(revision.action_type)
                record.action_model_coords = action.to_dict()
                logger.info("步骤 %d：%s", step_index, revision.reason)
        return {"action": action}

    def _node_execute(self, state: LoopState) -> dict:
        """--- 6. 执行 ---"""
        record, latency, action = state["record"], state["latency"], state["action"]

        start = time.perf_counter()
        outcome = self.executor.execute(action)
        latency.execute_ms = (time.perf_counter() - start) * 1000.0

        record.action_real_coords = self._real_coords(outcome)
        record.execution_status = self._status_of(outcome)
        record.error, record.error_type = outcome.error, outcome.error_type
        return {"outcome": outcome}

    def _node_observe(self, state: LoopState) -> dict:
        """--- 7. 等界面稳定，再拍一张 --- 与 --- 7.5 这一步到底有没有用 ---"""
        record, latency = state["record"], state["latency"]
        action, outcome = state["action"], state["outcome"]
        before, step_index = state["before"], state["step_index"]

        after = None
        if outcome.success:
            start = time.perf_counter()
            # **复用 AgentLoop 的方法，不再抄一份。** 固定等待与自适应等待的
            # 选择是单步逻辑，两个引擎之间只允许在编排上不同。
            after = self._settle_and_capture(record)
            latency.screenshot_ms += (time.perf_counter() - start) * 1000.0
            record.screenshot_after = self._save_frame(after, step_index, "after")

        # 两个开关共用这一次帧差。拍到了执行后的截图就**一律记下**，
        # 开关只决定要不要据此改变行为——「首步成功率」要看第一个动作有没有让
        # 屏幕变化，而评测默认两个开关都关。逐字对应 `AgentLoop._run_one_step`
        # 第 7.5 段：两份实现之间的差异只允许在编排上，不允许在单步逻辑上。
        if after is not None or self.config.escalate_on_no_change or self.config.reflector:
            from perception.change import compare

            report = compare(before, after, self.config.change_threshold)
            record.meta["change"] = report.as_dict()
        if self.config.escalate_on_no_change or self.config.reflector:
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

        if outcome.error_type == "emergency_stopped":
            stop = (STOP_EMERGENCY, outcome.error)
        elif not outcome.success:
            stop = (STOP_ACTION_FAILED, outcome.error)
        else:
            stop = None
        return {"after": after, "stop": stop, "push_history": True}

    def _node_finish_step(self, state: LoopState) -> dict:
        """收尾本步：落盘、计入结果，决定结束还是进下一步。

        对应 legacy 每个分支末尾的 ``record.latency = ...`` → （动作步才有的
        ``_push_history``）→ ``_commit`` → ``run_subtask`` 里的累加，**顺序不变**。
        """
        record, result, iteration = state["record"], state["result"], state["iteration"]

        record.latency = state["latency"].as_dict()
        if state.get("push_history"):
            self._push_history(
                state["action"],
                state["intent"].thinking,
                state["after"] or state["before"],
                state["outcome"],
            )
        self._commit(record)

        result.records.append(record)
        result.steps = iteration
        result.cost_cny += record.cost_cny

        stop = state.get("stop")
        if stop is not None:
            return {"terminal": stop}
        return {"iteration": iteration + 1}

    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"<GraphAgentLoop llm={self.llm.name!r} grounding={self.grounding.name!r} "
            f"max_iter={self.config.max_iterations}>"
        )


def build_agent_loop(**kwargs) -> AgentLoop:
    """按 ``config.engine`` 挑执行引擎。参数与 `AgentLoop` 的构造参数一致。

    上层（`agent.session.Session`、实测脚本）一律经这里创建循环，
    **不直接实例化某个引擎**——否则切引擎要改好几处，漏一处就是 A/B 对照里
    一边悄悄跑着旧引擎。
    """
    config = kwargs.get("config") or LoopConfig()
    kwargs["config"] = config
    if config.engine == "langgraph":
        return GraphAgentLoop(**kwargs)
    return AgentLoop(**kwargs)
