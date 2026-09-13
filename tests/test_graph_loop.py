"""LangGraph 引擎与 legacy 引擎的**逐字段**对照。

`tests/test_loop.py` 已经让每条用例在两个引擎上各跑一遍，但那些用例只断言
各自关心的几个字段。这里做更严的一件事：**同一个脚本分别喂给两个引擎，把
落盘的 `steps.jsonl` 逐条逐字段比对**，并比对模型实际收到的指令（含 Reflector
与重试策略追加的提示）、历史长度、截图次数。

为什么必须这么严：轨迹是 M3~M5 全部分析的唯一数据源。迁移引擎后只要有一个
字段悄悄变了——多了、少了、换了顺序、换了取值时机——前面几周积累的分析脚本
就会在不报错的情况下算出另一套数字。这正是本项目反复踩的那个坑。

比对时只剔除**天然随时间变化**的字段：时间戳、轨迹 ID、各段耗时。
"""

from __future__ import annotations

import json

import pytest

from agent.session import SessionConfig
from control.executor import ActionExecutor
from core.graph_loop import NODES_PER_STEP_MAX, GraphAgentLoop, build_agent_loop
from core.loop import (
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
)
from core.trajectory import TrajectoryWriter
from grounding.native import NativeGrounding
from llm.base import LLMBackendError, PriceSheet
from llm.fake import ScriptedBackend
from perception.coordinate import CoordinateScaler
from perception.types import BBox
from tests.test_loop import MODEL_H, MODEL_W, SCREEN, FakeCapturer
from tests.test_session import build as build_session
from tests.test_session import two_subtask_script

ENGINES = ("legacy", "langgraph")

#: 天然随时间变化、两次运行必然不同的字段
VOLATILE_KEYS = {"timestamp", "trajectory_id", "latency", "request_id"}


@pytest.fixture(autouse=True)
def _no_settle(monkeypatch):
    monkeypatch.setattr("core.loop.time.sleep", lambda _seconds: None)


def _strip_volatile(value):
    if isinstance(value, dict):
        return {
            k: _strip_volatile(v)
            for k, v in value.items()
            if k not in VOLATILE_KEYS and not k.endswith("_ms")
        }
    if isinstance(value, list):
        return [_strip_volatile(v) for v in value]
    return value


def _read_steps(path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [_strip_volatile(json.loads(line)) for line in lines if line.strip()]


def _run(engine, script, tmp_path, config, *, price=None, on_exhausted="done", setup=None):
    scaler = CoordinateScaler(SCREEN)
    scaler.register("planner", MODEL_W, MODEL_H)
    executor = ActionExecutor(scaler, space_name="planner", dry_run=True)
    backend = ScriptedBackend(script, price=price, on_exhausted=on_exhausted)
    writer = TrajectoryWriter("对照任务", root=tmp_path / engine)
    loop = build_agent_loop(
        llm=backend,
        grounding=NativeGrounding(MODEL_W, MODEL_H),
        executor=executor,
        capturer=FakeCapturer(),
        writer=writer,
        config=LoopConfig(**{**config, "engine": engine}),
    )
    if setup:
        setup(loop)
    result = loop.run_subtask("对照子任务", subtask_id=1)
    return {
        "result": result.as_dict(),
        "steps": _read_steps(writer.root / "steps.jsonl"),
        "instructions": [call["instruction"] for call in backend.calls],
        "history_lens": [call["history_len"] for call in backend.calls],
        "captures": loop.capturer.calls,
        "history": len(loop.history),
    }


def _trigger_emergency(loop) -> None:
    loop.executor.emergency_stop.trigger()


def _change_resolution(loop) -> None:
    other = CoordinateScaler(BBox(0, 0, SCREEN.width + 160, SCREEN.height + 90))
    other.register("planner", MODEL_W, MODEL_H)
    loop.executor.scaler = other


CLICK = {"action": "left_click", "x": 100, "y": 200, "thinking": "点开始菜单"}
BASE = {"max_iterations": 5, "save_frames": False}

#: 每一种停止路径至少一个场景，外加三个会改变单步内部走向的开关
SCENARIOS = {
    "正常完成": dict(
        script=[CLICK, {"action": "type", "text": "记事本"}, {"done": True, "thinking": "已打开"}],
        config=BASE,
        expect=STOP_DONE,
    ),
    "步数用尽": dict(
        script=[CLICK],
        config={**BASE, "max_iterations": 4},
        on_exhausted="repeat",
        expect=STOP_MAX_ITERATIONS,
    ),
    "坐标越界": dict(
        script=[{"action": "left_click", "x": 5000, "y": 10}],
        config=BASE,
        expect=STOP_GROUNDING_FAILED,
    ),
    "漏给坐标": dict(
        script=[{"action": "left_click", "thinking": "点地址栏"}],
        config=BASE,
        expect=STOP_GROUNDING_FAILED,
    ),
    "未知动作": dict(
        script=[{"action": "teleport", "x": 1, "y": 1}],
        config=BASE,
        expect=STOP_PARSE_ERROR,
    ),
    "不可重试的后端错误": dict(
        script=[LLMBackendError("余额不足", retryable=False, kind="insufficient_balance")],
        config=BASE,
        expect=STOP_BACKEND_ERROR,
    ),
    "可重试的后端错误后成功": dict(
        script=[LLMBackendError("超时", retryable=True, kind="timeout"), {"done": True}],
        config=BASE,
        expect=STOP_DONE,
    ),
    "成本熔断": dict(
        script=[CLICK],
        config={**BASE, "max_iterations": 20, "cost_limit_cny": 0.005},
        price=PriceSheet("m", input_per_1k=1.0, output_per_1k=1.0),
        on_exhausted="repeat",
        expect=STOP_COST_LIMIT,
    ),
    "分辨率被改": dict(
        script=[CLICK],
        config=BASE,
        setup=_change_resolution,
        expect=STOP_RESOLUTION_CHANGED,
    ),
    "急停": dict(
        script=[CLICK] * 3,
        config={**BASE, "max_iterations": 3},
        setup=_trigger_emergency,
        expect=(STOP_EMERGENCY, STOP_ACTION_FAILED),
    ),
    "Reflector 否决两次后兜底接受": dict(
        # 假截图恒为全黑，点击后帧差恒为「没变化」，于是 done 被否决，直到上限
        script=[CLICK, {"done": True}, {"done": True}, {"done": True}],
        config={**BASE, "max_iterations": 6, "reflector": True},
        expect=STOP_DONE,
    ),
    "Reflector 否决后模型改做动作": dict(
        # 否决提示只该回传一轮。上一个场景里每次否决都会重新设置提示，
        # 「用完不清」的错误会被下一次否决盖住、两个引擎产出一模一样——
        # 注入变异测试时就漏过了。这里否决之后插一个动作，提示若没清，
        # 第 4 步的指令就会多出一段。
        script=[CLICK, {"done": True}, CLICK, {"done": True}, {"done": True}],
        config={**BASE, "max_iterations": 6, "reflector": True},
        expect=STOP_DONE,
    ),
    "重试升级": dict(
        script=[CLICK, CLICK, CLICK, {"done": True}],
        config={**BASE, "escalate_on_no_change": True},
        expect=STOP_DONE,
    ),
    "保存前后帧": dict(
        script=[CLICK, {"done": True}],
        config={**BASE, "save_frames": True},
        expect=STOP_DONE,
    ),
}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_both_engines_persist_identical_trajectories(name, tmp_path) -> None:
    spec = SCENARIOS[name]
    kwargs = {k: spec[k] for k in ("price", "on_exhausted", "setup") if k in spec}
    runs = {
        engine: _run(engine, spec["script"], tmp_path, spec["config"], **kwargs)
        for engine in ENGINES
    }
    legacy, graph = runs["legacy"], runs["langgraph"]

    expect = spec["expect"]
    expected = expect if isinstance(expect, tuple) else (expect,)
    assert legacy["result"]["status"] in expected, "场景没有走到预期的停止路径，对照失去意义"

    assert graph["result"] == legacy["result"]
    assert len(graph["steps"]) == len(legacy["steps"])
    for index, (g, lg) in enumerate(zip(graph["steps"], legacy["steps"], strict=True), start=1):
        assert g == lg, f"第 {index} 步的落盘记录不一致"
    assert graph["instructions"] == legacy["instructions"], "模型收到的指令（含追加提示）不一致"
    assert graph["history_lens"] == legacy["history_lens"]
    assert graph["captures"] == legacy["captures"]
    assert graph["history"] == legacy["history"]


def test_reflector_scenario_really_rejects(tmp_path) -> None:
    """守卫：Reflector 场景必须真的走到否决边，否则「回到预算检查」那条边没被对照到。"""
    spec = SCENARIOS["Reflector 否决两次后兜底接受"]
    run = _run("langgraph", spec["script"], tmp_path, spec["config"])
    verdicts = [s["meta"]["reflector"]["verdict"] for s in run["steps"] if "reflector" in s["meta"]]
    assert verdicts.count("reject") == 2


def test_escalation_scenario_really_escalates(tmp_path) -> None:
    """守卫：重试升级场景必须真的发生升级，否则 5.5 那段没被对照到。"""
    spec = SCENARIOS["重试升级"]
    run = _run("langgraph", spec["script"], tmp_path, spec["config"])
    assert any(s["meta"].get("retry", {}).get("escalated") for s in run["steps"])


# ===================================================================== #
# LangGraph 特有的风险
# ===================================================================== #


def test_default_step_budget_does_not_hit_langgraph_recursion_limit(tmp_path) -> None:
    """**LangGraph 默认 25 个超步就抛 GraphRecursionError。**

    一步要走最多 8 个节点，而子任务默认允许 25 步——不放宽上限的话第 4 步
    左右就会被框架掐断，报错看起来像死循环。这里用默认步数跑满。
    """
    run = _run("langgraph", [CLICK], tmp_path, {"save_frames": False}, on_exhausted="repeat")
    assert run["result"]["status"] == STOP_MAX_ITERATIONS
    assert run["result"]["steps"] == LoopConfig().max_iterations


def test_recursion_limit_covers_the_longest_step() -> None:
    loop = build_agent_loop(
        llm=ScriptedBackend([]),
        grounding=NativeGrounding(MODEL_W, MODEL_H),
        executor=ActionExecutor(CoordinateScaler(SCREEN), dry_run=True),
        capturer=FakeCapturer(),
        config=LoopConfig(max_iterations=7, engine="langgraph"),
    )
    assert loop.recursion_limit > 7 * NODES_PER_STEP_MAX


def test_graph_has_one_node_per_numbered_block_of_the_legacy_step() -> None:
    """图的形状即文档。节点改名或增删时，模块文档与报告里的图要同步改。"""
    loop = build_agent_loop(
        llm=ScriptedBackend([]),
        grounding=NativeGrounding(MODEL_W, MODEL_H),
        executor=ActionExecutor(CoordinateScaler(SCREEN), dry_run=True),
        capturer=FakeCapturer(),
        config=LoopConfig(engine="langgraph"),
    )
    nodes = set(loop.graph.get_graph().nodes) - {"__start__", "__end__"}
    assert nodes == {
        "check_budget",
        "capture",
        "predict",
        "judge_done",
        "locate",
        "build_action",
        "execute",
        "observe",
        "finish_step",
    }
    assert len(nodes) == NODES_PER_STEP_MAX + 1  # judge_done 与 locate 之后四个节点二选一


# ===================================================================== #
# 单元测试的假对象暴露不了、真机上才会出事的三件事
# ===================================================================== #


def _spy_loop(script, monkeypatch, node, probe):
    """跑一个 langgraph 子任务，在指定节点里调用 probe() 并收集返回值。"""
    seen: list = []
    original = getattr(GraphAgentLoop, node)

    def spy(self, state):
        seen.append(probe())
        return original(self, state)

    monkeypatch.setattr(GraphAgentLoop, node, spy)
    scaler = CoordinateScaler(SCREEN)
    scaler.register("planner", MODEL_W, MODEL_H)
    loop = build_agent_loop(
        llm=ScriptedBackend(script),
        grounding=NativeGrounding(MODEL_W, MODEL_H),
        executor=ActionExecutor(scaler, space_name="planner", dry_run=True),
        capturer=FakeCapturer(),
        config=LoopConfig(max_iterations=5, save_frames=False, engine="langgraph"),
    )
    return loop, seen


def test_nodes_run_on_the_calling_thread(monkeypatch) -> None:
    """**节点必须在调用方线程里执行。**

    换到工作线程会出三件事：dxcam 截图与 UIA（COM 组件）通常绑定创建它们的
    线程；Ctrl+C 只打断主线程，工作线程里的键鼠操作会继续执行。

    现在成立是因为图是线性的，每个超步只有一个任务，LangGraph 内联执行。
    **这不是框架保证。** 后续阶段若加并行分支，LangGraph 会改用线程池——这条
    测试就是为那一刻准备的。
    """
    import threading

    main = threading.get_ident()
    loop, seen = _spy_loop(
        [CLICK, {"done": True}], monkeypatch, "_node_execute", threading.get_ident
    )
    loop.run_subtask("x")
    assert seen and set(seen) == {main}


@pytest.mark.parametrize("exc_type", [RuntimeError, KeyboardInterrupt])
def test_executor_errors_propagate_and_are_never_retried(exc_type) -> None:
    """**执行节点被重试，一次点击就会被执行两遍。**

    LangGraph 支持给节点挂重试策略，默认不挂。这里钉住「默认不挂」这件事：
    异常原样抛出，执行器只被调用一次；Ctrl+C 也要原样穿透，不能被框架吞掉。
    """
    scaler = CoordinateScaler(SCREEN)
    scaler.register("planner", MODEL_W, MODEL_H)
    loop = build_agent_loop(
        llm=ScriptedBackend([CLICK, {"done": True}]),
        grounding=NativeGrounding(MODEL_W, MODEL_H),
        executor=ActionExecutor(scaler, space_name="planner", dry_run=True),
        capturer=FakeCapturer(),
        config=LoopConfig(max_iterations=5, save_frames=False, engine="langgraph"),
    )
    calls = []

    def boom(action):
        calls.append(action)
        raise exc_type("执行器内部出错")

    loop.executor.execute = boom
    with pytest.raises(exc_type):
        loop.run_subtask("x")
    assert len(calls) == 1


_TRACING_PROBE = """
import sys
from langsmith.utils import tracing_is_enabled
import core.loop
core.loop.time.sleep = lambda _s: None
from control.executor import ActionExecutor
from core.graph_loop import GraphAgentLoop, build_agent_loop
from core.loop import LoopConfig
from grounding.native import NativeGrounding
from llm.fake import ScriptedBackend
from perception.coordinate import CoordinateScaler
from tests.test_loop import MODEL_H, MODEL_W, SCREEN, FakeCapturer

seen = []
original = GraphAgentLoop._node_capture
def spy(self, state):
    seen.append(tracing_is_enabled())
    return original(self, state)
GraphAgentLoop._node_capture = spy

scaler = CoordinateScaler(SCREEN)
scaler.register("planner", MODEL_W, MODEL_H)
loop = build_agent_loop(
    llm=ScriptedBackend([{"done": True}]),
    grounding=NativeGrounding(MODEL_W, MODEL_H),
    executor=ActionExecutor(scaler, space_name="planner", dry_run=True),
    capturer=FakeCapturer(),
    config=LoopConfig(max_iterations=2, save_frames=False, engine="langgraph"),
)
loop.run_subtask("x")
print("outside", tracing_is_enabled())
print("inside", seen[0])
"""


def test_langsmith_tracing_is_forced_off_inside_the_graph() -> None:
    """**轨迹截图不出客机。** 设了 LANGSMITH_TRACING 的机器上，节点状态不许被追踪上传。

    **必须在全新子进程里、启动时就设好变量。** langsmith 会缓存首次读到的
    环境变量——在当前进程里临时设置读到的是缓存值。排查时第一个探针就被这个
    缓存骗了，得出「设了也不激活」的错误结论。

    端点指向一个不可达地址，保证这条测试本身绝不会真的上传任何东西。
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    env = {
        **os.environ,
        "LANGSMITH_TRACING": "true",
        "LANGSMITH_API_KEY": "fake-key-for-test",
        "LANGSMITH_ENDPOINT": "http://127.0.0.1:9",
    }
    proc = subprocess.run(
        [sys.executable, "-c", _TRACING_PROBE],
        cwd=Path(__file__).resolve().parent.parent,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    lines = dict(line.split() for line in proc.stdout.strip().splitlines()[-2:])
    assert lines["outside"] == "True", "变量没生效，这条测试失去意义"
    assert lines["inside"] == "False", "图的节点内追踪仍是开着的"


# ===================================================================== #
# 没装 langgraph 的机器
# ===================================================================== #


def _hide_langgraph(monkeypatch) -> None:
    """模拟没装 langgraph：让 `import langgraph.graph` 抛 ImportError。"""
    import sys

    for name in [m for m in sys.modules if m == "langgraph" or m.startswith("langgraph.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "langgraph", None)
    monkeypatch.setitem(sys.modules, "langgraph.graph", None)


_WITHOUT_LANGGRAPH = """
import sys
sys.modules["langgraph"] = None
sys.modules["langgraph.graph"] = None

import agent.session  # 会话层在模块顶部导入了引擎工厂——这一行不许失败
from control.executor import ActionExecutor
from core.graph_loop import build_agent_loop
from core.loop import LoopConfig
from grounding.native import NativeGrounding
from llm.fake import ScriptedBackend
from perception.coordinate import CoordinateScaler
from tests.test_loop import MODEL_H, MODEL_W, SCREEN, FakeCapturer
import core.loop
core.loop.time.sleep = lambda _s: None

scaler = CoordinateScaler(SCREEN)
scaler.register("planner", MODEL_W, MODEL_H)
loop = build_agent_loop(
    llm=ScriptedBackend([{"action": "left_click", "x": 1, "y": 1}, {"done": True}]),
    grounding=NativeGrounding(MODEL_W, MODEL_H),
    executor=ActionExecutor(scaler, space_name="planner", dry_run=True),
    capturer=FakeCapturer(),
    config=LoopConfig(max_iterations=3, save_frames=False),
)
print(loop.run_subtask("x").status)
"""


def test_legacy_engine_still_works_without_langgraph() -> None:
    """**新代码不能把旧路径拖下水。**

    会话层在模块顶部导入了本引擎的工厂。若 langgraph 也在模块顶部导入，
    没装它的客机上连 legacy 引擎都跑不起来。

    **必须在全新子进程里测。** 本文件顶部早已导入过 `core.graph_loop`，在当前
    进程里再藏起 langgraph 不会重新触发模块导入——最初这条测试就是这么写的，
    用变异测试把导入改回模块顶部后它照样通过，等于什么都没守住。
    """
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [sys.executable, "-c", _WITHOUT_LANGGRAPH],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().splitlines()[-1] == STOP_DONE


def test_langgraph_engine_explains_the_missing_dependency(tmp_path, monkeypatch) -> None:
    _hide_langgraph(monkeypatch)
    with pytest.raises(RuntimeError, match="pip install langgraph"):
        _run("langgraph", [{"done": True}], tmp_path, BASE)


# ===================================================================== #
# 引擎选择
# ===================================================================== #


def test_default_engine_is_legacy() -> None:
    """默认必须是 legacy：M2~M4 的全部实测都跑在它上面，默认切换等于让复现命令悄悄换引擎。"""
    assert LoopConfig().engine == "legacy"


def test_unknown_engine_is_rejected() -> None:
    with pytest.raises(ValueError, match="engine"):
        LoopConfig(engine="autogen")


@pytest.mark.parametrize(("engine", "cls"), [("legacy", AgentLoop), ("langgraph", GraphAgentLoop)])
def test_factory_picks_the_engine(engine, cls) -> None:
    loop = build_agent_loop(
        llm=ScriptedBackend([]),
        grounding=NativeGrounding(MODEL_W, MODEL_H),
        executor=ActionExecutor(CoordinateScaler(SCREEN), dry_run=True),
        capturer=FakeCapturer(),
        config=LoopConfig(engine=engine),
    )
    assert type(loop) is cls


def test_session_runs_on_the_configured_engine(tmp_path, monkeypatch) -> None:
    """会话层必须经工厂建循环。直接实例化某个引擎的话，切引擎会漏掉这一处。"""
    built: list[type] = []
    original = GraphAgentLoop.run_subtask

    def spy(self, *args, **kwargs):
        built.append(type(self))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(GraphAgentLoop, "run_subtask", spy)
    config = SessionConfig(loop=LoopConfig(max_iterations=4, save_frames=False, engine="langgraph"))
    session, _ = build_session(two_subtask_script(), tmp_path, config=config)
    assert session.run("打开记事本").status == "completed"
    assert built and all(cls is GraphAgentLoop for cls in built)


def test_session_trajectories_match_across_engines(tmp_path) -> None:
    """会话层（拆解 + 多子任务）落盘的轨迹，两个引擎逐字段一致。"""
    steps = {}
    results = {}
    for engine in ENGINES:
        config = SessionConfig(loop=LoopConfig(max_iterations=4, save_frames=False, engine=engine))
        session, _ = build_session(two_subtask_script(), tmp_path / engine, config=config)
        result = session.run("打开记事本")
        results[engine] = (result.status, result.total_steps, [o.goal for o in result.outcomes])
        (traj,) = [p for p in (tmp_path / engine).iterdir() if p.is_dir()]
        steps[engine] = _read_steps(traj / "steps.jsonl")

    assert results["langgraph"] == results["legacy"]
    assert steps["langgraph"] == steps["legacy"]
