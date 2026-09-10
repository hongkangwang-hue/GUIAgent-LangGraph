"""Agent Loop 的单元测试。

测的是**编排逻辑**，不是模型能力：三道刹车会不会按时踩、坐标有没有在
正确的时机转换、失败时轨迹落没落盘、历史窗口截没截。这些和模型聪不
聪明毫无关系，却每一条都能毁掉一次任务。

因此这里用 `ScriptedBackend` 把模型输出钉死，用 ``dry_run`` 的执行器和
假 capturer 隔掉真实键鼠与屏幕。整个文件不需要 API key、不需要桌面，
在 CI 上跑得完。
"""

from __future__ import annotations

import numpy as np
import pytest

from control.executor import ActionExecutor
from core.loop import (
    GROUNDING_SKIPPED,
    STOP_ACTION_FAILED,
    STOP_BACKEND_ERROR,
    STOP_COST_LIMIT,
    STOP_DONE,
    STOP_GROUNDING_FAILED,
    STOP_MAX_ITERATIONS,
    STOP_PARSE_ERROR,
    STOP_RESOLUTION_CHANGED,
    AgentLoop,
    LoopConfig,
)
from core.trajectory import TrajectoryReader, TrajectoryWriter
from grounding.native import NativeGrounding
from llm.base import LLMBackendError, PriceSheet, TokenUsage
from llm.fake import ScriptedBackend
from perception.capture import Screenshot
from perception.coordinate import CoordinateScaler
from perception.types import BBox

SCREEN = BBox(0, 0, 2560, 1600)
MODEL_W, MODEL_H = 1024, 768


class FakeCapturer:
    """返回合成截图。不碰真实屏幕，CI 上也能跑。"""

    def __init__(self) -> None:
        self.calls = 0

    def capture(self, monitor: int = 1, fresh: bool = False) -> Screenshot:
        self.calls += 1
        return Screenshot(
            image=np.zeros((SCREEN.height, SCREEN.width, 3), dtype=np.uint8),
            region=SCREEN,
            engine="fake",
        )


@pytest.fixture(autouse=True)
def _no_settle(monkeypatch):
    """把动作后的稳定等待掐掉，否则每个用例要多等 0.8 秒。

    等待本身的行为由 `test_settle_wait_happens` 单独验证。
    """
    monkeypatch.setattr("core.loop.time.sleep", lambda _seconds: None)


def build_loop(script, tmp_path=None, config=None, price=None, on_exhausted="done"):
    scaler = CoordinateScaler(SCREEN)
    scaler.register("planner", MODEL_W, MODEL_H)
    executor = ActionExecutor(scaler, space_name="planner", dry_run=True)
    backend = ScriptedBackend(script, price=price, on_exhausted=on_exhausted)
    writer = TrajectoryWriter("测试任务", root=tmp_path) if tmp_path else None
    loop = AgentLoop(
        llm=backend,
        grounding=NativeGrounding(MODEL_W, MODEL_H),
        executor=executor,
        capturer=FakeCapturer(),
        writer=writer,
        config=config or LoopConfig(max_iterations=5, save_frames=False),
    )
    return loop, backend, writer


# ===================================================================== #
# 正常路径
# ===================================================================== #


def test_runs_until_model_reports_done() -> None:
    loop, _, _ = build_loop(
        [
            {"action": "left_click", "x": 100, "y": 200, "thinking": "点开始菜单"},
            {"action": "type", "text": "记事本"},
            {"done": True, "thinking": "已打开"},
        ]
    )
    result = loop.run_subtask("打开记事本")
    assert result.status == STOP_DONE
    assert result.succeeded
    assert result.steps == 3


def test_screenshot_taken_before_and_after_each_action() -> None:
    """动作后必须重新截图回传，否则模型看到的永远是第一帧。"""
    loop, _, _ = build_loop([{"action": "left_click", "x": 1, "y": 1}, {"done": True}])
    loop.run_subtask("x")
    # 第 1 步：前 + 后；第 2 步（done）：只有前
    assert loop.capturer.calls == 3


def test_settle_wait_happens(monkeypatch) -> None:
    """不等界面稳定就截图，会拍到菜单展开一半的中间态，模型下一步必错。"""
    slept: list[float] = []
    monkeypatch.setattr("core.loop.time.sleep", slept.append)

    loop, _, _ = build_loop(
        [{"action": "left_click", "x": 1, "y": 1}, {"done": True}],
        config=LoopConfig(max_iterations=5, settle_seconds=1.2, save_frames=False),
    )
    loop.run_subtask("x")
    assert slept == [1.2]


def test_history_is_capped_at_k() -> None:
    """历史窗口要截断：旧帧价值衰减快，而每帧都要付 token 费。"""
    loop, backend, _ = build_loop(
        [{"action": "left_click", "x": 1, "y": 1}] * 6,
        config=LoopConfig(max_iterations=6, history_k=2, save_frames=False),
    )
    loop.run_subtask("x")
    assert [call["history_len"] for call in backend.calls] == [0, 1, 2, 2, 2, 2]


def test_history_k_zero_sends_nothing() -> None:
    loop, backend, _ = build_loop(
        [{"action": "left_click", "x": 1, "y": 1}] * 3,
        config=LoopConfig(max_iterations=3, history_k=0, save_frames=False),
    )
    loop.run_subtask("x")
    assert all(call["history_len"] == 0 for call in backend.calls)


def test_subtask_goal_is_what_reaches_the_model() -> None:
    """进 Loop 的是**单个子任务**，不是整个任务。"""
    loop, backend, _ = build_loop([{"done": True}])
    loop.run_subtask("点击开始菜单", subtask_id=2)
    assert backend.calls[0]["instruction"] == "点击开始菜单"


# ===================================================================== #
# 三道刹车
# ===================================================================== #


def test_max_iterations_stops_a_spinning_model() -> None:
    """模型原地打转时必须自己停下来。"""
    loop, _, _ = build_loop(
        [{"action": "left_click", "x": 1, "y": 1}],
        config=LoopConfig(max_iterations=4, save_frames=False),
        on_exhausted="repeat",
    )
    result = loop.run_subtask("x")
    assert result.status == STOP_MAX_ITERATIONS
    assert result.steps == 4


def test_cost_limit_breaks_the_circuit() -> None:
    price = PriceSheet("m", input_per_1k=1.0, output_per_1k=1.0)
    loop, _, _ = build_loop(
        [{"action": "left_click", "x": 1, "y": 1}],
        config=LoopConfig(max_iterations=20, cost_limit_cny=0.005, save_frames=False),
        price=price,
        on_exhausted="repeat",
    )
    result = loop.run_subtask("x")
    assert result.status == STOP_COST_LIMIT
    assert result.steps < 20


def test_cost_limit_inactive_when_price_unknown() -> None:
    """单价未知时熔断无从判起。

    拿一个恒为 0 的成本去比上限，等于悄悄关掉了熔断——那还不如老实地
    不熔断，并在 CostInfo.priced 上标明这一路没有成本保护。
    """
    loop, _, _ = build_loop(
        [{"action": "left_click", "x": 1, "y": 1}],
        config=LoopConfig(max_iterations=3, cost_limit_cny=0.0001, save_frames=False),
        price=None,
        on_exhausted="repeat",
    )
    result = loop.run_subtask("x")
    assert result.status == STOP_MAX_ITERATIONS
    assert loop.cost().priced is False


def test_emergency_stop_ends_the_loop() -> None:
    """急停触发后必须立刻退出，不能再发下一个动作。"""
    loop, _, _ = build_loop(
        [{"action": "left_click", "x": 1, "y": 1}] * 3,
        config=LoopConfig(max_iterations=3, save_frames=False),
    )
    loop.executor.emergency_stop.trigger()
    result = loop.run_subtask("x")
    assert result.steps == 1
    assert result.status in ("emergency_stopped", STOP_ACTION_FAILED)


# ===================================================================== #
# 模型不听话
# ===================================================================== #


def test_missing_coordinates_ends_with_grounding_failed() -> None:
    """模式 A 下模型漏给坐标。终止原因要具体到 grounding_failed。"""
    loop, _, _ = build_loop([{"action": "left_click", "thinking": "点地址栏"}])
    result = loop.run_subtask("x")
    assert result.status == STOP_GROUNDING_FAILED
    assert result.records[-1].error_type == "grounding_failed"


def test_out_of_space_coordinates_end_with_grounding_failed() -> None:
    """越界要归到 grounding，不是归到执行器的安全拦截。

    两者的根因完全不同：前者是模型输出不合规，后者是动作本身危险。
    M3 统计模式 A 失败原因时必须分得开。
    """
    loop, _, _ = build_loop([{"action": "left_click", "x": 5000, "y": 10}])
    result = loop.run_subtask("x")
    assert result.status == STOP_GROUNDING_FAILED
    assert result.records[-1].grounding["meta"]["reason"] == "out_of_space"


def test_unknown_action_type_ends_with_parse_error() -> None:
    loop, _, _ = build_loop([{"action": "teleport", "x": 1, "y": 1}])
    assert loop.run_subtask("x").status == STOP_PARSE_ERROR


def test_coordinate_free_action_skips_grounding() -> None:
    """打字、等待没有目标可定位，送进 grounding 只会污染定位成功率统计。"""
    loop, _, _ = build_loop([{"action": "type", "text": "你好"}, {"done": True}])
    result = loop.run_subtask("x")
    assert result.records[0].grounding["source"] == GROUNDING_SKIPPED


def test_non_retryable_backend_error_stops_immediately() -> None:
    boom = LLMBackendError("余额不足", retryable=False, kind="insufficient_balance")
    loop, backend, _ = build_loop([boom])
    result = loop.run_subtask("x")
    assert result.status == STOP_BACKEND_ERROR
    assert len(backend.calls) == 1  # 没有白重试


def test_retryable_error_is_retried_once() -> None:
    """限流值得再试一次；余额不足重试多少次都一样。"""
    loop, backend, _ = build_loop(
        [
            LLMBackendError("限流", retryable=True, kind="rate_limit"),
            {"action": "left_click", "x": 1, "y": 1},
            {"done": True},
        ]
    )
    result = loop.run_subtask("x")
    assert result.status == STOP_DONE
    assert result.records[0].retry_count == 1
    assert len(backend.calls) == 3


def test_retry_budget_is_finite() -> None:
    """不能无限重试——那会把钱花在一个明显不听话的模型上。"""
    loop, backend, _ = build_loop(
        [LLMBackendError("限流", retryable=True, kind="rate_limit")] * 5,
        config=LoopConfig(max_iterations=3, parse_retries=1, save_frames=False),
    )
    assert loop.run_subtask("x").status == STOP_BACKEND_ERROR
    assert len(backend.calls) == 2


# ===================================================================== #
# 多子任务
# ===================================================================== #


def test_subtasks_run_in_order() -> None:
    loop, backend, _ = build_loop([{"done": True}], on_exhausted="done")
    results = loop.run_subtasks(["第一步", "第二步", "第三步"])
    assert len(results) == 3
    assert [call["instruction"] for call in backend.calls] == ["第一步", "第二步", "第三步"]


def test_subtasks_stop_at_first_failure() -> None:
    """GUI 操作有强顺序依赖，前一步没成往下走全是无效点击。"""
    loop, _, _ = build_loop(
        [{"action": "left_click", "thinking": "漏了坐标"}],
        on_exhausted="repeat",
    )
    results = loop.run_subtasks(["第一步", "第二步", "第三步"])
    assert len(results) == 1
    assert results[0].status == STOP_GROUNDING_FAILED


def test_cost_accumulates_across_subtasks() -> None:
    price = PriceSheet("m", input_per_1k=0.001, output_per_1k=0.0)
    loop, backend, _ = build_loop([{"done": True}], price=price)
    loop.run_subtasks(["a", "b", "c"])
    assert backend.get_cost().requests == 3


# ===================================================================== #
# 轨迹落盘
# ===================================================================== #


def test_trajectory_records_both_coordinate_systems(tmp_path) -> None:
    """验收标准第 7 条：模型坐标与实际点击坐标的映射要可追溯。

    1024×768 的 (512, 384) 映射到 2560×1600 的屏幕上应当是中心附近。
    """
    loop, _, writer = build_loop(
        [{"action": "left_click", "x": 512, "y": 384}, {"done": True}], tmp_path=tmp_path
    )
    loop.run_subtask("x")
    writer.finish()

    step = TrajectoryReader(writer.root).steps()[0]
    assert step.action_model_coords["x"] == 512
    assert step.action_real_coords["x"] == pytest.approx(1280, abs=2)
    assert step.action_real_coords["y"] == pytest.approx(800, abs=2)


def test_trajectory_records_four_latency_segments(tmp_path) -> None:
    loop, _, writer = build_loop(
        [{"action": "left_click", "x": 1, "y": 1}, {"done": True}], tmp_path=tmp_path
    )
    loop.run_subtask("x")
    writer.finish()

    latency = TrajectoryReader(writer.root).steps()[0].latency
    for segment in ("api_ms", "grounding_ms", "execute_ms", "screenshot_ms", "total_ms"):
        assert segment in latency


def test_failed_step_is_still_recorded(tmp_path) -> None:
    """失败的那一步尤其要落盘——它正是 M4 要分析的样本。"""
    loop, _, writer = build_loop(
        [{"action": "left_click", "thinking": "漏坐标"}], tmp_path=tmp_path
    )
    loop.run_subtask("x")
    writer.finish()

    steps = TrajectoryReader(writer.root).steps()
    assert len(steps) == 1
    assert steps[0].execution_status == "failed"


def test_step_numbers_continue_across_subtasks(tmp_path) -> None:
    """步号在整条轨迹里连续，子任务号单独记。回放时才不会出现两个 #1。"""
    loop, _, writer = build_loop(
        [{"action": "left_click", "x": 1, "y": 1}, {"done": True}],
        tmp_path=tmp_path,
        on_exhausted="done",
    )
    loop.run_subtasks(["第一步", "第二步"])
    writer.finish()

    steps = TrajectoryReader(writer.root).steps()
    assert [s.step for s in steps] == list(range(1, len(steps) + 1))
    assert {s.subtask_id for s in steps} == {1, 2}


def test_thinking_and_raw_output_are_kept(tmp_path) -> None:
    """解析失败时 raw_output 是唯一线索；thinking 是复盘时最有价值的字段。"""
    loop, _, writer = build_loop(
        [{"action": "left_click", "x": 1, "y": 1, "thinking": "点开始菜单", "raw_text": "{...}"}],
        tmp_path=tmp_path,
        config=LoopConfig(max_iterations=1, save_frames=False),
    )
    loop.run_subtask("x")
    writer.finish()

    step = TrajectoryReader(writer.root).steps()[0]
    assert step.model_thinking == "点开始菜单"
    assert step.raw_output == "{...}"


def test_frames_are_saved_when_enabled(tmp_path) -> None:
    loop, _, writer = build_loop(
        [{"action": "left_click", "x": 1, "y": 1}, {"done": True}],
        tmp_path=tmp_path,
        config=LoopConfig(max_iterations=3, save_frames=True),
    )
    loop.run_subtask("x")
    writer.finish()

    step = TrajectoryReader(writer.root).steps()[0]
    assert step.screenshot_before == "frames/step001-before.png"
    assert (writer.root / step.screenshot_before).exists()
    assert (writer.root / step.screenshot_after).exists()


def test_frame_save_failure_does_not_abort_the_task(tmp_path, monkeypatch) -> None:
    """图是给事后看的，存不下也不该中断正在进行的任务。"""
    loop, _, writer = build_loop(
        [{"action": "left_click", "x": 1, "y": 1}, {"done": True}],
        tmp_path=tmp_path,
        config=LoopConfig(max_iterations=3, save_frames=True),
    )
    monkeypatch.setattr(
        Screenshot, "save", lambda self, path: (_ for _ in ()).throw(OSError("磁盘满"))
    )
    result = loop.run_subtask("x")
    assert result.status == STOP_DONE
    assert result.records[0].screenshot_before == ""


def test_loop_works_without_a_writer() -> None:
    """不落盘也要能跑——写单元测试和快速试验时不该被迫建目录。"""
    loop, _, _ = build_loop([{"done": True}])
    assert loop.run_subtask("x").status == STOP_DONE


# ===================================================================== #
# 配置
# ===================================================================== #


def test_config_rejects_nonsense() -> None:
    with pytest.raises(ValueError):
        LoopConfig(max_iterations=0)
    with pytest.raises(ValueError):
        LoopConfig(history_k=-1)


def test_loop_result_as_dict() -> None:
    loop, _, _ = build_loop([{"done": True}])
    payload = loop.run_subtask("x").as_dict()
    assert payload["status"] == STOP_DONE
    assert "cost_cny" in payload


def test_repr_shows_wiring() -> None:
    loop, _, _ = build_loop([{"done": True}])
    text = repr(loop)
    assert "scripted" in text and "native" in text


def test_usage_is_recorded_on_each_step(tmp_path) -> None:
    loop, _, writer = build_loop(
        [{"action": "left_click", "x": 1, "y": 1}, {"done": True}], tmp_path=tmp_path
    )
    loop.run_subtask("x")
    writer.finish()

    step = TrajectoryReader(writer.root).steps()[0]
    assert step.tokens["total_tokens"] == TokenUsage(1000, 50).total_tokens
    assert step.backend == "scripted"


# ===================================================================== #
# 分辨率被改掉
# ===================================================================== #


def test_loop_stops_when_screenshot_size_leaves_the_mapped_region() -> None:
    """截图尺寸与坐标系区域不符时必须立刻停。

    虚拟机上这事的成因通常是 VMware Tools 让客机分辨率跟随窗口大小——
    拖一下窗口边框，1920×1080 就悄悄变成 1760×990，**没有任何提示**。
    继续跑不会崩，只会产出一串系统性偏移的点击，看起来像模型定位不准。
    """
    from perception.types import BBox

    loop, _, _ = build_loop([{"action": "left_click", "x": 10, "y": 20}])
    # 假截图仍是 SCREEN 尺寸，把 scaler 换成按另一个分辨率建立的映射，
    # 等价于"运行中分辨率被改掉了"
    other = CoordinateScaler(BBox(0, 0, SCREEN.width + 160, SCREEN.height + 90))
    other.register("planner", MODEL_W, MODEL_H)
    loop.executor.scaler = other

    result = loop.run_subtask("点击开始按钮")

    assert result.status == STOP_RESOLUTION_CHANGED
    assert f"{SCREEN.width + 160}×{SCREEN.height + 90}" in result.reason
    assert result.records[0].error_type == "resolution_changed"


def test_loop_runs_normally_when_region_matches() -> None:
    """默认配置下区域是吻合的，闸门不该误伤——其余全部用例也都走这条路径。"""
    loop, _, _ = build_loop([{"done": True}])
    assert loop.run_subtask("随便什么").succeeded


def test_scaler_region_matches() -> None:
    from perception.types import BBox

    scaler = CoordinateScaler(BBox(0, 0, 1920, 1080))
    assert scaler.region_matches(1920, 1080)
    assert not scaler.region_matches(1760, 990)
    assert scaler.region.width == 1920
