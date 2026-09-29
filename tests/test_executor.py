"""执行器与急停测试。

全部跑在 `dry_run=True` 下：走完校验、坐标转换、安全检查的完整流程，
但不真的发键鼠事件。**测试不能夺走开发者的鼠标**——那会让本地跑测试
变成一件危险的事，久而久之就没人跑了。
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from control.actions import Action, ActionType
from control.emergency_stop import EmergencyStop, EmergencyStopped
from control.executor import ActionExecutor, ActionResult
from control.safety import SafetyGuard
from perception.coordinate import CoordinateScaler
from perception.types import BBox, Point

SCREEN = BBox(0, 0, 2560, 1600)


@pytest.fixture
def executor() -> ActionExecutor:
    scaler = CoordinateScaler(SCREEN)
    scaler.register("planner", 1024, 768)
    return ActionExecutor(scaler, space_name="planner", dry_run=True)


def test_type_keeps_clipboard_until_paste_has_been_processed(
    executor: ActionExecutor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """模拟 Edge 延后读取 Ctrl+V：不能读到运行前复制的 PowerShell 命令。"""
    original = "cd C:\\Users\\22900\\GUIAgent-LangGraph\ngit pull origin main"
    clipboard = {"text": original}
    queued_pastes: list = []
    fake_clipboard = SimpleNamespace(
        copy=lambda value: clipboard.__setitem__("text", value),
        paste=lambda: clipboard["text"],
    )
    monkeypatch.setitem(sys.modules, "pyperclip", fake_clipboard)
    monkeypatch.setattr("control.executor.time.sleep", lambda _: None)
    executor._pyautogui = SimpleNamespace(
        hotkey=lambda *_: queued_pastes.append(lambda: clipboard["text"])
    )
    result = ActionResult(Action(ActionType.TYPE, text="Python 官方文档"), False)

    executor._type_text("Python 官方文档", result)
    assert len(queued_pastes) == 1
    assert queued_pastes[0]() == "Python 官方文档"
    assert clipboard["text"] == "Python 官方文档"

    executor.stop()
    assert clipboard["text"] == original


def test_type_rejects_clipboard_changed_by_another_program(
    executor: ActionExecutor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """VMware 共享剪贴板若覆盖搜索词，应在 Ctrl+V 前失败而非粘错文本。"""
    clipboard = {"text": "原内容"}

    def copy_then_overwrite(_: str) -> None:
        clipboard["text"] = "外部覆盖"

    monkeypatch.setitem(
        sys.modules,
        "pyperclip",
        SimpleNamespace(copy=copy_then_overwrite, paste=lambda: clipboard["text"]),
    )
    monkeypatch.setattr("control.executor.time.sleep", lambda _: None)
    pasted: list[tuple[str, ...]] = []
    executor._pyautogui = SimpleNamespace(hotkey=lambda *keys: pasted.append(keys))

    executor.dry_run = False
    result = executor.execute(Action(ActionType.TYPE, text="Python 官方文档"))
    assert not result.success
    assert "粘贴前被其他程序改写" in result.error
    assert pasted == []


# --------------------------------------------------------------------- #
# 坐标转换的可追溯性 —— M1 验收标准
# --------------------------------------------------------------------- #


def test_result_records_real_click_point(executor: ActionExecutor) -> None:
    """验收要求"模型输出坐标与实际点击坐标的映射关系可追溯"。"""
    result = executor.execute(Action(ActionType.LEFT_CLICK, x=512, y=384))
    assert result.success
    assert result.real_point is not None
    assert abs(result.real_point.x - 1280) <= 3
    assert abs(result.real_point.y - 800) <= 3


def test_real_point_appears_in_log_payload(executor: ActionExecutor) -> None:
    payload = executor.execute(Action(ActionType.LEFT_CLICK, x=100, y=100)).as_dict()
    assert payload["real_point"] is not None
    assert payload["action"]["action"] == "left_click"


def test_drag_records_both_endpoints(executor: ActionExecutor) -> None:
    result = executor.execute(Action(ActionType.LEFT_CLICK_DRAG, x=100, y=100, to_x=500, to_y=400))
    assert result.real_point is not None and result.real_point_to is not None
    assert result.real_point_to.x > result.real_point.x


def test_non_coordinate_action_has_no_real_point(executor: ActionExecutor) -> None:
    assert executor.execute(Action(ActionType.WAIT, duration=0.1)).real_point is None


# --------------------------------------------------------------------- #
# 失败以返回值表达，不抛异常
# --------------------------------------------------------------------- #


def test_blocked_action_returns_failure_not_exception(executor: ActionExecutor) -> None:
    """Agent Loop 需要的是可判定的结果对象，不是满地的 try/except。"""
    result = executor.execute(Action(ActionType.TYPE, text="shutdown /s /t 0"))
    assert not result.success
    assert result.error_type == "blocked"
    assert result.verdict is not None and result.verdict.rule == "dangerous_text"


def test_out_of_bounds_action_blocked(executor: ActionExecutor) -> None:
    result = executor.execute(Action(ActionType.LEFT_CLICK, x=5000, y=100))
    assert not result.success
    assert result.verdict.rule == "out_of_bounds"


def test_stub_action_reports_not_implemented(executor: ActionExecutor) -> None:
    result = executor.execute(Action(ActionType.MIDDLE_CLICK, x=1, y=1))
    assert not result.success
    assert result.verdict.rule == "stub_action"


def test_screenshot_without_capturer_fails_cleanly(executor: ActionExecutor) -> None:
    executor.dry_run = False
    executor._pyautogui = object()  # 避免真的导入 pyautogui
    result = executor.execute(Action(ActionType.SCREENSHOT))
    assert not result.success
    assert "ScreenCapturer" in result.error


# --------------------------------------------------------------------- #
# 历史与统计
# --------------------------------------------------------------------- #


def test_history_records_every_attempt(executor: ActionExecutor) -> None:
    """失败的尝试也要进历史——M4 的错误分类正是靠这些失败样本建立的。"""
    executor.execute(Action(ActionType.LEFT_CLICK, x=1, y=1))
    executor.execute(Action(ActionType.TYPE, text="format c:"))
    assert len(executor.history) == 2


def test_stats_breaks_down_failures(executor: ActionExecutor) -> None:
    executor.execute(Action(ActionType.LEFT_CLICK, x=1, y=1))
    executor.execute(Action(ActionType.TYPE, text="format c:"))
    executor.execute(Action(ActionType.LEFT_CLICK, x=9999, y=1))

    stats = executor.stats()
    assert stats["total"] == 3
    assert stats["succeeded"] == 1
    assert stats["failures_by_type"]["blocked"] == 2
    assert stats["blocked_count"] == 2


def test_execute_all_stops_on_first_failure(executor: ActionExecutor) -> None:
    """GUI 操作有强顺序依赖，前一步没成功后面全是无意义的点击。"""
    results = executor.execute_all(
        [
            Action(ActionType.LEFT_CLICK, x=1, y=1),
            Action(ActionType.TYPE, text="shutdown /s"),
            Action(ActionType.LEFT_CLICK, x=2, y=2),
        ]
    )
    assert len(results) == 2


def test_execute_all_can_continue_on_failure(executor: ActionExecutor) -> None:
    results = executor.execute_all(
        [
            Action(ActionType.TYPE, text="shutdown /s"),
            Action(ActionType.LEFT_CLICK, x=2, y=2),
        ],
        stop_on_failure=False,
    )
    assert len(results) == 2
    assert results[1].success


# --------------------------------------------------------------------- #
# 急停
# --------------------------------------------------------------------- #


def test_emergency_stop_blocks_subsequent_actions(executor: ActionExecutor) -> None:
    assert executor.execute(Action(ActionType.LEFT_CLICK, x=1, y=1)).success

    executor.emergency_stop.trigger()

    result = executor.execute(Action(ActionType.LEFT_CLICK, x=2, y=2))
    assert not result.success
    assert result.error_type == "emergency_stopped"


def test_emergency_stop_precedes_safety_check(executor: ActionExecutor) -> None:
    """急停触发后连安全检查都不该再跑——任何动作都要立刻拒绝。"""
    executor.emergency_stop.trigger()
    result = executor.execute(Action(ActionType.TYPE, text="shutdown /s"))
    assert result.error_type == "emergency_stopped"


def test_emergency_stop_requires_manual_reset(executor: ActionExecutor) -> None:
    executor.emergency_stop.trigger()
    assert not executor.execute(Action(ActionType.LEFT_CLICK, x=1, y=1)).success

    executor.emergency_stop.reset()
    assert executor.execute(Action(ActionType.LEFT_CLICK, x=1, y=1)).success


def test_emergency_stop_records_trigger_time() -> None:
    stop = EmergencyStop()
    assert stop.triggered_at is None
    stop.trigger()
    assert stop.triggered_at is not None


def test_emergency_stop_fires_callback() -> None:
    fired = []
    stop = EmergencyStop(on_trigger=lambda: fired.append(True))
    stop.trigger()
    assert fired == [True]


def test_emergency_stop_callback_error_does_not_break_stop() -> None:
    """回调里出错不能让急停本身失效——那是最不能失效的东西。"""

    def boom() -> None:
        raise RuntimeError("回调炸了")

    stop = EmergencyStop(on_trigger=boom)
    stop.trigger()
    assert stop.is_triggered


def test_emergency_stop_is_idempotent() -> None:
    calls = []
    stop = EmergencyStop(on_trigger=lambda: calls.append(1))
    stop.trigger()
    stop.trigger()
    assert len(calls) == 1


def test_raise_if_triggered() -> None:
    stop = EmergencyStop()
    stop.raise_if_triggered()
    stop.trigger()
    with pytest.raises(EmergencyStopped):
        stop.raise_if_triggered()


def test_wait_returns_true_after_trigger() -> None:
    stop = EmergencyStop()
    stop.trigger()
    assert stop.wait(timeout=0.01)


def test_wait_times_out_when_not_triggered() -> None:
    assert EmergencyStop().wait(timeout=0.01) is False


def test_interruptible_sleep_aborts_on_stop(executor: ActionExecutor) -> None:
    """长 wait 期间按急停必须立刻生效，不能等睡完。"""
    executor.emergency_stop.trigger()
    with pytest.raises(EmergencyStopped):
        executor._interruptible_sleep(5.0)


# --------------------------------------------------------------------- #
# 自定义 guard
# --------------------------------------------------------------------- #


def test_custom_guard_is_used() -> None:
    scaler = CoordinateScaler(SCREEN)
    scaler.register("planner", 1024, 768)
    permissive = SafetyGuard(rules=[], raise_on_block=False)
    executor = ActionExecutor(scaler, guard=permissive, dry_run=True)

    # 规则清空后高危文本不再被拦，但坐标越界检查仍在（它不在 rules 里）
    assert executor.execute(Action(ActionType.TYPE, text="shutdown /s")).success
    assert not executor.execute(Action(ActionType.LEFT_CLICK, x=9999, y=1)).success


@pytest.mark.parametrize("raise_on_block", [True, False])
def test_blocking_works_regardless_of_raise_mode(raise_on_block: bool) -> None:
    """`raise_on_block` 只该决定报错方式，绝不能决定拦不拦。

    这条测试锁住的是一个真实出现过的漏洞：执行器最初只 `except ActionBlocked`，
    于是把开关设成 False 时，被判定为不允许的动作照样执行了。
    """
    scaler = CoordinateScaler(SCREEN)
    scaler.register("planner", 1024, 768)
    guard = SafetyGuard(raise_on_block=raise_on_block)
    executor = ActionExecutor(scaler, guard=guard, dry_run=True)

    result = executor.execute(Action(ActionType.TYPE, text="format c: /q"))
    assert not result.success
    assert result.error_type == "blocked"
    assert result.verdict.rule == "dangerous_text"


def test_scaler_region_check(executor: ActionExecutor) -> None:
    assert executor.scaler.is_in_region(Point(1280, 800))
    assert not executor.scaler.is_in_region(Point(-1, 800))
