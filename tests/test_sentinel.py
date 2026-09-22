"""安全哨兵与自动急停测试。

每条规则都对应 M2 实测里的一次安全事件（`docs/m2-basic-tasks-report.md` §3.2、§7.5）。
全部不碰真实桌面：前台窗口、窗口列表、进程列表都由测试注入，执行器跑 `dry_run`。
"""

from __future__ import annotations

import pytest

from control.actions import Action, ActionType
from control.emergency_stop import EmergencyStop
from control.executor import ActionExecutor
from control.sentinel import (
    SafetySentinel,
    SentinelEvent,
    WindowInfo,
    match_window,
)
from perception.coordinate import CoordinateScaler
from perception.types import BBox

SCREEN = BBox(0, 0, 1024, 768)


def sentinel_with(title: str = "", process: str = "", **kwargs) -> SafetySentinel:
    window = WindowInfo(title, process) if (title or process) else None
    return SafetySentinel(foreground=lambda: window, windows=list, processes=list, **kwargs)


def enter() -> Action:
    return Action(ActionType.KEY, keys="enter")


# --------------------------------------------------------------------- #
# 敏感窗口：三次实测事件
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("title", "rule"),
    [
        (".env - 记事本", "protected_file"),  # 事件 1：记事本会话恢复把 .env 拉到前台
        ("*.env - 记事本", "protected_file"),  # 改过未保存时标题带星号
        (".env – Notepad", "protected_file"),
        ("登录 Microsoft 帐户", "account_signin"),  # 事件 2：对着验证界面连按回车
        ("Sign in to your account", "account_signin"),
        ("导入浏览器数据", "browser_data_import"),  # 事件 3：「保存的密码」勾选着
        ("Microsoft 365 Copilot", "copilot"),  # 事件 2 的起点：任务栏上紧挨 Edge
        ("用户帐户控制", "uac"),
        ("保存的密码 - 设置", "credential"),
    ],
)
def test_sensitive_titles_hit(title: str, rule: str) -> None:
    hit = match_window(WindowInfo(title))
    assert hit is not None and hit[0] == rule


@pytest.mark.parametrize(
    "title",
    [
        "Python 官方文档 - 搜索 - Microsoft Edge",  # search_content 任务的正常终态
        "测试文档.txt - 记事本",  # open_file 任务的正常终态
        "测试消息",  # send_message 的 mock 程序
        "计算器",
        "environment.py - Visual Studio Code",  # 含 env 字样但不是 .env
        "Catalog index",  # 含 "log in" 字母序列，但不是登录
        "",
    ],
)
def test_normal_task_windows_do_not_hit(title: str) -> None:
    """**误报的代价是停一整批。** 五个基础任务的正常窗口一个都不能命中。"""
    assert match_window(WindowInfo(title)) is None


def test_sensitive_process_hits_even_without_title() -> None:
    """标题被本地化或取不到时，进程名是更稳的判据。"""
    hit = match_window(WindowInfo("", "CredentialUIBroker.exe"))
    assert hit is not None and hit[0] == "credential"


def test_evidence_masks_email() -> None:
    """**登录框标题里可能带着真实邮箱**，证据进存档前必须脱敏。"""
    hit = match_window(WindowInfo("登录 someone.real@outlook.com"))
    assert hit is not None
    assert "outlook" not in hit[2] and "<email>" in hit[2]


def test_no_foreground_window_passes() -> None:
    assert match_window(None) is None


# --------------------------------------------------------------------- #
# 重复输入熔断：事件 2 的「连按 12 次回车」
# --------------------------------------------------------------------- #


def test_third_identical_key_trips() -> None:
    sentinel = sentinel_with()
    assert sentinel.check(enter()) is None
    assert sentinel.check(enter()) is None
    event = sentinel.check(enter())
    assert event is not None and event.rule == "repeated_input"


def test_different_input_resets_the_count() -> None:
    sentinel = sentinel_with()
    for action in (enter(), enter(), Action(ActionType.KEY, keys="tab"), enter(), enter()):
        assert sentinel.check(action) is None


def test_click_between_inputs_resets_the_count() -> None:
    """只盯键盘输入；中间夹了点击就说明不是对着同一个框原地打转。"""
    sentinel = sentinel_with()
    sentinel.check(enter())
    sentinel.check(enter())
    sentinel.check(Action(ActionType.LEFT_CLICK, x=10, y=10))
    assert sentinel.check(enter()) is None


def test_key_names_are_normalized() -> None:
    sentinel = sentinel_with()
    sentinel.check(Action(ActionType.KEY, keys="Ctrl+S"))
    sentinel.check(Action(ActionType.KEY, keys="ctrl + s"))
    assert sentinel.check(Action(ActionType.KEY, keys="CTRL+s")) is not None


def test_repeated_text_trips() -> None:
    """事件 1 里写进 .env 的正是反复输入的「你好世界」。"""
    sentinel = sentinel_with()
    typed = Action(ActionType.TYPE, text="你好世界")
    sentinel.check(typed)
    sentinel.check(typed)
    assert sentinel.check(typed) is not None


def test_reset_clears_repeat_state() -> None:
    """每轮开始要清零——上一轮最后两次回车不该让这一轮第一次回车就触发。"""
    sentinel = sentinel_with()
    sentinel.check(enter())
    sentinel.check(enter())
    sentinel.reset()
    assert sentinel.check(enter()) is None


def test_repeat_limit_must_allow_one_input() -> None:
    with pytest.raises(ValueError):
        SafetySentinel(repeat_limit=1)


def test_events_are_recorded() -> None:
    sentinel = sentinel_with(title=".env - 记事本")
    sentinel.check(enter())
    assert [e.rule for e in sentinel.events] == ["protected_file"]


# --------------------------------------------------------------------- #
# 开跑前环境扫描
# --------------------------------------------------------------------- #


def test_scan_environment_reports_all_hazards() -> None:
    sentinel = SafetySentinel(
        foreground=lambda: None,
        windows=lambda: [
            WindowInfo("计算器"),
            WindowInfo("登录 Microsoft 帐户"),
            WindowInfo(".env - 记事本"),
        ],
        processes=lambda: ["explorer.exe", "Microsoft365Copilot.exe"],
    )
    rules = sorted(e.rule for e in sentinel.scan_environment())
    assert rules == ["account_signin", "copilot", "protected_file"]
    assert all(e.source == "environment" for e in sentinel.events)


def test_scan_environment_clean_desktop() -> None:
    sentinel = SafetySentinel(
        foreground=lambda: None,
        windows=lambda: [WindowInfo("计算器"), WindowInfo("测试消息")],
        processes=lambda: ["explorer.exe", "CalculatorApp.exe"],
    )
    assert sentinel.scan_environment() == []


def test_scan_environment_deduplicates() -> None:
    sentinel = SafetySentinel(
        foreground=lambda: None,
        windows=lambda: [WindowInfo("Microsoft 365 Copilot"), WindowInfo("Microsoft 365 Copilot")],
        processes=list,
    )
    assert len(sentinel.scan_environment()) == 1


# --------------------------------------------------------------------- #
# 接入执行器：命中即急停
# --------------------------------------------------------------------- #


def build_executor(sentinel: SafetySentinel | None, dry_run: bool = True) -> ActionExecutor:
    scaler = CoordinateScaler(SCREEN)
    scaler.register("planner", 1024, 768)
    return ActionExecutor(scaler, space_name="planner", dry_run=dry_run, sentinel=sentinel)


def test_sentinel_hit_triggers_emergency_stop() -> None:
    executor = build_executor(sentinel_with(title=".env - 记事本"))
    result = executor.execute(Action(ActionType.TYPE, text="你好世界"))

    assert not result.success
    assert result.error_type == "emergency_stopped"
    assert result.verdict is not None and result.verdict.rule == "sentinel:protected_file"
    assert executor.emergency_stop.is_triggered
    assert executor.emergency_stop.trigger_reason == "sentinel:protected_file"


def test_after_sentinel_stop_every_action_is_refused() -> None:
    """**不是拒绝一个动作然后继续**——急停保持，直到人工复位。"""
    window = {"title": ".env - 记事本"}
    sentinel = SafetySentinel(
        foreground=lambda: WindowInfo(window["title"]), windows=list, processes=list
    )
    executor = build_executor(sentinel)
    executor.execute(Action(ActionType.LEFT_CLICK, x=1, y=1))

    window["title"] = "计算器"  # 敏感窗口消失了也不能自动恢复
    result = executor.execute(Action(ActionType.LEFT_CLICK, x=1, y=1))
    assert result.error_type == "emergency_stopped"


def test_sentinel_verdict_is_in_history() -> None:
    """批量脚本从 `executor.history` 里取安全事件。"""
    executor = build_executor(sentinel_with(title="登录 Microsoft 帐户"))
    executor.execute(enter())
    assert executor.history[-1].verdict.rule == "sentinel:account_signin"


def test_clean_window_lets_action_through() -> None:
    executor = build_executor(sentinel_with(title="计算器"))
    assert executor.execute(Action(ActionType.LEFT_CLICK, x=1, y=1)).success


def test_sentinel_on_by_default_only_when_executing() -> None:
    """演练不发键鼠事件，默认不开哨兵；真执行时默认开。"""
    assert build_executor(None, dry_run=True).sentinel is None
    assert isinstance(build_executor(None, dry_run=False).sentinel, SafetySentinel)


def test_failsafe_is_an_emergency_stop_not_an_action_failure() -> None:
    """**FAILSAFE 是人把鼠标甩到角落。** 原来它被记成 `FailSafeException` 类型的动作失败，
    本轮结束后下一轮照常开跑，也不计入人工干预。"""

    class FailSafeException(Exception):
        pass

    class FakePyAutoGUI:
        easeInOutQuad = None

        def moveTo(self, *_args, **_kwargs):  # noqa: N802 —— 与 pyautogui 同名
            raise FailSafeException("鼠标到了屏幕角落")

    executor = build_executor(sentinel_with(), dry_run=False)
    executor._pyautogui = FakePyAutoGUI()
    result = executor.execute(Action(ActionType.LEFT_CLICK, x=1, y=1))

    assert result.error_type == "emergency_stopped"
    assert executor.emergency_stop.trigger_reason == "failsafe"


# --------------------------------------------------------------------- #
# 急停记录触发来源
# --------------------------------------------------------------------- #


def test_hotkey_callback_records_hotkey_reason() -> None:
    """pynput 的回调不带参数，走默认值。"""
    stop = EmergencyStop()
    stop._handle_trigger()
    assert stop.trigger_reason == "hotkey"


def test_programmatic_trigger_records_reason() -> None:
    stop = EmergencyStop()
    stop.trigger(reason="sentinel:copilot")
    assert stop.trigger_reason == "sentinel:copilot"


def test_first_reason_wins() -> None:
    """已经停了再触发不改来源——记录的应当是**第一个**刹车的人或规则。"""
    stop = EmergencyStop()
    stop.trigger(reason="sentinel:copilot")
    stop.trigger(reason="hotkey")
    assert stop.trigger_reason == "sentinel:copilot"


def test_reset_clears_reason() -> None:
    stop = EmergencyStop()
    stop.trigger(reason="failsafe")
    stop.reset()
    assert stop.trigger_reason == "" and not stop.is_triggered


def test_event_as_dict_round_trip() -> None:
    event = SentinelEvent("copilot", "说明", "证据", source="environment")
    assert event.as_dict() == {
        "rule": "copilot",
        "reason": "说明",
        "evidence": "证据",
        "source": "environment",
    }
