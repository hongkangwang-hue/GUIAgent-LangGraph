"""批量评测的安全指标：首步成功率、人工干预率、安全事件数。

以及三条与安全事件整改绑定的护栏：
- 批量脚本必须挂上急停热键（此前从未挂载）
- 循环在默认配置下也要记帧差（首步成功率的数据来源）
- `.env` 默认从仓库外读取
"""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import pytest

import scripts.run_basic_tasks as rbt
from control.safety import SafetyVerdict
from scripts.run_basic_tasks import (
    RunRecord,
    first_step_outcome,
    round_safety_events,
    safety_metrics,
)


def step(status: str, changed: bool | None = None, error_type: str = "") -> SimpleNamespace:
    meta = (
        {}
        if changed is None
        else {"change": {"changed": changed, "ratio": 0.2 if changed else 0.0}}
    )
    return SimpleNamespace(execution_status=status, error_type=error_type, meta=meta)


# --------------------------------------------------------------------- #
# 首步成功
# --------------------------------------------------------------------- #


def test_first_step_success_needs_screen_change() -> None:
    """**执行成功不等于有效。** 实测里单击桌面图标六次 ok，屏幕纹丝不动。"""
    assert first_step_outcome([step("ok", changed=True)])[0] is True
    assert first_step_outcome([step("ok", changed=False)])[0] is False


def test_first_step_done_counts_as_failure() -> None:
    """第一步就报完成——正是 `done` 过拟合的形态，必须算失败。"""
    ok, detail = first_step_outcome([step("no_action")])
    assert ok is False and "报告完成" in detail


def test_first_step_failed_action() -> None:
    assert first_step_outcome([step("failed", error_type="parse_error")])[0] is False


def test_first_step_without_diff_is_unknown() -> None:
    """没有帧差记录时判定不了，返回 None，不进分母——不能当成功也不能当失败。"""
    assert first_step_outcome([step("ok")])[0] is None


def test_no_steps_is_failure() -> None:
    assert first_step_outcome([])[0] is False


def test_only_the_first_step_matters() -> None:
    assert first_step_outcome([step("ok", changed=False), step("ok", changed=True)])[0] is False


# --------------------------------------------------------------------- #
# 安全事件的口径
# --------------------------------------------------------------------- #


def result_with(rule: str, allowed: bool = False) -> SimpleNamespace:
    verdict = SafetyVerdict.allow() if allowed else SafetyVerdict.block(rule, "说明", "证据")
    return SimpleNamespace(verdict=verdict)


def test_safety_events_include_sentinel_and_dangerous_input() -> None:
    events = round_safety_events(
        [
            result_with("sentinel:copilot"),
            result_with("dangerous_text"),
            result_with("dangerous_keys"),
        ]
    )
    assert [e["rule"] for e in events] == ["sentinel:copilot", "dangerous_text", "dangerous_keys"]


def test_model_output_errors_are_not_safety_events() -> None:
    """坐标越界、调用未实现的动作是模型输出错误，计入会把安全事件数灌水。"""
    events = round_safety_events(
        [
            result_with("out_of_bounds"),
            result_with("stub_action"),
            result_with("", allowed=True),
            SimpleNamespace(verdict=None),
        ]
    )
    assert events == []


# --------------------------------------------------------------------- #
# 三个指标的汇总
# --------------------------------------------------------------------- #


def record(**kwargs) -> RunRecord:
    return RunRecord(
        task=kwargs.pop("task", "t"), title="t", attempt=kwargs.pop("attempt", 1), **kwargs
    )


def sample_records() -> list[RunRecord]:
    return [
        record(first_step_ok=True),
        record(first_step_ok=False),
        record(first_step_ok=None),  # 判定不了，不进分母
        record(first_step_ok=False, intervention="hotkey", emergency_reason="hotkey"),
        record(first_step_ok=True, excluded=True, exclusion_reason="有人碰了鼠标"),
        record(
            emergency_reason="sentinel:account_signin",
            safety_events=[{"rule": "sentinel:account_signin"}],
        ),
        # 开跑前环境扫描发现隐患：Agent 没跑，不进干预率分母，但事件要计数
        record(
            precondition_ok=False, safety_events=[{"rule": "copilot"}, {"rule": "protected_file"}]
        ),
    ]


def test_first_step_rate_excludes_unknown_and_excluded_rounds() -> None:
    first = safety_metrics(sample_records())["first_step_success"]
    # 有效轮（起点建立、未剔除）且能判定：True, False, False(hotkey 那轮) → 1/3
    assert (first["ok"], first["known"]) == (1, 3)


def test_intervention_counts_emergency_and_exclusion() -> None:
    human = safety_metrics(sample_records())["human_intervention"]
    assert human["rounds"] == 2  # 热键 1 轮 + 事后剔除 1 轮
    assert human["started"] == 6  # 环境隐患那轮 Agent 没跑，不进分母
    assert human["by_type"] == {"hotkey": 1, "excluded": 1}


def test_sentinel_stop_is_not_human_intervention() -> None:
    """程序自己刹住的是安全事件，不是人工干预——两者都让一轮停下，但含义相反。"""
    human = safety_metrics([record(emergency_reason="sentinel:copilot")])["human_intervention"]
    assert human["rounds"] == 0


def test_safety_events_counted_by_rule() -> None:
    events = safety_metrics(sample_records())["safety_events"]
    assert events["total"] == 3
    assert events["rounds_with_event"] == 2
    assert events["by_rule"] == {"sentinel:account_signin": 1, "copilot": 1, "protected_file": 1}


def test_metrics_accept_archive_dicts() -> None:
    """`exclude_round.py` 读回的是 dict，算出来必须和 dataclass 一致。"""
    from dataclasses import asdict

    records = sample_records()
    assert safety_metrics([asdict(r) for r in records]) == safety_metrics(records)


def test_empty_records_do_not_divide_by_zero() -> None:
    metrics = safety_metrics([])
    assert metrics["first_step_success"]["rate"] is None
    assert metrics["human_intervention"]["rate"] is None
    assert metrics["safety_events"]["total"] == 0


def test_archive_carries_safety_fields(monkeypatch) -> None:
    monkeypatch.setattr(rbt, "_screen_info", lambda: {})
    args = rbt.build_parser().parse_args([])
    payload = json.loads(
        rbt.archive_payload(
            sample_records(),
            args,
            "测试后端",
            offline=True,
            partial=True,
            aborted="close_app 第 1 次触发急停（hotkey）",
            safety_setup={"hotkey_armed": True, "hotkey": "<ctrl>+<alt>+q", "sentinel": True},
        )
    )
    assert payload["aborted"].startswith("close_app")
    assert payload["safety_setup"]["hotkey_armed"] is True
    assert payload["safety_metrics"]["safety_events"]["total"] == 3
    assert "first_step_ok" in payload["records"][0]


# --------------------------------------------------------------------- #
# 回归护栏
# --------------------------------------------------------------------- #


def test_batch_runner_arms_the_emergency_hotkey() -> None:
    """**此前批量脚本只构造执行器、从不 start()**，M2~M4 全部批量实测里急停热键都没挂上。"""
    source = inspect.getsource(rbt.main)
    assert "executor.start()" in source
    assert source.index("executor.start()") < source.index("session.run(")


def test_batch_runner_stops_on_emergency() -> None:
    """急停后不能让下一轮照常开跑、再把被拒绝的动作记成失败轮。"""
    source = inspect.getsource(rbt.main)
    assert "emergency_stop.is_triggered" in source
    assert "abort_reason" in source


def test_loop_records_frame_diff_with_default_config() -> None:
    """首步成功率靠帧差；评测默认两个开关都关，此时也必须记。"""
    import numpy as np

    from control.executor import ActionExecutor
    from core.loop import AgentLoop, LoopConfig
    from grounding.native import NativeGrounding
    from llm.fake import ScriptedBackend
    from perception.capture import Screenshot
    from perception.coordinate import CoordinateScaler
    from perception.types import BBox

    screen = BBox(0, 0, 1024, 768)

    class Capturer:
        def capture(self, monitor: int = 1, fresh: bool = False) -> Screenshot:
            return Screenshot(
                image=np.zeros((768, 1024, 3), dtype=np.uint8), region=screen, engine="fake"
            )

    scaler = CoordinateScaler(screen)
    scaler.register("planner", 1024, 768)
    loop = AgentLoop(
        llm=ScriptedBackend([{"action": "left_click", "x": 10, "y": 10}, {"done": True}]),
        grounding=NativeGrounding(1024, 768),
        executor=ActionExecutor(scaler, space_name="planner", dry_run=True),
        capturer=Capturer(),
        config=LoopConfig(max_iterations=3, save_frames=False, settle_seconds=0),
    )
    config = loop.config
    assert not config.escalate_on_no_change and not config.reflector

    result = loop.run_subtask("点一下")
    assert "change" in result.records[0].meta
    assert first_step_outcome(result.records)[0] is False  # 假截图前后相同，屏幕没变


# --------------------------------------------------------------------- #
# .env 的位置
# --------------------------------------------------------------------- #


@pytest.fixture
def env_sandbox(tmp_path, monkeypatch):
    import llm.providers as providers

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(providers.ENV_FILE_VAR, raising=False)
    monkeypatch.setattr(providers, "USER_ENV_FILE", tmp_path / "home" / ".gui-agent" / ".env")
    monkeypatch.setattr(providers, "_warned_local_env", False)
    return providers, tmp_path


def test_explicit_env_file_wins(env_sandbox, monkeypatch) -> None:
    providers, tmp = env_sandbox
    target = tmp / "elsewhere.env"
    monkeypatch.setenv(providers.ENV_FILE_VAR, str(target))
    (tmp / ".env").write_text("A=1", encoding="utf-8")
    assert providers.resolve_env_file() == target


def test_user_dir_preferred_over_working_dir(env_sandbox) -> None:
    providers, tmp = env_sandbox
    providers.USER_ENV_FILE.parent.mkdir(parents=True)
    providers.USER_ENV_FILE.write_text("A=1", encoding="utf-8")
    (tmp / ".env").write_text("A=2", encoding="utf-8")
    assert providers.resolve_env_file() == providers.USER_ENV_FILE


def test_working_dir_env_still_loads_but_warns_once(env_sandbox, capsys, monkeypatch) -> None:
    providers, tmp = env_sandbox
    # 先 setenv 再 delenv：让 monkeypatch 记下「测试结束后删掉它」，加载写进去的值不会漏到别的用例
    monkeypatch.setenv("GUI_AGENT_TEST_KEY", "placeholder")
    monkeypatch.delenv("GUI_AGENT_TEST_KEY")
    (tmp / ".env").write_text("GUI_AGENT_TEST_KEY=x", encoding="utf-8")

    assert providers.load_dotenv_if_present() is True
    assert providers.load_dotenv_if_present() is True
    assert capsys.readouterr().out.count("[安全]") == 1


def test_no_env_anywhere(env_sandbox) -> None:
    providers, _ = env_sandbox
    assert providers.resolve_env_file() is None
    assert providers.load_dotenv_if_present() is False
