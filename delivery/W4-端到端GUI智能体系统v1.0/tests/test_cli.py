"""CLI 的单元测试。

不测 `run` 的真实执行（那要花钱、要真桌面），测的是它**周边那些容易悄悄
坏掉的东西**：任务清单解析、面板计数、轨迹定位、打标写回。

这些出错的方式都很安静——清单少读一条任务、面板成本少算一次、打标写到
了别的步骤上，都不会报错，只会让最后的数字不对。而那些数字是要进汇报的。
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from cli.main import _load_tasks, _open_trajectory, app
from cli.panel import MAX_LOG_ROWS, PanelState
from core.trajectory import StepRecord, TrajectoryWriter

runner = CliRunner()


def _step(**kwargs) -> StepRecord:
    payload = {
        "step": 1,
        "subtask_id": 1,
        "subtask": "点击开始按钮",
        "action_model_coords": {"action": "left_click", "x": 10, "y": 20},
        "action_real_coords": {"action": "left_click", "x": 25, "y": 41},
        "execution_status": "ok",
        "tokens": {"prompt_tokens": 1000, "completion_tokens": 50},
        "cost_cny": 0.003,
        "latency": {"total_ms": 1500.0},
    }
    payload.update(kwargs)
    return StepRecord(**payload)


def _trajectory(tmp_path, steps=None):
    writer = TrajectoryWriter("打开记事本", root=tmp_path)
    writer.meta.subtasks = ["点击开始按钮"]
    writer.meta.backend = "dashscope"
    writer.meta.model = "qwen3-vl-8b-instruct"
    for record in steps or [_step(step=0)]:
        writer.append(record)
    writer.finish()
    return writer


# ===================================================================== #
# 任务清单
# ===================================================================== #


def test_load_tasks_from_yaml(tmp_path) -> None:
    path = tmp_path / "tasks.yaml"
    path.write_text(
        "tasks:\n"
        "  - name: open_notepad\n"
        "    instruction: 打开记事本\n"
        "  - name: close_app\n"
        "    instruction: 关闭当前窗口\n",
        encoding="utf-8",
    )
    tasks = _load_tasks(path)
    assert [t["name"] for t in tasks] == ["open_notepad", "close_app"]
    assert tasks[0]["instruction"] == "打开记事本"


def test_load_tasks_from_json(tmp_path) -> None:
    path = tmp_path / "tasks.json"
    path.write_text(
        json.dumps({"tasks": [{"name": "a", "instruction": "打开浏览器"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    assert _load_tasks(path)[0]["instruction"] == "打开浏览器"


def test_load_tasks_accepts_bare_list(tmp_path) -> None:
    """最省事的写法也要认，否则写清单本身成了负担。"""
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(["打开记事本", "关闭窗口"], ensure_ascii=False), encoding="utf-8")
    assert len(_load_tasks(path)) == 2


def test_load_tasks_accepts_task_key_alias(tmp_path) -> None:
    path = tmp_path / "tasks.yaml"
    path.write_text("tasks:\n  - task: 打开记事本\n", encoding="utf-8")
    assert _load_tasks(path)[0]["instruction"] == "打开记事本"


def test_missing_task_file_exits(tmp_path) -> None:
    from typer import Exit

    with pytest.raises(Exit):
        _load_tasks(tmp_path / "查无此文件.yaml")


# ===================================================================== #
# 面板计数
# ===================================================================== #


def test_panel_accumulates_tokens_and_cost() -> None:
    """成本计数错了不会报错，只会让汇报里的数字不对。"""
    state = PanelState()
    state.record_step(_step(step=1))
    state.record_step(_step(step=2, cost_cny=0.004))

    assert state.total_steps == 2
    assert state.prompt_tokens == 2000
    assert state.completion_tokens == 100
    assert state.total_tokens == 2100
    assert state.cost_cny == pytest.approx(0.007)


def test_panel_counts_failures_only() -> None:
    """no_action 是正常收尾，不该计入失败数。"""
    state = PanelState()
    state.record_step(_step(step=1))
    state.record_step(_step(step=2, execution_status="failed", error="越界"))
    state.record_step(_step(step=3, execution_status="no_action", action_model_coords={}))

    assert state.total_steps == 3
    assert state.failed_steps == 1


def test_panel_marks_three_states() -> None:
    state = PanelState()
    state.record_step(_step(step=1))
    state.record_step(_step(step=2, execution_status="failed"))
    state.record_step(_step(step=3, execution_status="no_action", action_model_coords={}))
    assert [row["mark"] for row in state.steps] == ["成功", "失败", "完成"]


def test_panel_shows_both_coordinate_systems() -> None:
    """验收标准 7 要求坐标映射可追溯。面板上直接看得见的话，坐标错位
    第一时间就能发现，不用等回放。"""
    state = PanelState()
    state.record_step(_step())
    action = state.steps[0]["action"]
    assert "10,20" in action and "25,41" in action


def test_panel_tracks_current_subtask() -> None:
    state = PanelState(subtasks=["甲", "乙", "丙"])
    state.record_step(_step(step=1, subtask_id=1, subtask="甲"))
    state.record_step(_step(step=2, subtask_id=2, subtask="乙"))

    assert state.current_subtask == 2
    assert state.current_goal == "乙"
    assert state.progress == pytest.approx(2 / 3)


def test_panel_progress_without_plan() -> None:
    assert PanelState().progress == 0.0


def test_panel_log_is_bounded() -> None:
    """日志表要有上限，否则长任务会把屏幕撑爆。"""
    state = PanelState()
    for index in range(MAX_LOG_ROWS + 8):
        state.record_step(_step(step=index))
    assert len(state.steps) == MAX_LOG_ROWS
    assert state.total_steps == MAX_LOG_ROWS + 8  # 计数不受显示条数影响


def test_panel_renders_without_crashing() -> None:
    """渲染失败绝不能变成任务中断——那时正在操作真实桌面。"""
    from cli.panel import LivePanel

    state = PanelState(instruction="打开记事本", subtasks=["甲"])
    state.record_step(_step())
    with LivePanel(state, enabled=False) as panel:
        panel.refresh()


# ===================================================================== #
# 轨迹定位
# ===================================================================== #


def test_open_trajectory_by_id(tmp_path) -> None:
    writer = _trajectory(tmp_path)
    reader = _open_trajectory(writer.trajectory_id, tmp_path)
    assert reader.meta.trajectory_id == writer.trajectory_id


def test_open_trajectory_by_directory(tmp_path) -> None:
    writer = _trajectory(tmp_path)
    assert _open_trajectory(str(writer.root), tmp_path).meta.instruction == "打开记事本"


def test_open_latest_trajectory(tmp_path) -> None:
    """省略参数用最近一条：调试时最常见的动作就是"刚跑完那条给我看看"。"""
    TrajectoryWriter("旧的", trajectory_id="traj-20260101-000000-aaaaaa", root=tmp_path).append(
        _step()
    )
    TrajectoryWriter("新的", trajectory_id="traj-20260301-000000-bbbbbb", root=tmp_path).append(
        _step()
    )
    assert _open_trajectory(None, tmp_path).meta.instruction == "新的"


def test_open_missing_trajectory_exits(tmp_path) -> None:
    from typer import Exit

    with pytest.raises(Exit):
        _open_trajectory("查无此轨迹", tmp_path)


# ===================================================================== #
# 命令
# ===================================================================== #


def test_config_show_runs() -> None:
    result = runner.invoke(app, ["config", "--show"])
    assert result.exit_code == 0
    assert "动作空间" in result.output


def test_config_lists_prompts() -> None:
    result = runner.invoke(app, ["config", "--prompts"])
    assert result.exit_code == 0
    assert "executor_v1" in result.output


def test_config_lists_trajectories(tmp_path) -> None:
    _trajectory(tmp_path)
    result = runner.invoke(app, ["config", "--trajectories", "--root", str(tmp_path)])
    assert result.exit_code == 0
    assert "打开记事本" in result.output


def test_replay_shows_the_essentials(tmp_path) -> None:
    """回放要能回答"这一步为什么这样"——思考、坐标映射、延迟拆分缺一不可。"""
    writer = _trajectory(tmp_path, [_step(step=1, model_thinking="任务栏最左侧就是开始按钮")])
    result = runner.invoke(app, ["replay", writer.trajectory_id, "--root", str(tmp_path)])

    assert result.exit_code == 0
    assert "任务栏最左侧就是开始按钮" in result.output
    assert "模型 10,20" in result.output and "屏幕 25,41" in result.output
    assert "延迟" in result.output


def test_replay_single_step(tmp_path) -> None:
    writer = _trajectory(
        tmp_path,
        [_step(step=1, model_thinking="第一步"), _step(step=2, model_thinking="第二步")],
    )
    result = runner.invoke(
        app, ["replay", writer.trajectory_id, "--root", str(tmp_path), "--step", "2"]
    )
    assert "第二步" in result.output
    assert "第一步" not in result.output


def test_label_lists_vocabulary() -> None:
    result = runner.invoke(app, ["label", "--list"])
    assert result.exit_code == 0
    assert "grounding_off" in result.output


def test_label_non_interactive_writes_back(tmp_path) -> None:
    writer = _trajectory(tmp_path, [_step(step=1, execution_status="failed", error="点偏了")])
    result = runner.invoke(
        app,
        [
            "label",
            writer.trajectory_id,
            "--root",
            str(tmp_path),
            "--step",
            "1",
            "--labels",
            "grounding_off",
            "--note",
            "点到了旁边的图标",
        ],
    )
    assert result.exit_code == 0

    from core.trajectory import TrajectoryReader

    step = TrajectoryReader(writer.root).steps()[0]
    assert step.labels == ["grounding_off"]
    assert step.label_note == "点到了旁边的图标"


def test_label_warns_on_unknown_vocabulary(tmp_path) -> None:
    """不拒绝——现场打标时卡住人不合适；但要提醒，自由文本会让 M4
    统计前还要人工归并同义写法。"""
    writer = _trajectory(tmp_path, [_step(step=1, execution_status="failed")])
    result = runner.invoke(
        app,
        [
            "label",
            writer.trajectory_id,
            "--root",
            str(tmp_path),
            "--step",
            "1",
            "--labels",
            "自创标签",
        ],
    )
    assert result.exit_code == 0
    assert "未知标签" in result.output

    from core.trajectory import TrajectoryReader

    assert TrajectoryReader(writer.root).steps()[0].labels == ["自创标签"]


def test_label_skips_clean_trajectory(tmp_path) -> None:
    writer = _trajectory(tmp_path, [_step(step=1)])
    result = runner.invoke(app, ["label", writer.trajectory_id, "--root", str(tmp_path)])
    assert "没有需要打标的步骤" in result.output


def test_run_without_instruction_fails() -> None:
    result = runner.invoke(app, ["run"])
    assert result.exit_code == 1
    assert "没有任务可执行" in result.output
