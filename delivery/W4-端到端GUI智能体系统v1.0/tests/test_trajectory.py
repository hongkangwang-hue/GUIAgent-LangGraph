"""轨迹日志的单元测试。

轨迹是 M3、M4、M5 唯一的数据源，因此这里测的重点是**在最坏情况下还剩
多少数据**：进程被急停杀掉、最后一行写了一半、字段是后来才加的、打标
过程中断。这些不是假想——急停热键就是设计来在 Agent 跑飞时随时按的，
它触发的时刻恰恰是最需要日志的时刻。
"""

from __future__ import annotations

import json

import pytest

from core.trajectory import (
    ERROR_LABELS,
    LatencyBreakdown,
    StepRecord,
    TrajectoryMeta,
    TrajectoryReader,
    TrajectoryWriter,
    describe_labels,
    list_trajectories,
    new_trajectory_id,
    validate_labels,
)


def _writer(tmp_path, instruction="打开浏览器"):
    return TrajectoryWriter(instruction, root=tmp_path)


def _step(**kwargs) -> StepRecord:
    payload = {
        "subtask_id": 1,
        "action_model_coords": {"action": "left_click", "x": 10, "y": 20},
        "execution_status": "ok",
    }
    payload.update(kwargs)
    return StepRecord(**payload)


# ===================================================================== #
# 延迟拆分
# ===================================================================== #


def test_latency_breakdown_totals() -> None:
    latency = LatencyBreakdown(api_ms=1200.0, grounding_ms=0.5, execute_ms=300.0, screenshot_ms=4.0)
    assert latency.total_ms == pytest.approx(1504.5)
    assert latency.as_dict()["total_ms"] == pytest.approx(1504.5)


def test_latency_four_segments_present() -> None:
    """验收标准第 3 条要求四段分解，字段名固定，分析脚本按名取值。"""
    keys = LatencyBreakdown().as_dict().keys()
    for segment in ("api_ms", "grounding_ms", "execute_ms", "screenshot_ms"):
        assert segment in keys


# ===================================================================== #
# 写入与读取
# ===================================================================== #


def test_writer_creates_layout(tmp_path) -> None:
    writer = _writer(tmp_path)
    assert (writer.root / "meta.json").exists()
    assert writer.frames_dir.is_dir()


def test_trajectory_id_is_unique() -> None:
    """同一秒起两条批量任务不能撞 ID。"""
    assert new_trajectory_id() != new_trajectory_id()


def test_append_and_read_back(tmp_path) -> None:
    writer = _writer(tmp_path)
    writer.append(_step())
    writer.append(_step(execution_status="failed", error="越界"))
    writer.finish()

    steps = TrajectoryReader(writer.root).steps()
    assert [s.step for s in steps] == [1, 2]
    assert steps[0].succeeded is True
    assert steps[1].succeeded is False


def test_append_flushes_immediately(tmp_path) -> None:
    """未 finish 就能读到已写的行。

    急停热键把进程杀掉时，已执行的步骤必须还在文件里——那正是最需要
    日志的时刻。这条测的就是"没有优雅退出也不丢数据"。
    """
    writer = _writer(tmp_path)
    writer.append(_step())
    assert len(TrajectoryReader(writer.root).steps()) == 1


def test_meta_accumulates_cost_and_tokens(tmp_path) -> None:
    writer = _writer(tmp_path)
    writer.append(_step(cost_cny=0.003, tokens={"total_tokens": 1000}))
    writer.append(_step(cost_cny=0.004, tokens={"total_tokens": 1200}))
    meta = writer.finish()
    assert meta.total_steps == 2
    assert meta.total_cost_cny == pytest.approx(0.007)
    assert meta.total_tokens == 2200


def test_context_manager_marks_success(tmp_path) -> None:
    with _writer(tmp_path) as writer:
        writer.append(_step())
    assert TrajectoryReader(writer.root).meta.status == "success"


def test_context_manager_marks_abort_on_exception(tmp_path) -> None:
    """异常退出也要落状态，否则轨迹停在 running 上，事后分不清跑崩还是在跑。"""
    writer = _writer(tmp_path)
    with pytest.raises(RuntimeError), writer:
        writer.append(_step())
        raise RuntimeError("急停触发")

    meta = TrajectoryReader(writer.root).meta
    assert meta.status == "aborted"
    assert "急停触发" in meta.error


def test_explicit_finish_not_overwritten_by_context_exit(tmp_path) -> None:
    with _writer(tmp_path) as writer:
        writer.append(_step())
        writer.finish(status="max_iterations")
    assert TrajectoryReader(writer.root).meta.status == "max_iterations"


def test_both_coordinate_systems_recorded(tmp_path) -> None:
    """验收标准第 7 条：坐标映射要可追溯，两套坐标缺一不可。"""
    writer = _writer(tmp_path)
    writer.append(
        _step(
            action_model_coords={"action": "left_click", "x": 512, "y": 384},
            action_real_coords={"action": "left_click", "x": 1280, "y": 800},
        )
    )
    step = TrajectoryReader(writer.root).steps()[0]
    assert step.action_model_coords["x"] == 512
    assert step.action_real_coords["x"] == 1280


def test_frame_paths_are_relative(tmp_path) -> None:
    """帧路径存相对形式：轨迹目录拷到别的机器上分析时，绝对路径全废。"""
    writer = _writer(tmp_path)
    absolute = writer.frame_path(1, "before")
    assert writer.relative(absolute) == "frames/step001-before.png"


def test_relative_tolerates_outside_path(tmp_path) -> None:
    writer = _writer(tmp_path)
    outside = tmp_path / "elsewhere.png"
    assert writer.relative(outside)  # 不抛异常即可


# ===================================================================== #
# 容错
# ===================================================================== #


def test_truncated_last_line_does_not_kill_the_trajectory(tmp_path) -> None:
    """进程被杀在写一半时，最后一行天然残缺——那不该毁掉前面所有步骤。"""
    writer = _writer(tmp_path)
    writer.append(_step())
    writer.append(_step())
    with writer.steps_path.open("a", encoding="utf-8") as handle:
        handle.write('{"step": 3, "execution_stat')

    steps = TrajectoryReader(writer.root).steps()
    assert [s.step for s in steps] == [1, 2]


def test_blank_lines_skipped(tmp_path) -> None:
    writer = _writer(tmp_path)
    writer.append(_step())
    with writer.steps_path.open("a", encoding="utf-8") as handle:
        handle.write("\n\n")
    assert len(TrajectoryReader(writer.root).steps()) == 1


def test_unknown_fields_preserved(tmp_path) -> None:
    """旧轨迹里的陌生字段不丢——分析脚本可能还要用。"""
    record = StepRecord.from_dict({"step": 1, "reflector_verdict": "retry"})
    assert record.meta["_unknown_fields"]["reflector_verdict"] == "retry"


def test_missing_steps_file_reads_as_empty(tmp_path) -> None:
    writer = _writer(tmp_path)
    assert TrajectoryReader(writer.root).steps() == []


def test_missing_meta_file_yields_placeholder(tmp_path) -> None:
    writer = _writer(tmp_path)
    (writer.root / "meta.json").unlink()
    assert TrajectoryReader(writer.root).meta.trajectory_id == writer.trajectory_id


def test_reader_rejects_missing_directory(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        TrajectoryReader(tmp_path / "查无此轨迹")


def test_non_serializable_field_does_not_lose_the_step(tmp_path) -> None:
    """某个字段序列化不了，不该让整步丢掉。"""
    writer = _writer(tmp_path)

    class Opaque:
        def __repr__(self) -> str:
            return "<opaque>"

    writer.append(_step(meta={"engine": Opaque()}))
    steps = TrajectoryReader(writer.root).steps()
    assert len(steps) == 1
    assert "opaque" in steps[0].meta["engine"]


def test_jsonl_is_one_line_per_step(tmp_path) -> None:
    writer = _writer(tmp_path)
    for _ in range(3):
        writer.append(_step(model_thinking="含\n换行的思考"))
    assert len(writer.steps_path.read_text(encoding="utf-8").strip().splitlines()) == 3


def test_chinese_is_not_escaped(tmp_path) -> None:
    """ensure_ascii=False：日志要人能直接读，不能满屏 \\uXXXX。"""
    writer = _writer(tmp_path)
    writer.append(_step(model_thinking="点击地址栏"))
    assert "点击地址栏" in writer.steps_path.read_text(encoding="utf-8")


# ===================================================================== #
# 打标
# ===================================================================== #


def test_write_labels(tmp_path) -> None:
    writer = _writer(tmp_path)
    writer.append(_step())
    writer.append(_step(execution_status="failed", error="点偏了"))

    reader = TrajectoryReader(writer.root)
    assert reader.write_labels(2, ["grounding_off"], note="点到了旁边的图标") is True

    steps = reader.steps()
    assert steps[0].labels == []
    assert steps[1].labels == ["grounding_off"]
    assert steps[1].label_note == "点到了旁边的图标"


def test_write_labels_preserves_other_steps(tmp_path) -> None:
    """整文件重写不能改动别的行。"""
    writer = _writer(tmp_path)
    writer.append(_step(model_thinking="第一步", cost_cny=0.001))
    writer.append(_step(execution_status="failed"))

    reader = TrajectoryReader(writer.root)
    reader.write_labels(2, ["wrong_action"])
    first = reader.steps()[0]
    assert first.model_thinking == "第一步"
    assert first.cost_cny == pytest.approx(0.001)


def test_write_labels_on_missing_step(tmp_path) -> None:
    writer = _writer(tmp_path)
    writer.append(_step())
    assert TrajectoryReader(writer.root).write_labels(99, ["other"]) is False


def test_write_labels_leaves_no_temp_file(tmp_path) -> None:
    writer = _writer(tmp_path)
    writer.append(_step(execution_status="failed"))
    reader = TrajectoryReader(writer.root)
    reader.write_labels(1, ["timeout"])
    assert not list(writer.root.glob("*.tmp"))


def test_failed_steps_filter(tmp_path) -> None:
    writer = _writer(tmp_path)
    writer.append(_step())
    writer.append(_step(execution_status="failed"))
    writer.append(_step(execution_status="rejected"))
    assert [s.step for s in TrajectoryReader(writer.root).failed_steps()] == [2, 3]


def test_validate_labels_splits_known_and_unknown() -> None:
    known, unknown = validate_labels(["grounding_off", "自创标签"])
    assert known == ["grounding_off"]
    assert unknown == ["自创标签"]


def test_label_vocabulary_is_fixed() -> None:
    """固定候选而非自由文本：否则 M4 统计前还要人工归并同义写法。"""
    assert "grounding_off" in ERROR_LABELS
    assert "other" in ERROR_LABELS
    assert all(desc for desc in ERROR_LABELS.values())


def test_describe_labels_lists_every_option() -> None:
    text = describe_labels()
    assert all(key in text for key in ERROR_LABELS)


# ===================================================================== #
# 摘要与清单
# ===================================================================== #


def test_summary_is_ascii_safe() -> None:
    """Windows 控制台是 cp936，摘要里混进非 GBK 字符会让 print 崩掉。

    摘要是出问题时用来看的东西，它本身不能成为新的崩溃源。
    """
    record = _step(execution_status="failed", error="越界", labels=["grounding_off"])
    record.step = 3
    summary = record.summary()
    summary.encode("gbk")  # 编不出来就是回归
    assert "FAIL" in summary and "grounding_off" in summary


def test_summary_marks_success() -> None:
    assert _step().summary().startswith("OK")


def test_list_trajectories_newest_first(tmp_path) -> None:
    for name in ("traj-20260101-000000-aaaaaa", "traj-20260301-000000-bbbbbb"):
        writer = TrajectoryWriter("x", trajectory_id=name, root=tmp_path)
        writer.append(_step())
    assert [p.name for p in list_trajectories(tmp_path)][0].startswith("traj-20260301")


def test_list_trajectories_ignores_incomplete_dirs(tmp_path) -> None:
    """只有 meta.json 没有 steps.jsonl 的目录不算轨迹。"""
    TrajectoryWriter("x", root=tmp_path)
    assert list_trajectories(tmp_path) == []


def test_list_trajectories_on_missing_root(tmp_path) -> None:
    assert list_trajectories(tmp_path / "不存在") == []


def test_meta_round_trip(tmp_path) -> None:
    writer = _writer(tmp_path, instruction="在记事本里写一句话")
    writer.meta.backend = "qwen_vl_api"
    writer.meta.mode = "A"
    writer.append(_step())
    writer.finish()

    raw = json.loads((writer.root / "meta.json").read_text(encoding="utf-8"))
    meta = TrajectoryMeta.from_dict(raw)
    assert meta.instruction == "在记事本里写一句话"
    assert meta.backend == "qwen_vl_api"
    assert meta.duration_s >= 0


# ===================================================================== #
# no_action 不是失败 —— 回归测试
#
# 模型报告子任务完成的那一步 execution_status 是 no_action：它没有动作可
# 执行，既不算"成功执行了动作"，也不是失败。最初把 succeeded 的反面当成
# 失败，后果是每条轨迹的收尾步都被拉进待打标清单，逼人给一次正常完成打
# 错误标签；按步统计成功率时每条轨迹还凭空多一次失败。
# ===================================================================== #


def test_done_step_is_not_a_failure() -> None:
    record = _step(execution_status="no_action", action_model_coords={})
    assert record.failed is False
    assert record.is_terminal is True
    assert record.succeeded is False  # 它确实没执行动作，这一条仍然为假


def test_failed_and_succeeded_are_not_complementary() -> None:
    """三种状态，不是二元的。"""
    assert _step(execution_status="ok").succeeded
    assert _step(execution_status="failed").failed
    assert not _step(execution_status="no_action").failed
    assert not _step(execution_status="no_action").succeeded


@pytest.mark.parametrize("status", ["failed", "rejected", "error"])
def test_real_failures_are_flagged(status) -> None:
    assert _step(execution_status=status).failed


def test_done_steps_excluded_from_labeling_queue(tmp_path) -> None:
    """待打标清单里不该出现正常收尾的那一步。"""
    writer = _writer(tmp_path)
    writer.append(_step())
    writer.append(_step(execution_status="failed", error="点偏了"))
    writer.append(_step(execution_status="no_action", action_model_coords={}))

    assert [s.step for s in TrajectoryReader(writer.root).failed_steps()] == [2]


def test_done_step_summary_reads_as_done() -> None:
    record = _step(execution_status="no_action", action_model_coords={})
    summary = record.summary()
    assert "DONE" in summary
    assert "FAIL" not in summary
    summary.encode("gbk")  # 仍然要 cp936 可打印
