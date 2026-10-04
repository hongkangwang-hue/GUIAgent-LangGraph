"""失败轨迹摘要只包含动作诊断字段，不泄漏模型输入或截图内容。"""

from __future__ import annotations

import json

from scripts.summarize_run_traces import safe_keys, summarize


def test_summarize_failed_trace_redacts_text_and_raw_output(tmp_path):
    archive = tmp_path / "run.json"
    archive.write_text(
        json.dumps(
            {
                "git_commit": "example-commit",
                "records": [
                    {
                        "task": "copy_paste_text",
                        "verified": False,
                        "trajectory_id": "t1",
                        "unchanged_click_limit": 4,
                        "final_check_enabled": True,
                        "reset_errors": ["reset 第 2 条命令退出码 1"],
                    },
                    {"task": "write_note", "verified": True, "trajectory_id": "t2"},
                ],
            }
        ),
        encoding="utf-8",
    )
    trace = tmp_path / "trajectories" / "t1"
    trace.mkdir(parents=True)
    (trace / "steps.jsonl").write_text(
        json.dumps(
            {
                "step": 1,
                "action_intent": {
                    "action_type": "type",
                    "params": {"text": "sk-SuperSecretExample123456"},
                    "thinking": "sk-SuperSecretExample123456",
                },
                "raw_output": "sk-SuperSecretExample123456",
                "screenshot_before": "frames/secret.png",
                "execution_status": "ok",
            }
        ),
        encoding="utf-8",
    )

    report = summarize(archive, tmp_path / "trajectories")
    output = json.dumps(report)
    assert len(report["failures"]) == 1
    assert report["git_commit"] == "example-commit"
    assert report["failures"][0]["unchanged_click_limit"] == 4
    assert report["failures"][0]["final_check_enabled"] is True
    assert report["failures"][0]["reset_errors"] == ["reset 第 2 条命令退出码 1"]
    assert report["failures"][0]["steps"][0]["typed_chars"] == 27
    assert "SuperSecret" not in output
    assert "secret.png" not in output


def test_only_known_shortcuts_are_exported():
    assert safe_keys(["ctrl", "s"]) == "ctrl+s"
    assert safe_keys("sk-SuperSecretExample123456") == "[redacted]"
