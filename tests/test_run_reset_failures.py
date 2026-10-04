"""重置脚本出错时不得让模型从未建立的起点开跑。"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

from scripts import run_basic_tasks


def test_missing_taskkill_is_benign_but_setup_failure_is_reported(monkeypatch):
    responses = iter((SimpleNamespace(returncode=128), SimpleNamespace(returncode=2)))
    monkeypatch.setattr(run_basic_tasks.subprocess, "run", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(run_basic_tasks.time, "sleep", lambda _: None)

    errors = run_basic_tasks.run_reset(
        ["taskkill /F /IM notepad.exe /T", "python tasks/setup_env.py --clean"],
        dry_run=False,
    )

    assert errors == ["reset 第 2 条命令退出码 2"]


def test_reset_timeout_is_reported_without_command_output(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="python tasks/setup_env.py", timeout=30)

    monkeypatch.setattr(run_basic_tasks.subprocess, "run", timeout)
    monkeypatch.setattr(run_basic_tasks.time, "sleep", lambda _: None)

    assert run_basic_tasks.run_reset(["python tasks/setup_env.py"], dry_run=False) == [
        "reset 第 1 条命令超时"
    ]
