"""输出失败任务的非敏感动作摘要，供客机复制到聊天中分析。

只读批次 JSON 与 steps.jsonl；不输出模型原文、思考、输入文本、截图或密钥。
在 Windows PowerShell 5.1 中也能复制：输出使用 ASCII JSON。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SAFE_KEY_PARTS = frozenset(
    {
        "alt",
        "backspace",
        "ctrl",
        "delete",
        "down",
        "end",
        "enter",
        "esc",
        "escape",
        "home",
        "left",
        "pageup",
        "pagedown",
        "right",
        "shift",
        "space",
        "tab",
        "up",
        "win",
        "a",
        "c",
        "d",
        "e",
        "n",
        "o",
        "s",
        "v",
        "x",
    }
)


def safe_keys(value: object) -> str:
    if isinstance(value, list | tuple):
        value = "+".join(str(part) for part in value)
    if not isinstance(value, str):
        return ""
    parts = value.lower().replace(" ", "").split("+")
    return value if all(part in SAFE_KEY_PARTS for part in parts) else "[redacted]"


def summarize_step(step: dict) -> dict:
    intent = step.get("action_intent") or {}
    params = intent.get("params") or {}
    real = step.get("action_real_coords") or {}
    change = (step.get("meta") or {}).get("change") or {}
    action = intent.get("action_type") or real.get("action") or ""
    typed = params.get("text")
    return {
        "step": step.get("step"),
        "subtask_id": step.get("subtask_id"),
        "action": action,
        "done": bool(intent.get("done")),
        "x": real.get("x", params.get("x")),
        "y": real.get("y", params.get("y")),
        "keys": safe_keys(params.get("keys") or params.get("key")),
        "typed_chars": len(typed) if action == "type" and isinstance(typed, str) else 0,
        "status": step.get("execution_status"),
        "error_type": step.get("error_type"),
        "screen_change_ratio": change.get("ratio"),
    }


def summarize(archive: Path, trajectories: Path) -> dict:
    run = json.loads(archive.read_text(encoding="utf-8"))
    failures = []
    for record in run.get("records", []):
        if record.get("verified"):
            continue
        item = {
            "task": record.get("task"),
            "precondition_ok": record.get("precondition_ok"),
            "reset_errors": record.get("reset_errors") or [],
            "unchanged_click_limit": record.get("unchanged_click_limit"),
            "final_check_enabled": record.get("final_check_enabled"),
            "loop_status": record.get("loop_status"),
            "model_said_done": record.get("model_said_done"),
            "trajectory_id": record.get("trajectory_id"),
            "emergency_reason": record.get("emergency_reason"),
            "steps": [],
        }
        traj_id = record.get("trajectory_id")
        if traj_id:
            steps_path = trajectories / traj_id / "steps.jsonl"
            if steps_path.is_file():
                item["steps"] = [
                    summarize_step(json.loads(line))
                    for line in steps_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            else:
                item["trace_missing"] = True
        failures.append(item)
    return {
        "archive": archive.name,
        "git_commit": run.get("git_commit"),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="导出不含输入文本的失败轨迹摘要")
    parser.add_argument("archive", type=Path, help="批次存档 JSON")
    parser.add_argument("--trajectories", type=Path, default=Path("outputs/trajectories"))
    args = parser.parse_args()
    print(json.dumps(summarize(args.archive, args.trajectories), ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
