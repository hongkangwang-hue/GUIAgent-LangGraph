"""在 Windows 客机连续运行演示任务，并录制一段本地 MP4 视频。

用法（在仓库根目录）：
    python scripts/run_demo_suite.py basics
    python scripts/run_demo_suite.py w7-passed

视频保存在被 Git 忽略的 outputs/demo-videos/；原始判定仍以
docs/m2-runs/ 中的运行存档为准。此脚本不会上传视频或密钥。
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "docs" / "m2-runs"
VIDEOS = ROOT / "outputs" / "demo-videos"
W7_PASSED = ("open_notepad", "open_calculator", "open_explorer", "close_notepad")


class ScreenRecorder:
    """用低帧率 GDI 截屏录制演示，避免再占用 Agent 的 DXGI 摄像头。"""

    def __init__(self, output: Path, fps: int = 4, monitor_index: int = 1):
        self.output = output
        self.fps = fps
        self.monitor_index = monitor_index
        self.ready = threading.Event()
        self.stop_signal = threading.Event()
        self.thread = threading.Thread(target=self._record, daemon=True)
        self.error: Exception | None = None
        self.frames = 0

    def start(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.thread.start()
        if not self.ready.wait(12):
            self.stop_signal.set()
            raise RuntimeError("录屏启动超时")
        if self.error:
            raise RuntimeError(f"录屏启动失败：{self.error}") from self.error

    def stop(self) -> None:
        self.stop_signal.set()
        self.thread.join(timeout=12)
        if self.thread.is_alive():
            raise RuntimeError("录屏未能正常结束；请检查视频文件")
        if self.error:
            raise RuntimeError(f"录屏失败：{self.error}") from self.error
        if self.frames < 2:
            raise RuntimeError("录屏帧数不足，视频不可作为演示")

    def _record(self) -> None:
        writer = None
        try:
            import cv2
            import mss
            import numpy as np

            with mss.mss() as grabber:
                if not 0 < self.monitor_index < len(grabber.monitors):
                    raise ValueError(f"无效显示器序号 {self.monitor_index}")
                monitor = grabber.monitors[self.monitor_index]
                width = monitor["width"] - monitor["width"] % 2
                height = monitor["height"] - monitor["height"] % 2
                writer = cv2.VideoWriter(
                    str(self.output),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self.fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError("OpenCV 无法创建 MP4；请检查客机的视频编码支持")
                self.ready.set()
                interval = 1.0 / self.fps
                while not self.stop_signal.is_set():
                    started = time.monotonic()
                    shot = np.asarray(grabber.grab(monitor))
                    writer.write(shot[:height, :width, :3])  # BGRA → BGR
                    self.frames += 1
                    self.stop_signal.wait(max(0.0, interval - (time.monotonic() - started)))
        except Exception as exc:  # noqa: BLE001
            self.error = exc
        finally:
            if writer is not None:
                writer.release()
            self.ready.set()


def console_window() -> int:
    if os.name != "nt":
        return 0
    return int(ctypes.windll.kernel32.GetConsoleWindow())


def show_console(handle: int, command: int) -> None:
    if handle:
        ctypes.windll.user32.ShowWindow(handle, command)


def find_archive(tag: str, started: float) -> tuple[Path, dict]:
    if not RUNS.exists():
        raise RuntimeError("运行存档目录不存在")
    for path in sorted(RUNS.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        if path.stat().st_mtime < started - 60:
            break
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("tag") == tag:
            return path, payload
    raise RuntimeError(f"没有找到本次运行存档：{tag}")


def run_batch(
    *,
    task_file: str,
    tag: str,
    model: str,
    planner: str,
    only: str = "",
) -> dict:
    command = [
        sys.executable,
        "scripts/run_basic_tasks.py",
        "--execute",
        "--tasks",
        task_file,
        "--repeats",
        "1",
        "--provider",
        "dashscope",
        "--model",
        model,
        "--engine",
        "langgraph",
        "--planner-template",
        planner,
        "--executor-template",
        "executor_v5",
        "--tag",
        tag,
    ]
    if only:
        command.extend(("--only", only))
    started = time.time()
    result = subprocess.run(  # noqa: S603
        command,
        cwd=ROOT,
        env=os.environ.copy(),
        input="yes\n",  # 外层已经由操作者确认一次；每个单项无需重复确认。
        text=True,
        check=False,
    )
    archive, payload = find_archive(tag, started)
    records = payload.get("records", [])
    expected = 1 if only else 5
    passed = sum(bool(record.get("verified")) for record in records)
    print(f"[演示] {tag}: {passed}/{len(records)}，存档 {archive}", flush=True)
    if result.returncode != 0 or payload.get("aborted"):
        raise RuntimeError(f"运行中止或触发急停；已保留存档 {archive}")
    if payload.get("partial") or len(records) != expected:
        raise RuntimeError(f"运行不完整；已保留存档 {archive}")
    if any(not record.get("precondition_ok", True) for record in records):
        raise RuntimeError(f"测试起点无效；已保留存档 {archive}")
    if passed != expected:
        raise RuntimeError(f"程序化判定仅通过 {passed}/{expected}；已保留存档 {archive}")
    return {"archive": str(archive), "tasks": [record["task"] for record in records]}


def main() -> int:
    parser = argparse.ArgumentParser(description="连续运行 GUI Agent 演示并录制本地视频")
    parser.add_argument("suite", choices=("basics", "w7-passed"))
    parser.add_argument("--model", default="qwen3-vl-8b-instruct")
    parser.add_argument("--planner-template", default="planner_v8")
    parser.add_argument("--monitor", type=int, default=1, help="MSS 显示器序号，默认 1")
    parser.add_argument("--list", action="store_true", help="只显示演示任务，不操作桌面")
    args = parser.parse_args()

    names = (
        ("open_browser", "search_content", "open_file", "send_message", "close_app")
        if args.suite == "basics"
        else W7_PASSED
    )
    print(f"演示部分：{args.suite}；任务：{', '.join(names)}")
    print(f"模型：{args.model}；LangGraph；{args.planner_template} + executor_v5")
    if args.list:
        return 0
    if os.name != "nt":
        raise SystemExit("实机演示仅在 Windows 客机运行")
    env_file = Path.home() / ".gui-agent" / ".env"
    if not env_file.is_file():
        raise SystemExit(f"缺少仓库外 API 配置：{env_file}")
    os.environ["GUI_AGENT_ENV_FILE"] = str(env_file)
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if input("将操作客机键鼠并录制屏幕。确认开始？(yes/N) ").strip().lower() != "yes":
        print("已取消")
        return 1

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    video = VIDEOS / f"{stamp}-{args.suite}.mp4"
    manifest = video.with_suffix(".json")
    recorder = ScreenRecorder(video, monitor_index=args.monitor)
    completed: list[dict] = []
    error = ""
    handle = console_window()
    try:
        show_console(handle, 6)  # 最小化终端，给 GUI Agent 留出桌面。
        time.sleep(0.7)
        recorder.start()
        if args.suite == "basics":
            completed.append(
                run_batch(
                    task_file="tasks/basic_tasks.yaml",
                    tag=f"demo-basic-{stamp}",
                    model=args.model,
                    planner=args.planner_template,
                )
            )
        else:
            for name in W7_PASSED:
                completed.append(
                    run_batch(
                        task_file="tasks/desktop_20.yaml",
                        only=name,
                        tag=f"demo-w7-{name}-{stamp}",
                        model=args.model,
                        planner=args.planner_template,
                    )
                )
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
    finally:
        if recorder.thread.ident is not None:
            try:
                recorder.stop()
            except Exception as exc:  # noqa: BLE001
                error = f"{error}；{exc}" if error else str(exc)
        show_console(handle, 9)  # 恢复终端，显示判定和文件位置。
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {"suite": args.suite, "video": str(video), "records": completed, "error": error},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"视频：{video}（{recorder.frames} 帧；无音频）")
    print(f"演示索引：{manifest}")
    if error:
        print(f"演示未全部通过：{error}", file=sys.stderr)
        return 2
    print(f"全部通过：{len(names)}/{len(names)}。视频仅保存在客机本地，分享前请检查画面。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
