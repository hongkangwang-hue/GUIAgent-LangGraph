"""重置消息模拟程序的状态 —— 基础任务 4 每轮之前跑。

做两件事：清空日志文件，把消息窗口提到前台。

**已经开着就不重启**：重启会改变窗口位置与大小，而那是 Agent 每轮面对的
初始条件之一，每轮都换一次就没有「相同起点」可言了。

**但完全没开时会自己拉起来。** 原来这里只报告「没找到窗口」，靠人记得
先去另一个窗口手动 `python tasks/mock_messenger.py`。2026-08-24 的在线组
实测就栽在这上面：5 轮全部因起点未建立作废，而作废是在 reset 之后才发现的
——**一个纯手工的前置条件，代价是一整个任务的数据**。

两件事要分清：
- 「每轮重启」会破坏起点一致性 → 不做
- 「一次都没起」是环境没准备好 → 该自动补上，而不是让人记着

## 为什么要提到前台

第一次 5×5 实测里这个任务 0/5。轨迹显示 Agent 从头到尾在操作**记事本**
——上一个任务留下的窗口压在最上面，消息程序被盖住了。

「窗口在前台」是一个**声明出来的初始条件**，不是给 Agent 放水：
任务是「在测试消息程序里发一条消息」，程序可见是这个任务成立的前提，
就像「关闭计算器」要求计算器先开着一样。真正的难点在于识别输入框、
打字、点发送，那三步一步没少。

反过来说，若把「从一堆窗口里找出目标程序」也算进来，测的就是两件事
混在一起的结果，失败了分不清是哪一环。M4 若要单独测「窗口切换」，
那该是一个独立任务。
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

TITLE = "测试消息"

LOG = (
    Path("C:/agent-test/messages.log")
    if sys.platform == "win32"
    else Path("/tmp/agent-test/messages.log")
)


#: 没开时最多等它把窗口画出来多久。tkinter 起窗口很快，5 秒足够宽松。
LAUNCH_TIMEOUT_S = 5.0


def _find_window(auto):
    """找标题含 TITLE 的顶层窗口，没有就返回 None。"""
    window = auto.GetRootControl().GetFirstChildControl()
    while window:
        if TITLE in (window.Name or ""):
            return window
        window = window.GetNextSiblingControl()
    return None


def launch() -> str:
    """把 `mock_messenger.py` 拉起来，等窗口出现。

    用 `Popen` 而不是 `run`：这是个 GUI 程序，会一直开着，等它退出就死锁了。
    """
    script = Path(__file__).with_name("mock_messenger.py")
    if not script.exists():
        return f"找不到 {script}"
    subprocess.Popen(  # noqa: S603 - 固定路径，无外部输入
        [sys.executable, str(script)],
        cwd=str(script.parent.parent),
        creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
    )
    return f"已启动 {script.name}"


def activate_window() -> str:
    """把消息窗口提到前台；完全没开时先启动它。

    失败只报告，不中断——起点检查会兜住，而它给的信息比这里更准确。
    """
    if sys.platform != "win32":
        return "非 Windows，跳过"
    try:
        import uiautomation as auto
    except ImportError:
        return "uiautomation 未安装，跳过"

    try:
        window = _find_window(auto)
        if window is None:
            note = launch()
            deadline = time.monotonic() + LAUNCH_TIMEOUT_S
            while window is None and time.monotonic() < deadline:
                time.sleep(0.3)
                window = _find_window(auto)
            if window is None:
                return f"{note}，但 {LAUNCH_TIMEOUT_S:.0f}s 内没等到窗口"
            window.SetActive()
            return f"{note}，窗口已就绪并激活"
        window.SetActive()
        return f"已激活 {window.Name!r}（本来就开着，未重启）"
    except Exception as exc:  # noqa: BLE001
        return f"激活失败：{exc}"


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    print(activate_window())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
