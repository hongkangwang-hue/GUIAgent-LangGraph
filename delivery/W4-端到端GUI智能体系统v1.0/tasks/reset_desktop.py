"""把桌面恢复到「什么都没开」的状态 —— 每个任务 reset 的第一步。

## taskkill 不够

reset 原本只有 `taskkill`。但实测里连续撞了三种它够不着的残留：

1. **记事本的会话恢复** —— 状态在磁盘上，杀进程再启动会原样恢复
2. **开始菜单 / 搜索浮层** —— 系统 UI，不属于任何可杀的进程
3. **Microsoft 帐户登录框** —— 模态对话框，盖住半个屏幕

第三种最贵：它让「搜索指定内容」连续四轮以完全相同的方式失败
（17 步、同一个终止原因），Agent 对着登录框按了 12 次回车。

**屏幕上的东西就是 Agent 的全部输入。** 清不掉它们，每一轮的起点就
不一样，5 次的成功率不可比 —— 这是实验设计问题,不是模型能力问题。

## 做两件事

- **Esc ×2**：关掉开始菜单、搜索浮层、下拉菜单、以及多数模态框
- **Win+D**：最小化所有窗口

Win+D 会把消息模拟程序也最小化，但 `mock_messenger_reset.py` 在这之后
跑，会把它重新激活 —— 顺序是有意的：**先清空，再摆好**。

## 这不算给 Agent 放水

清掉的都是**上一轮遗留的**东西，不是任务本身的难度。任务是「打开
浏览器」，不是「先关掉上一轮卡住的登录框再打开浏览器」。若要测后者，
那该是一个独立设计的任务，而不是随机混进来的干扰。

## 局限

清不掉需要点确认按钮的对话框（「是否保存更改？」这类），也清不掉
系统级弹窗。**根本解法是快照恢复** —— OSWorld 那类工作用整机快照做
reset 不是因为慢得可以接受,是因为任务级 reset 清不干净。M5 的 60 轮
正式评测要用快照。
"""

from __future__ import annotations

import argparse
import sys
import time

#: 不属于测试环境、却会自己弹出来的应用。**按需补充**。
#: Copilot 是实测里的元凶：它固定在任务栏上紧挨着 Edge,
#: 模型点岔了就会拉起一个真实账号的登录框。
STRAY_PROCESSES = (
    "Microsoft365Copilot.exe",
    "msedgewebview2.exe",
    "olk.exe",
)


def dismiss_overlays() -> str:
    """Esc 关浮层，Win+D 最小化所有窗口。"""
    try:
        import pyautogui
    except ImportError:
        return "pyautogui 未安装，跳过"

    pyautogui.FAILSAFE = False
    try:
        for _ in range(2):
            pyautogui.press("escape")
            time.sleep(0.15)
        pyautogui.hotkey("win", "d")
        time.sleep(0.5)
    except Exception as exc:  # noqa: BLE001 —— 清场失败不该中断评测
        return f"清场出错：{exc}"
    return "已关闭浮层并最小化所有窗口"


def kill_strays() -> str:
    """杀掉不属于测试环境的常驻应用。"""
    if sys.platform != "win32":
        return "非 Windows，跳过"
    import subprocess

    killed = []
    for name in STRAY_PROCESSES:
        result = subprocess.run(
            ["taskkill", "/F", "/IM", name, "/T"],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
        # 进程本就不存在时 taskkill 返回非零,那是正常情况不是错误
        if result.returncode == 0:
            killed.append(name)
    return f"已清理 {', '.join(killed)}" if killed else "无游离进程"


def main() -> int:
    parser = argparse.ArgumentParser(description="桌面清场（每轮 reset 的第一步）")
    parser.add_argument("--no-kill", action="store_true", help="只关浮层，不杀进程")
    args = parser.parse_args()

    if not args.no_kill:
        print(kill_strays())
    print(dismiss_overlays())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
