"""全局热键急停——第二道刹车。

## 为什么 PyAutoGUI 的 FAILSAFE 不够

`FAILSAFE` 的机制是"鼠标被移到屏幕角落时抛异常"。问题在于：**Agent 运行
时鼠标正被程序控制**。你想把鼠标甩到角落，程序在同一时刻把它移回目标位置，
人和程序会抢夺光标——在快速连续动作下你很可能甩不过去。

全局键盘监听不依赖鼠标，按下即触发，是更可靠的刹车。

## 两者同时保留，互为兜底

`FAILSAFE` 挡的是"程序还在跑但鼠标失控"，热键挡的是"程序在跑而我要它
立刻停"。两者的失效模式不同：热键在系统被高权限窗口（UAC 提权对话框）
抢占焦点时可能收不到；`FAILSAFE` 在鼠标不动的纯键盘动作序列里不起作用。
所以是互补关系，不是二选一。

M1 验收标准 7 要求**两道都验证有效**，其中热键必须在 Agent 正控制鼠标时
仍能立即中断。
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)

DEFAULT_HOTKEY = "<ctrl>+<alt>+q"


class EmergencyStopped(RuntimeError):
    """急停已触发，动作被中止。"""


class EmergencyStop:
    """全局热键急停。

    典型用法::

        stop = EmergencyStop()
        stop.arm()
        try:
            while running:
                stop.raise_if_triggered()   # 每个动作前检查
                do_something()
        finally:
            stop.disarm()

    或者用上下文管理器::

        with EmergencyStop() as stop:
            ...
    """

    def __init__(
        self,
        hotkey: str = DEFAULT_HOTKEY,
        on_trigger: Callable[[], None] | None = None,
    ) -> None:
        self.hotkey = hotkey
        self._on_trigger = on_trigger
        #: 用 Event 而不是 bool：跨线程可见性由标准库保证，不需要自己加锁
        self._triggered = threading.Event()
        self._listener = None
        self._triggered_at: float | None = None

    # ------------------------------------------------------------------ #

    @property
    def is_triggered(self) -> bool:
        return self._triggered.is_set()

    @property
    def is_armed(self) -> bool:
        return self._listener is not None

    @property
    def triggered_at(self) -> float | None:
        """触发时刻，用于计算从按键到实际停止的响应延迟。"""
        return self._triggered_at

    # ------------------------------------------------------------------ #

    def arm(self) -> bool:
        """启动监听。返回是否成功。

        失败不抛异常而是返回 False 并告警——**急停装不上时不应该让整个
        系统起不来**，但必须让操作者知道此刻只剩 FAILSAFE 一道刹车。
        """
        if self._listener is not None:
            return True
        try:
            from pynput import keyboard
        except ImportError as exc:
            logger.error("pynput 未安装，全局热键急停不可用：%s。当前只剩 FAILSAFE 一道刹车", exc)
            return False

        try:
            self._listener = keyboard.GlobalHotKeys({self.hotkey: self._handle_trigger})
            self._listener.daemon = True
            self._listener.start()
            logger.info("急停热键已启用：%s", self.hotkey)
            return True
        except Exception as exc:  # noqa: BLE001 —— 键位串解析失败、无输入权限等
            logger.error("急停热键 %r 启动失败：%s。当前只剩 FAILSAFE 一道刹车", self.hotkey, exc)
            self._listener = None
            return False

    def disarm(self) -> None:
        """停止监听。"""
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception as exc:  # noqa: BLE001
                logger.debug("停止热键监听时出错：%s", exc)
            self._listener = None
            logger.info("急停热键已关闭")

    def reset(self) -> None:
        """清除触发状态，允许继续执行。**只应由人工确认后调用。**"""
        self._triggered.clear()
        self._triggered_at = None
        logger.info("急停状态已复位")

    # ------------------------------------------------------------------ #

    def trigger(self) -> None:
        """以编程方式触发。测试用，也可供上层在检测到异常时主动刹车。"""
        self._handle_trigger()

    def raise_if_triggered(self) -> None:
        """已触发则抛 `EmergencyStopped`。每个动作执行前调用。"""
        if self._triggered.is_set():
            raise EmergencyStopped(
                f"急停已触发（{self.hotkey}），拒绝继续执行。人工确认后调用 reset() 恢复"
            )

    def wait(self, timeout: float | None = None) -> bool:
        """阻塞等待触发。返回是否在超时前触发。"""
        return self._triggered.wait(timeout)

    # ------------------------------------------------------------------ #

    def _handle_trigger(self) -> None:
        if self._triggered.is_set():
            return
        self._triggered.set()
        self._triggered_at = time.time()
        logger.critical("!!! 急停触发 !!! 后续动作全部拒绝执行")
        if self._on_trigger is not None:
            try:
                self._on_trigger()
            except Exception as exc:  # noqa: BLE001 —— 回调出错不能影响急停本身
                logger.error("急停回调出错：%s", exc)

    # ------------------------------------------------------------------ #

    def __enter__(self) -> EmergencyStop:
        self.arm()
        return self

    def __exit__(self, *exc_info) -> None:
        self.disarm()
