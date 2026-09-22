"""动作执行器：把结构化动作翻译成实际键鼠事件。

## 职责边界

执行器负责四件事，缺一不可：

1. **坐标转换**——模型坐标 → 屏幕坐标，只经 `CoordinateScaler`
2. **安全拦截**——执行前过 `SafetyGuard`，越界与高危动作拒绝
3. **急停检查**——每个动作前检查急停状态
4. **结果记录**——成败、耗时、实际点击的屏幕坐标全部落进 `ActionResult`

第 4 项不是附带的。M1 验收标准要求"模型输出坐标与实际点击坐标的映射
关系可追溯，无坐标错位"，而 M2 的轨迹日志、M4 的错误分类都建立在
`ActionResult` 上。**执行了但没记录，等于没执行。**

## 中文输入必须走剪贴板

PyAutoGUI 的 `typewrite()` 底层发送的是 ASCII 键码，打不出中文——它会
静默地什么都不输入，或者输出乱码。可靠做法是写入剪贴板再模拟 Ctrl+V。
英文也一并走剪贴板，避免两条路径行为不一致。
"""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass, field

from control.actions import Action, ActionType
from control.emergency_stop import EmergencyStop, EmergencyStopped
from control.safety import ActionBlocked, SafetyGuard, SafetyVerdict
from perception.capture import ScreenCapturer, Screenshot
from perception.coordinate import CoordinateScaler
from perception.types import Point

logger = logging.getLogger(__name__)

#: 鼠标移动时长。瞬移会让很多控件的 hover 效果不触发（下拉菜单、工具提示
#: 依赖 mouseenter 事件），0.3 秒加缓动是经验值，兼顾可靠性与速度。
DEFAULT_MOVE_DURATION = 0.3


@dataclass
class ActionResult:
    """一次动作执行的完整记录。"""

    action: Action
    success: bool
    duration_ms: float = 0.0
    #: 转换后实际操作的屏幕坐标。坐标错位排查的核心证据
    real_point: Point | None = None
    real_point_to: Point | None = None
    error: str = ""
    error_type: str = ""
    verdict: SafetyVerdict | None = None
    #: SCREENSHOT 动作的产出
    screenshot: Screenshot | None = None
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        """轨迹日志的一行。M2 直接把它写进 JSONL。"""
        payload = {
            "action": self.action.to_dict(),
            "success": self.success,
            "duration_ms": round(self.duration_ms, 2),
            "real_point": self.real_point.as_tuple() if self.real_point else None,
        }
        if self.real_point_to:
            payload["real_point_to"] = self.real_point_to.as_tuple()
        if not self.success:
            payload["error"] = self.error
            payload["error_type"] = self.error_type
        if self.verdict is not None and not self.verdict.allowed:
            payload["safety"] = self.verdict.as_dict()
        return payload


class ActionExecutor:
    """动作执行器。

    典型用法::

        scaler = CoordinateScaler(capturer.monitor_region(1))
        scaler.register("planner", 1024, 768)

        executor = ActionExecutor(scaler, space_name="planner")
        executor.start()
        try:
            executor.execute(Action(ActionType.LEFT_CLICK, x=512, y=384))
        finally:
            executor.stop()
    """

    def __init__(
        self,
        scaler: CoordinateScaler,
        space_name: str = "planner",
        guard: SafetyGuard | None = None,
        emergency_stop: EmergencyStop | None = None,
        capturer: ScreenCapturer | None = None,
        move_duration: float = DEFAULT_MOVE_DURATION,
        dry_run: bool = False,
    ) -> None:
        self.scaler = scaler
        self.space_name = space_name
        self.guard = guard if guard is not None else SafetyGuard()
        self.emergency_stop = emergency_stop if emergency_stop is not None else EmergencyStop()
        self.capturer = capturer
        self.move_duration = move_duration
        #: 只走完整个校验与转换流程但不真的发键鼠事件。写测试与演练时用
        self.dry_run = dry_run
        self.history: list[ActionResult] = []
        self._pyautogui = None

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """启动急停监听并配置 PyAutoGUI。"""
        self.emergency_stop.arm()
        if not self.dry_run:
            pyautogui = self._ensure_pyautogui()
            # 第一道刹车：鼠标甩到屏幕角落即抛异常
            pyautogui.FAILSAFE = True
            # 每个 PyAutoGUI 调用后的固定间隔。设 0 由我们自己控制节奏，
            # 否则每个动作都白等 0.1 秒，多步任务会明显变慢
            pyautogui.PAUSE = 0.0

    def stop(self) -> None:
        self.emergency_stop.disarm()

    def __enter__(self) -> ActionExecutor:
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()

    def _ensure_pyautogui(self):
        if self._pyautogui is None:
            import pyautogui

            self._pyautogui = pyautogui
        return self._pyautogui

    # ------------------------------------------------------------------ #
    # 执行
    # ------------------------------------------------------------------ #

    def execute(self, action: Action) -> ActionResult:
        """执行一个动作。

        **不抛异常**（急停除外）：失败以 `ActionResult.success=False` 返回，
        由上层决定重试还是终止。让执行器抛异常会导致每个调用点都要写
        try/except，而 M2 的 Agent Loop 需要的是可判定的结果对象。
        """
        start = time.perf_counter()
        space = self.scaler.get(self.space_name)

        # --- 急停优先于一切 ---
        try:
            self.emergency_stop.raise_if_triggered()
        except EmergencyStopped as exc:
            return self._finish(
                ActionResult(action, False, error=str(exc), error_type="emergency_stopped"), start
            )

        # --- 安全检查 ---
        # 两条路径都要拦：guard 可能配成抛异常，也可能配成只返回结论。
        # **绝不能只处理异常那条**——否则把 raise_on_block 设成 False
        # 就等于关掉了整个安全层，而那个开关本意只是控制报错方式。
        try:
            verdict = self.guard.check(action, space.width, space.height)
        except ActionBlocked as exc:
            verdict = exc.verdict
        if not verdict.allowed:
            return self._finish(
                ActionResult(
                    action,
                    False,
                    error=f"[{verdict.rule}] {verdict.reason}",
                    error_type="blocked",
                    verdict=verdict,
                ),
                start,
            )

        # --- 坐标转换 ---
        real_point = real_point_to = None
        if action.requires_coordinates():
            real_point = self.scaler.to_real(Point(action.x, action.y), self.space_name)
            if action.to_x is not None and action.to_y is not None:
                real_point_to = self.scaler.to_real(
                    Point(action.to_x, action.to_y), self.space_name
                )
            logger.debug(
                "%s 坐标转换：模型(%d,%d) → 屏幕%s",
                action.type.value, action.x, action.y, real_point.as_tuple(),
            )

        result = ActionResult(
            action=action, success=False, real_point=real_point,
            real_point_to=real_point_to, verdict=verdict,
        )

        # --- 分派 ---
        try:
            if self.dry_run:
                logger.info("[dry-run] %s → %s", action, real_point.as_tuple() if real_point else "—")
            else:
                self._dispatch(action, real_point, real_point_to, result)
            result.success = True
        except EmergencyStopped as exc:
            result.error, result.error_type = str(exc), "emergency_stopped"
        except NotImplementedError as exc:
            result.error, result.error_type = str(exc), "not_implemented"
        except Exception as exc:  # noqa: BLE001 —— 键鼠底层可能抛任何东西
            result.error, result.error_type = str(exc), type(exc).__name__
            logger.warning("动作执行失败 %s：%s", action, exc)

        return self._finish(result, start)

    def execute_all(self, actions: list[Action], stop_on_failure: bool = True) -> list[ActionResult]:
        """顺序执行多个动作。默认一步失败即停——GUI 操作有强顺序依赖，
        前一步没成功就往下走，后面全是无意义的点击。"""
        results = []
        for action in actions:
            result = self.execute(action)
            results.append(result)
            if not result.success and stop_on_failure:
                logger.info("动作序列在第 %d 步中止：%s", len(results), result.error)
                break
        return results

    def _finish(self, result: ActionResult, start: float) -> ActionResult:
        result.duration_ms = (time.perf_counter() - start) * 1000.0
        self.history.append(result)
        return result

    # ------------------------------------------------------------------ #
    # 各动作的实现
    # ------------------------------------------------------------------ #

    def _dispatch(
        self,
        action: Action,
        point: Point | None,
        point_to: Point | None,
        result: ActionResult,
    ) -> None:
        pyautogui = self._ensure_pyautogui()
        kind = action.type

        if kind is ActionType.SCREENSHOT:
            if self.capturer is None:
                raise RuntimeError("执行 screenshot 需要在构造时传入 ScreenCapturer")
            result.screenshot = self.capturer.capture_region(self.scaler.region)
            result.meta["latency_ms"] = round(result.screenshot.latency_ms, 2)

        elif kind is ActionType.MOUSE_MOVE:
            self._move(point)

        elif kind in (ActionType.LEFT_CLICK, ActionType.RIGHT_CLICK, ActionType.DOUBLE_CLICK):
            self._move(point)
            button = "right" if kind is ActionType.RIGHT_CLICK else "left"
            clicks = 2 if kind is ActionType.DOUBLE_CLICK else 1
            pyautogui.click(button=button, clicks=clicks, interval=0.05)

        elif kind is ActionType.LEFT_CLICK_DRAG:
            self._move(point)
            pyautogui.mouseDown(button="left")
            try:
                pyautogui.moveTo(
                    point_to.x, point_to.y,
                    duration=max(self.move_duration, 0.4),
                    tween=pyautogui.easeInOutQuad,
                )
            finally:
                # 无论中途出什么事都必须松开左键，否则鼠标会一直处于
                # 按下状态，后续所有操作都变成拖拽
                pyautogui.mouseUp(button="left")

        elif kind is ActionType.SCROLL:
            self._move(point)
            amount = action.amount if action.amount is not None else 3
            if action.direction in ("up", "down"):
                clicks = amount if action.direction == "up" else -amount
                pyautogui.scroll(clicks * 100)
            else:
                clicks = amount if action.direction == "right" else -amount
                pyautogui.hscroll(clicks * 100)

        elif kind is ActionType.KEY:
            keys = [part.strip().lower() for part in action.keys.split("+") if part.strip()]
            if len(keys) == 1:
                pyautogui.press(keys[0])
            else:
                pyautogui.hotkey(*keys)

        elif kind is ActionType.TYPE:
            self._type_text(action.text, result)

        elif kind is ActionType.WAIT:
            self._interruptible_sleep(action.duration)

        else:
            raise NotImplementedError(f"动作 {kind.value} 尚未实现（M1 只留接口桩）")

    def _move(self, point: Point | None) -> None:
        """人类化鼠标移动。"""
        if point is None:
            return
        pyautogui = self._ensure_pyautogui()
        pyautogui.moveTo(point.x, point.y, duration=self.move_duration, tween=pyautogui.easeInOutQuad)

    def _type_text(self, text: str, result: ActionResult) -> None:
        """经剪贴板输入文本，失败时退回逐字符输入（只对 ASCII 有效）。"""
        pyautogui = self._ensure_pyautogui()
        try:
            import pyperclip

            original = ""
            with contextlib.suppress(Exception):  # 剪贴板可能被其他进程独占
                original = pyperclip.paste()

            pyperclip.copy(text)
            time.sleep(0.05)  # 给剪贴板一点时间落定，否则偶发粘贴到旧内容
            pyautogui.hotkey("ctrl", "v")
            result.meta["input_method"] = "clipboard"

            # 还原剪贴板，避免污染用户环境（也避免下一次粘贴拿到本次内容）
            if original:
                time.sleep(0.05)
                with contextlib.suppress(Exception):
                    pyperclip.copy(original)

        except ImportError:
            if not text.isascii():
                raise RuntimeError(
                    "pyperclip 未安装，无法输入中文。PyAutoGUI 的 typewrite 只支持 ASCII"
                ) from None
            pyautogui.typewrite(text, interval=0.02)
            result.meta["input_method"] = "typewrite"

    def _interruptible_sleep(self, seconds: float) -> None:
        """可被急停打断的等待。

        直接 `time.sleep(10)` 会让急停在这 10 秒里完全失效——按了热键也
        要等睡完才生效。切成小片轮询，把急停响应延迟压到 50ms 以内。
        """
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            self.emergency_stop.raise_if_triggered()
            time.sleep(min(0.05, max(0.0, deadline - time.perf_counter())))

    # ------------------------------------------------------------------ #

    def stats(self) -> dict:
        """执行统计。M2 的轨迹复盘与 M4 的错误分类都从这里起步。"""
        total = len(self.history)
        succeeded = sum(1 for r in self.history if r.success)
        by_error: dict[str, int] = {}
        for r in self.history:
            if not r.success:
                by_error[r.error_type] = by_error.get(r.error_type, 0) + 1
        return {
            "total": total,
            "succeeded": succeeded,
            "failed": total - succeeded,
            "success_rate": round(succeeded / total, 4) if total else 0.0,
            "mean_duration_ms": (
                round(sum(r.duration_ms for r in self.history) / total, 2) if total else 0.0
            ),
            "failures_by_type": by_error,
            "blocked_count": len(self.guard.blocked_log),
        }
