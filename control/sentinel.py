"""安全哨兵 —— 看**动作发往哪里**，命中即自动急停。

## 为什么 `SafetyGuard` 不够

`control/safety.py` 只看动作本身：输入的文本像不像危险命令、组合键在不在黑名单。
M2 实测的三次安全事件，**每一个动作单独看都完全合法**：

| 事件 | 动作 | 为什么 guard 放行 |
|---|---|---|
| 修改并保存了仓库里的 `.env` | `type "你好世界"` | 文本无害；问题是前台窗口是 `.env` |
| 对真实微软账号的身份验证界面连按 12 次回车 | `key enter` × 12 | 回车无害；问题是对着登录框、且在原地打转 |
| 进入 Edge 的数据导入对话框，「保存的密码」处于勾选 | `left_click` | 点击无害；问题是点进了凭据相关界面 |

机制见 `docs/m2-basic-tasks-report.md` §3.2、§7.5：记事本会话恢复把 `.env` 拉回前台，
任务栏上 Microsoft 365 Copilot 紧挨着 Edge、点岔一次就弹出账号登录框。
**三次都发生在无人值守的批量跑里**——热键急停要有人按才有用。

所以这里补的是**上下文**这一层：动作执行前查一次前台窗口，查一次最近的输入是否在原地重复。
命中不是「拒绝这一个动作」，而是**触发急停、整批停下等人看**：
敏感窗口出现在屏幕上本身就说明环境已经偏离预期，换个动作继续跑不安全。

## 能力边界（必须写清楚）

1. **拦不住第一下。** 前台窗口是「动作执行前」的状态。任务栏上点岔的那一下，
   发出时前台还是 Edge，哨兵放行；登录框弹出来之后的**下一个**动作才被拦下。
   第一下的预防只能靠环境：快照里取消固定 Copilot、每轮开跑前 `scan_environment()`。
2. **标题匹配会误报，也会漏报。** 误报的代价是停一批、人看一眼；漏报的代价是
   又一次安全事件。这里偏向误报。规则表是开放的，新事件出现后往里加。
3. **与 `SafetyGuard` 一样，拦得住「犯傻」，拦不住「规避」。**
   真正的安全边界仍然是隔离虚拟机。
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from control.actions import Action, ActionType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WindowInfo:
    """一个窗口的标题与所属进程名。"""

    title: str
    process: str = ""


@dataclass(frozen=True)
class SentinelEvent:
    """一次哨兵命中。进存档，是「安全事件数」的计数单位。"""

    rule: str
    reason: str
    evidence: str = ""
    #: `action` 表示在执行某个动作前拦下；`environment` 表示开跑前扫描到的环境隐患
    source: str = "action"

    def as_dict(self) -> dict:
        return {
            "rule": self.rule,
            "reason": self.reason,
            "evidence": self.evidence,
            "source": self.source,
        }


#: (规则名, 标题正则, 说明)。**每一条都对应一次实测事件或同类界面**，不凭想象加。
SENSITIVE_TITLE_RULES: tuple[tuple[str, str, str], ...] = (
    (
        "protected_file",
        r"(?<![\w.])\.env\b|(?<![\w.])\.git\b|id_rsa|\.pem\b",
        "前台是受保护的配置或凭据文件（M2 实测：Agent 修改并保存了 .env）",
    ),
    (
        "account_signin",
        # 英文词必须带 \b：不加的话 "Catalog index" 里的 "log in" 也会命中
        r"登录|登入|\bsign\s*-?\s*in\b|\blog\s*-?\s*in\b|身份验证|验证你的身份|\bverify your identity\b"
        r"|\bmicrosoft\s*(帐户|账户|account)",
        "前台是账号登录或身份验证界面（M2 实测：对真实微软账号的验证界面连按 12 次回车）",
    ),
    (
        "credential",
        r"密码|\bpasswords?\b|凭据|\bcredentials?\b|\bpasskeys?\b|通行密钥",
        "前台是密码或凭据相关界面",
    ),
    (
        "browser_data_import",
        r"导入浏览器数据|导入收藏夹|import browser data|import favorites",
        "前台是浏览器数据导入界面（M2 实测：进入该对话框且「保存的密码」处于勾选）",
    ),
    (
        "uac",
        r"用户帐户控制|用户账户控制|user account control",
        "前台是 UAC 提权对话框",
    ),
    (
        "copilot",
        r"copilot",
        "前台是 Copilot（M2 实测：任务栏上它紧挨着 Edge，点岔后弹出账号登录框）",
    ),
)

#: 进程名（小写）。窗口标题取不到或被本地化时，进程名是更稳的判据。
SENSITIVE_PROCESSES: dict[str, str] = {
    "microsoft365copilot.exe": "copilot",
    "credentialuibroker.exe": "credential",
    "consent.exe": "uac",
}

_COMPILED_TITLE_RULES = tuple(
    (name, re.compile(pattern, re.IGNORECASE), reason)
    for name, pattern, reason in SENSITIVE_TITLE_RULES
)

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")

#: 同一输入连续多少次算原地打转。3：正常操作几乎不会连按三次同一个键、连打三次同一段字。
DEFAULT_REPEAT_LIMIT = 3


def _mask(text: str, limit: int = 80) -> str:
    """证据进存档之前先脱敏。**登录框的标题里可能带着真实邮箱。**"""
    return _EMAIL.sub("<email>", text)[:limit]


def match_window(window: WindowInfo | None) -> tuple[str, str, str] | None:
    """窗口命中哪条规则。返回 (规则名, 说明, 脱敏后的证据)，不命中返回 None。纯函数。"""
    if window is None:
        return None
    process = (window.process or "").lower()
    if process in SENSITIVE_PROCESSES:
        rule = SENSITIVE_PROCESSES[process]
        reason = next(r for n, _, r in _COMPILED_TITLE_RULES if n == rule)
        return rule, reason, _mask(f"{window.process} | {window.title}")
    for name, pattern, reason in _COMPILED_TITLE_RULES:
        if window.title and pattern.search(window.title):
            return name, reason, _mask(window.title)
    return None


# ---------------------------------------------------------------------- #
# Windows 取窗口信息。非 Windows 一律返回空，规则整体跳过而不是报错。
# ---------------------------------------------------------------------- #


def _process_name(pid: int) -> str:
    try:
        import psutil

        return psutil.Process(pid).name()
    except Exception:  # noqa: BLE001 —— 进程可能已退出或无权限
        return ""


def foreground_window() -> WindowInfo | None:
    """当前前台窗口。取不到返回 None。"""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return None
        length = user32.GetWindowTextLengthW(hwnd)
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return WindowInfo(buffer.value, _process_name(pid.value))
    except Exception as exc:  # noqa: BLE001
        logger.debug("取前台窗口失败：%s", exc)
        return None


def visible_windows() -> list[WindowInfo]:
    """所有可见的顶层窗口。取不到返回空列表。"""
    if sys.platform != "win32":
        return []
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        found: list[WindowInfo] = []
        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def _collect(hwnd, _lparam):
            if user32.IsWindowVisible(hwnd):
                length = user32.GetWindowTextLengthW(hwnd)
                if length:
                    buffer = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buffer, length + 1)
                    pid = wintypes.DWORD()
                    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                    found.append(WindowInfo(buffer.value, _process_name(pid.value)))
            return True

        user32.EnumWindows(callback_type(_collect), 0)
        return found
    except Exception as exc:  # noqa: BLE001
        logger.debug("枚举窗口失败：%s", exc)
        return []


def running_processes() -> list[str]:
    """正在运行的进程名。取不到返回空列表。"""
    try:
        import psutil

        return [p.info.get("name") or "" for p in psutil.process_iter(["name"])]
    except Exception:  # noqa: BLE001
        return []


# ---------------------------------------------------------------------- #


@dataclass
class SafetySentinel:
    """动作级的上下文检查。`ActionExecutor` 在每个动作执行前调用 `check`。

    三个取数函数都可替换，单元测试不碰真实桌面。
    """

    foreground: Callable[[], WindowInfo | None] = foreground_window
    windows: Callable[[], Iterable[WindowInfo]] = visible_windows
    processes: Callable[[], Iterable[str]] = running_processes
    repeat_limit: int = DEFAULT_REPEAT_LIMIT
    #: 全部命中记录，按时间顺序。
    events: list[SentinelEvent] = field(default_factory=list)
    _last_input: tuple | None = field(default=None, repr=False)
    _repeat_count: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if self.repeat_limit < 2:
            raise ValueError("repeat_limit 至少为 2，否则第一次输入就会被拦")

    def reset(self) -> None:
        """清空重复计数。**每轮开始时调用**——上一轮最后按的回车不该算进这一轮。"""
        self._last_input = None
        self._repeat_count = 0

    def check(self, action: Action) -> SentinelEvent | None:
        """动作执行前的检查。命中返回事件（并记入 `events`），否则返回 None。"""
        hit = match_window(self.foreground())
        if hit is not None:
            rule, reason, evidence = hit
            return self._record(SentinelEvent(rule, reason, evidence))

        event = self._check_repeat(action)
        if event is not None:
            return self._record(event)
        return None

    def scan_environment(self) -> list[SentinelEvent]:
        """开跑前扫一遍：屏幕上有没有敏感窗口、有没有敏感进程在跑。

        返回全部命中（不止第一条），便于一次看清环境里有几处问题。
        """
        found: list[SentinelEvent] = []
        seen: set[tuple[str, str]] = set()
        for window in self.windows():
            hit = match_window(window)
            if hit is not None and (hit[0], hit[2]) not in seen:
                seen.add((hit[0], hit[2]))
                found.append(SentinelEvent(hit[0], hit[1], hit[2], source="environment"))
        for name in self.processes():
            rule = SENSITIVE_PROCESSES.get((name or "").lower())
            if rule is not None and (rule, name) not in seen:
                seen.add((rule, name))
                reason = next(r for n, _, r in _COMPILED_TITLE_RULES if n == rule)
                found.append(SentinelEvent(rule, reason, name, source="environment"))
        self.events.extend(found)
        return found

    # ------------------------------------------------------------------ #

    def _check_repeat(self, action: Action) -> SentinelEvent | None:
        """同一个按键 / 同一段文字连续发出 `repeat_limit` 次。

        只看键盘输入：原地重复点击由 `core/retry.py` 负责升级策略，危害也小得多；
        **对着一个模态框反复敲键盘才是实测里出事的形态。**
        """
        if action.type is ActionType.KEY and action.keys:
            signature: tuple | None = (
                "key",
                "+".join(p.strip().lower() for p in action.keys.split("+")),
            )
        elif action.type is ActionType.TYPE and action.text:
            signature = ("type", action.text)
        else:
            signature = None

        if signature is None:
            self.reset()
            return None
        if signature == self._last_input:
            self._repeat_count += 1
        else:
            self._last_input, self._repeat_count = signature, 1

        if self._repeat_count >= self.repeat_limit:
            what = f"{signature[0]} {signature[1]!r}"
            return SentinelEvent(
                "repeated_input",
                f"同一输入连续第 {self._repeat_count} 次（M2 实测：对着登录框连按 12 次回车）",
                _mask(what),
            )
        return None

    def _record(self, event: SentinelEvent) -> SentinelEvent:
        self.events.append(event)
        logger.critical("安全哨兵命中 [%s] %s —— %s", event.rule, event.reason, event.evidence)
        return event
