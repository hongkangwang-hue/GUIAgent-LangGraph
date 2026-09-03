"""动作安全拦截。

## 这是第二道防线，不是第一道

**第一道防线是隔离虚拟机**——M0 的硬性前置条件：Agent 的一切键鼠执行只
发生在虚拟机内，宿主机不受控制。本模块存在的意义是在虚拟机内部再挡一层，
降低"跑坏测试环境导致重装"的概率。

必须诚实地看待它的能力边界：**文本模式匹配是可以被绕过的**。模型完全
可以把 `shutdown /s` 拆成两次 `type` 动作，或者先打开记事本写好再复制
粘贴。因此本模块拦得住的是"模型犯傻"，拦不住"模型有意规避"。真正的
安全边界永远是虚拟机。

把这一点写清楚，比假装白名单是万无一失的更重要——技术报告里也应当这么写。

## 拦截记录会进 M4 的错误分类

每次拦截都返回结构化的 `SafetyVerdict`，由执行器写进轨迹日志。M4 建立
错误分类体系时，"被安全规则拦下"是一个独立的失败类别，不能和"动作执行
失败"混为一谈。
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from control.actions import STUB_ACTIONS, Action, ActionType
from perception.types import Point

logger = logging.getLogger(__name__)


@dataclass
class SafetyVerdict:
    """一次安全检查的结论。"""

    allowed: bool
    rule: str = ""
    reason: str = ""
    #: 命中的具体内容（如匹配到的危险命令片段），便于复盘
    evidence: str = ""

    @classmethod
    def allow(cls) -> SafetyVerdict:
        return cls(allowed=True)

    @classmethod
    def block(cls, rule: str, reason: str, evidence: str = "") -> SafetyVerdict:
        return cls(allowed=False, rule=rule, reason=reason, evidence=evidence)

    def as_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "rule": self.rule,
            "reason": self.reason,
            "evidence": self.evidence,
        }


class ActionBlocked(RuntimeError):
    """动作被安全规则拦截。"""

    def __init__(self, verdict: SafetyVerdict) -> None:
        super().__init__(f"[{verdict.rule}] {verdict.reason}" + (f" — {verdict.evidence}" if verdict.evidence else ""))
        self.verdict = verdict


# ---------------------------------------------------------------------- #
# 危险命令模式
# ---------------------------------------------------------------------- #

#: 每条是 (规则名, 正则, 人话说明)。正则一律大小写不敏感。
DANGEROUS_TEXT_PATTERNS: tuple[tuple[str, str, str], ...] = (
    ("shutdown", r"\bshutdown\b|\bStop-Computer\b|\bRestart-Computer\b", "关机或重启系统"),
    ("format", r"\bformat\s+[a-z]:|\bdiskpart\b|\bFormat-Volume\b", "格式化磁盘"),
    (
        "delete_system_dir",
        r"(?:\bdel\b|\brd\b|\brmdir\b|\bRemove-Item\b)[^\n]*"
        r"(?:c:\\windows|c:\\program\s+files|system32|%systemroot%)",
        "删除系统目录",
    ),
    ("rm_rf_root", r"\brm\s+-[rf]{1,2}\s+/(?:\s|$)|\brm\s+-rf\s+~", "递归删除根目录或家目录"),
    ("registry_hive", r"\breg\s+delete\b[^\n]*\bHKLM\b|\bRemove-Item\b[^\n]*\bHKLM:", "删除系统注册表项"),
    ("shadow_copy", r"\bvssadmin\b[^\n]*\bdelete\s+shadows\b|\bwbadmin\s+delete\b", "删除卷影副本或备份"),
    ("wipe", r"\bcipher\s+/w\b|\bsdelete\b", "擦除磁盘空闲空间"),
    ("bcd", r"\bbcdedit\b[^\n]*\b(?:delete|set)\b", "修改启动配置"),
    ("account", r"\bnet\s+user\b[^\n]*\/(?:delete|add)\b", "增删系统账户"),
    (
        "download_execute",
        r"(?:Invoke-WebRequest|curl|wget)[^\n]*\|\s*(?:iex|Invoke-Expression|bash|sh)\b",
        "下载并直接执行远程脚本",
    ),
    ("disable_defender", r"Set-MpPreference[^\n]*DisableRealtimeMonitoring\s*\$?true", "关闭安全防护"),
)

#: 危险组合键。ctrl+alt+del 由系统直接接管，程序发不出去，列在这里只为记录意图。
DANGEROUS_KEY_COMBOS: frozenset[str] = frozenset(
    {
        "ctrl+alt+del",
        "ctrl+alt+delete",
        "win+l",  # 锁屏后 Agent 就瞎了，且需要人工解锁
        "win+d",  # 最小化全部窗口，会打断任务上下文
    }
)

_COMPILED_PATTERNS = tuple(
    (name, re.compile(pattern, re.IGNORECASE), reason)
    for name, pattern, reason in DANGEROUS_TEXT_PATTERNS
)


# ---------------------------------------------------------------------- #
# 规则
# ---------------------------------------------------------------------- #

Rule = Callable[[Action], SafetyVerdict]


def rule_stub_action(action: Action) -> SafetyVerdict:
    """接口桩动作直接拒绝，避免"看起来执行了其实没有"。"""
    if action.type in STUB_ACTIONS:
        return SafetyVerdict.block(
            "stub_action",
            f"动作 {action.type.value} 在 M1 只有接口桩，尚未实现",
        )
    return SafetyVerdict.allow()


def rule_dangerous_text(action: Action) -> SafetyVerdict:
    """输入文本中的高危命令。"""
    if action.type is not ActionType.TYPE or not action.text:
        return SafetyVerdict.allow()
    for _name, pattern, reason in _COMPILED_PATTERNS:
        match = pattern.search(action.text)
        if match:
            return SafetyVerdict.block("dangerous_text", reason, evidence=match.group(0)[:80])
    return SafetyVerdict.allow()


def rule_dangerous_keys(action: Action) -> SafetyVerdict:
    """高危或会让 Agent 失去上下文的组合键。"""
    if action.type is not ActionType.KEY or not action.keys:
        return SafetyVerdict.allow()
    normalized = "+".join(part.strip().lower() for part in action.keys.split("+"))
    if normalized in DANGEROUS_KEY_COMBOS:
        return SafetyVerdict.block("dangerous_keys", f"组合键 {normalized} 被禁止", evidence=normalized)
    return SafetyVerdict.allow()


@dataclass
class SafetyGuard:
    """动作安全检查器。

    坐标越界检查需要坐标系上下文，因此不做成无状态规则，而是由
    `check()` 额外接收模型画布尺寸。
    """

    rules: list[Rule] = field(
        default_factory=lambda: [rule_stub_action, rule_dangerous_text, rule_dangerous_keys]
    )
    #: 命中拦截时是否抛异常。M1 手工测试时设 False 便于批量验证规则
    raise_on_block: bool = True
    #: 全部拦截记录，M4 错误分类的数据来源之一
    blocked_log: list[dict] = field(default_factory=list)

    def check(
        self,
        action: Action,
        canvas_width: int | None = None,
        canvas_height: int | None = None,
    ) -> SafetyVerdict:
        """检查一个动作。返回结论；`raise_on_block` 为真且被拦时抛 `ActionBlocked`。"""
        verdict = self._evaluate(action, canvas_width, canvas_height)
        if not verdict.allowed:
            record = {"action": action.to_dict(), **verdict.as_dict()}
            self.blocked_log.append(record)
            logger.warning("动作被拦截：%s | %s", action, verdict.reason)
            if self.raise_on_block:
                raise ActionBlocked(verdict)
        return verdict

    def _evaluate(
        self, action: Action, canvas_width: int | None, canvas_height: int | None
    ) -> SafetyVerdict:
        for rule in self.rules:
            verdict = rule(action)
            if not verdict.allowed:
                return verdict

        if action.requires_coordinates() and canvas_width and canvas_height:
            for x_name, y_name in (("x", "y"), ("to_x", "to_y")):
                x, y = getattr(action, x_name, None), getattr(action, y_name, None)
                if x is None or y is None:
                    continue
                if not (0 <= x < canvas_width and 0 <= y < canvas_height):
                    return SafetyVerdict.block(
                        "out_of_bounds",
                        f"坐标 ({x}, {y}) 超出模型画布 {canvas_width}×{canvas_height}",
                        evidence=f"{x_name}={x}, {y_name}={y}",
                    )
        return SafetyVerdict.allow()

    def check_real_point(self, point: Point, region_contains: Callable[[Point], bool]) -> SafetyVerdict:
        """转换后的屏幕坐标是否仍在截图区域内。

        坐标转换本身会做夹取，因此正常情况下不会越界；这一层是防御性的
        ——如果它触发了，说明 `CoordinateScaler` 的配置和实际截图区域
        对不上，属于配置错误而非模型错误，值得单独报出来。
        """
        if not region_contains(point):
            return SafetyVerdict.block(
                "out_of_screen",
                f"转换后的屏幕坐标 {point.as_tuple()} 不在截图区域内，检查 CoordinateScaler 配置",
            )
        return SafetyVerdict.allow()
