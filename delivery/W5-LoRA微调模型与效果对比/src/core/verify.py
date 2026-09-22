"""任务成功的**程序化判定** —— M2 验收标准 2 的地基。

M2 任务 8 写死了这条：

> **成功判定必须程序化**：进程检查、窗口标题检查、文件内容检查，
> 不用人工目测。

## 为什么不能人工目测

三个理由，按严重程度排：

1. **模型自报的 `done` 不等于任务完成。** 循环收尾与任务达成是两件事，
   开源规划模型在多步任务上尤其容易提前报完成。若拿 `done` 当成功，
   成功率会被系统性高估，而且高得看不出来。
2. **人工目测不可复现。** M5 要跑 20 任务 × 3 次共 60 轮，M2 是 5 × 5，
   跨越数周。同一个人在不同时间对「算不算成功」的判断会漂移，
   而这个漂移会直接进入成功率。
3. **人工目测挡不住挂机。** 大部分评测应当无人值守地跑，
   需要有人盯着的判定形同虚设。

思路直接来自 WebArena 的功能正确性评测方法（M1 调研阶段读的）。

## 判定器只回答一个问题

「任务完成后，系统处于预期状态了吗」——不关心过程。Agent 用什么路径
达成的不重要；点了十次还是一次，只要终态对就算成功。

这也意味着**判定器必须在客机内运行**：它要查客机的进程、窗口、文件。
宿主机看不到那些，这是 M0 把 Agent 放进客机的三条理由之一。
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

IS_WINDOWS = sys.platform == "win32"


@dataclass
class CheckResult:
    """一次判定的结果。`detail` 要能让人看懂为什么判成这样。"""

    passed: bool
    detail: str
    kind: str = ""

    def __bool__(self) -> bool:
        return self.passed


# ---------------------------------------------------------------------- #
# 单项判据
# ---------------------------------------------------------------------- #


def _running(name_pattern: str) -> list[str]:
    """返回匹配的进程名列表。匹配不区分大小写、按子串。"""
    try:
        import psutil
    except ImportError:
        return []
    pattern = name_pattern.lower()
    hits = []
    for proc in psutil.process_iter(["name"]):
        try:
            name = (proc.info.get("name") or "").lower()
        except Exception:  # noqa: BLE001 —— 进程可能在遍历途中消失
            continue
        if pattern in name:
            hits.append(name)
    return hits


def check_process(name: str, should_run: bool = True) -> CheckResult:
    """进程在 / 不在。

    `should_run=False` 用于「关闭应用」这类任务——**它的判据是进程消失**，
    而不是「点到了关闭按钮」。后者用截图也能看，但点了不一定关掉。
    """
    hits = _running(name)
    ok = bool(hits) if should_run else not hits
    verb = "应在运行" if should_run else "应已退出"
    state = f"找到 {len(hits)} 个：{', '.join(sorted(set(hits))[:3])}" if hits else "未找到"
    return CheckResult(ok, f"进程 {name!r} {verb}；{state}", "process")


def _own_console_handle() -> int:
    """评测脚本自己那个控制台窗口的句柄，没有则 0。"""
    if not IS_WINDOWS:
        return 0
    try:
        import ctypes

        return int(ctypes.windll.kernel32.GetConsoleWindow())
    except Exception:  # noqa: BLE001
        return 0


def check_window_title(pattern: str, regex: bool = False, should_match: bool = True) -> CheckResult:
    """存在（或不存在）标题匹配的窗口。

    ``should_match=False`` 是给**起点检查**用的。另外三个判据早就有反方向
    （`should_run` / `should_contain` / `should_exist`），只有这个缺——
    而 M2 的假成功**全部**来自起点没清干净：判据一上来就满足，Agent 什么
    都没做也打勾。

    两类任务非它不可：

        「打开系统设置」   判据是窗口标题含「设置」，起点就得是没有这个窗口。
                          不能用进程判据——SystemSettings.exe 在 Windows 11
                          上常驻后台（挂起态），进程判据恒为真。
        「显示桌面」       判据是某窗口不再可见。最小化不杀进程，
                          只有窗口判据看得出来。

    **遍历所有顶层窗口，不只看前台。** 只看前台会让判定受「任务结束时
        恰好哪个窗口在最上面」影响——那是与任务成败无关的噪声。

        **但要跳过评测脚本自己的控制台窗口。** 这是实测踩出来的：
        「搜索 Python 官方文档」的判据是标题含 `Python`，而运行评测的那个
        命令行窗口标题正是

            命令提示符 - python  scripts
    un_basic_tasks.py --execute --repeats 5

        于是任务明明失败（模型自己都报了没做完），判定却打勾。25 轮里
        这样虚增了 1 次成功，成功率从 52% 变成 56%——而且从输出里看不出来，
        因为 `detail` 里写的是「窗口标题命中」，看着完全正常。

        教训比这条 bug 本身重要：**评测工具必须看不见自己。** 判定器读的是
        全局状态，而评测进程本身也在那个全局状态里。
    """
    if not IS_WINDOWS:
        return CheckResult(False, "窗口标题判据仅支持 Windows", "window_title")
    try:
        import uiautomation as auto
    except ImportError:
        return CheckResult(False, "uiautomation 未安装，无法做窗口标题判定", "window_title")

    own = _own_console_handle()
    titles: list[str] = []
    try:
        window = auto.GetRootControl().GetFirstChildControl()
        while window:
            name = (window.Name or "").strip()
            if name and (not own or window.NativeWindowHandle != own):
                titles.append(name)
            window = window.GetNextSiblingControl()
    except Exception as exc:  # noqa: BLE001
        return CheckResult(False, f"枚举窗口失败：{exc}", "window_title")

    if regex:
        matcher = re.compile(pattern, re.IGNORECASE)
        hit = next((t for t in titles if matcher.search(t)), "")
    else:
        low = pattern.lower()
        hit = next((t for t in titles if low in t.lower()), "")

    sample = "、".join(titles[:5]) or "（无）"
    if bool(hit) == should_match:
        if should_match:
            return CheckResult(True, f"窗口标题命中 {hit!r}", "window_title")
        return CheckResult(True, f"没有标题含 {pattern!r} 的窗口，符合预期", "window_title")

    # 失败的说法要跟着方向变：要求「有」而没有，与要求「没有」却有，
    # 是两种完全不同的故障，报同一句话会把排查引到反方向去。
    if should_match:
        return CheckResult(
            False, f"没有标题含 {pattern!r} 的窗口。当前窗口：{sample}", "window_title"
        )
    return CheckResult(False, f"不该存在的窗口仍在：{hit!r}", "window_title")


def check_file_contains(
    path: str, text: str, encoding: str = "utf-8", should_contain: bool = True
) -> CheckResult:
    """文件存在且含（或不含）指定文本。

    **这是最可靠的一类判据**：它检查的是任务真正产生的副作用，
    不受窗口 Z 序、焦点、动画的影响。能设计成文件判定的任务就别用别的。

    ``should_contain=False`` 是给**起点检查**用的：跑「发送消息」之前，
    日志里不该已经有「你好世界」——上一轮的内容若没被 reset 清掉，
    Agent 什么都不做判据也会打勾。`check_process` 早就有这个反方向
    （`should_run=False`），这里补齐。
    """
    target = Path(path)
    if not target.exists():
        # 文件不存在时：要求「含」是失败；要求「不含」则**成立**
        # ——一个不存在的文件当然不含任何东西。
        if should_contain:
            return CheckResult(False, f"文件不存在：{target}", "file_contains")
        return CheckResult(True, f"文件不存在，故不含 {text!r}：{target}", "file_contains")
    try:
        content = target.read_text(encoding=encoding, errors="replace")
    except OSError as exc:
        return CheckResult(False, f"读不出 {target}：{exc}", "file_contains")
    if (text in content) == should_contain:
        verb = "含" if should_contain else "不含"
        return CheckResult(True, f"{target.name} {verb} {text!r}", "file_contains")
    preview = content.strip()[:60].replace("\n", "⏎")
    # 失败的说法要跟着方向变：要求「含」而没有，与要求「不含」却有，
    # 是两种完全不同的故障，报同一句话会把排查引到反方向去。
    verb = "不含" if should_contain else "却含有"
    return CheckResult(
        False, f"{target.name} {verb} {text!r}；实际内容：{preview!r}", "file_contains"
    )


def check_file_exists(path: str, should_exist: bool = True) -> CheckResult:
    target = Path(path)
    exists = target.exists()
    ok = exists if should_exist else not exists
    verb = "应存在" if should_exist else "应不存在"
    return CheckResult(ok, f"{target} {verb}；实际{'存在' if exists else '不存在'}", "file_exists")


#: 判据名 → 实现。任务清单里按名字引用。
CHECKERS = {
    "process": check_process,
    "window_title": check_window_title,
    "file_contains": check_file_contains,
    "file_exists": check_file_exists,
}


# ---------------------------------------------------------------------- #
# 组合
# ---------------------------------------------------------------------- #


@dataclass
class SuccessCheck:
    """一个任务的成功判据，可以由多条组成。

    `mode="all"` 时全部通过才算成功（默认）；`"any"` 时任意一条即可。

    多条判据不是为了保险，是为了**堵住不同的伪成功**。
    比如「打开指定文件」：只查进程，记事本开着但没打开那个文件也算成功；
    只查窗口标题，标题对但可能是别人开的。两条同时要求才排除得掉。
    """

    checks: list[dict] = field(default_factory=list)
    mode: str = "all"

    def run(self) -> tuple[bool, str]:
        """执行全部判据，返回 (是否成功, 可读说明)。"""
        if not self.checks:
            return False, "未定义成功判据——**不能默认算成功**"

        results = []
        for spec in self.checks:
            spec = dict(spec)
            kind = spec.pop("type", "")
            checker = CHECKERS.get(kind)
            if checker is None:
                results.append(CheckResult(False, f"未知判据类型 {kind!r}", kind))
                continue
            try:
                results.append(checker(**spec))
            except TypeError as exc:
                results.append(CheckResult(False, f"{kind} 参数不对：{exc}", kind))
            except Exception as exc:  # noqa: BLE001 —— 判定出错就是判定失败
                results.append(CheckResult(False, f"{kind} 执行出错：{exc}", kind))

        passed = all(results) if self.mode == "all" else any(results)
        lines = [f"{'✓' if r.passed else '✗'} {r.detail}" for r in results]
        return passed, "; ".join(lines)

    @classmethod
    def from_spec(cls, spec: Any) -> SuccessCheck:
        """从任务清单里的 `success_check` 字段构造。

        支持三种写法：单条 dict、多条 list、带 mode 的 dict。
        """
        if isinstance(spec, dict) and "checks" in spec:
            return cls(checks=list(spec["checks"]), mode=spec.get("mode", "all"))
        if isinstance(spec, dict):
            return cls(checks=[spec])
        if isinstance(spec, list):
            return cls(checks=list(spec))
        return cls(checks=[])


__all__ = [
    "CHECKERS",
    "CheckResult",
    "SuccessCheck",
    "check_file_contains",
    "check_file_exists",
    "check_process",
    "check_window_title",
]
