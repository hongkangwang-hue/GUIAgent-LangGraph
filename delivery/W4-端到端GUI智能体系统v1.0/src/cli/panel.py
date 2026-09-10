"""Rich 实时监控面板。

M2 任务 7 要求"任务进度条、当前子任务、动作日志表格、token/成本计数器"。

## 为什么值得做一个面板，而不是打日志了事

Agent 跑起来是**慢**的：单步几秒到十几秒，一个任务几十秒到几分钟。这段
时间里人要判断的是"它是在正常推进，还是已经跑偏了、该按急停了"。

滚动的日志行做不到这件事——你得在心里把散落的行拼成状态。面板把"当前
在第几个子任务、这一步做了什么、花了多少钱"钉在固定位置，扫一眼就知道
要不要伸手。**急停是人按的，人得先看得出来该按。**

## 显示层不能拖垮执行层

面板里任何异常都被 `AgentLoop._commit` 兜住了（回调抛错只记日志）。这里
再自己防一道：终端宽度、编码、Unicode 宽字符，都可能在别人的机器上出
意外，而那时任务正在操作真实桌面——**渲染失败绝不能变成任务中断**。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

#: 动作日志表最多显示多少行。再多屏幕放不下，且早期步骤已经没参考价值
MAX_LOG_ROWS = 12


@dataclass
class PanelState:
    """面板要显示的全部内容。

    独立成一个纯数据类，是为了**渲染与状态分开**：状态可以单测（断言
    计数对不对），渲染只管把状态画出来。混在一起的话，测面板就得去解析
    终端输出。
    """

    instruction: str = ""
    trajectory_id: str = ""
    dry_run: bool = True
    backend: str = ""
    model: str = ""

    subtasks: list[str] = field(default_factory=list)
    current_subtask: int = 0
    current_goal: str = ""

    steps: list[dict] = field(default_factory=list)
    total_steps: int = 0
    failed_steps: int = 0

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_cny: float = 0.0
    priced: bool = True

    status: str = "运行中"

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def progress(self) -> float:
        if not self.subtasks:
            return 0.0
        return min(self.current_subtask / len(self.subtasks), 1.0)

    def record_step(self, record) -> None:
        """吸收一条 `StepRecord`。"""
        self.total_steps += 1
        if record.failed:
            self.failed_steps += 1

        self.current_subtask = max(self.current_subtask, record.subtask_id)
        if record.subtask:
            self.current_goal = record.subtask

        tokens = record.tokens or {}
        self.prompt_tokens += int(tokens.get("prompt_tokens", 0) or 0)
        self.completion_tokens += int(tokens.get("completion_tokens", 0) or 0)
        self.cost_cny += record.cost_cny

        if record.is_terminal:
            mark, action = "完成", "—"
        elif record.succeeded:
            mark, action = "成功", _action_text(record)
        else:
            mark, action = "失败", _action_text(record)

        self.steps.append(
            {
                "step": record.step,
                "subtask": record.subtask_id,
                "mark": mark,
                "action": action,
                "thinking": (record.model_thinking or "")[:60],
                "latency": (record.latency or {}).get("total_ms", 0.0),
                "error": record.error[:50],
            }
        )
        del self.steps[:-MAX_LOG_ROWS]


def _action_text(record) -> str:
    payload = record.action_model_coords or {}
    name = payload.get("action", "—")
    if "x" in payload:
        real = record.action_real_coords or {}
        # 同时显示两套坐标：验收标准 7 要求映射关系可追溯，面板上直接
        # 看得见的话，坐标错位第一时间就能发现，不用等回放
        return f"{name}({payload['x']},{payload['y']}) → 屏幕({real.get('x', '?')},{real.get('y', '?')})"
    for key in ("text", "keys", "direction", "duration"):
        if key in payload:
            return f"{name}({payload[key]!r})"
    return str(name)


# ---------------------------------------------------------------------- #


class LivePanel:
    """把 `PanelState` 画成一个会自动刷新的终端面板。

    没装 rich 或终端不支持时**自动退化成逐行打印**，不报错——面板是
    锦上添花，不该成为跑任务的前提条件。
    """

    def __init__(self, state: PanelState, enabled: bool = True) -> None:
        self.state = state
        self.enabled = enabled
        self._live = None
        self._console = None

    def __enter__(self) -> LivePanel:
        if not self.enabled:
            return self
        try:
            from rich.console import Console
            from rich.live import Live

            self._console = Console()
            self._live = Live(
                self._render(),
                console=self._console,
                refresh_per_second=4,
                transient=False,
            )
            self._live.__enter__()
        except Exception:  # noqa: BLE001 - 退化成纯文本即可
            logger.debug("实时面板不可用，退化为逐行输出", exc_info=True)
            self._live = None
        return self

    def __exit__(self, *exc_info) -> None:
        if self._live is not None:
            try:
                self._live.update(self._render())
                self._live.__exit__(*exc_info)
            except Exception:  # noqa: BLE001
                logger.debug("面板收尾失败", exc_info=True)

    def refresh(self) -> None:
        """状态变了之后调用。"""
        if self._live is None:
            if self.state.steps:
                last = self.state.steps[-1]
                print(f"  [{last['mark']}] #{last['step']} {last['action']}")
            return
        try:
            self._live.update(self._render())
        except Exception:  # noqa: BLE001 - 渲染失败绝不能中断任务
            logger.debug("面板刷新失败", exc_info=True)

    # ------------------------------------------------------------------ #

    def _render(self):
        from rich.console import Group
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        state = self.state

        header = Table.grid(padding=(0, 2))
        header.add_column(style="bold")
        header.add_column()
        header.add_row("任务", state.instruction)
        header.add_row("模型", f"{state.backend} / {state.model}")
        header.add_row("轨迹", state.trajectory_id or "—")
        mode = (
            Text("演练（不实际操作）", style="green")
            if state.dry_run
            else Text("实机执行 —— 急停热键 Ctrl+Alt+Q", style="bold red")
        )
        header.add_row("模式", mode)

        if state.subtasks:
            done = min(state.current_subtask, len(state.subtasks))
            bar = "█" * done + "░" * (len(state.subtasks) - done)
            header.add_row("进度", f"{bar}  {done}/{len(state.subtasks)}")
            header.add_row("当前", state.current_goal or "—")

        counters = Table.grid(padding=(0, 3))
        counters.add_column()
        counters.add_column()
        counters.add_column()
        cost = f"{state.cost_cny:.4f} 元" if state.priced else "未配单价（成本熔断已关闭）"
        counters.add_row(
            f"步数 {state.total_steps}（失败 {state.failed_steps}）",
            f"tokens {state.total_tokens}",
            f"成本 {cost}",
        )

        log = Table(expand=True, show_edge=False, pad_edge=False)
        log.add_column("#", width=4, justify="right")
        log.add_column("子", width=3, justify="right")
        log.add_column("状态", width=4)
        log.add_column("动作")
        log.add_column("耗时", width=8, justify="right")

        for row in state.steps:
            style = {"成功": "green", "完成": "cyan", "失败": "red"}.get(row["mark"], "")
            detail = row["action"]
            if row["error"]:
                detail = f"{detail}  [{row['error']}]"
            log.add_row(
                str(row["step"]),
                str(row["subtask"]),
                Text(row["mark"], style=style),
                detail,
                f"{row['latency'] / 1000:.1f}s" if row["latency"] else "—",
            )

        return Panel(
            Group(header, "", counters, "", log),
            title=f"GUI Agent — {state.status}",
            border_style="cyan" if state.dry_run else "red",
        )
