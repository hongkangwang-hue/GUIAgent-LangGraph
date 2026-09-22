"""子任务完成判定 —— M4 任务 1 / 大纲 W6「错误检测与自动重试」。

> **位置与里程碑文档不同。** M4 文档写的是 `agent/reflector.py`，
> 实际落在 `core/`。原因有二：`agent/__init__.py` 会导入 `session`，
> 而 `session` 导入 `core.loop`，放在 `agent/` 下会造成循环导入；
> 且本模块与 `core/retry.py` 是同一类东西——同样的
> reset / observe / 判定生命周期，同样只服务于循环内部，不知道
> 后端和提示词。**里程碑文档写在代码之前，以实际结构为准。**

## 为什么需要它

M3 档 A 的端到端实测（`docs/m4-错误分类体系.md`）：

    94.4% 的子任务在执行 0~1 个动作之后就宣布完成
    21/25 轮以 `done` 收尾但程序化判定失败
    动作执行 ok 之后立刻报 `done`：78 次

模型的完成判据是「我发出了动作」，不是「屏幕变了」。`model_thinking`
的句式直接暴露这一点：

    因为发送按钮已经被点击，所以任务完成
    因为地址栏已经被点击，所以任务完成

**这不能靠换数据或加训练解决。** ScreenAgent 的标注是「这一步该做什么」，
不含「上一步做成了没有」——模型没有可学的信号去区分「点了」和「点中了」。
所以必须在系统层加一道校验。

## 与 M2 原则的关系

M2 定下「程序化成功判定，绝不采信模型自报」，但只用在**轮次级验收**上；
子任务级的推进仍然无条件相信模型的 `done`。本模块把同一条原则下推一层。

## 级联设计（里程碑 M4）

    1. OpenCV 帧差      免费、毫秒级：界面完全没变 → 动作未生效
    2. 程序化判据      免费、毫秒级：进程 / 窗口标题 / 文件 / UIA 控件
    3. 语义判定        仅在前两级冲突或都判不了时触发

**本版只实现级联 1。** 理由是频次：A 类的主峰是「1 个动作后报 done」
（81.1%），而那个动作若真的命中，屏幕通常会变——级联 1 就能挡住。
级联 2 需要把任务级判据下推到子任务级，级联 3 要多一次推理，
两者都等级联 1 的实测触发率出来再定。

## 两条刻意的保守选择

**① 「0 个动作就 done」不否决，只标记。**
实测 12 个子任务（13.3%）属于这种。它**可能是合法的**——子任务的目标
本来就已经满足了（这正是 M2 那批假成功的成因：precondition 没重置好，
判据一开始就满足）。现有数据分不开「合法的已满足」和「纯粹偷懒」。
贸然否决会挡掉合法情况，制造一个新的失败模式。

**② 连续否决有上限。**
否决之后模型可能原样再报一次 `done`。没有上限就是死循环。
到上限仍报 `done` 时接受它并标记，让程序化判定去兜底——
**Reflector 的职责是提高证据要求，不是替代最终判据。**
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: 判定结论。
VERDICT_ACCEPT = "accept"  #: 有依据，接受 done
VERDICT_REJECT = "reject"  #: 无依据，否决 done，让模型继续
VERDICT_UNKNOWN = "unknown"  #: 判不了（缺数据），按接受处理但标记

#: 回传给模型的反馈。**必须说清"上一步没生效"，而不只是"再试一次"。**
#:
#: `core/retry.py` 的实测记录：只把 left_click 升级成 double_click，
#: 模型下一轮多半还给同一个坐标——探针量到的重复率 75.3%。
#: 要让它换目标，得把失败信息写进子任务描述里。
REJECT_HINT = (
    "上一步动作执行后屏幕没有发生变化，说明它很可能没有命中目标。"
    "子任务尚未完成，请换一个位置或换一种方式再试，不要报告完成。"
)


@dataclass
class Verdict:
    """一次判定的结果。"""

    verdict: str
    reason: str
    level: int = 0  #: 在第几级级联出的结论；0 表示没进级联
    hint: str = ""  #: 要回传给模型的话，空表示没有

    @property
    def rejected(self) -> bool:
        return self.verdict == VERDICT_REJECT

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "level": self.level,
            "hint": self.hint,
        }


@dataclass
class Reflector:
    """按子任务维护状态，判定模型自报的 `done` 是否可信。

    调用顺序与 `RetryPolicy` 一致：
    每个子任务开始时 `reset()`；每次动作执行完 `observe()`；
    模型报 `done` 时 `judge()`。
    """

    #: 同一子任务里最多连续否决几次。到上限就接受，交给程序化判定兜底。
    max_rejects: int = 2

    #: 已执行的动作数（不含 done 本身）
    actions_taken: int = field(default=0, init=False)
    #: 最近一次动作之后屏幕有没有变；None = 没测到
    last_changed: bool | None = field(default=None, init=False)
    #: 本子任务里已经否决过几次
    rejects: int = field(default=0, init=False)

    def reset(self) -> None:
        self.actions_taken = 0
        self.last_changed = None
        self.rejects = 0

    def observe(self, *, changed: bool | None) -> None:
        """记下一次已经执行完的动作及其效果。"""
        self.actions_taken += 1
        self.last_changed = changed

    def judge(self) -> Verdict:
        """模型报了 `done`，判断要不要接受。**不改状态之外的东西。**"""
        if self.rejects >= self.max_rejects:
            return Verdict(
                VERDICT_ACCEPT,
                f"已连续否决 {self.rejects} 次仍报完成，接受并交由程序化判定兜底",
                level=1,
            )

        if self.actions_taken == 0:
            # 见模块文档「两条刻意的保守选择」①
            return Verdict(
                VERDICT_UNKNOWN,
                "本子任务尚未执行任何动作即报完成；可能是目标本来就已满足，不否决",
                level=0,
            )

        if self.last_changed is None:
            return Verdict(
                VERDICT_UNKNOWN,
                "缺少帧差数据，无法判定上一步是否生效",
                level=0,
            )

        if self.last_changed:
            return Verdict(
                VERDICT_ACCEPT,
                "上一步动作后屏幕发生了变化，动作已生效",
                level=1,
            )

        self.rejects += 1
        return Verdict(
            VERDICT_REJECT,
            f"上一步动作后屏幕无变化，动作未生效（第 {self.rejects} 次否决）",
            level=1,
            hint=REJECT_HINT,
        )
