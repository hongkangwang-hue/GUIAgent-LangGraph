"""自适应界面稳定等待 —— 大纲第 6 周任务 3「提升感知速度」的落点。

## 它替掉的是什么

动作发出之后不能立刻截图：界面可能正在重绘，拿到的是动作前的画面。
v1.0 的做法是**固定睡 0.8 秒**（`LoopConfig.settle_seconds`），然后拍。

固定值两头不讨好：

- **多数步骤根本不需要 0.8 秒。** 点一下按钮、按一个键，界面几十毫秒就定了，
  剩下的时间纯等。
- **少数步骤 0.8 秒不够。** 浏览器冷启动、UWP 应用拉起来要好几秒，
  拍早了拍到的是半张白屏——而那张图会成为下一步的观测，模型据此判断
  「什么都没发生」。

## 换成什么

连拍几帧，**直到连续两帧之间不再变化**就返回，最多等 `max_wait`：

    发出动作 → 拍一帧 → 隔 poll 秒再拍 → 两帧一样？ → 是：返回
                  ↑                              ↓ 否
                  └──────────────────────────────┘（直到 max_wait）

界面停了就立刻走，没停就继续等——**这比固定值两个方向都更准**。

## 为什么不默认打开

它**改变模型看到的图**：更早拍到的界面和睡满 0.8 秒拍到的可能不一样。
M2~M5 的全部端到端数字都是在固定等待下跑出来的，默认改掉的话，
新旧数据不可比，而这条对照链是这个项目最值钱的资产。

所以是 `--adaptive-settle` 开关，默认关，交付物里用 A/B 对照给出实测收益。

## 省下的时间要用实测说，不要估算

`StabilityReport` 记下实际等了多久、拍了几帧、是稳定退出还是超时退出。
报告里的「每步省了多少毫秒」直接从这些数算，不写估计值——本项目
栽过的口径错误里，有好几处都是「看起来合理的估算」。

## 已知边界

1. **动画会让它等满。** 视频、loading 转圈、光标闪烁都在持续改变像素。
   `PIXEL_TOLERANCE`（沿用 `perception.change` 的）压得住抗锯齿抖动，
   压不住真动画。所以 `max_wait` 是必须有的，不是保险。
2. **拍帧本身要花时间。** dxcam 客机实测 p50 0.043ms，可以忽略；
   但 `fresh=True` 要等一帧新画面，poll 间隔不该小于一帧的时间。
3. **它只保证「不再变」，不保证「变对了」。** 判断动作有没有起效是
   帧差和 Reflector 的事，两者用的是动作前后的对比，不是这里的连续两帧。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from perception.change import DEFAULT_THRESHOLD, compare

logger = logging.getLogger(__name__)

#: 两次采样的间隔。比一帧（60Hz 约 16.7ms）略大，避免拍到同一帧误判为「稳了」。
DEFAULT_POLL_SECONDS = 0.05

#: 最长等待。超过就带着「没稳定」的结论返回，**不抛异常**：
#: 等不稳是常态（有动画），上层该拿这张图继续跑，而不是中断任务。
DEFAULT_MAX_WAIT_SECONDS = 2.0

#: 至少要等这么久再开始判稳。**0 是不行的**：动作发出到界面开始响应之间
#: 有延迟，立刻连拍两帧会拍到两张「还没开始变」的相同画面，
#: 误判成「已经稳定」——这正是固定等待存在的原因，不能因为换了策略就丢掉。
DEFAULT_MIN_WAIT_SECONDS = 0.15


@dataclass
class StabilityReport:
    """一次稳定等待的实测记录。

    **每个字段都是报告里要用的数**：省了多少时间靠 `waited_ms` 与固定
    等待相减，等满的比例靠 `stable` 统计，拍帧开销靠 `frames`。
    """

    #: 最终有没有稳定下来。False 表示等到 `max_wait` 仍在变
    stable: bool
    #: 实际等了多久（毫秒），含采样时间
    waited_ms: float
    #: 一共拍了几帧
    frames: int
    #: 最后一次比较的变化比例
    last_ratio: float
    #: 稳定判据的阈值，跟着一起记——阈值是拍的，不记就看不出对不对
    threshold: float

    def as_dict(self) -> dict:
        return {
            "stable": self.stable,
            "waited_ms": round(self.waited_ms, 2),
            "frames": self.frames,
            "last_ratio": round(self.last_ratio, 5),
            "threshold": self.threshold,
        }


def wait_until_stable(
    capture,
    *,
    max_wait: float = DEFAULT_MAX_WAIT_SECONDS,
    poll: float = DEFAULT_POLL_SECONDS,
    min_wait: float = DEFAULT_MIN_WAIT_SECONDS,
    threshold: float = DEFAULT_THRESHOLD,
    sleep=time.sleep,
    clock=time.perf_counter,
):
    """连拍到界面不再变化为止。返回 (最后一帧, StabilityReport)。

    `capture` 是一个**无参可调用对象**，每次调用返回一张新截图。
    调用方负责绑定好显示器/区域与 `fresh=True`：这个模块不该知道
    `ScreenCapturer` 的参数长什么样，否则截图接口一改这里就得跟着改。

    `sleep` 与 `clock` 可注入，测试里用假的，**不真的睡**——一条用例睡
    两秒，几十条用例就是一分钟，没人会愿意频繁跑这样的测试套件。

    等不稳时返回 `stable=False` 而**不是抛异常**：界面上有动画是常态，
    上层要做的是拿着这张图继续，不是中断一次真实桌面操作。
    """
    started = clock()
    if min_wait > 0:
        sleep(min_wait)

    previous = capture()
    frames = 1
    last_ratio = 1.0

    while True:
        elapsed = clock() - started
        if elapsed >= max_wait:
            logger.debug("等待界面稳定超时：%.0fms，最后变化 %.4f", elapsed * 1000, last_ratio)
            return previous, StabilityReport(
                stable=False,
                waited_ms=elapsed * 1000.0,
                frames=frames,
                last_ratio=last_ratio,
                threshold=threshold,
            )

        sleep(poll)
        current = capture()
        frames += 1
        report = compare(previous, current, threshold=threshold)
        last_ratio = report.ratio
        previous = current

        if not report.changed:
            return current, StabilityReport(
                stable=True,
                waited_ms=(clock() - started) * 1000.0,
                frames=frames,
                last_ratio=last_ratio,
                threshold=threshold,
            )


def savings_ms(reports: list[StabilityReport], fixed_seconds: float) -> dict:
    """相对固定等待省下了多少。**报告里的数字从这里来，不手算。**

    `fixed_seconds` 传 v1.0 的 `settle_seconds`。返回的 `mean_saved_ms`
    可能是负数——界面确实慢的时候自适应会等得更久，**那不是失败**，
    是拍到了一张更可用的图。负数要如实报出来，不能只报正的那些。
    """
    if not reports:
        return {"n": 0, "mean_waited_ms": 0.0, "mean_saved_ms": 0.0, "timeout_rate": 0.0}
    fixed_ms = fixed_seconds * 1000.0
    waited = [r.waited_ms for r in reports]
    mean_waited = sum(waited) / len(waited)
    return {
        "n": len(reports),
        "mean_waited_ms": round(mean_waited, 2),
        "mean_saved_ms": round(fixed_ms - mean_waited, 2),
        "timeout_rate": round(sum(1 for r in reports if not r.stable) / len(reports), 4),
        "mean_frames": round(sum(r.frames for r in reports) / len(reports), 2),
        "fixed_baseline_ms": fixed_ms,
    }


__all__ = [
    "DEFAULT_MAX_WAIT_SECONDS",
    "DEFAULT_MIN_WAIT_SECONDS",
    "DEFAULT_POLL_SECONDS",
    "StabilityReport",
    "savings_ms",
    "wait_until_stable",
]
