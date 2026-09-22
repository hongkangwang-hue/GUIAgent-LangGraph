"""自适应稳定等待 —— 大纲第 6 周任务 3。

## 这个文件守什么

固定睡 0.8 秒换成「连拍到不再变化」，风险有两头：

- **退得太早**：拍到还没画完的界面，那张图成了下一步的观测，
  模型据此认为「什么都没发生」
- **退不出来**：界面上有动画（视频、loading 转圈），永远等不到不变

所以测试的重点不是「能不能判稳」，而是**两头的边界**：最短等待不能省，
最长等待必须兜住。

时钟与 sleep 全部注入假的，**测试不真的睡**——一条用例睡两秒，
几十条就是一分钟，那样的套件没人愿意频繁跑。
"""

from __future__ import annotations

import numpy as np
import pytest

from perception.stability import (
    DEFAULT_MIN_WAIT_SECONDS,
    StabilityReport,
    savings_ms,
    wait_until_stable,
)


class FakeClock:
    """假时钟。`sleep` 只推进时间，不真的等。"""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds

    def __call__(self) -> float:
        return self.now


def frame(value: int, size=(8, 8)) -> np.ndarray:
    return np.full((*size, 3), value, dtype=np.uint8)


def frames_then_still(changing: int, still_value: int = 100):
    """前 `changing` 帧每帧都不一样，之后固定不变。"""
    state = {"i": 0}

    def capture():
        i = state["i"]
        state["i"] += 1
        return frame((i * 40) % 250 if i < changing else still_value)

    return capture


class TestStableExit:
    def test_界面不动时两帧就返回(self):
        clock = FakeClock()
        shot, report = wait_until_stable(
            lambda: frame(100), sleep=clock.sleep, clock=clock, poll=0.05
        )
        assert report.stable
        assert report.frames == 2, "静止界面不该多拍"
        assert shot is not None

    def test_界面停下后立刻返回不等满(self):
        """**这就是这个模块的全部价值**：界面早就定了就别再等。"""
        clock = FakeClock()
        _, report = wait_until_stable(
            frames_then_still(2), sleep=clock.sleep, clock=clock, poll=0.05, max_wait=2.0
        )
        assert report.stable
        assert report.waited_ms < 800, f"等了 {report.waited_ms}ms，比固定的 800ms 还久"

    def test_返回的是最后拍到的那一帧(self):
        """返回旧帧的话，下一步的观测就是过时画面——比等不稳更糟。"""
        clock = FakeClock()
        shot, _ = wait_until_stable(
            frames_then_still(1, still_value=77), sleep=clock.sleep, clock=clock
        )
        assert int(shot.mean()) == 77


class TestTimeout:
    def test_一直在动就等到上限返回(self):
        """视频、loading 转圈会让它永远等不到「不变」。**上限是必须，不是保险。**"""
        clock = FakeClock()
        counter = {"i": 0}

        def always_changing():
            counter["i"] += 1
            return frame((counter["i"] * 37) % 250)

        _, report = wait_until_stable(
            always_changing, sleep=clock.sleep, clock=clock, poll=0.05, max_wait=0.5
        )
        assert not report.stable
        assert report.waited_ms >= 500

    def test_超时不抛异常(self):
        """等不稳是常态。抛异常会把一次真实桌面操作中断在半路。"""
        clock = FakeClock()
        counter = {"i": 0}

        def always_changing():
            counter["i"] += 1
            return frame((counter["i"] * 37) % 250)

        shot, report = wait_until_stable(
            always_changing, sleep=clock.sleep, clock=clock, max_wait=0.3
        )
        assert shot is not None and not report.stable

    def test_超时也返回一帧可用的图(self):
        clock = FakeClock()
        shot, _ = wait_until_stable(
            frames_then_still(999), sleep=clock.sleep, clock=clock, max_wait=0.2
        )
        assert shot is not None


class TestMinWait:
    def test_默认会先等一小会儿再判稳(self):
        """**不能从 0 开始连拍。** 动作发出到界面开始响应之间有延迟，
        立刻拍两帧会拿到两张「还没开始变」的相同画面，误判成已经稳定——
        那正是固定等待当初存在的理由，换了策略也不能把它丢掉。
        """
        clock = FakeClock()
        wait_until_stable(lambda: frame(50), sleep=clock.sleep, clock=clock)
        assert clock.slept[0] == pytest.approx(DEFAULT_MIN_WAIT_SECONDS)

    def test_最短等待可以调到0(self):
        """给测试和确定性很强的场景留口子，但默认不是 0。"""
        clock = FakeClock()
        wait_until_stable(lambda: frame(50), sleep=clock.sleep, clock=clock, min_wait=0.0)
        assert clock.slept[0] != DEFAULT_MIN_WAIT_SECONDS


class TestReport:
    def test_报告带上阈值(self):
        """阈值是拍的。不跟着记，事后看不出当时判稳的标准是什么。"""
        clock = FakeClock()
        _, report = wait_until_stable(
            lambda: frame(10), sleep=clock.sleep, clock=clock, threshold=0.05
        )
        assert report.as_dict()["threshold"] == 0.05

    def test_报告能直接进轨迹(self):
        clock = FakeClock()
        _, report = wait_until_stable(lambda: frame(10), sleep=clock.sleep, clock=clock)
        payload = report.as_dict()
        assert set(payload) == {"stable", "waited_ms", "frames", "last_ratio", "threshold"}


class TestSavings:
    def test_省下的时间是算出来的不是估的(self):
        reports = [
            StabilityReport(stable=True, waited_ms=200.0, frames=2, last_ratio=0.0, threshold=0.01),
            StabilityReport(stable=True, waited_ms=300.0, frames=3, last_ratio=0.0, threshold=0.01),
        ]
        stats = savings_ms(reports, fixed_seconds=0.8)
        assert stats["mean_waited_ms"] == 250.0
        assert stats["mean_saved_ms"] == 550.0

    def test_等更久时如实报负数(self):
        """界面真的慢时自适应会等得更久。**那不是失败，是拍到了更可用的图**，
        但必须如实报出来，不能只报好看的那一半。
        """
        reports = [
            StabilityReport(
                stable=False, waited_ms=2000.0, frames=40, last_ratio=0.3, threshold=0.01
            )
        ]
        assert savings_ms(reports, fixed_seconds=0.8)["mean_saved_ms"] == -1200.0

    def test_超时比例单独报(self):
        """等满上限的比例高，说明阈值或上限该调了——这个信号不能被平均值盖住。"""
        reports = [
            StabilityReport(stable=True, waited_ms=200.0, frames=2, last_ratio=0.0, threshold=0.01),
            StabilityReport(
                stable=False, waited_ms=2000.0, frames=40, last_ratio=0.2, threshold=0.01
            ),
        ]
        assert savings_ms(reports, fixed_seconds=0.8)["timeout_rate"] == 0.5

    def test_没有数据时不炸(self):
        assert savings_ms([], fixed_seconds=0.8)["n"] == 0


class TestLoopIntegration:
    """接进循环的那一层。"""

    def test_默认走固定等待(self):
        """**默认必须是 v1.0 的行为。** 改默认等于让 M2~M5 的全部端到端
        数字失去可比性，而那条对照链是这个项目最值钱的资产。
        """
        from core.loop import LoopConfig

        assert LoopConfig().adaptive_settle is False

    def test_两个引擎共用同一个等待实现(self):
        """上一次合并就是栽在「单步逻辑有两份拷贝」上。

        图版的 observe 节点必须调 `AgentLoop._settle_and_capture`，
        而不是自己再写一遍。
        """
        import inspect

        from core.graph_loop import GraphAgentLoop

        source = inspect.getsource(GraphAgentLoop._node_observe)
        assert "_settle_and_capture" in source
        assert "settle_seconds" not in source, "图版不该自己再实现一遍等待"

    def test_自适应时把实测记进轨迹(self, monkeypatch):
        """省了多少时间要能从轨迹里算回来，不写估算值。"""
        from core.loop import AgentLoop, LoopConfig
        from core.trajectory import StepRecord

        loop = AgentLoop.__new__(AgentLoop)
        loop.config = LoopConfig(adaptive_settle=True)
        loop.capturer = type("C", (), {"capture": staticmethod(lambda **kw: frame(9))})()

        record = StepRecord()
        shot = loop._settle_and_capture(record)
        assert shot is not None
        assert "settle" in record.meta
        assert record.meta["settle"]["frames"] >= 2

    def test_固定等待时不写settle字段(self):
        """没开的功能不该在轨迹里留痕，否则事后分不清哪轮开了哪轮没开。"""
        import time as time_module

        from core.loop import AgentLoop, LoopConfig
        from core.trajectory import StepRecord

        loop = AgentLoop.__new__(AgentLoop)
        loop.config = LoopConfig(settle_seconds=0.0)
        loop.capturer = type("C", (), {"capture": staticmethod(lambda **kw: frame(9))})()
        original = time_module.sleep
        try:
            record = StepRecord()
            loop._settle_and_capture(record)
            assert "settle" not in record.meta
        finally:
            time_module.sleep = original
