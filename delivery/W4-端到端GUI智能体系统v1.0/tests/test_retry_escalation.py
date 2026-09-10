"""错误检测与自动重试 —— 大纲 W6 任务 2。

## 这套机制要解决的具体问题

2026-08-25 实测（`open_browser`，微调 3B，10 轮）：

    出现过 double_click 的轮次    2/2 成功
    全程只有 left_click 的轮次    0/8 成功        Fisher p = 0.0222

失败形态是同一个动作重复到步数耗尽：

    子任务「点击任务栏上的Microsoft Edge图标」
       left_click (15, 190) ×6      execution_status 全是 ok

单击桌面图标只是选中它，不会打开。而"选中"和"打开"在 `execution_status`
里长得一模一样——**系统没有任何一处能发现"点了六次，什么都没发生"。**
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from core.retry import ESCALATION, SAME_SPOT_RADIUS, RetryPolicy  # noqa: E402


def screen(fill=0, box=None):
    image = np.full((108, 192, 3), fill, dtype=np.uint8)
    if box:
        top, left, height, width = box
        image[top : top + height, left : left + width] = 255
    return image


class TestChangeDetection:
    def test_完全没变判为没变(self):
        from perception.change import compare

        assert not compare(screen(), screen()).changed

    def test_开一个窗口判为变了(self):
        from perception.change import compare

        assert compare(screen(), screen(box=(20, 30, 60, 120))).changed

    def test_图标高亮不算变(self):
        """**这条是关键。**

        单击桌面图标会给它加个高亮框——那也是像素变化，但任务没有推进。
        判成"变了"的话，升级机制就永远不会触发，而那正是它要抓的场景。
        """
        from perception.change import compare

        # 8×8 的高亮块占 192×108 的 0.3%，与本机 1920×1080 上的图标高亮同量级
        report = compare(screen(), screen(box=(10, 10, 8, 8)))
        assert not report.changed
        assert 0 < report.ratio < report.threshold

    def test_比例要带出来(self):
        """只返回布尔值的话，阈值定得对不对就再也看不出来了——**而它是拍的**。"""
        from perception.change import compare

        assert "ratio" in compare(screen(), screen()).as_dict()

    def test_尺寸不一致标出来而不是当成有效变化(self):
        from perception.change import compare

        report = compare(screen(), np.zeros((50, 50, 3), dtype=np.uint8))
        assert report.changed and report.size_mismatch

    def test_拍不到就当没变化(self):
        """**宁可漏报也不要误报。**

        误报（明明没动却以为动了）会让升级永远不触发；
        漏报只是继续用原策略，代价小得多。
        """
        from perception.change import compare

        assert not compare(screen(), None).changed
        assert not compare(None, None).changed

    def test_小噪点不算变化(self):
        """JPEG 噪点与抗锯齿抖动不该触发升级。"""
        from perception.change import compare

        noisy = (screen(fill=100).astype(np.int16) + 10).astype(np.uint8)
        assert not compare(screen(fill=100), noisy).changed


class TestEscalation:
    def test_第一次不升级(self):
        assert not RetryPolicy().revise("left_click", 15, 190).escalated

    def test_同一位置无效之后升级(self):
        policy = RetryPolicy()
        policy.observe("left_click", 15, 190, changed=False)
        revision = policy.revise("left_click", 15, 190)
        assert revision.escalated
        assert revision.action_type == "double_click"
        assert revision.reason

    def test_屏幕动过就不升级(self):
        """**动作起效了就不该改写它。**

        模型第二次给同一个坐标，可能是在确认目标，而第一次点击其实已经
        打开了一个还在渲染的窗口。
        """
        policy = RetryPolicy()
        policy.observe("left_click", 15, 190, changed=True)
        assert not policy.revise("left_click", 15, 190).escalated

    def test_效果未知时不升级(self):
        """截图失败等情况下 `changed` 是 None——不知道就别动。"""
        policy = RetryPolicy()
        policy.observe("left_click", 15, 190, changed=None)
        assert not policy.revise("left_click", 15, 190).escalated

    def test_换了位置就不升级(self):
        policy = RetryPolicy()
        policy.observe("left_click", 15, 190, changed=False)
        assert not policy.revise("left_click", 900, 100).escalated

    def test_抖动几个像素算同一个位置(self):
        """**GUI 按钮几十像素宽。**

        逐字比对会把抖动当成"改主意了"从而不升级，而探针数据显示那恰恰
        是死循环的形态。
        """
        policy = RetryPolicy()
        policy.observe("left_click", 15, 190, changed=False)
        assert policy.revise("left_click", 15 + SAME_SPOT_RADIUS - 1, 190).escalated

    def test_双击到头了不再改写但要说明(self):
        """**不静默重复。** 阶梯到头要说清楚，让上层决定换目标还是放弃。"""
        policy = RetryPolicy()
        policy.observe("double_click", 15, 190, changed=False)
        revision = policy.revise("double_click", 15, 190)
        assert not revision.escalated
        assert revision.reason and "没有可升级" in revision.reason

    def test_阶梯只有一级(self):
        """更长的阶梯需要证据。**现在只有「单击变双击」有实测**（p=0.0222）。"""
        assert ESCALATION == {"left_click": "double_click"}

    def test_revise_不改状态(self):
        policy = RetryPolicy()
        policy.observe("left_click", 15, 190, changed=False)
        first = policy.revise("left_click", 15, 190)
        assert policy.revise("left_click", 15, 190) == first

    def test_reset_清掉上个子任务的历史(self):
        """不重置的话，上个子任务在同一坐标上的尝试会让这个子任务的
        **第一次点击就被升级**——而那时屏幕状态已经完全不同了。
        """
        policy = RetryPolicy()
        policy.observe("left_click", 15, 190, changed=False)
        policy.reset()
        assert not policy.revise("left_click", 15, 190).escalated

    def test_patience_可调(self):
        policy = RetryPolicy(patience=2)
        policy.observe("left_click", 15, 190, changed=False)
        assert not policy.revise("left_click", 15, 190).escalated
        policy.observe("left_click", 15, 190, changed=False)
        assert policy.revise("left_click", 15, 190).escalated


class TestHint:
    """**升级只改这一步的动作，改不了模型下一步的判断。**

    不把"你刚才点的地方没反应"告诉它，它下一步多半还给同一个坐标——
    探针量到的重复率 75.3% 就是这么来的。
    """

    def test_没有无效尝试时不说话(self):
        assert RetryPolicy().hint() == ""
        policy = RetryPolicy()
        policy.observe("left_click", 15, 190, changed=True)
        assert policy.hint() == ""

    def test_说清楚在哪没反应(self):
        policy = RetryPolicy()
        policy.observe("left_click", 15, 190, changed=False)
        hint = policy.hint()
        assert "15" in hint and "190" in hint
        assert "没有任何变化" in hint

    def test_提示要求换目标而不只是重试(self):
        """**「再试一次」是没用的建议**——模型会照原样再给一遍。"""
        policy = RetryPolicy()
        policy.observe("left_click", 15, 190, changed=False)
        assert "换" in policy.hint()


class TestActionRewrite:
    def test_with_type_只换类型(self):
        from control.actions import Action, ActionType

        action = Action(type=ActionType.LEFT_CLICK, x=15, y=190, reasoning="点图标")
        upgraded = action.with_type("double_click")
        assert upgraded.type is ActionType.DOUBLE_CLICK
        assert (upgraded.x, upgraded.y) == (15, 190)
        assert upgraded.reasoning == "点图标"

    def test_with_type_返回新对象(self):
        """原地改的话，`StepRecord` 里已经记下的改写前动作会跟着变，
        事后就分不清"模型自己选了双击"和"策略改的"。
        """
        from control.actions import Action, ActionType

        action = Action(type=ActionType.LEFT_CLICK, x=15, y=190)
        assert action.with_type("double_click") is not action
        assert action.type is ActionType.LEFT_CLICK


class TestLoopWiring:
    def test_默认关闭(self):
        """**打开会改变行为，而 M2/M3 的实测都是关着跑的。**

        默认打开等于让那些复现命令悄悄跑出另一套结果。
        """
        from core.loop import LoopConfig

        assert LoopConfig().escalate_on_no_change is False

    def test_每个子任务开始时重置(self):
        import inspect

        from core.loop import AgentLoop

        assert "self.retry_policy.reset()" in inspect.getsource(AgentLoop.run_subtask)

    def test_关闭时不进策略分支(self):
        """默认路径不该为一个关着的功能付出代价，也不该改变任何记录。

        三处入口（升级、变化检测、回传提示）都必须挂在开关下。
        """
        import inspect

        from core.loop import AgentLoop

        src = inspect.getsource(AgentLoop._run_one_step)
        for marker in ("self.retry_policy.revise", "from perception.change import compare"):
            index = src.index(marker)
            preceding = src[:index]
            assert "if self.config.escalate_on_no_change:" in preceding, marker

    def test_开关进存档(self):
        from agent.session import SessionConfig
        from core.loop import LoopConfig

        config = SessionConfig(loop=LoopConfig(escalate_on_no_change=True))
        assert config.as_dict()["escalate_on_no_change"] is True

    def test_开关只有一个来源(self):
        """`SessionConfig` 不再自己存一份。

        两处各存一份的话，一处改了另一处不跟着改，存档记的就不是实际生效
        的那个值——**本项目栽过两次**（提示词模板、屏幕分辨率）。
        """
        from agent.session import SessionConfig

        assert not hasattr(SessionConfig(), "escalate_on_no_change")

    def test_cli_默认关闭(self):
        from scripts.run_basic_tasks import build_parser

        assert build_parser().parse_args([]).escalate_on_no_change is False

    def test_cli_开关进运行存档(self):
        import json

        from scripts.run_basic_tasks import archive_payload
        from tests.test_run_provenance import make_args

        args = make_args(escalate_on_no_change=True)
        payload = json.loads(archive_payload([], args, "离线", offline=True, partial=False))
        assert payload["escalate_on_no_change"] is True
