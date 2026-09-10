"""执行器与规划器只能用模型真的会的动作。

## 这条守的是什么

2026-08-25 查出的结构性错位：微调模型的训练集里 `type` / `key` 各 **0** 条
（ScreenAgent 的键盘动作因不带坐标被整批过滤），而提示词照样告诉它
「你可以 `type`」。五个基础任务里四个需要打字——**模型连掷骰子的机会
都没有**，25 轮跑完只看到一片 0/5，系统里没有任何一处发现原因。

**收窄规划器比收窄执行器更要紧。** 执行器面对一个「输入 Microsoft Edge」
的子任务，无论提示词怎么写都做不了；只有让规划器一开始就不产出这种步骤，
整轮才不会卡在那儿烧完 `max_steps`。
"""

from __future__ import annotations

import pytest

from agent.prompts import PromptError, load_template, render_action_reference

CLICK_ONLY = ("left_click", "double_click", "right_click", "mouse_move")


class TestActionReferenceFilter:
    def test_不给_allowed_时列出全部核心动作(self):
        text = render_action_reference()
        for name in ("left_click", "type", "key", "scroll"):
            assert f"`{name}`" in text

    def test_给了之后只列那几个(self):
        text = render_action_reference(allowed=CLICK_ONLY)
        for name in CLICK_ONLY:
            assert f"`{name}`" in text
        for name in ("type", "key", "scroll", "left_click_drag"):
            assert f"`{name}`" not in text, f"{name} 不该出现——模型不会它"

    def test_空元组等同于不限制(self):
        """**「没声明」和「声明为空」必须区别对待。**

        空元组当成「一个动作都不许用」的话，所有既有调用都会炸。
        """
        assert render_action_reference(allowed=()) == render_action_reference()

    def test_动作名拼错要报错而不是静默忽略(self):
        """拼错一个名字就等于悄悄放宽了限制——**正是这个参数要防的事**。"""
        with pytest.raises(PromptError, match="不存在的动作"):
            render_action_reference(allowed=("left_click", "typ"))

    def test_非核心动作也算拼错(self):
        """`only_core=True` 下,非核心动作不在候选里——要报错,不能静默丢掉。"""
        with pytest.raises(PromptError, match="不存在的动作"):
            render_action_reference(allowed=("left_click", "cursor_position"))

    def test_正文里手写的动作名也被删掉(self):
        """**光过滤动作清单不够。**

        模板的使用规则那几条是手写的，直接点了动作名：

            3. **中文输入用 `type`。** 它内部走剪贴板，可以正确输入中文。

        不删的话，同一份提示词一边说「你只有这四个动作」，一边说
        「中文输入用 type」——正是本项目一直在防的那种自相矛盾。
        这条是写 `allowed_actions` 时被测试先撞出来的。
        """
        template = load_template("executor_v1")
        narrow = template.render_system(width=1000, height=1000, allowed_actions=CLICK_ONLY)
        assert "中文输入用" not in narrow
        assert "`wait`" not in narrow

    def test_不点名动作的规则要留着(self):
        """**只删点名了不可用动作的行**，别把整段规则都削掉。"""
        template = load_template("executor_v1")
        narrow = template.render_system(width=1000, height=1000, allowed_actions=CLICK_ONLY)
        assert "一次只做一步" in narrow
        assert "点击目标要具体" in narrow

    def test_参数说明跟着动作一起被过滤(self):
        """不能只删动作名、留下参数说明——那比不过滤更糟。"""
        text = render_action_reference(allowed=CLICK_ONLY)
        assert "`text`(string)" not in text
        assert "`keys`(string)" not in text


class TestExecutorPrompt:
    def test_渲染系统提示时能收窄(self):
        template = load_template("executor_v1")
        wide = template.render_system(width=1000, height=1000)
        narrow = template.render_system(width=1000, height=1000, allowed_actions=CLICK_ONLY)
        assert "`type`" in wide
        assert "`type`" not in narrow
        assert len(narrow) < len(wide)

    def test_不传时与从前逐字相同(self):
        """**既有行为不能变。** 在线 8B 那一路不收窄，它的提示词必须原样。"""
        template = load_template("executor_v1")
        assert template.render_system(width=1000, height=1000) == template.render_system(
            width=1000, height=1000, allowed_actions=None
        )


class TestPlannerConstraint:
    class _Backend:
        name = "fake"
        model = "fake"
        system_prompt = ""
        few_shot: list = []
        user_template = "{instruction}"

        def __init__(self):
            self.seen = ""

        def complete(self, prompt, screenshot=None, history=None):
            from llm.base import RawResponse

            self.seen = self.system_prompt
            return RawResponse(text='{"subtasks": ["点击任务栏上的浏览器图标"]}')

    def _plan(self, allowed):
        from agent.planner import Planner

        backend = self._Backend()
        Planner(backend, allowed_actions=allowed).plan("打开浏览器")
        return backend.seen

    def test_约束段进了规划器的系统提示(self):
        system = self._plan(CLICK_ONLY)
        assert "硬约束" in system
        for name in CLICK_ONLY:
            assert f"`{name}`" in system

    def test_约束段说明了该怎么换路走(self):
        """**光说「不许用 type」不够。**

        模型需要一个可行的替代路径，否则它只会把同一个计划再写一遍。
        """
        system = self._plan(CLICK_ONLY)
        assert "点击任务栏上的浏览器图标" in system

    def test_不给约束时提示词里没有那一段(self):
        assert "硬约束" not in self._plan(())

    def test_约束段是模块级常量而不是内联字符串(self):
        """写成常量,测试才能逐字断言它;内联的话改一个字测试也看不出来。"""
        from agent.planner import CONSTRAINT_LINES

        assert any("硬约束" in line for line in CONSTRAINT_LINES)


class TestSessionWiring:
    def test_SessionConfig_默认不限制(self):
        from agent.session import SessionConfig

        assert SessionConfig().allowed_actions == ()

    def test_动作集进轨迹存档(self):
        """**能左右结论的变量必须进存档。**

        与 `executor_template`、`screen` 同一条护栏——不记下来，事后分不清
        「它没变」和「它变了但没人看见」。
        """
        from agent.session import SessionConfig

        payload = SessionConfig(allowed_actions=CLICK_ONLY).as_dict()
        assert payload["allowed_actions"] == list(CLICK_ONLY)

    def test_规划器拿到的是_config_里那一份(self):
        import inspect

        from agent.session import Session

        src = inspect.getsource(Session)
        assert "allowed_actions=self.config.allowed_actions" in src

    def test_执行器提示词也收窄(self):
        import inspect

        from agent.session import Session

        src = inspect.getsource(Session._prepare_backend)
        assert "allowed_actions=self.config.allowed_actions" in src


class TestCli:
    def test_逗号分隔解析(self):
        from scripts.run_basic_tasks import build_parser

        args = build_parser().parse_args(["--allowed-actions", "left_click, mouse_move"])
        parsed = tuple(x.strip() for x in args.allowed_actions.split(",") if x.strip())
        assert parsed == ("left_click", "mouse_move")

    def test_默认留空(self):
        from scripts.run_basic_tasks import build_parser

        assert build_parser().parse_args([]).allowed_actions == ""


class TestWholeRuleDropped:
    """**一条规则要整条删，连同它的续行。**

    2026-08-25 的真实事故。初版按行删，而模板的规则 2 是两行：

        2. **看不清就先观察。** 如果界面还在加载、菜单还在展开，用
           `wait` 等一下，不要对着中间态乱点。

    只有第二行带 `wait`，删完剩下 **「…菜单还在展开，用」**——一句话被
    砍成半截。实测 `close_app` 从 4/4 掉到 0/2，两轮都在第一次调用上
    `parse_error`。

    而 `_drop_rules_naming` 的初版文档里明明写着「改写留下的半句话比整行
    删掉更危险」。**文档描述了危险，实现造出了它。**
    """

    def _narrow(self, name="executor_v1"):
        return load_template(name).render_system(
            width=1000, height=1000, allowed_actions=CLICK_ONLY
        )

    def test_多行规则整条消失(self):
        narrow = self._narrow()
        assert "`wait`" not in narrow
        assert "看不清就先观察" not in narrow, "第一行留下了——那就是半截句子"
        assert "菜单还在展开" not in narrow

    def test_过滤没有造出新的半截行(self):
        """**只抓过滤造出来的。**

        模板本来就有以「，」结尾的自然换行（`坐标范围是 x ∈ [0,1000)，`），
        一刀切地禁止连词结尾会误报。真正的判据是：收窄之后**每一行的后继
        关系都必须在原文里也成立**——一行在原文里有续行、收窄后没有了，
        那就是被砍了半截。
        """
        template = load_template("executor_v1")
        wide = template.render_system(width=1000, height=1000).splitlines()
        narrow = self._narrow().splitlines()

        wide_next = {line: wide[i + 1] for i, line in enumerate(wide[:-1])}
        narrow_next = {line: narrow[i + 1] for i, line in enumerate(narrow[:-1])}
        for line, following in narrow_next.items():
            if line not in wide_next:
                continue
            original = wide_next[line]
            if original == following:
                continue
            # 后继变了：只有当原后继整块被删时才允许
            assert original not in narrow, f"{line!r} 的续行 {original!r} 不见了，句子被砍断"

    def test_三份模板都不留半截(self):
        for name in ("executor_v0", "executor_v1", "executor_v2"):
            narrow = self._narrow(name)
            assert "看不清就先观察" not in narrow, f"{name} 留下了半截"
            assert "中文输入用" not in narrow

    def test_相邻的完整规则没被误删(self):
        narrow = self._narrow()
        assert "一次只做一步" in narrow
        assert "每步执行后你会看到新的截图" in narrow, "规则 1 的续行不该被牵连"
        assert "点击目标要具体" in narrow

    def test_分组只吃缩进更深的续行(self):
        from agent.prompts import _blocks

        lines = ["1. 头一条", "   续行", "2. 第二条", "不缩进的普通段落"]
        blocks = _blocks(lines)
        assert [len(b) for b in blocks] == [2, 1, 1]

    def test_不是列表项的行不吸收后续(self):
        """普通段落后面跟缩进行时，不该被当成列表项的续行吞掉。"""
        from agent.prompts import _blocks

        assert [len(b) for b in _blocks(["普通段落", "   缩进行"])] == [1, 1]
