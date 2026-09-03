"""切断规划器与执行器之间的**示例回路**。

## 事故

2026-08-25，`open_browser` 五轮全挂，每轮六次点击**完全相同**：

    子任务「点击任务栏的开始按钮」
       left_click (470, 750) ×6

链条是这样的：

    planner_v1.yaml:29    示例教规划器拆出「点击任务栏的开始按钮」
            ↓ 逐字产出
    executor_v1.yaml:75   执行器 few-shot 的输入正是这句话
            ↓ 逐字吐出
                          (470, 750) → 1920×1080 上的 (902, 810) → 桌面壁纸

两份提示词的示例当初是配套写的，于是串成一条固定回路——不管指令是什么，
3B 把整条链原样回放。

## 同一批数据里的反证

子任务**没撞上示例**时，模型的坐标相当准：

    「点击计算器窗口右上角的关闭按钮」  第一步就点中，2/2 通过
    「在搜索框输入 Microsoft Edge」     (428, 936)，离目标约 42

**墙的很大一部分是提示词砌的，不是模型不行。**

所以这里守两件事：两份新模板的**系统提示与旧版逐字相同**（否则差值不可
归因），以及**示例之间不再撞车**。
"""

from __future__ import annotations

import pytest

from agent.prompts import load_template

CLICK_ONLY = ("left_click", "double_click", "right_click", "mouse_move")


class TestSystemPromptUnchanged:
    """新版只该差示例。系统提示动一个字，差值里就混进了那个字的影响。"""

    def test_executor_v3_的系统提示与_v1_逐字相同(self):
        a = load_template("executor_v1").render_system(width=1000, height=1000)
        b = load_template("executor_v3").render_system(width=1000, height=1000)
        assert a == b

    def test_planner_v2_的系统提示与_v1_逐字相同(self):
        a = load_template("planner_v1").render_system(width=0, height=0)
        b = load_template("planner_v2").render_system(width=0, height=0)
        assert a == b

    def test_旧版没被改动(self):
        """`executor_v1` 是 M3 §8 三档消融的交付数据，`planner_v1` 是 M2/M3

        全部实测用的那份。**改了就不可复现**，所以新建而不是原地改。
        """
        assert '"x": 470' in load_template("executor_v1").few_shot[0].output
        assert any("开始按钮" in e.output for e in load_template("planner_v1").few_shot)


class TestLoopIsBroken:
    """规划器产出的措辞，不能再逐字落进执行器的示例输入里。"""

    def _planner_goals(self, name: str) -> list[str]:
        import json
        import re

        goals: list[str] = []
        for example in load_template(name).few_shot:
            for match in re.finditer(r'"goal":\s*"([^"]+)"', example.output):
                goals.append(match.group(1))
        assert goals, f"{name} 的示例里没找到 goal，测试自己坏了"
        assert json is not None
        return goals

    #: 撞车的判据：这几个词在两边同时出现。
    #:
    #: 逐字比对太脆——事故里规划器写的是「点击任务栏的开始按钮」，执行器
    #: 示例写的是「点击任务栏上的"开始"按钮」，**只差一个「上」和一对引号**，
    #: 而 3B 照样把它们当成同一件事，原样吐出示例的答案。
    COLLISION_WORDS = ("任务栏", "开始")

    def test_旧的一对确实撞车(self):
        """先钉住事故本身——**这条描述的是修之前的状态**。"""
        goals = " ".join(self._planner_goals("planner_v1"))
        inputs = "\n".join(e.input for e in load_template("executor_v1").few_shot)
        assert all(w in goals for w in self.COLLISION_WORDS)
        assert all(w in inputs for w in self.COLLISION_WORDS)

    def test_新的一对不撞车(self):
        goals = " ".join(self._planner_goals("planner_v2"))
        inputs = "\n".join(e.input for e in load_template("executor_v3").few_shot)
        overlap = [w for w in self.COLLISION_WORDS if w in goals and w in inputs]
        assert not overlap, f"这些词仍然两边都有：{overlap}"

    def test_那个害人的常数不在新示例里(self):
        outputs = "\n".join(e.output for e in load_template("executor_v3").few_shot)
        assert "470" not in outputs
        assert "750" not in outputs

    def test_新执行器示例避开了规划器的高频词(self):
        """「开始按钮」「任务栏」「搜索框」是 planner 示例里的词。"""
        text = "\n".join(e.input + e.output for e in load_template("executor_v3").few_shot)
        for word in ("开始按钮", "任务栏", "搜索框"):
            assert word not in text


class TestExamplesAreDiverse:
    def test_坐标彼此差得远(self):
        """三个例子给同一个坐标的话，模型学到的是「永远输出那个坐标」。"""
        import math
        import re

        points = []
        for example in load_template("executor_v3").few_shot:
            found = re.search(r'"x":\s*(\d+),\s*"y":\s*(\d+)', example.output)
            if found:
                points.append((int(found.group(1)), int(found.group(2))))
        assert len(points) >= 2
        for i, a in enumerate(points):
            for b in points[i + 1 :]:
                assert math.dist(a, b) > 200, f"{a} 与 {b} 太近"

    def test_动作类型不止一种(self):
        """只给 `left_click` 的话，模型学到的是「无论问什么都点击」。"""
        import re

        kinds = set()
        for example in load_template("executor_v3").few_shot:
            found = re.search(r'"action":\s*"(\w+)"', example.output)
            if found:
                kinds.add(found.group(1))
        assert len(kinds) >= 2

    def test_规划器第一条示例演示直接路径(self):
        """**顺序有意义。** 第一条示例的影响最大，它该演示一步到位那条路。"""
        first = load_template("planner_v2").few_shot[0]
        assert first.output.count('"id"') == 1, "第一条示例应当只有一个子任务"


class TestFewShotFiltering:
    """示例里的动作也要按 `allowed_actions` 过滤。

    不过滤的话，同一份提示词一边说「你只有这四个动作」，一边给出一条
    `type` 的示例。而实测表明 **3B 更听示例的**——`allowed_actions` 的硬
    约束段明说了「不要拆成点开始→输入→回车」，规划器照样这么拆。
    """

    def test_不可用动作的示例被丢掉(self):
        template = load_template("executor_v3")
        assert len(template.few_shot_pairs(CLICK_ONLY)) < len(template.few_shot_pairs())
        text = "\n".join(p["output"] for p in template.few_shot_pairs(CLICK_ONLY))
        assert '"action": "type"' not in text

    def test_可用动作的示例留着(self):
        text = "\n".join(
            p["output"] for p in load_template("executor_v3").few_shot_pairs(CLICK_ONLY)
        )
        assert '"action": "left_click"' in text
        assert '"action": "double_click"' in text

    def test_done_示例不受影响(self):
        """`done` 不是动作，是完成信号——不该被动作过滤牵连。"""
        text = "\n".join(
            p["output"] for p in load_template("executor_v3").few_shot_pairs(CLICK_ONLY)
        )
        assert '"done": true' in text

    def test_不给约束时一条不少(self):
        template = load_template("executor_v3")
        assert len(template.few_shot_pairs()) == len(template.few_shot)
        assert template.few_shot_pairs(None) == template.few_shot_pairs()

    def test_session_把约束传给示例(self):
        import inspect

        from agent.session import Session

        src = inspect.getsource(Session._prepare_backend)
        assert "few_shot_pairs(" in src and "allowed_actions" in src

    def test_planner_把约束传给示例(self):
        import inspect

        from agent.planner import Planner

        src = inspect.getsource(Planner)
        assert "few_shot_pairs(self.allowed_actions" in src


class TestConfigurable:
    def test_两份新模板都能被加载(self):
        from agent.prompts import list_templates

        available = list_templates()
        assert "executor_v3" in available
        assert "planner_v2" in available

    def test_SessionConfig_能指向新模板(self):
        from agent.session import SessionConfig

        config = SessionConfig(executor_template="executor_v3", planner_template="planner_v2")
        assert config.as_dict()["executor_template"] == "executor_v3"
        assert config.as_dict()["planner_template"] == "planner_v2"

    def test_默认仍是旧模板(self):
        """**默认不变。** 新模板要显式选，否则 M2/M3 的复现命令会悄悄换了行为。"""
        from agent.session import SessionConfig

        config = SessionConfig()
        assert config.executor_template == "executor_v1"
        assert config.planner_template == "planner_v1"


@pytest.mark.parametrize("name", ["executor_v3", "planner_v2"])
def test_模板能渲染且不留占位符(name):
    template = load_template(name)
    text = template.render_system(width=1000, height=1000)
    assert "{action_reference}" not in text
    assert "{width}" not in text and "{height}" not in text


class TestExecutorV4:
    """v4 = v3 + 一条 GUI 常识：**桌面图标要双击**。

    v3 把 `open_browser` 从 0/5 提到 2/5 之后，探针显示剩下的失败形态是：

        子任务「点击任务栏上的Microsoft Edge图标」
           left_click (15, 190) ×3   ← 桌面上的 Edge 图标，不是任务栏的

    模型**认对了目标类别**，但用了单击——而单击桌面图标只是选中，不会启动。
    那 2 次成功很可能是连点三下恰好构成了双击。

    v3 保留不动：它的系统提示与 v1 逐字相同，那次 0/5→2/5 的对照靠这一点
    才可归因。
    """

    def _system(self, name):
        return load_template(name).render_system(width=1000, height=1000)

    def test_v4_只比_v3_多几行(self):
        """**只该多，不该改。** 改了措辞，2/5→? 的差值里就混进了那处改动。"""
        v3 = set(self._system("executor_v3").splitlines())
        extra = [line for line in self._system("executor_v4").splitlines() if line not in v3]
        assert 1 <= len(extra) <= 6, f"多出来 {len(extra)} 行，超出「加一条规则」的范围"

    def test_v4_的示例与_v3_逐字相同(self):
        assert (
            load_template("executor_v4").few_shot_pairs()
            == load_template("executor_v3").few_shot_pairs()
        )

    def test_规则说明了桌面与任务栏的区别(self):
        """**光说「要双击」不够。** 任务栏图标单击就启动，一刀切会引入新错误。"""
        system = self._system("executor_v4")
        assert "双击" in system
        assert "任务栏" in system and "单击就启动" in system

    def test_这条规则用的动作在_CLICK_ONLY_里(self):
        """规则点名了 `double_click`。**它若不在允许集合里，整条会被删掉**——

        那样 v4 就退化成 v3 了，而表格上看不出来。
        """
        narrow = load_template("executor_v4").render_system(
            width=1000, height=1000, allowed_actions=CLICK_ONLY
        )
        assert "桌面上的图标要双击" in narrow

    def test_规则点名了要用哪个动作(self):
        """**「要双击」是自然语言，`double_click` 才是它要输出的东西。**

        规则里只说「双击」而不点名动作，模型得自己完成这层映射——而它
        在别处已经证明过更听示例、更听字面。变异测试里把
        ``` `double_click` 才会 ``` 改成「单击才会」，规则当场自相矛盾，
        而只查「双击」这个词的测试抓不到。
        """
        system = self._system("executor_v4")
        assert "`double_click`" in system
        rule = next(line for line in system.splitlines() if "桌面上的图标要双击" in line)
        following = system.split(rule, 1)[1][:120]
        assert "`double_click`" in rule + following, "规则正文里没点名 double_click"

    def test_规则没有自相矛盾(self):
        """「单击只是选中」和「单击才会打开」不能同时出现。"""
        system = self._system("executor_v4")
        assert "单击桌面图标只是选中" in system
        squashed = "".join(system.split())
        assert "单击才会打开" not in squashed, "既说单击只是选中，又说单击才会打开"

    def test_v3_没被改动(self):
        assert "桌面上的图标要双击" not in self._system("executor_v3")
