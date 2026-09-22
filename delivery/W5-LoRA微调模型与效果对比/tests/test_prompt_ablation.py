"""提示词消融的**阶梯性质**测试。

## 这里守的是什么

消融要能解释，前提是三档之间**只差一个变量**：

    P1 executor_v0  =  基线
    P2 executor_v1  =  P1 + few-shot          ← 系统提示必须与 P1 逐字相同
    P3 executor_v2  =  P2 + 思维链            ← few-shot 必须与 P2 逐字相同

哪一档被顺手改了措辞，P1→P2 的差就不再是「few-shot 的净效果」，
而是「few-shot + 某人改的那句话」——**而这种污染在结果表上完全看不出来**，
三列数字照样整整齐齐。

所以这些用例全是「必须逐字相同」的形式。
"""

from __future__ import annotations

import pytest

from agent.prompts import load_template

#: 与 `scripts/ablate_prompts.py` 的 LADDER 对应。
V0, V1, V2 = "executor_v0", "executor_v1", "executor_v2"

#: 渲染系统提示用的坐标系尺寸，与 `SessionConfig.coordinate_space` 一致。
SPACE = (1000, 1000)


def _system(name: str) -> str:
    return load_template(name).render_system(width=SPACE[0], height=SPACE[1])


class TestLadder:
    def test_三档都存在(self):
        from agent.prompts import list_templates

        available = list_templates()
        for name in (V0, V1, V2):
            assert name in available, f"{name} 不存在，消融跑不起来"

    def test_P1_与_P2_的系统提示逐字相同(self):
        """**它们之间只该差 few-shot。**

        系统提示但凡差一个字，P1→P2 的差值里就混进了那个字的影响，
        而报告会把整个差值都归给 few-shot。
        """
        assert _system(V0) == _system(V1)

    def test_P1_没有_few_shot_而_P2_有(self):
        assert load_template(V0).few_shot == []
        assert len(load_template(V1).few_shot) == 3

    def test_P3_的_few_shot_与_P2_逐字相同(self):
        """P2→P3 只该差思维链那一段。"""
        assert load_template(V2).few_shot_pairs() == load_template(V1).few_shot_pairs()

    def test_P3_的系统提示是_P2_加一段而不是改写(self):
        """把 CoT 那段抠掉之后，必须与 P2 一模一样。

        这条比「P3 更长」强得多：更长可能是因为顺手改了别的地方。
        """
        import re

        stripped = re.sub(r"## 先想再答.*?(?=## 输出格式)", "", _system(V2), flags=re.S)
        assert stripped == _system(V1)

    def test_只有_P3_声明了思维链特性(self):
        """`features` 是给报告用的标签，不能和实际内容脱节。"""
        assert "chain_of_thought" not in load_template(V1).features
        assert "chain_of_thought" in load_template(V2).features


class TestParrotDetection:
    """逐字照抄的判定。

    3B 会把 few-shot 示例的答案原样吐出来，坐标是示例里的常数。这种输出
    **格式 100% 合规、动作类型也可能对**——`format_ok` 与 `action_ok`
    一个都拦不住。不单独量，few-shot 那一档看起来会只有好处。
    """

    FEW = [{"input": "x", "output": '{"action": "left_click", "x": 470, "y": 750}'}]

    def test_逐字相同判为照抄(self):
        from eval.action import parroted

        assert parroted('{"action": "left_click", "x": 470, "y": 750}', self.FEW)

    def test_前后空白不影响(self):
        from eval.action import parroted

        assert parroted('  {"action": "left_click", "x": 470, "y": 750}\n', self.FEW)

    def test_同格式不同坐标不算照抄(self):
        """**这条是关键。** 学会格式正是 few-shot 该起的作用，

        把它判成照抄会让指标反过来惩罚 few-shot 的正当效果。
        """
        from eval.action import parroted

        assert not parroted('{"action": "left_click", "x": 15, "y": 978}', self.FEW)

    def test_没有_few_shot_时恒为假(self):
        """零样本档不存在「照抄」这回事，那一格该是 0 而不是 N/A。"""
        from eval.action import parroted

        assert not parroted('{"action": "left_click", "x": 470, "y": 750}', [])

    def test_照抄的输出同时是格式合规的(self):
        """把「照抄」与「格式合规」的关系钉死：

        照抄的输出**一定**合规，所以 `format_acc` 永远抓不到它。
        这正是要单独报 `parrot_rate` 的理由。
        """
        from eval.action import format_compliant, parroted

        answer = self.FEW[0]["output"]
        assert format_compliant(answer)
        assert parroted(answer, self.FEW)


class TestAblationRunner:
    def test_阶梯定义与模板对得上(self):
        from scripts.ablate_prompts import LADDER

        assert [t for _, t in LADDER] == [V0, V1, V2]

    def test_prefix_区分两轮消融(self):
        """基座与微调各跑一轮，结果文件不能互相覆盖。"""
        from scripts.ablate_prompts import tag_of

        assert tag_of("", V1) == V1
        assert tag_of("lora", V1) == f"lora-{V1}"

    def test_缺档时报告不崩也不用零填(self, capsys, monkeypatch, tmp_path):
        """**缺数据要说缺**，用 0 填会被读成「这一档全错」。"""
        from scripts import ablate_prompts

        monkeypatch.setattr(ablate_prompts, "RESULT_DIR", tmp_path, raising=False)
        monkeypatch.setattr("eval.action.RESULT_DIR", tmp_path)
        ablate_prompts.report("不存在的前缀")
        out = capsys.readouterr().out
        assert "缺这几档" in out
        assert "—" in out

    def test_main_读到的每个属性都存在(self):
        """与 serve_local_model / compare_runs 同一条护栏。"""
        import inspect
        import re

        from scripts import ablate_prompts

        used = set(
            re.findall(r"\bargs\.([a-zA-Z_][a-zA-Z0-9_]*)", inspect.getsource(ablate_prompts))
        )
        parsed = vars(ablate_prompts.build_parser().parse_args([]))
        assert not used - parsed.keys()

    def test_不给模型时明确报错(self):
        """跑三档要十几分钟，参数缺了要立刻说，不能跑到一半才发现。"""
        from scripts import ablate_prompts

        args = ablate_prompts.build_parser().parse_args([])
        assert not args.local and not args.provider
        with pytest.raises(SystemExit):
            import sys

            saved = sys.argv
            try:
                sys.argv = ["ablate_prompts.py"]
                ablate_prompts.main()
            finally:
                sys.argv = saved


class TestLimitSampling:
    """`--limit` 取的必须是**样本**，不是一小撮 session。

    2026-08-25 的实测教训：`val.jsonl` 按 session_id 排序，`--limit 20`
    切前缀只覆盖到 6 个 session，把 few-shot 的效果放大了约 15 倍
    （25% vs 真值 4.9%），还配上了 p=0.047 的显著性。
    """

    ROWS = [{"sample_id": f"{s}-{i}", "session_id": s} for s in "abcde" for i in range(10)]

    def test_前_N_条覆盖_N_个不同_session(self):
        from eval.action import spread_by_session

        picked = spread_by_session(self.ROWS)[:5]
        assert len({r["session_id"] for r in picked}) == 5

    def test_切前缀会漏掉大部分_session(self):
        """把缺陷本身钉住：这条描述的是**修之前**的行为。"""
        assert len({r["session_id"] for r in self.ROWS[:5]}) == 1

    def test_不丢样本也不重复(self):
        from eval.action import spread_by_session

        out = spread_by_session(self.ROWS)
        assert sorted(r["sample_id"] for r in out) == sorted(r["sample_id"] for r in self.ROWS)

    def test_确定性(self):
        """同一份 val 跑两次要选中同一批，否则探路结果不可比。"""
        from eval.action import spread_by_session

        assert spread_by_session(self.ROWS) == spread_by_session(self.ROWS)

    def test_session_长度不均也不崩(self):
        from eval.action import spread_by_session

        rows = [{"sample_id": "a-0", "session_id": "a"}] + [
            {"sample_id": f"b-{i}", "session_id": "b"} for i in range(4)
        ]
        assert len(spread_by_session(rows)) == 5


class TestLimitKeepsActionMix:
    """`--limit` 抽出来的动作分布必须与全集一致。

    ## 这是同一个坑的第二次

    2026-08-25 修好了「切前缀只覆盖 6 个 session」。但那个实现是
    `buckets[key].pop(0)`——第一轮取每个 session 的**第 0 条**，
    而验证集有 43 个 session，`--limit 20` 就只取得到第 0 条。

    **动作类型和位置强相关**：`done` 是一段动作序列的最后一个，
    永远不在第 0 位。2026-08-26 实测 `--limit 20` 抽出的 20 条里
    `done` 是 **0 条**，而全集里它占 **41.6%**——随机抽中 0 条的概率
    约十万分之三。探路样本完全没测到占四成的那类动作。

    教训：**「按 X 铺开」只保证 X 这一维无偏。** 换一个与位置相关的维度，
    偏差就原样回来了。
    """

    #: 造一个与真实验证集同形态的集合：done 占四成且**总在每段末尾**。
    ROWS = [
        {"sample_id": f"{s}-{i}", "session_id": s, "action": a}
        for s in "abcdefghij"
        for i, a in enumerate(["left_click", "type", "done", "done"])
    ]

    def test_前_N_条里_done_的占比接近全集(self):
        import collections

        from eval.action import spread_by_session

        full = collections.Counter(r["action"] for r in self.ROWS)
        picked = spread_by_session(self.ROWS)[:20]
        got = collections.Counter(r["action"] for r in picked)
        for name, n in full.items():
            want = n / len(self.ROWS) * 20
            assert abs(got.get(name, 0) - want) <= 1, (
                f"{name}: 抽到 {got.get(name, 0)}，期望 {want}"
            )

    def test_切前缀与旧实现都漏掉_done(self):
        """把缺陷钉住：**这两条描述的是修之前的行为。**"""
        assert all(r["action"] != "done" for r in self.ROWS[:2])

        # 旧实现等价于「每个 session 取第 0 条」
        first_of_each = [r for r in self.ROWS if r["sample_id"].endswith("-0")]
        assert all(r["action"] != "done" for r in first_of_each[:20])

    def test_小类不被过采样(self):
        """**过采样也是偏差，只是方向反了。**

        中途试过「每轮各层至少取 1 条」，结果占 2.7% 的 `wait` 在前 20 条里
        拿到 2 条，过采样 4 倍。这个函数存在的全部理由就是不让探路样本
        给出偏差的数——哪个方向的偏差都不行。
        """
        import collections

        from eval.action import spread_by_session

        rows = [
            {"sample_id": f"{i}", "session_id": f"s{i % 5}", "action": "left_click"}
            for i in range(97)
        ] + [{"sample_id": "rare", "session_id": "s0", "action": "scroll"}]
        got = collections.Counter(r["action"] for r in spread_by_session(rows)[:20])
        assert got.get("scroll", 0) <= 1, "占 1% 的动作不该在前 20 条里出现多次"

    def test_没有_action_字段时退化为纯_session_轮流(self):
        """向后兼容：旧的调用方（与旧测试）不带 action 字段。"""
        from eval.action import spread_by_session

        rows = [{"sample_id": f"{s}-{i}", "session_id": s} for s in "abcde" for i in range(3)]
        assert len({r["session_id"] for r in spread_by_session(rows)[:5]}) == 5

    def test_没有_session_id_字段时退化为原序(self):
        from eval.action import spread_by_session

        rows = [{"sample_id": str(i)} for i in range(4)]
        assert spread_by_session(rows) == rows

    def test_evaluate_用的是轮流取而不是切片(self):
        """护栏：有人把 `spread_by_session(...)` 删回 `todo[:limit]` 就会红。"""
        import inspect

        import eval.action

        src = inspect.getsource(eval.action.evaluate)
        assert "spread_by_session(todo)[:limit]" in src
        assert "todo = todo[:limit]" not in src
