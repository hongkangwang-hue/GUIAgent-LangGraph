"""M3 三个新模块的纯函数测试。

覆盖 `finetune.dataset` 的坐标归一化与样本构建，以及
`grounding.local_vlm` 的坐标解析。

**训练与推理的坐标空间一致性是这里最要紧的一条。** 不一致时 loss 会
正常下降、评测分数却莫名其妙地低，因为模型学的坐标系和评测换算用的
不是一个。这种错误只能靠单测钉住，跑训练是发现不了的。
"""

from __future__ import annotations

import json
import types

from finetune.dataset import COORD_SPACE, build_record, executable_action, normalize_point, task_of
from grounding.local_vlm import MODEL_SPACE, parse_point
from perception.types import BBox, Point

#: 默认落点。提到模块级是为了避开「函数默认值里做调用」这条 lint，
#: 顺带也让「所有用例共享同一个基准点」这件事显式化。
_CENTER = Point(960, 540)


class _Kind:
    """假的 ActionType 枚举成员，只需要 `.value`。"""

    def __init__(self, value: str) -> None:
        self.value = value


def _sample(
    sample_id="s1",
    instruction="在互联网上查找冯·诺依曼的相关信息",
    resolution=(1920, 1080),
    point=_CENTER,
    bbox=None,
    meta=None,
    action_type="click",
    subtype="click",
    raw_type="MouseAction",
    params=None,
    subtask=None,
):
    """造一个够用的假样本。不 import UnifiedSample 是为了不牵扯整条数据链。

    `subtask` 不传时跟 `instruction` 一致——真实数据里指令**就是**子任务
    （装载器把它写进 `instruction`，同时留一份在 `meta["subtask"]`）。
    """
    base = {"raw_action_subtype": subtype} if subtype else {}
    if raw_type:
        base["raw_action_type"] = raw_type
    base["subtask"] = instruction if subtask is None else subtask
    base.update(meta or {})
    return types.SimpleNamespace(
        sample_id=sample_id,
        instruction=instruction,
        resolution=resolution,
        point=point,
        bbox=bbox,
        params=params or {},
        screenshot_path="/tmp/x.png",
        source_dataset="screenagent",
        action_type=_Kind(action_type),
        meta=base,
    )


class TestCoordSpaceConsistency:
    def test_训练与推理用同一个坐标空间(self):
        """`finetune.dataset.COORD_SPACE` 必须等于 `local_vlm.MODEL_SPACE`。

        这条是整个 M3 最容易犯又最难查的错误的护栏。两边不一致时：
        训练正常、loss 正常下降、adapter 正常保存，只有评测分数低得
        莫名其妙 —— 而没人会想到去比这两个常量。
        """
        assert COORD_SPACE == MODEL_SPACE


class TestNormalizePoint:
    def test_中心点(self):
        assert normalize_point(960, 540, (1920, 1080)) == (500, 500)

    def test_原点(self):
        assert normalize_point(0, 0, (1920, 1080)) == (0, 0)

    def test_右下角不越界(self):
        """归一化空间是左闭右开的 [0,1000)。

        坐标落在图像最右一列时 `x/width*1000` 会等于 1000，越界。
        几千条样本里这种边界值必然出现，夹到 999 而不是让它溢出。
        """
        nx, ny = normalize_point(1919, 1079, (1920, 1080))
        assert nx == 999
        assert ny == 999
        assert nx < COORD_SPACE[0]
        assert ny < COORD_SPACE[1]

    def test_各轴独立(self):
        assert normalize_point(400, 300, (800, 600)) == (500, 500)

    def test_分辨率非法直接抛(self):
        """宁可抛也不要静默产生垃圾样本 —— 那会污染整个训练集。"""
        import pytest

        with pytest.raises(ValueError):
            normalize_point(1, 1, (0, 100))


class TestTaskOf:
    def test_取当前子任务(self):
        """**不是会话级总目标。**

        一个会话十几步共用一句总目标，实测 716 条只对应 169 个不重复指令，
        「Insert a circle」出现 17 次却对应 17 个不同坐标——那是矛盾监督。
        """
        assert task_of(_sample(subtask="点击工具栏的圆形按钮")) == "点击工具栏的圆形按钮"

    def test_空任务返回空(self):
        assert task_of(_sample(subtask="   ")) == ""

    def test_取不到子任务时不退回总目标(self):
        """**退回去等于把矛盾监督又放回训练集**，而条数看起来还是满的。"""
        assert task_of(_sample(instruction="在互联网上查找冯·诺依曼", subtask="")) == ""


class TestExecutableAction:
    def test_原始子类型优先(self):
        """`raw_action_subtype` 比统一 schema 的 `action_type` 更细。

        后者把 `move` 归进了 `other`，只看它会丢掉 74 条 mouse_move 样本。
        """
        sample = _sample(action_type="other", subtype="move")
        assert executable_action(sample) == "mouse_move"

    def test_常见映射(self):
        assert executable_action(_sample(subtype="click")) == "left_click"
        assert executable_action(_sample(subtype="double_click")) == "double_click"

    def test_不可执行的动作返回空(self):
        """拖拽只有起点没有终点，down/up 是半个动作。

        **丢弃而不是硬凑**——拿只有起点的拖拽去训练，
        教出来的是「拖拽 = 点一下」。
        """
        assert executable_action(_sample(action_type="drag", subtype="drag")) == ""
        assert executable_action(_sample(action_type="other", subtype="down")) == ""


class TestBuildRecord:
    def test_基本构建(self):
        record = build_record(_sample())
        assert record is not None
        assert record["point_norm"] == [500, 500]
        assert record["point_px"] == [960, 540]
        # 子任务原样进指令，不再包装成「找到：xxx」
        assert record["instruction"] == "在互联网上查找冯·诺依曼的相关信息"

    def test_输出格式与推理时一致(self):
        """answer 必须能被 `ActionIntent` 的解析路径直接读懂。

        训练时输出一种格式、推理时要求另一种，模型就得学两套 ——
        而这种不一致在 loss 曲线上完全看不出来。
        """
        import json

        record = build_record(_sample())
        payload = json.loads(record["answer"])
        assert payload == {"action": "left_click", "x": 500, "y": 500}

    def test_没有_point_时用_bbox_中心(self):
        sample = _sample(point=None, bbox=BBox(100, 100, 300, 300))
        record = build_record(sample)
        assert record is not None
        assert record["point_px"] == [200, 200]
        assert record["bbox_px"] == [100, 100, 300, 300]

    def test_任务为空的样本丢掉(self):
        """**不填占位符。** 没有指令的样本会教模型在缺信息时也硬输出动作。"""
        assert build_record(_sample(instruction="", subtask="")) is None

    def test_不可执行的动作丢掉(self):
        assert build_record(_sample(action_type="drag", subtype="drag")) is None

    def test_需要坐标的动作没有坐标就丢掉(self):
        assert build_record(_sample(point=None, bbox=None)) is None

    def test_不需要坐标的动作没有坐标照样保留(self):
        """**这是这次放宽的核心。**

        以前这里无条件要求 `point` 不为 None，于是 `type` / `key` / `wait` /
        `done` 即使通过了上游池子筛选，也会在这一行被第二次挡掉——
        两道门，改一道等于没改。
        """
        record = build_record(
            _sample(
                action_type="type",
                subtype="text",
                raw_type="KeyboardAction",
                point=None,
                bbox=None,
                params={"text": "冯诺依曼"},
            )
        )
        assert record is not None
        assert "point_norm" not in record
        assert json.loads(record["answer"]) == {"action": "type", "text": "冯诺依曼"}

    def test_done_是完成信号不是动作(self):
        """`done` 不带 `action` 字段——`llm.parsing` 走的是 `DONE_KEYS` 分支。"""
        record = build_record(
            _sample(
                action_type="other",
                subtype="",
                raw_type="EvaluateSubTaskAction",
                point=None,
                bbox=None,
                params={"situation": "sub_task_success"},
            )
        )
        assert record is not None
        assert json.loads(record["answer"]) == {"done": True}

    def test_训练输出能被真解析器读回来(self):
        """**训练与推理同构的唯一靠谱证明方式：拿真解析器跑一遍。**

        比对字符串只能证明"格式没变"，证明不了"推理时读得懂"。
        `llm.parsing.PARAM_ALIASES` 会把 `key` 归一成 `keys`、`seconds` 归一
        成 `duration`——训练时输出别名的话，模型学的是将来可能被改掉的名字。
        """
        from llm.parsing import parse_action_payload

        cases = [
            (_sample(), "left_click", {"x": 500, "y": 500}),
            (
                _sample(
                    action_type="type",
                    subtype="text",
                    raw_type="KeyboardAction",
                    point=None,
                    params={"text": "你好"},
                ),
                "type",
                {"text": "你好"},
            ),
            (
                _sample(
                    action_type="key",
                    subtype="press",
                    raw_type="KeyboardAction",
                    point=None,
                    params={"key": "Return"},
                ),
                "key",
                {"keys": "Return"},
            ),
            (
                _sample(
                    action_type="wait",
                    subtype="",
                    raw_type="WaitAction",
                    point=None,
                    params={"seconds": 0.5},
                ),
                "wait",
                {"duration": 0.5},
            ),
            (
                _sample(
                    action_type="scroll",
                    subtype="scroll_down",
                    raw_type="MouseAction",
                    point=None,
                    params={"direction": "down", "repeat": 3},
                ),
                "scroll",
                {"direction": "down", "amount": 3},
            ),
        ]
        for sample, want_type, want_params in cases:
            record = build_record(sample)
            assert record is not None, want_type
            back = parse_action_payload(record["answer"])
            assert back["action_type"] == want_type
            assert back["params"] == want_params

    def test_图片路径一律用正斜杠(self):
        r"""**Windows 上生成的训练集要能拿到 Linux 服务器上跑。**

        `data\raw\...` 在 Linux 上不是路径，是一个含反斜杠的文件名——
        图片一张都打不开，而报错指向「文件不存在」，离根因很远。
        正斜杠在 Windows 上同样有效。
        """
        record = build_record(_sample())
        assert "\\" not in record["image"]

    def test_分辨率缺失丢掉(self):
        assert build_record(_sample(resolution=(0, 0))) is None


class TestParsePoint:
    def test_标准_json(self):
        point = parse_point('{"x": 456, "y": 310}')
        assert point == Point(456, 310)

    def test_json_被截断也能救回来(self):
        """微调后的模型格式遵从度是可预期的薄弱环节 —— 训练集只有 569 条。

        一个能解析出坐标的回答，不该因为少个花括号被判成定位失败。
        """
        assert parse_point('{"x":456,"y":310') == Point(456, 310)

    def test_裸的数字对(self):
        assert parse_point("坐标是 (456, 310)") == Point(456, 310)
        assert parse_point("456，310") == Point(456, 310)

    def test_夹在文字里的_json(self):
        assert parse_point('好的，答案是 {"x": 12, "y": 34} 。') == Point(12, 34)

    def test_没有坐标返回_None(self):
        assert parse_point("我找不到这个元素") is None
        assert parse_point("") is None

    def test_负数不被当成坐标丢掉(self):
        """解析阶段不判合法性，越界检查在 `locate()` 里做。

        两件事分开：解析回答「模型说了什么」，检查回答「说的能不能用」。
        混在一起的话，一个越界坐标会被报成「输出里没有坐标」，
        而那两种失败的修法完全不同。
        """
        assert parse_point('{"x": -5, "y": 10}') == Point(-5, 10)


class TestPromptDataContract:
    """**动作空间是提示词与数据之间的契约。**

    2026-08-26 的事故：训练数据从 3 种动作放宽到 8 种，`SYSTEM_PROMPT`
    没跟着改，仍然只列四个鼠标动作并要求「必须输出 x/y」。于是 88 分钟的
    训练是在**提示词与七成目标直接矛盾**的条件下做的——

        提示词说     只有 4 个鼠标动作，必须给 x/y
        训练目标却是  {"done": true}          占 41.6%
                     {"action":"type",...}    占 12.3%

    后果实测：`done` 8 条全错、误报率 0.0%（提示词从没说过它存在）。
    **两边都不报错，loss 曲线完全正常。**

    这是同一根源的第三次（前两次：`data.split` 的坐标过滤、`eval.action`
    的动作集合），所以这条契约必须由测试守着。
    """

    def _prompt(self) -> str:
        from finetune.train_lora import SYSTEM_PROMPT

        return SYSTEM_PROMPT

    def test_提示词列全了数据里会出现的动作(self):
        """凡是 `finetune.dataset` 会生成的动作，提示词里都要有。"""
        from finetune.dataset import ANSWER_FIELDS

        prompt = self._prompt()
        missing = [name for name in ANSWER_FIELDS if name not in prompt]
        assert not missing, f"这些动作会出现在训练目标里，但提示词没提：{missing}"

    def test_提示词说明了_done(self):
        """`done` 占验证集 41.6%，提示词不提它，模型就没有理由输出它。"""
        prompt = self._prompt()
        assert '"done"' in prompt or "done" in prompt

    def test_提示词给出的键名与训练目标一致(self):
        """**键名对不上，模型学的就是提示词里那个用不了的名字。**

        `key` 动作的输出键是 `keys`（复数），而装载器存的是 `key`（单数）
        ——这两套词表混用过一次，导致命中率恒为 0。
        """
        from finetune.dataset import ANSWER_FIELDS

        prompt = self._prompt()
        for action, fields in ANSWER_FIELDS.items():
            for out_key, _ in fields:
                assert f'"{out_key}"' in prompt, f"{action} 的输出键 {out_key} 没在提示词里出现"

    def test_不再无条件要求坐标(self):
        """`type` / `key` / `done` 没有坐标。提示词若说「必须输出 x/y」，
        就是在要求模型做一件对七成样本不成立的事。
        """
        squashed = "".join(self._prompt().split())
        assert "只输出一个JSON对象" in squashed or "只输出一个 JSON 对象" in self._prompt()
        # 旧提示词的那句模板：{"action": "动作名", "x": …, "y": …}
        assert '"动作名"' not in self._prompt(), "还留着「一个模板套所有动作」的旧写法"

    def test_评测器认得所有训练动作(self):
        """评测侧的 `ALLOWED_ACTIONS` 也要跟上，否则合规率会凭空低一截。

        实测：`--limit 20` 报格式合规率 70%，正好等于 20 条里坐标类的 14 条
        ——非坐标输出被全判为不合规，而它们其实完全合规。
        """
        from eval.action import ALLOWED_ACTIONS
        from finetune.dataset import ANSWER_FIELDS

        missing = set(ANSWER_FIELDS) - set(ALLOWED_ACTIONS)
        assert not missing, f"评测器不认这些动作：{missing}"

    def test_真值与预测的参数词表不混用(self):
        """`key` 的真值键是 `key`，模型输出的键是 `keys`。

        混用过一次：5 条 key 样本类型准确率 100%、命中率 0%——真值侧按
        `keys` 去取，取到空，于是永远和预测对不上。
        """
        from eval.action import key_params

        truth = key_params("key", {"key": "Enter"})
        pred = key_params("key", {"keys": "enter"}, predicted=True)
        assert truth and truth == pred, f"真值 {truth} 与预测 {pred} 应当相等"


class TestLabelMasking:
    """**标签掩码错了不报错，loss 曲线照样好看，模型学的却是别的东西。**

    2026-08-26 实测出的 bug：原来按「留最后 len(answer) 个 token」掩码，
    而 chat template 会在答案后追加结束符与换行，于是窗口整体后移——

        期望训练  {"action": "left_click", "x": 475, "y": 251}
        实际训练  ": "left_click", "x": 475, "y": 251}<|im_end|>…
                  ↑ 开头的 {"action 被当成上下文遮掉了

    旧数据没暴露它：答案 23 个 token，丢开头 3 个还剩动作名和坐标。
    **答案越短损失比例越大**——`{"done": true}` 只有 5 个 token，
    丢 3 个就是六成信号，模型从没被训练过输出 `{"done"` 这个开头。
    这解释了放宽后 done 8 条全错、误报率 0.0%。
    """

    def test_定位函数取最后一次出现(self):
        """多轮对话里 assistant 会出现多次，要的是最后那一轮。"""
        from finetune.train_lora import _last_subsequence_end

        assert _last_subsequence_end([1, 2, 3, 1, 2, 9], [1, 2]) == 5
        assert _last_subsequence_end([1, 2, 3], [9]) is None
        assert _last_subsequence_end([1], [1, 2]) is None
        assert _last_subsequence_end([1, 2], []) is None

    def test_掩码按_assistant_起始而不是答案长度(self):
        """钉住实现：**按长度倒推就是那个 bug 本身。**"""
        import inspect

        from finetune.train_lora import train

        src = inspect.getsource(train)
        assert "_last_subsequence_end" in src
        assert "labels.shape[1] - keep" not in src, "又退回按答案长度倒推了"

    def test_找不到起始标记时整条屏蔽(self):
        """**宁可这条样本不产生梯度，也不要用一个偏移的窗口训练。**

        后者不报错，只会安静地学错东西。
        """
        import inspect

        from finetune.train_lora import train

        src = inspect.getsource(train)
        index = src.index("if start is None")
        assert "labels[index, :] = -100" in src[index : index + 400]
