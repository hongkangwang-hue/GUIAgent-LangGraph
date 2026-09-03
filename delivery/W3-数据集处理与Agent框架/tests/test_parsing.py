"""容错解析层的单元测试。

这里的每个用例都是**开源模型真会输出的形状**，不是假想的边界条件。

判断一条抖动该不该兼容，标准只有一条：**意图能不能无歧义地还原**。
能，就兼容——否则等于用解析层的脆弱去惩罚模型，把它的能力测低，而
M3 的横评正是要测这三个模型的真实能力边界。不能，就老实失败——猜一个
坐标出来会把可诊断的失败变成一次乱点，后果比失败严重得多。
"""

from __future__ import annotations

import pytest

from llm.parsing import (
    OutputParseError,
    coerce_bool,
    extract_json,
    extract_point,
    normalize_action_type,
    parse_action_payload,
)

# ===================================================================== #
# 抠 JSON
# ===================================================================== #


def test_plain_json() -> None:
    assert extract_json('{"action": "left_click"}') == {"action": "left_click"}


def test_json_fence() -> None:
    text = '```json\n{"action": "left_click", "x": 1, "y": 2}\n```'
    assert extract_json(text)["x"] == 1


def test_bare_fence_without_language() -> None:
    assert extract_json('```\n{"action": "wait"}\n```')["action"] == "wait"


def test_prose_around_json() -> None:
    """模型爱在 JSON 前后说两句。这是最常见的一种抖动。"""
    text = '好的，我先点击开始菜单。\n{"action":"left_click","x":10,"y":20}\n这样就能打开了。'
    assert extract_json(text)["y"] == 20


def test_braces_inside_string_do_not_confuse_the_scanner() -> None:
    """字符串字面量里的花括号不参与配平。"""
    text = 'blah {"action":"type","text":"{}"} blah'
    assert extract_json(text)["text"] == "{}"


def test_escaped_quote_inside_string() -> None:
    text = r'{"action":"type","text":"他说\"你好\""}'
    assert "你好" in extract_json(text)["text"]


def test_fullwidth_punctuation() -> None:
    """中文语境下输入法把标点带成全角，是中文模型最常见的脏输出。

    百分之百可以无歧义还原，没有理由判它失败。
    """
    assert extract_json('{"action"："type"，"text"："你好"}')["text"] == "你好"


def test_single_quotes() -> None:
    """模型见过的训练数据里 Python 风格和 JSON 风格都有，混着输出很常见。"""
    assert extract_json("{'action': 'scroll', 'direction': 'down'}")["direction"] == "down"


def test_python_literals() -> None:
    assert extract_json("{'done': True, 'x': None}")["done"] is True


def test_trailing_comma() -> None:
    assert extract_json('{"action":"wait","duration":1.5,}')["duration"] == 1.5


def test_action_list_takes_the_first() -> None:
    """模型给了一串动作。只取第一个——Loop 是一步一动的，多给的部分
    下一轮会重新决策，那时界面已经变了，照着旧计划走必错。"""
    text = '[{"action":"left_click","x":1,"y":2},{"action":"type","text":"a"}]'
    assert extract_json(text)["x"] == 1


def test_empty_output_fails() -> None:
    with pytest.raises(OutputParseError):
        extract_json("   ")


def test_pure_prose_fails() -> None:
    with pytest.raises(OutputParseError):
        extract_json("我觉得应该点一下那个按钮。")


def test_parse_error_keeps_the_raw_text() -> None:
    """原文必须留着——解析失败时它是唯一的线索。"""
    with pytest.raises(OutputParseError) as info:
        extract_json("完全不是 JSON")
    assert info.value.raw == "完全不是 JSON"


# ===================================================================== #
# 动作名归一
# ===================================================================== #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("click", "left_click"),
        ("CLICK", "left_click"),
        ("left-click", "left_click"),
        ("tap", "left_click"),
        ("Double_Click", "double_click"),
        ("type_text", "type"),
        ("hotkey", "key"),
        ("hover", "mouse_move"),
        ("sleep", "wait"),
    ],
)
def test_action_aliases(raw, expected) -> None:
    assert normalize_action_type(raw) == expected


def test_unknown_action_is_passed_through() -> None:
    """认不出来就原样返回，让 Action 去报"未知动作类型"。

    在这里静默改成某个已知动作，等于替模型做了它没做的决定。
    """
    assert normalize_action_type("teleport") == "teleport"


def test_nested_action_object() -> None:
    assert normalize_action_type({"name": "click"}) == "left_click"


def test_ambiguous_alias_not_guessed() -> None:
    """``select`` 既可能是点击也可能是拖选——宁可失败也不猜。"""
    assert normalize_action_type("select") == "select"


# ===================================================================== #
# 布尔
# ===================================================================== #


@pytest.mark.parametrize("raw", [True, "true", "True", "yes", "1", 1, "完成", "是"])
def test_truthy(raw) -> None:
    assert coerce_bool(raw) is True


@pytest.mark.parametrize("raw", [False, "false", "False", "no", "0", 0, "", None, "未完成"])
def test_falsy(raw) -> None:
    """``"false"`` 是个非空字符串，直接 bool() 会得到 True。这个坑必须堵死。"""
    assert coerce_bool(raw) is False


# ===================================================================== #
# 坐标
# ===================================================================== #


def test_point_from_xy() -> None:
    assert extract_point({"x": 100, "y": 200}) == (100, 200)


@pytest.mark.parametrize("key", ["point", "coordinate", "coordinates", "position", "location"])
def test_point_from_list_under_various_keys(key) -> None:
    assert extract_point({key: [512, 384]}) == (512, 384)


def test_point_from_nested_object() -> None:
    assert extract_point({"point": {"x": 5, "y": 6}}) == (5, 6)


def test_point_from_string() -> None:
    assert extract_point({"position": "(300, 400)"}) == (300, 400)


def test_bbox_collapses_to_center() -> None:
    """模型给了框而不是点。取中心是唯一合理的还原。"""
    assert extract_point({"coordinate": [100, 200, 150, 240]}) == (125, 220)


def test_bbox_string_collapses_to_center() -> None:
    assert extract_point({"point": "[100, 200, 150, 240]"}) == (125, 220)


def test_float_coordinates_become_int() -> None:
    """像素是离散的，Point 强制 int。"""
    assert extract_point({"x": 100.7, "y": 200.2}) == (100, 200)


def test_no_point() -> None:
    assert extract_point({"action": "type", "text": "a"}) is None


def test_non_numeric_point() -> None:
    assert extract_point({"x": "左边", "y": "上面"}) is None


def test_point_packed_into_x_with_no_y() -> None:
    """x 里装着整个坐标对，y 缺席。

    这不是假想：qwen3-vl-8b-instruct 实测三次里两次是这个形状——
        {"action": "left_click", "x": [314, 48], "thinking": "..."}
    写法古怪但无歧义。不捞的话这个模型 2/3 的输出会被判成"没给坐标"，
    能力被严重低估，而 M3 横评正是要测它的真实能力边界。
    """
    assert extract_point({"x": [314, 48]}) == (314, 48)


def test_packed_point_as_string() -> None:
    assert extract_point({"x": "[319, 48]"}) == (319, 48)


def test_packed_bbox_in_x_collapses_to_center() -> None:
    assert extract_point({"x": [100, 200, 150, 240]}) == (125, 220)


def test_packed_x_with_conflicting_y_is_not_guessed() -> None:
    """x=[314,48] 与 y=50 互相矛盾，挑一个不如判失败。"""
    assert extract_point({"x": [314, 48], "y": 50}) is None


def test_scalar_x_without_y_is_still_dropped() -> None:
    """单个标量不构成点，这条不能被上面的放宽波及。"""
    assert extract_point({"x": 100}) is None


def test_three_numbers_is_ambiguous() -> None:
    """三个数既不是点也不是框，不猜。"""
    assert extract_point({"point": [1, 2, 3]}) is None


# ===================================================================== #
# 完整载荷
# ===================================================================== #


def test_full_payload() -> None:
    payload = parse_action_payload('{"action":"left_click","x":100,"y":200,"thinking":"点地址栏"}')
    assert payload["action_type"] == "left_click"
    assert payload["params"] == {"x": 100, "y": 200}
    assert payload["thinking"] == "点地址栏"
    assert payload["done"] is False


@pytest.mark.parametrize("key", ["thinking", "thought", "reasoning", "analysis", "observation"])
def test_thinking_aliases(key) -> None:
    payload = parse_action_payload(f'{{"action":"wait","duration":1,"{key}":"在想"}}')
    assert payload["thinking"] == "在想"


def test_nested_parameters() -> None:
    payload = parse_action_payload('{"action":"key","arguments":{"keys":"ctrl+s"}}')
    assert payload["params"]["keys"] == "ctrl+s"


@pytest.mark.parametrize(
    ("alias", "canonical", "value"),
    [
        ("content", "text", "你好"),
        ("value", "text", "abc"),
        ("key", "keys", "ctrl+c"),
        ("seconds", "duration", 2),
        ("scroll_amount", "amount", 3),
    ],
)
def test_param_aliases(alias, canonical, value) -> None:
    payload = parse_action_payload(f'{{"action":"type","{alias}":{value!r}}}'.replace("'", '"'))
    assert payload["params"][canonical] is not None


def test_mode_b_target_description() -> None:
    """模式 B 的核心字段。没有坐标是正常的，不该被当成错误。"""
    payload = parse_action_payload(
        '{"action":"left_click","target_description":"浏览器地址栏","thinking":"要输网址"}'
    )
    assert payload["target_description"] == "浏览器地址栏"
    assert "x" not in payload["params"]


def test_done_without_action() -> None:
    """模型报告完成时可以没有动作。"""
    payload = parse_action_payload('{"done":"true","thinking":"任务已完成"}')
    assert payload["done"] is True
    assert payload["action_type"] == ""


def test_half_a_coordinate_is_dropped() -> None:
    """只给了 x 没给 y。

    留一个孤零零的 x 会让 needs_grounding 判成"不缺坐标"，然后一路走到
    Action.validate 才炸，错误信息离根因太远。这里直接清掉，让它老实
    走 grounding。
    """
    payload = parse_action_payload('{"action":"left_click","x":100}')
    assert "x" not in payload["params"]
    assert "y" not in payload["params"]


def test_no_action_and_no_done_fails() -> None:
    with pytest.raises(OutputParseError) as info:
        parse_action_payload('{"thinking":"我在想"}')
    assert "thinking" in str(info.value)


def test_extra_fields_are_dropped() -> None:
    """模型多给的字段不进 params，否则 Action 构造会带上一堆垃圾。"""
    payload = parse_action_payload(
        '{"action":"left_click","x":1,"y":2,"confidence":0.9,"element_id":"btn-3"}'
    )
    assert set(payload["params"]) == {"x", "y"}


def test_direction_is_normalized() -> None:
    payload = parse_action_payload('{"action":"scroll","direction":" DOWN ","amount":3}')
    assert payload["params"]["direction"] == "down"


def test_numeric_strings_are_coerced() -> None:
    """模型把数字写成字符串很常见。"""
    payload = parse_action_payload('{"action":"wait","duration":"1.5"}')
    assert payload["params"]["duration"] == pytest.approx(1.5)


def test_payload_feeds_action_intent() -> None:
    """解析出来的载荷要能直接喂给 ActionIntent 并构造出合法 Action。

    这条把解析层和执行链路接上了——两边字段名对不齐的话，单测各自都
    过，一跑真任务就崩。
    """
    from llm.base import ActionIntent

    payload = parse_action_payload(
        '```json\n{"action":"click","point":[512,384],"thought":"点它"}\n```'
    )
    intent = ActionIntent(**payload)
    action = intent.to_action()
    assert action.x == 512 and action.y == 384
    assert intent.needs_grounding is False


# ===================================================================== #
# 边界框形态的定位输出（M3 本地 grounding 模型）
# ===================================================================== #


def test_bbox_keyed_output_folds_to_center() -> None:
    """视觉模型做定位时给框比给点更常见。

    Qwen2.5-VL 的 grounding 示例输出就是 `{"bbox_2d": [...], "label": ...}`。
    不认这个键的后果很隐蔽——模式 B 接上本地模型后每次定位都解析成"没有
    坐标"，看着像模型不会定位，其实是它答对了没人听懂。
    """
    for key in ("bbox_2d", "bbox", "box_2d", "box"):
        assert extract_point({key: [100, 200, 150, 240]}) == (125, 220), key


def test_bbox_output_carries_label_without_confusing_the_parser() -> None:
    assert extract_point({"bbox_2d": [10, 20, 30, 40], "label": "关闭按钮"}) == (20, 30)


def test_explicit_xy_still_wins_over_bbox() -> None:
    """两者都给时以 x/y 为准——那是模型明确指的点，框只是它的依据。"""
    assert extract_point({"x": 7, "y": 8, "bbox_2d": [100, 200, 150, 240]}) == (7, 8)
