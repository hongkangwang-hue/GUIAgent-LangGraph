"""把模型的自由文本抠成结构化动作 —— 容错解析层。

## 为什么这值得单独一个模块

M2 任务拆解第 4 条写着"输出解析要容错：开源模型的结构化输出稳定性低于
前沿模型，解析失败时应重试一次并记录，而非直接崩溃"。

重试一次治不了根本问题。真实情况是**模型几乎每次都"差不多对"**：JSON
是对的但裹在 ```json 围栏里；字段名对但用了 ``thought`` 而不是
``thinking``；坐标对但写成了 ``[x, y]`` 而不是两个字段；甚至整体都对，
只是中文输入法把引号打成了全角的 ``"``。

这些都不是"模型不会做"，是格式抖动。**每一种抖动都直接判失败，会把模型
的能力测低**——而 M3 的横评正是要测这几个模型的真实能力边界，解析层太脆
会让结论失真：你以为在比较模型的规划能力，实际在比较它们对 JSON 围栏的
偏好。

所以这里的原则是：**只要意图能无歧义地还原出来，就还原。** 有歧义的才失败。

## 不做的事

不"猜"动作。模型说不清点哪就是说不清，这里不会替它编一个坐标——那会
把一个可诊断的失败变成一次乱点，后果比失败严重得多。
"""

from __future__ import annotations

import ast
import json
import logging
import re

logger = logging.getLogger(__name__)


class OutputParseError(ValueError):
    """模型输出无法还原成动作意图。"""

    def __init__(self, message: str, raw: str = "") -> None:
        super().__init__(message)
        self.raw = raw


# ---------------------------------------------------------------------- #
# 字段别名
# ---------------------------------------------------------------------- #

#: 动作类型的字段名。不同模型的习惯不一样，在这里一次性吸收掉，
#: 好过在每个后端里各写一遍
ACTION_KEYS = ("action", "action_type", "type", "name", "tool", "function")

#: 思考过程。``thought`` 与 ``reasoning`` 出现频率和 ``thinking`` 相当
THINKING_KEYS = ("thinking", "thought", "reasoning", "reason", "analysis", "observation")

#: 元素描述（模式 B 的核心字段）
TARGET_KEYS = ("target_description", "target", "element", "description", "element_description")

#: 完成标志
DONE_KEYS = ("done", "finished", "complete", "completed", "is_done", "task_complete")

#: 参数容器。有些模型把参数套一层再给
PARAM_KEYS = ("params", "parameters", "arguments", "args", "action_input", "input")

#: 坐标对。``point`` / ``coordinate`` 是 Qwen-VL 系列的常见写法。
#:
#: **末尾几个 bbox 键名是给 M3 的本地 grounding 模型留的。** 视觉模型做定位
#: 时给框比给点更常见，Qwen2.5-VL 的 grounding 示例输出就是
#: ``{"bbox_2d": [x1, y1, x2, y2], "label": "…"}`` 这种形状。四个数会被
#: `extract_point` 折成中心点——点击目标元素的中心正是我们要的行为。
#:
#: 不收这几个键的后果很隐蔽：模式 B 接上本地模型后，**每一次定位都会解析成
#: "没有坐标"**，表现为 grounding_failed，看上去像"本地模型不会定位"，
#: 而实际上它答对了、只是没人听懂。M3 那周排查这个要花掉半天。
#:
#: M3 接线时仍应拿真实输出核对一遍键名，这里只是把已知的几种先兜住。
POINT_KEYS = (
    "point",
    "coordinate",
    "coordinates",
    "position",
    "pos",
    "xy",
    "location",
    "bbox_2d",
    "bbox",
    "box_2d",
    "box",
)

#: 动作类型的常见同义写法 → 本项目的动作名。
#:
#: 只收**无歧义**的：``click`` 就是左键单击，这没有第二种解释。像
#: ``select`` 这种既可能是点击也可能是拖选的，不收——宁可解析失败，
#: 也不能猜错动作。
ACTION_ALIASES = {
    "click": "left_click",
    "left_click": "left_click",
    "leftclick": "left_click",
    "tap": "left_click",
    "click_element": "left_click",
    "right_click": "right_click",
    "rightclick": "right_click",
    "double_click": "double_click",
    "doubleclick": "double_click",
    "left_click_drag": "left_click_drag",
    "drag": "left_click_drag",
    "mouse_move": "mouse_move",
    "move": "mouse_move",
    "hover": "mouse_move",
    "scroll": "scroll",
    "key": "key",
    "keypress": "key",
    "key_press": "key",
    "hotkey": "key",
    "press": "key",
    "type": "type",
    "type_text": "type",
    "input": "type",
    "write": "type",
    "wait": "wait",
    "sleep": "wait",
    "screenshot": "screenshot",
    "middle_click": "middle_click",
    "triple_click": "triple_click",
    "hold_key": "hold_key",
}

#: 全角标点 → 半角。中文模型在中文语境里输出 JSON 时，标点被输入法带成
#: 全角是最常见的一类脏输出，而它百分之百可以无歧义地还原
FULLWIDTH_MAP = str.maketrans(
    {
        "＂": '"',  # ＂
        "“": '"',  # “
        "”": '"',  # ”
        "‘": "'",  # ‘
        "’": "'",  # ’
        "：": ":",  # ：
        "，": ",",  # ，
        "｛": "{",  # ｛
        "｝": "}",  # ｝
        "［": "[",  # ［
        "］": "]",  # ］
        "（": "(",  # （
        "）": ")",  # ）
    }
)


# ---------------------------------------------------------------------- #
# 抠 JSON
# ---------------------------------------------------------------------- #

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


def extract_json(text: str) -> dict:
    """从模型输出里抠出一个 JSON 对象。

    按"损伤从小到大"的顺序试，第一个成功的就返回：

    1. 整段就是 JSON
    2. ```json 围栏里的内容
    3. 花括号配平扫描出的第一个完整对象（应对前后有解说文字）
    4. 上面三种各自再做一遍"清洗"（全角转半角、去尾逗号、Python 字面量）

    清洗放在最后而不是一上来就做，是为了**别把本来好的输入改坏**——
    比如文本字段里合法的中文引号，不该被无差别替换掉。
    """
    if not text or not text.strip():
        raise OutputParseError("模型输出为空", raw=text)

    for candidate in _candidates(text):
        for loader in (_load_strict, _load_cleaned, _load_pythonic):
            try:
                data = loader(candidate)
            except (ValueError, SyntaxError, TypeError):
                continue
            if isinstance(data, dict):
                return data
            if isinstance(data, list) and data and isinstance(data[0], dict):
                # 模型给了动作列表。取第一个——Loop 是一步一动的，
                # 多给的部分下一轮会重新决策，那时界面已经变了
                logger.debug("模型返回了 %d 个动作，只取第一个", len(data))
                return data[0]

    raise OutputParseError("模型输出中找不到可解析的 JSON 对象", raw=text)


def _candidates(text: str):
    """按损伤从小到大产出候选片段。"""
    stripped = text.strip()
    yield stripped

    for match in _FENCE.findall(text):
        if match.strip():
            yield match.strip()

    yield from _brace_scan(text)


def _brace_scan(text: str):
    """花括号配平扫描，应对 JSON 前后裹着解说文字的情况。

    要跳过字符串字面量里的花括号——``{"text": "{}"}`` 里那对不算。
    """
    depth = 0
    start = -1
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                yield text[start : index + 1]
                start = -1
            elif depth < 0:
                depth = 0


def _load_strict(text: str):
    return json.loads(text)


def _load_cleaned(text: str):
    """全角转半角 + 去掉尾逗号后再试。"""
    cleaned = text.translate(FULLWIDTH_MAP)
    cleaned = _TRAILING_COMMA.sub(r"\1", cleaned)
    return json.loads(cleaned)


def _load_pythonic(text: str):
    """按 Python 字面量解析。

    应对单引号包裹的"JSON"，以及 ``True`` / ``None`` 这类 Python 写法——
    模型见过的训练数据里两种风格都有，混着输出很常见。

    用 ``literal_eval`` 而不是 ``eval``：只认字面量，不执行任何代码。
    """
    cleaned = _TRAILING_COMMA.sub(r"\1", text.translate(FULLWIDTH_MAP))
    return ast.literal_eval(cleaned)


# ---------------------------------------------------------------------- #
# 取字段
# ---------------------------------------------------------------------- #


def first_key(data: dict, keys, default=None):
    """按别名顺序取第一个存在的键。大小写不敏感。"""
    lowered = {str(k).lower(): v for k, v in data.items()}
    for key in keys:
        if key in lowered and lowered[key] is not None:
            return lowered[key]
    return default


def normalize_action_type(value) -> str:
    """把模型给的动作名归一到本项目的动作集。

    认不出来的**原样返回**，让 `Action` 去报"未知动作类型"——在这里
    静默改成某个已知动作，等于替模型做了它没做的决定。
    """
    if value is None:
        return ""
    if isinstance(value, dict):
        # {"action": {"name": "click", ...}} 这种嵌套
        value = first_key(value, ACTION_KEYS, "")
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return ACTION_ALIASES.get(text, text)


def coerce_bool(value) -> bool:
    """模型的布尔值可能是字符串。``"false"`` 是 True 这种坑必须堵死。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1", "是", "已完成", "完成")
    return False


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def extract_point(data: dict) -> tuple[int, int] | None:
    """从各种写法里抠出 (x, y)。

    见过的写法：

    - ``{"x": 100, "y": 200}``
    - ``{"point": [100, 200]}`` / ``{"coordinate": [100, 200]}``
    - ``{"point": {"x": 100, "y": 200}}``
    - ``{"coordinate": "(100, 200)"}``
    - ``{"point": [100, 200, 150, 240]}`` —— 给的是框，取中心

    浮点坐标向下取整到 int：像素是离散的，而 `Point` 强制 int。
    """
    x, y = data.get("x"), data.get("y")
    if x is not None and y is not None:
        pair = _to_int_pair(x, y)
        if pair:
            return pair

    # x 里装着整个坐标对，y 缺席。
    #
    # 实测 qwen3-vl-8b-instruct 有 2/3 的输出是这个形状：
    #     {"action": "left_click", "x": [314, 48], "thinking": "..."}
    # 写法古怪，但**无歧义**——只有一个数值序列、没有 y，它只可能是点。
    # 按"能无歧义还原就还原"的原则必须捞回来：不捞的话这个模型三次里
    # 两次会被判成"没给坐标"，能力被严重低估。
    #
    # y 同时存在时不猜：那种情况下 x=[314,48] 与 y=50 互相矛盾，
    # 交给下面的 POINT_KEYS 或直接判失败，好过挑一个。
    if x is not None and y is None:
        packed = _sequence_to_point(x)
        if packed:
            return packed

    raw = first_key(data, POINT_KEYS)
    if raw is None:
        return None

    if isinstance(raw, dict):
        return _to_int_pair(raw.get("x"), raw.get("y"))

    if isinstance(raw, list | tuple):
        numbers = [n for n in raw if isinstance(n, int | float)]
        if len(numbers) == 2:
            return _to_int_pair(numbers[0], numbers[1])
        if len(numbers) == 4:
            # 框 → 中心点
            left, top, right, bottom = numbers
            return _to_int_pair((left + right) / 2, (top + bottom) / 2)
        return None

    if isinstance(raw, str):
        numbers = _NUMBER.findall(raw)
        if len(numbers) == 2:
            return _to_int_pair(float(numbers[0]), float(numbers[1]))
        if len(numbers) == 4:
            left, top, right, bottom = (float(n) for n in numbers)
            return _to_int_pair((left + right) / 2, (top + bottom) / 2)
    return None


def _sequence_to_point(value) -> tuple[int, int] | None:
    """把"一串数"解释成点：两个数是点，四个数是框取中心，其余不猜。

    列表、元组、以及 ``"[314, 48]"`` 这类字符串都认。
    """
    if isinstance(value, list | tuple):
        numbers = [n for n in value if isinstance(n, int | float)]
    elif isinstance(value, str):
        numbers = [float(n) for n in _NUMBER.findall(value)]
    else:
        return None

    if len(numbers) == 2:
        return _to_int_pair(numbers[0], numbers[1])
    if len(numbers) == 4:
        left, top, right, bottom = numbers
        return _to_int_pair((left + right) / 2, (top + bottom) / 2)
    return None


def _to_int_pair(x, y) -> tuple[int, int] | None:
    try:
        return int(float(x)), int(float(y))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------- #
# 组装
# ---------------------------------------------------------------------- #

#: `Action` 认识的参数名。抠参数时按这张表过滤，多余字段丢进 meta
ACTION_PARAM_KEYS = (
    "x",
    "y",
    "to_x",
    "to_y",
    "text",
    "keys",
    "direction",
    "amount",
    "duration",
)

#: 参数的别名
PARAM_ALIASES = {
    "content": "text",
    "value": "text",
    "string": "text",
    "key": "keys",
    "hotkey": "keys",
    "key_combination": "keys",
    "clicks": "amount",
    "scroll_amount": "amount",
    "distance": "amount",
    "seconds": "duration",
    "time": "duration",
    "dx": "to_x",
    "dy": "to_y",
    "end_x": "to_x",
    "end_y": "to_y",
    "to": "to_x",
}


def parse_action_payload(text: str) -> dict:
    """模型输出 → 归一化的动作载荷。

    返回 ``{action_type, params, thinking, target_description, done}``。
    这里不构造 `Action`：参数合法性由 `Action.validate` 负责，两处都校验
    会让错误信息出现在两个地方，排查时反而更费劲。
    """
    data = extract_json(text)

    # 参数可能套了一层
    nested = first_key(data, PARAM_KEYS)
    merged: dict = {}
    if isinstance(nested, dict):
        merged.update(nested)
    merged.update({k: v for k, v in data.items() if k not in PARAM_KEYS})

    action_type = normalize_action_type(
        first_key(data, ACTION_KEYS) or first_key(merged, ACTION_KEYS)
    )
    thinking = first_key(merged, THINKING_KEYS, "") or ""
    target = first_key(merged, TARGET_KEYS, "") or ""
    done = coerce_bool(first_key(merged, DONE_KEYS, False))

    params: dict = {}
    for key, value in merged.items():
        name = PARAM_ALIASES.get(str(key).lower(), str(key).lower())
        if name in ACTION_PARAM_KEYS and value is not None:
            params[name] = value

    point = extract_point(merged)
    if point:
        params["x"], params["y"] = point
    else:
        # 抠不出成对坐标就把半截的也清掉：留一个孤零零的 x 会让
        # needs_grounding 判成"不缺坐标"，然后在 Action.validate 那里
        # 才炸，错误信息离根因太远
        params.pop("x", None)
        params.pop("y", None)

    if "duration" in params:
        params["duration"] = _to_float(params["duration"])
    if "amount" in params:
        params["amount"] = _to_int(params["amount"])
    for key in ("text", "keys", "direction"):
        if key in params and params[key] is not None:
            params[key] = str(params[key])
    if "direction" in params:
        params["direction"] = params["direction"].strip().lower()

    if not action_type and not done:
        raise OutputParseError(
            f"模型输出里没有动作类型，也没有完成标志。可用键：{sorted(data)}", raw=text
        )

    return {
        "action_type": action_type,
        "params": params,
        "thinking": str(thinking).strip(),
        "target_description": str(target).strip(),
        "done": done,
    }


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
