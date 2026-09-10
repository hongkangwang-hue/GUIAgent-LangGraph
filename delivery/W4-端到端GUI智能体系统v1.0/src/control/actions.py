"""动作空间定义与 JSON Schema。

## 为什么自己定义动作空间

官方大纲没有指定动作集，UI-TARS 与 Claude Computer Use 各有一套。本项目
自定义，但**保证每个动作都能被 JSON Schema 完整描述**——M2 会把这些
schema 直接转成大模型的工具定义，中间不做第二次映射。二次映射是错位的
温床：参数名在两处不一致时，模型输出的是一套、执行的是另一套。

## 两种工具渲染方式都提供

- `to_tool_schemas()`：一个动作一个工具。标准 function calling 的形态。
- `to_unified_tool_schema()`：单个 `computer` 工具 + `action` 枚举。
  Claude Computer Use 与 UI-TARS 的形态。

**开源规划模型在工具数量多时选错工具的概率明显更高**，统一形态只需要它
选一个工具再填一个枚举值，通常更稳。M2 用哪种由实测决定，因此两种都留着。

## 坐标一律是模型坐标

动作里的 x / y 是**模型坐标系**下的值，由 `ActionExecutor` 经
`CoordinateScaler` 转成屏幕坐标。动作对象本身不携带坐标系信息——
坐标系是执行器的上下文，混进动作里会让同一个动作在不同上下文中含义不同。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActionType(str, Enum):
    """完整动作集。M1 实现前十个，后三个留接口桩。"""

    SCREENSHOT = "screenshot"
    LEFT_CLICK = "left_click"
    RIGHT_CLICK = "right_click"
    DOUBLE_CLICK = "double_click"
    LEFT_CLICK_DRAG = "left_click_drag"
    MOUSE_MOVE = "mouse_move"
    SCROLL = "scroll"
    KEY = "key"
    TYPE = "type"
    WAIT = "wait"
    # --- 低频动作，M1 只留接口桩，M4 按需补齐 ---
    MIDDLE_CLICK = "middle_click"
    TRIPLE_CLICK = "triple_click"
    HOLD_KEY = "hold_key"


#: M1 必须实现的动作
CORE_ACTIONS = frozenset(
    {
        ActionType.SCREENSHOT,
        ActionType.LEFT_CLICK,
        ActionType.RIGHT_CLICK,
        ActionType.DOUBLE_CLICK,
        ActionType.LEFT_CLICK_DRAG,
        ActionType.MOUSE_MOVE,
        ActionType.SCROLL,
        ActionType.KEY,
        ActionType.TYPE,
        ActionType.WAIT,
    }
)

#: 接口桩，调用时抛 NotImplementedError
STUB_ACTIONS = frozenset({ActionType.MIDDLE_CLICK, ActionType.TRIPLE_CLICK, ActionType.HOLD_KEY})

SCROLL_DIRECTIONS = ("up", "down", "left", "right")


@dataclass(frozen=True)
class ParamSpec:
    """一个动作参数的定义。description 会原样进入给模型的 schema。"""

    name: str
    json_type: str
    description: str
    required: bool = True
    enum: tuple[str, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None


_XY = (
    ParamSpec("x", "integer", "目标点的横坐标（模型坐标系，左上角为原点）"),
    ParamSpec("y", "integer", "目标点的纵坐标（模型坐标系，左上角为原点）"),
)

#: 动作 → (说明, 参数列表)。JSON Schema 与参数校验都从这张表生成，
#: 保证"给模型看的"和"实际校验的"永远是同一份定义。
ACTION_SPECS: dict[ActionType, tuple[str, tuple[ParamSpec, ...]]] = {
    ActionType.SCREENSHOT: ("截取当前屏幕，用于观察界面当前状态", ()),
    ActionType.LEFT_CLICK: ("在指定位置单击鼠标左键", _XY),
    ActionType.RIGHT_CLICK: ("在指定位置单击鼠标右键，通常用于打开上下文菜单", _XY),
    ActionType.DOUBLE_CLICK: ("在指定位置双击鼠标左键，通常用于打开文件或程序", _XY),
    ActionType.MOUSE_MOVE: ("把鼠标移动到指定位置，不点击。用于触发悬停效果", _XY),
    ActionType.LEFT_CLICK_DRAG: (
        "按住鼠标左键从起点拖到终点，用于拖动、框选、调整大小",
        (
            ParamSpec("x", "integer", "拖动起点的横坐标"),
            ParamSpec("y", "integer", "拖动起点的纵坐标"),
            ParamSpec("to_x", "integer", "拖动终点的横坐标"),
            ParamSpec("to_y", "integer", "拖动终点的纵坐标"),
        ),
    ),
    ActionType.SCROLL: (
        "在指定位置滚动鼠标滚轮",
        (
            *_XY,
            ParamSpec("direction", "string", "滚动方向", enum=SCROLL_DIRECTIONS),
            ParamSpec(
                "amount",
                "integer",
                "滚动格数，通常 3 格约等于一屏的三分之一",
                required=False,
                minimum=1,
                maximum=30,
            ),
        ),
    ),
    ActionType.KEY: (
        "按下组合键。单键直接写键名，组合键用加号连接",
        (
            ParamSpec(
                "keys",
                "string",
                "键名或组合键，例如 'enter'、'ctrl+c'、'alt+f4'、'ctrl+shift+n'",
            ),
        ),
    ),
    ActionType.TYPE: (
        "输入一段文本。支持中文",
        (ParamSpec("text", "string", "要输入的文本内容"),),
    ),
    ActionType.WAIT: (
        "等待一段时间，用于让界面完成加载或动画",
        (ParamSpec("duration", "number", "等待秒数", minimum=0.1, maximum=10.0),),
    ),
    ActionType.MIDDLE_CLICK: ("在指定位置单击鼠标中键", _XY),
    ActionType.TRIPLE_CLICK: ("在指定位置三击鼠标左键，通常用于选中整行", _XY),
    ActionType.HOLD_KEY: (
        "按住某个键一段时间",
        (
            ParamSpec("keys", "string", "要按住的键名"),
            ParamSpec("duration", "number", "按住秒数", minimum=0.1, maximum=10.0),
        ),
    ),
}


class ActionValidationError(ValueError):
    """动作参数不合法。模型输出解析失败时抛这个，由上层决定重试还是终止。"""


@dataclass
class Action:
    """一个待执行的动作。

    坐标是**模型坐标**，由执行器转换。见模块文档。
    """

    type: ActionType
    x: int | None = None
    y: int | None = None
    to_x: int | None = None
    to_y: int | None = None
    text: str | None = None
    keys: str | None = None
    direction: str | None = None
    amount: int | None = None
    duration: float | None = None
    #: 模型给出的自然语言理由，只进日志不影响执行。轨迹复盘时很有用
    reasoning: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.type, str):
            try:
                object.__setattr__(self, "type", ActionType(self.type))
            except ValueError:
                raise ActionValidationError(
                    f"未知动作类型 {self.type!r}。可用：{[a.value for a in ActionType]}"
                ) from None
        self.validate()

    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        """按 `ACTION_SPECS` 校验参数。校验规则与给模型的 schema 同源。"""
        _, specs = ACTION_SPECS[self.type]
        for spec in specs:
            value = getattr(self, spec.name, None)
            if value is None:
                if spec.required:
                    raise ActionValidationError(
                        f"动作 {self.type.value} 缺少必填参数 {spec.name!r}"
                    )
                continue
            if spec.enum and value not in spec.enum:
                raise ActionValidationError(
                    f"动作 {self.type.value} 的 {spec.name}={value!r} 不在允许值 {spec.enum} 中"
                )
            if spec.minimum is not None and value < spec.minimum:
                raise ActionValidationError(
                    f"动作 {self.type.value} 的 {spec.name}={value} 小于下限 {spec.minimum}"
                )
            if spec.maximum is not None and value > spec.maximum:
                raise ActionValidationError(
                    f"动作 {self.type.value} 的 {spec.name}={value} 大于上限 {spec.maximum}"
                )

    def requires_coordinates(self) -> bool:
        """本动作是否涉及屏幕坐标——决定是否需要做越界检查与坐标转换。"""
        _, specs = ACTION_SPECS[self.type]
        return any(spec.name == "x" for spec in specs)

    def with_type(self, action_type: ActionType | str) -> Action:
        """换一个动作类型，其余字段照抄，返回**新对象**。

        重试策略升级动作时用（`core/retry.py`）：同一个位置单击没反应，
        改成双击。**只换类型不换坐标**——坐标是模型对目标位置的判断，
        策略没有比它更好的信息；要改的是"用什么方式碰它"。

        返回新对象而不是原地改：`StepRecord` 已经记下了改写前的
        `action_model_coords`，原地改会让那份记录跟着变，事后就分不清
        "模型自己选了双击"和"策略改的"。
        """
        import dataclasses

        return dataclasses.replace(self, type=ActionType(action_type))

    def to_dict(self) -> dict:
        """只含本动作实际用到的参数，便于写进轨迹日志。"""
        _, specs = ACTION_SPECS[self.type]
        payload: dict[str, Any] = {"action": self.type.value}
        for spec in specs:
            value = getattr(self, spec.name, None)
            if value is not None:
                payload[spec.name] = value
        if self.reasoning:
            payload["reasoning"] = self.reasoning
        return payload

    @classmethod
    def from_dict(cls, data: dict) -> Action:
        """从模型输出的 dict 构造。未知字段忽略，缺失字段由校验兜住。

        兼容 ``action`` 与 ``type`` 两种键名——不同模型的输出习惯不一样，
        在这里吸收掉，比在每个后端里各写一遍好。
        """
        payload = dict(data)
        action_type = payload.pop("action", None) or payload.pop("type", None)
        if action_type is None:
            raise ActionValidationError(f"动作缺少 action / type 字段：{data!r}")

        known = {f for f in cls.__dataclass_fields__ if f != "type"}
        kwargs = {k: v for k, v in payload.items() if k in known}
        extra = {k: v for k, v in payload.items() if k not in known}
        if extra:
            kwargs.setdefault("meta", {}).update(extra)
        return cls(type=action_type, **kwargs)

    def __str__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.to_dict().items() if k != "action")
        return f"{self.type.value}({params})"


# ---------------------------------------------------------------------- #
# JSON Schema 生成
# ---------------------------------------------------------------------- #


def _params_schema(specs: tuple[ParamSpec, ...]) -> dict:
    properties: dict[str, dict] = {}
    required: list[str] = []
    for spec in specs:
        prop: dict[str, Any] = {"type": spec.json_type, "description": spec.description}
        if spec.enum:
            prop["enum"] = list(spec.enum)
        if spec.minimum is not None:
            prop["minimum"] = spec.minimum
        if spec.maximum is not None:
            prop["maximum"] = spec.maximum
        properties[spec.name] = prop
        if spec.required:
            required.append(spec.name)
    return {"type": "object", "properties": properties, "required": required}


def action_schema(action_type: ActionType) -> dict:
    """单个动作的 JSON Schema。"""
    description, specs = ACTION_SPECS[action_type]
    return {
        "name": action_type.value,
        "description": description,
        "parameters": _params_schema(specs),
    }


def to_tool_schemas(only_core: bool = True) -> list[dict]:
    """一个动作一个工具。标准 function calling 形态。"""
    types = sorted(CORE_ACTIONS if only_core else set(ActionType), key=lambda a: a.value)
    return [action_schema(t) for t in types]


def to_unified_tool_schema(only_core: bool = True) -> dict:
    """单个 `computer` 工具 + `action` 枚举。Claude CU / UI-TARS 形态。

    所有参数合并成一个可选参数表，由 `action` 的取值决定哪些生效——
    JSON Schema 的 `oneOf` 表达力更强，但**开源模型对 `oneOf` 的遵循度
    很差**，扁平参数表加上描述里的说明反而更可靠。
    """
    types = sorted(CORE_ACTIONS if only_core else set(ActionType), key=lambda a: a.value)

    merged: dict[str, dict] = {}
    usage_lines = []
    for action_type in types:
        description, specs = ACTION_SPECS[action_type]
        param_names = [s.name for s in specs]
        usage_lines.append(
            f"- {action_type.value}: {description}"
            + (f"（需要参数：{', '.join(param_names)}）" if param_names else "（无需参数）")
        )
        for spec in specs:
            if spec.name not in merged:
                prop: dict[str, Any] = {"type": spec.json_type, "description": spec.description}
                if spec.enum:
                    prop["enum"] = list(spec.enum)
                if spec.minimum is not None:
                    prop["minimum"] = spec.minimum
                if spec.maximum is not None:
                    prop["maximum"] = spec.maximum
                merged[spec.name] = prop

    properties = {
        "action": {
            "type": "string",
            "enum": [t.value for t in types],
            "description": "要执行的动作类型。各动作所需参数：\n" + "\n".join(usage_lines),
        },
        **merged,
    }
    return {
        "name": "computer",
        "description": "操作桌面：截图、点击、输入、按键、滚动、等待。每次只执行一个动作。",
        "parameters": {"type": "object", "properties": properties, "required": ["action"]},
    }
