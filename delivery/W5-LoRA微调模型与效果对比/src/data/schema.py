"""统一样本 schema —— 所有公开数据集在此收敛成同一种形状。

## 为什么必须统一

M3 要做的事有两件：拿 grounding 训练集微调，拿 ScreenSpot 做零样本评测。
两者的原始格式毫不相干——ScreenSpot 是 parquet + 内嵌 PNG + 归一化 bbox，
ScreenAgent 是一堆按会话分目录的 JSON + jpg + 点击点。如果不先收敛，
M3 那周会同时面对"训练脚本读不了评测集"和"评测脚本读不了训练集"两个问题，
而 M3 是全项目最挤的一周。

## 坐标一律存**绝对像素**

各家原始格式不同：ScreenSpot 存归一化 [0,1]，ScreenAgent 存 1024×768 下的
绝对值，模型输出又是 [0,1000)。三套并存的后果 M1 已经吃过一次亏了——
第一次真机跑 Agent 时模型给出 (275, 969)，被当成越界拒绝，实际那是归一化坐标。

所以本模块的规则是：**入库即转绝对像素，并强制同时记录 `resolution`**。
任何归一化需求都由使用方按 `resolution` 现算，不留第二份真值。

## bbox 与 point 都可以缺，但不能都缺

- ScreenSpot 给的是元素框（有 bbox，point 取中心）
- ScreenAgent 给的是鼠标落点（有 point，无 bbox）

强行给 ScreenAgent 编一个 bbox 会污染"元素尺寸分布"这张统计图；强行要求
ScreenSpot 只留点又会丢掉评测要用的框。两个字段都设为可选，清洗时校验
"至少有一个"。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

from perception.types import BBox, Point

#: 统一 schema 的字段顺序。大纲第 3 周 D1-D2 点名的九个字段在前，
#: 之后是本项目为了可追溯补的三个。写 CSV / 建 DataFrame 都以此为准。
SCHEMA_FIELDS: tuple[str, ...] = (
    # —— 大纲指定 ——
    "sample_id",
    "screenshot_path",
    "resolution",
    "instruction",
    "action_type",
    "bbox",
    "point",
    "platform",
    "app",
    # —— 本项目补充，用于追溯与划分 ——
    "source_dataset",
    "split",
    "meta",
)


class Platform(str, Enum):
    """样本所属的界面形态。

    大纲把 Mind2Web / WebArena 列为"仅抽样查看"，理由正是它们是 WEB 而本
    项目目标是 DESKTOP。这个字段让"对不对齐"从一句论断变成一个可以 groupby
    的事实。
    """

    DESKTOP = "desktop"
    MOBILE = "mobile"
    WEB = "web"
    UNKNOWN = "unknown"


class ActionType(str, Enum):
    """归一化后的动作类型。

    有意保持粗粒度。各数据集的原始标注词表互不相同（ScreenSpot 只区分
    text/icon 这种**元素类别**而非动作，ScreenAgent 有 click/drag/scroll/
    type/key 等），细分反而会造出一堆只在单个数据集里出现的类别，让
    "动作类型分布"这张图变成三个数据集各画各的。

    原始标签一律保留在 `meta["raw_action_type"]` 里，需要细分时回去取。
    """

    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    DRAG = "drag"
    SCROLL = "scroll"
    TYPE = "type"
    KEY = "key"
    WAIT = "wait"
    #: 数据集标的是"这是个什么元素"而不是"做什么动作"（ScreenSpot 即如此）。
    #: 不硬塞成 CLICK——那等于凭空断言这些元素都该被点击。
    LOCATE = "locate"
    OTHER = "other"


#: 训练动作 → 它必须具备的字段。**唯一的一份口径**，`data.split` 与
#: `finetune.dataset` 都从这里读。
#:
#: 两处各写一份的话，一处放宽了另一处没跟着放宽，池子里有的样本会在生成
#: 训练记录时被静默丢掉，而两边的计数都"看起来对"——**本项目栽过两次**
#: （提示词模板、屏幕分辨率）。
#:
#: `"point"` 是特殊值，表示需要 `resolve_point()` 拿得到坐标；其余都是
#: `params` 里的键名。空元组表示这个动作不需要额外字段。
ACTION_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "left_click": ("point",),
    "double_click": ("point",),
    "right_click": ("point",),
    "mouse_move": ("point",),
    "scroll": ("direction",),
    "type": ("text",),
    "key": ("key",),
    "wait": (),
    #: 子任务完成信号。不是键鼠动作，但执行器必须学会在什么时候发出它——
    #: M3 实测：模型不会发 `done`，于是每个子任务都跑到步数上限才停，
    #: 中途撞上过登录框与密码导入对话框。**这是安全缺陷，不只是效率问题。**
    "done": (),
    #: 规划器的训练目标，不进执行器训练集。
    "plan": ("element",),
}

#: ScreenAgent 的 `raw_action_subtype` → 训练动作名。
#: 只在 `action_type` 粗粒度不够用时查这张表（`other` 底下混着三样东西）。
_SUBTYPE_TO_ACTION = {
    "click": "left_click",
    "double_click": "double_click",
    "right_click": "right_click",
    "move": "mouse_move",
    "scroll_up": "scroll",
    "scroll_down": "scroll",
    "text": "type",
    "press": "key",
}

#: `raw_action_type` → 训练动作名。
#:
#: 前两个在 `action_type` 里都是 `other`，非查不可。`WaitAction` 靠粗粒度
#: 类型也能对上，但仍然写在这里——**兜底路径能不能用取决于 `action_type`
#: 的粒度，而那是为画图定的，不是为训练定的**。显式写死，粒度改了也不受影响。
_RAW_TYPE_TO_ACTION = {
    "EvaluateSubTaskAction": "done",
    "PlanAction": "plan",
    "WaitAction": "wait",
}


def training_action(sample: UnifiedSample) -> str:
    """这条样本对应哪个训练动作。不可训练返回空串。

    **`action_type` 一个人不够用。** 它有意是粗粒度的（见 `ActionType`
    文档），`other` 底下同时混着 `done`（1184 条）、`plan`（1168 条）和
    `mouse_move`（83 条）——三样完全不同的东西。所以先看原始标签。

    `drag` 一律不可训练：ScreenAgent 的拖拽标注只有起点没有终点，
    构不成可执行的动作。**丢弃而不是硬凑**——拿只有起点的拖拽去训练，
    教出来的是「拖拽 = 点一下」。
    """
    meta = sample.meta or {}
    raw_type = str(meta.get("raw_action_type") or "")
    if raw_type in _RAW_TYPE_TO_ACTION:
        return _RAW_TYPE_TO_ACTION[raw_type]

    subtype = str(meta.get("raw_action_subtype") or "").lower()
    if subtype in _SUBTYPE_TO_ACTION:
        return _SUBTYPE_TO_ACTION[subtype]

    # 没有原始标签时（非 ScreenAgent 来源）退回粗粒度类型
    value = sample.action_type.value
    if value == "click":
        return "left_click"
    return value if value in ACTION_REQUIREMENTS else ""


def is_trainable(sample: UnifiedSample) -> bool:
    """这条样本的字段是否齐全到可以生成一条训练记录。

    **这是"字段齐不齐"，不是"样本好不好"。** 被判为 False 的样本没有质量
    问题，只是缺了该动作必需的参数——比如一条 `type` 没带 `keyboard_text`。
    """
    action = training_action(sample)
    if not action:
        return False
    # 每条训练记录都需要一句指令——**没有指令的样本会教模型在缺信息时
    # 也硬输出一个动作**。ScreenAgent 的 `instruction` 是当前子任务（见该
    # 装载器的 `_subtask_of`），取不到时为空，这里就把它挡住。
    if not (sample.instruction or "").strip():
        return False
    for need in ACTION_REQUIREMENTS[action]:
        if need == "point":
            if sample.resolve_point() is None:
                return False
        elif not (sample.params or {}).get(need):
            return False
    return True


@dataclass
class UnifiedSample:
    """统一后的单条样本。

    一条样本 = 一张截图 + 一句指令 + 一个目标位置。多步会话（ScreenAgent）
    在装载时被拆成逐步的独立样本，会话归属记在 `meta` 里。
    """

    sample_id: str
    screenshot_path: str
    #: (width, height)，**截图的真实像素尺寸**，不是模型输入尺寸。
    resolution: tuple[int, int]
    instruction: str
    action_type: ActionType
    platform: Platform
    source_dataset: str
    #: 绝对像素的元素框。ScreenAgent 这类只有落点的样本为 None。
    bbox: BBox | None = None
    #: 绝对像素的目标点。为 None 时由 `resolve_point()` 从 bbox 中心补。
    point: Point | None = None
    #: 坐标之外的动作参数。**没有它，非鼠标动作就只剩一个类型名，训不了。**
    #:
    #: 2026-08-26 补。此前本类只有 bbox / point 两个参数位，因为它写于
    #: 「grounding 层」时期——那时坐标就是任务定义本身，别的参数无处可放也
    #: 无人需要。目标改为动作生成后，`type` 的文本、`key` 的键名、`wait` 的
    #: 秒数、`done` 的判定结果在装载时被静默丢弃，于是 4012 条样本里只有
    #: 716 条能进训练。原始标注里这些字段一直都在。
    #:
    #: 各动作的键名见 `data.loaders.screenagent._params_of`。空 dict 表示
    #: 这个动作除坐标外不需要参数（click / double_click / move）。
    params: dict = field(default_factory=dict)
    app: str = ""
    #: 数据集自带的划分（test / train / …）。本项目自己的三分见 `data.split`，
    #: 那个写在独立的 split 文件里，不覆盖此字段——原始划分要留着对照。
    split: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def width(self) -> int:
        return self.resolution[0]

    @property
    def height(self) -> int:
        return self.resolution[1]

    def resolve_point(self) -> Point | None:
        """拿到可用于训练 / 评测的目标点。

        显式给出的 point 优先于 bbox 中心：ScreenAgent 的落点是标注者真实
        点过的位置，比几何中心更能代表"可点击的地方"（想想一个很长的菜单项）。
        """
        if self.point is not None:
            return self.point
        if self.bbox is not None:
            return self.bbox.center
        return None

    def normalized_point(self, scale: int = 1000) -> tuple[float, float] | None:
        """按 `resolution` 折算成 [0, scale) 的归一化坐标。

        M3 训练样本要喂给 Qwen2.5-VL，它的坐标约定就是 0-1000 归一化——
        这一点在 M1 真机验证里被实测确认过（告诉模型图高 768 或 640，
        它给出的 y 都在 96x 附近，不随声明的尺寸变化）。
        """
        p = self.resolve_point()
        if p is None or self.width <= 0 or self.height <= 0:
            return None
        return (p.x / self.width * scale, p.y / self.height * scale)

    def to_row(self) -> dict:
        """摊平成一行，供 pandas 直接建 DataFrame。

        嵌套结构（bbox / point / resolution / meta）在这里拆开或序列化，
        因为 DataFrame 里放 dataclass 会让 groupby 与绘图都没法用。
        """
        return {
            "sample_id": self.sample_id,
            "screenshot_path": self.screenshot_path,
            "width": self.width,
            "height": self.height,
            "instruction": self.instruction,
            "action_type": self.action_type.value,
            "platform": self.platform.value,
            "app": self.app,
            "source_dataset": self.source_dataset,
            "split": self.split,
            "bbox_left": self.bbox.left if self.bbox else None,
            "bbox_top": self.bbox.top if self.bbox else None,
            "bbox_right": self.bbox.right if self.bbox else None,
            "bbox_bottom": self.bbox.bottom if self.bbox else None,
            "bbox_width": self.bbox.width if self.bbox else None,
            "bbox_height": self.bbox.height if self.bbox else None,
            "bbox_area": self.bbox.area if self.bbox else None,
            "point_x": (p := self.resolve_point()) and p.x,
            "point_y": p.y if p else None,
            "has_bbox": self.bbox is not None,
        }

    def to_json_dict(self) -> dict:
        """无损序列化，用于 JSONL 落盘。`from_json_dict` 是它的逆。"""
        return {
            "sample_id": self.sample_id,
            "screenshot_path": self.screenshot_path,
            "resolution": list(self.resolution),
            "instruction": self.instruction,
            "action_type": self.action_type.value,
            "platform": self.platform.value,
            "app": self.app,
            "source_dataset": self.source_dataset,
            "split": self.split,
            "bbox": list(self.bbox.as_tuple()) if self.bbox else None,
            "point": list(self.point.as_tuple()) if self.point else None,
            "params": self.params,
            "meta": self.meta,
        }

    @classmethod
    def from_json_dict(cls, payload: dict) -> UnifiedSample:
        bbox = payload.get("bbox")
        point = payload.get("point")
        return cls(
            sample_id=payload["sample_id"],
            screenshot_path=payload["screenshot_path"],
            resolution=tuple(payload["resolution"]),  # type: ignore[arg-type]
            instruction=payload.get("instruction", ""),
            action_type=ActionType(payload.get("action_type", "other")),
            platform=Platform(payload.get("platform", "unknown")),
            app=payload.get("app", ""),
            source_dataset=payload.get("source_dataset", ""),
            split=payload.get("split", ""),
            bbox=BBox(*bbox) if bbox else None,
            point=Point(*point) if point else None,
            params=payload.get("params", {}),
            meta=payload.get("meta", {}),
        )


# ===================================================================== #
# 落盘 / 装载
# ===================================================================== #


def write_jsonl(samples: Iterable[UnifiedSample], path: str | Path) -> int:
    """写 JSONL。逐条写而不是先攒成 list —— 装载器都是生成器，攒起来会把
    几万条样本连同 meta 一起顶在内存里。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_json_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> Iterator[UnifiedSample]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield UnifiedSample.from_json_dict(json.loads(line))


def to_dataframe(samples: Iterable[UnifiedSample]):
    """建 DataFrame。pandas 在函数内导入，让 schema 本身保持可无依赖测试。"""
    import pandas as pd

    return pd.DataFrame([s.to_row() for s in samples])


__all__ = [
    "SCHEMA_FIELDS",
    "ActionType",
    "Platform",
    "UnifiedSample",
    "asdict",
    "read_jsonl",
    "to_dataframe",
    "write_jsonl",
]
