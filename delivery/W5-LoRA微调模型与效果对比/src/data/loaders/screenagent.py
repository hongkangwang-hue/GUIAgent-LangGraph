"""ScreenAgent —— M3 的动作生成训练样本来源。

## 数据不在 HuggingFace 上

HF 的 `niurl/ScreenAgent` 只有 CogAgent 格式的权重 zip。**标注数据在作者的
GitHub 仓库里**（`niuzaisheng/ScreenAgent`，`data/ScreenAgent/`），train 是
按会话分的目录树，test 是一个 50MB 的 `test.zip`。这一点值得写进报告：
按 HF 名称去找会一无所获。

许可证：数据集 Apache-2.0（代码 MIT，模型权重走 CogVLM 协议——三者不同，
《模型与数据集许可证对照表》要分开记）。

## 三个必须知道的格式细节

**1. train 与 test 的文件命名不一样。**
train 是 `<时间戳>_translate.json`，test 是 `<时间戳>.json`。两者都还有
`_neg_plan` / `_neg_eval` 后缀的负样本（RLHF 用的 reject 回答），共 1264 个，
**这些不是标注，是模型的错误回答**，必须排除。只按 `*.json` 通配会把负样本
一起当成训练数据装进来。

**2. `mouse_position` 用 `width` / `height` 当 x / y 的键名。**
`{"mouse_position": {"width": 543, "height": 133}}` 里 width 是 x、height 是 y，
和字面意思毫无关系。这里翻译成 `Point(x, y)` 之后不再让这个命名外泄。

**3. `clickable_area`（元素框）只有 test 划分里有。**
实测：test 的 218 个带落点鼠标动作全都有框；train 的 757 个鼠标动作**一个也没有**。
也就是说 ScreenAgent 能给 M3 的训练监督信号只有点，没有框。

## 一个动作一条样本

按动作展开而不是按步，是因为训练要的正是"一句指令 → 一个动作"。

## 为什么 `PlanAction.element` **不能**用作指令

这一条是踩过坑之后写的。`element` 字段看上去正是我们要的东西——标注者
手写的自然语言描述，形如 "Choose the blue color from the color palette"、
"点击save"。很自然会想把它当成鼠标动作的 `instruction`。

**但数据结构不支持这么做。** 实测确认了三件事：

1. **Plan 与 Mouse 从不出现在同一步里。** ScreenAgent 的一步是单一类型的：
   要么整步都是 PlanAction，要么整步都是 MouseAction。实测 716 个带坐标的
   鼠标动作，**同一步内前方有 PlanAction 的是 0 个**。典型会话长这样：

       step1 (255913.jpg)  PlanAction ×2   规划整个任务的几个子目标
       step2 (140419.jpg)  MouseAction click(487,193) + 输入 + 回车
       step3 (168194.jpg)  EvaluateSubTaskAction  sub_task_success

2. **就算跨步去找最近的 Plan，它描述的也是另一张截图。** 上例里 Plan 写在
   `255913.jpg` 上，点击发生在 `140419.jpg` 上。把前者的描述配后者的坐标，
   等于给模型一对错配的监督信号。

3. **`element` 中英混杂且没有语言变体。** 实测 1168 条里 73% 是中文、
   27% 是英文，而这个字段不像 `task_prompt` 那样有 `_en` / `_zh` 两套。
   用它当指令会让 `language` 选项形同虚设——选 en 的训练集里照样混进
   七成中文。

所以本装载器**绝不用 `element` 作为 `instruction`**。

## 指令取哪一句，取决于这条样本是给谁训的（2026-08-26 修正）

    PlanAction   → 规划器。活是「总目标 → 子任务列表」，指令用 `task_prompt`
    其他动作     → 执行器。活是「当前子任务 → 下一个动作」，指令用子任务

**此前一律用 `task_prompt`。** 那对执行器是矛盾监督：一个会话十几步共用
同一句总目标，716 条样本只对应 **169 条不重复**指令，「Insert a circle」
出现 17 次却配着 17 个不同坐标。改取子任务后，不重复指令涨到 1068 条。

子任务的抽取见 `_subtask_of`——优先中文原文（英文那份是机翻且引号已经坏了，
而且本项目部署时模型收到的本来就是中文）。**取不到时留空，不退回总目标**：
退回去等于把矛盾监督又放回训练集，而条数看起来还是满的。

`element` 文本仍然保留在 PlanAction 样本的 `meta["plan_element"]` 里——
它对指令没用，但**是模式 B `target_description` 的现成参考语料**
（M3 前置件④）。同一份内容也进 `params["element"]`，那是规划器的训练目标。

## 坐标之外的动作参数（2026-08-26 补）

`type` 的文本、`key` 的键名、`wait` 的秒数、`done` 的判定结果，此前在装载时
被静默丢弃——`UnifiedSample` 只有 bbox / point 两个参数位。现在统一进
`params`，见 `_params_of`。键名还要做 **X11 keysym → pyautogui** 的翻译，
理由见 `_KEYSYM_TO_PYAUTOGUI`：不翻译不只是执行失败，危险键安全规则会静默失效。
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from pathlib import Path

from data.loaders.base import DatasetLoader
from data.schema import ActionType, Platform, UnifiedSample
from perception.types import BBox, Point

logger = logging.getLogger(__name__)

#: 负样本（RLHF 的 reject 回答）。**不是标注数据。**
_NEGATIVE = re.compile(r"_neg_(plan|eval)\.json$")

#: ScreenAgent 的 mouse_action_type → 统一动作类型。
#: `down` / `up` 是拖拽的两端，单独出现时归入 DRAG；`move` 是纯移动（悬停），
#: 它带坐标所以对 grounding 有用，但不该和 click 混为一谈。
_MOUSE_ACTIONS = {
    "click": ActionType.CLICK,
    "double_click": ActionType.DOUBLE_CLICK,
    "right_click": ActionType.RIGHT_CLICK,
    "move": ActionType.OTHER,
    "drag": ActionType.DRAG,
    "down": ActionType.DRAG,
    "up": ActionType.DRAG,
    "scroll_up": ActionType.SCROLL,
    "scroll_down": ActionType.SCROLL,
}

_KEYBOARD_ACTIONS = {
    "text": ActionType.TYPE,
    "press": ActionType.KEY,
}


def _point_of(payload: dict | None) -> Point | None:
    """`{"width": x, "height": y}` → `Point(x, y)`。见模块文档第 2 点。"""
    if not isinstance(payload, dict):
        return None
    x, y = payload.get("width"), payload.get("height")
    if x is None or y is None:
        return None
    return Point(int(round(float(x))), int(round(float(y))))


#: X11 keysym → pyautogui 键名。
#:
#: ScreenAgent 是通过 VNC 采集的，键名是 **X11 keysym**（`Return`、`Control_L`）；
#: 本项目执行器用 pyautogui，认的是 `enter` / `ctrl`（见 `control.actions` 里
#: `keys` 的说明：「例如 'enter'、'ctrl+c'、'alt+f4'」）。
#:
#: **不翻译的后果不止是执行失败。** `control.safety.rule_dangerous_keys` 靠
#: 按 `+` 切分来匹配危险组合键——喂进去一个未翻译的键名，危险键规则一条也
#: 匹配不上，**那是安全缺口，不是格式问题**。
#:
#: 全量实测只有 24 个不同 token（见下），所以逐个列出而不是写规则去猜。
#: 表里没有的原样小写透传：单字母与数字本来就一致。
_KEYSYM_TO_PYAUTOGUI = {
    "return": "enter",
    "control_l": "ctrl",
    "control_r": "ctrl",
    "control": "ctrl",
    "shift_l": "shift",
    "shift_r": "shift",
    "alt_l": "alt",
    "alt_r": "alt",
    "super_l": "win",
    "super_r": "win",
    "escape": "esc",
    "backspace": "backspace",
    "print": "printscreen",
    "prior": "pageup",
    "next": "pagedown",
    "up": "up",
    "down": "down",
    "left": "left",
    "right": "right",
    "end": "end",
    "home": "home",
    "tab": "tab",
    "delete": "delete",
    "space": "space",
}


def _keys_of(raw) -> str:
    """`keyboard_key` → 执行器认的组合键字符串。取不到返回空串。

    输入有三种形态，实测都出现过：

        "Return"                        单键
        ["Control_L", "a"]              **列表**，323 条里有 85 条
        "Ctrl+S"                        已经是 + 连接的（1 条）

    列表如果不合并，`params["key"]` 会存成 ``"['Control_L', 'a']"``——
    Python 列表的字符串形式**原样进训练目标**，教模型输出一串它自己都
    解析不了的东西。
    """
    parts: list[str] = []
    for token in raw if isinstance(raw, list) else [raw]:
        if token is None:
            continue
        for piece in str(token).split("+"):
            piece = piece.strip()
            if piece:
                parts.append(_KEYSYM_TO_PYAUTOGUI.get(piece.lower(), piece.lower()))
    return "+".join(parts)


def _params_of(action: dict) -> dict:
    """坐标之外的动作参数 → `UnifiedSample.params`。

    **这些字段以前被静默丢弃。** 装载层只取了 point / bbox，于是 `type` 只剩
    动作类型名、`key` 不知道按哪个键、`done` 不知道判定结果——4012 条样本里
    只有 716 条能进训练，而缺的正是这些。原始标注里它们一直都在。

    键名口径（下游 `finetune.dataset` 按这个生成训练目标）：

        type    {"text": "Hello, world!"}          ← keyboard_text
        key     {"key": "Ctrl+A"}                  ← keyboard_key
        wait    {"seconds": 1.0}                   ← wait_time
        scroll  {"direction": "down", "repeat": 3} ← mouse_action_type + scroll_repeat
        done    {"situation": "sub_task_success"}  ← EvaluateSubTaskAction
        plan    {"element": "在搜索框输入……"}      ← PlanAction

    click / double_click / move 返回空 dict——它们除坐标外不需要参数。

    **缺字段就不写这个键**，而不是填默认值：下游要能分辨"标注里没有"和
    "标注里是空字符串"。前者该丢弃样本，后者是一条真实的空输入。
    """
    kind = action.get("action_type")

    if kind == "MouseAction":
        subtype = action.get("mouse_action_type", "")
        if subtype in ("scroll_up", "scroll_down"):
            params: dict = {"direction": subtype.removeprefix("scroll_")}
            if action.get("scroll_repeat") is not None:
                params["repeat"] = int(action["scroll_repeat"])
            return params
        return {}

    if kind == "KeyboardAction":
        subtype = action.get("keyboard_action_type", "")
        if subtype == "text" and action.get("keyboard_text") is not None:
            return {"text": str(action["keyboard_text"])}
        if subtype == "press":
            keys = _keys_of(action.get("keyboard_key"))
            return {"key": keys} if keys else {}
        return {}

    if kind == "WaitAction":
        return {} if action.get("wait_time") is None else {"seconds": float(action["wait_time"])}

    if kind == "EvaluateSubTaskAction":
        situation = action.get("situation")
        return {"situation": str(situation)} if situation else {}

    if kind == "PlanAction":
        element = action.get("element")
        # 与 `meta["plan_element"]` 重复是有意的：meta 那份是 M3 前置件④的
        # 模式 B 参考语料（已被文档与测试引用），params 这份是规划器的训练目标。
        return {"element": str(element)} if element else {}

    return {}


#: 中文标注模板里「这一步要干什么」那句话。句式高度一致，只有引号有几种写法：
#:
#:     现在的子任务是「在浏览器中输入"冯诺依曼"作为搜索关键词」。
#:     现在的子任务是 "在搜索框中输入…"，请仔细描述下一步动作。
_SUBTASK_ZH = re.compile(r"现在的子任务是\s*[「『\"“”‘']?\s*(.+?)\s*[」』\"“”’']?\s*[。，,]")


def _subtask_of(payload: dict) -> tuple[str, str]:
    """当前子任务，以及它是从哪来的。取不到返回 `("", "")`。

    ## 为什么需要它

    `task_prompt` 是**会话级总目标**，一个会话十几步共用同一句。原来的训练
    集拿它当指令，于是 716 条样本只对应 169 个不重复指令，「Insert a circle」
    出现 17 次却对应 17 个不同坐标——**对模型是矛盾监督**。

    ## 为什么优先中文，哪怕装载语言设的是 en

    两个理由，第二个是决定性的：

    1. **中文是原文，英文是机翻且引号已经坏了**——
       ``Now the subtask is "enter" Von Neumann "as the search keyword"``，
       引号嵌套错位，按引号切会切出半句话。
    2. **本项目部署时模型收到的就是中文**：`tasks/basic_tasks.yaml` 写的是
       「打开 Microsoft Edge 浏览器」，`prompts/executor_v1.yaml` 通篇中文。
       训练指令与推理指令不同语言，等于白训一半。

    `current_task` 字段（test 划分 639/639 有）是英文，作为退路——那个划分
    的原始文件里根本没有中文变体。**这构成 train / test 之间的指令语言差异，
    必须在报告里写明**，不能装作没有。
    """
    found = _SUBTASK_ZH.search(str(payload.get("send_prompt_zh") or ""))
    if found:
        return found.group(1).strip(), "send_prompt_zh"

    current = str(payload.get("current_task") or "").strip()
    if current:
        return current, "current_task"

    return "", ""


def _bbox_of(payload: dict | None) -> BBox | None:
    """`clickable_area` → BBox。只有 test 划分有这个字段。"""
    if not isinstance(payload, dict):
        return None
    upper = _point_of(payload.get("upper_left_position"))
    lower = _point_of(payload.get("lower_right_position"))
    if upper is None or lower is None:
        return None
    if lower.x < upper.x or lower.y < upper.y:
        return None
    return BBox(upper.x, upper.y, lower.x, lower.y)


class ScreenAgentLoader(DatasetLoader):
    """装载 ScreenAgent 的 train / test 两个划分。"""

    name = "screenagent"

    #: 仓库整体 clone 下来，数据在这个子路径下
    _DATA_SUBPATH = "data/ScreenAgent"

    def __init__(self, root: str | Path = "data/raw", **kwargs) -> None:
        super().__init__(root, **kwargs)
        #: 语言。数据集自带 en / zh 两套文本，训练用哪套要显式选，
        #: 混着来会让模型同时见到两种语言的同一批样本。
        self.language: str = kwargs.get("language", "en")

    @property
    def dataset_dir(self) -> Path:
        return self.root / "screenagent_repo" / self._DATA_SUBPATH

    def available(self) -> tuple[bool, str]:
        if not self.dataset_dir.is_dir():
            return False, (
                f"未找到 {self.dataset_dir}。ScreenAgent 数据集不在 HuggingFace 上，"
                "需 clone github.com/niuzaisheng/ScreenAgent，"
                "见 `python scripts/prepare_datasets.py --download`"
            )
        if not (self.dataset_dir / "train").is_dir():
            return False, f"{self.dataset_dir} 下没有 train 目录"
        if not (self.dataset_dir / "test").is_dir():
            return False, (
                f"test 未解压：{self.dataset_dir / 'test.zip'} → "
                f"{self.dataset_dir / 'test'}（用 --extract 解压）"
            )
        return True, ""

    def load(self) -> Iterator[UnifiedSample]:
        for split in ("train", "test"):
            split_dir = self.dataset_dir / split
            for path in sorted(split_dir.glob("*/*.json")):
                if _NEGATIVE.search(path.name):
                    self._skip("RLHF 负样本")
                    continue
                yield from self._load_step(path, split)

    # ----------------------------------------------------------------- #

    def _load_step(self, path: Path, split: str) -> Iterator[UnifiedSample]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._skip("标注文件无法解析")
            return

        width = payload.get("video_width")
        height = payload.get("video_height")
        if not width or not height:
            self._skip("缺少分辨率")
            return

        image_name = payload.get("saved_image_name")
        if not image_name:
            self._skip("缺少截图文件名")
            return
        image_path = path.parent / "images" / image_name
        if not image_path.exists():
            self._skip("截图文件不存在")
            return

        suffix = "_zh" if self.language == "zh" else "_en"
        task_prompt = payload.get(f"task_prompt{suffix}") or payload.get("task_prompt") or ""
        # `session_id` 只有 train 划分有；test 划分靠目录名。两个划分的文件
        # 结构本来就不同（见模块文档），这里不能只信字段。
        session_id = payload.get("session_id") or path.parent.name
        stem = path.stem.replace("_translate", "")

        subtask, subtask_source = _subtask_of(payload)

        for index, action in enumerate(payload.get("actions") or []):
            kind = action.get("action_type")

            parsed = self._parse_action(action)
            if parsed is None:
                self._skip(f"未识别的动作类型：{kind}")
                continue
            action_type, point, bbox = parsed

            # 指令取哪一句，**取决于这条样本是给谁训的**：
            #
            #   PlanAction  → 规划器。它的活是「总目标 → 子任务列表」，
            #                 指令当然是总目标。给它子任务等于把答案当题目。
            #   其他动作    → 执行器。它的活是「当前子任务 → 下一个动作」。
            #
            # 2026-08-26 之前一律取 `task_prompt`，于是执行器样本一个会话
            # 十几步共用一句，造成矛盾监督（716 条只对应 169 个不重复指令）。
            # 改成子任务后一度**连规划样本也要求子任务**，把 884 条 plan
            # 挡到只剩 215 条——一刀切的过滤会挡掉本来就不该被过滤的那一类。
            #
            # 执行器样本**取不到子任务时留空，不退回总目标**。退回去等于把
            # 矛盾监督又放回训练集，而条数看起来还是满的，事后查不出来。
            #
            # 仍然**不用 `PlanAction.element`** 当指令——理由见模块文档：
            # Plan 和 Mouse 不在同一步、描述的不是同一张截图、且中英混杂。
            if kind == "PlanAction":
                instruction, source = task_prompt, "task_prompt"
            else:
                instruction, source = subtask, (subtask_source or "缺失")

            # element 文本对 grounding 没用，但别丢——它是模式 B
            # `target_description` 的参考语料（M3 前置件④）。
            plan_element = str(action.get("element") or "") if kind == "PlanAction" else ""

            yield UnifiedSample(
                sample_id=f"screenagent-{split}-{session_id}-{stem}-{index}",
                screenshot_path=str(image_path),
                resolution=(int(width), int(height)),
                instruction=instruction,
                action_type=action_type,
                platform=Platform.DESKTOP,
                source_dataset=self.name,
                bbox=bbox,
                point=point,
                params=_params_of(action),
                app="",  # 数据集未标注具体应用，留空而不是猜
                split=split,
                meta={
                    "session_id": session_id,
                    "task_prompt": task_prompt,
                    "instruction_source": source,
                    #: 「这一步要干什么」。见 `_subtask_of`。
                    **({"subtask": subtask, "subtask_source": subtask_source} if subtask else {}),
                    "raw_action_type": kind,
                    "raw_action_subtype": (
                        action.get("mouse_action_type") or action.get("keyboard_action_type") or ""
                    ),
                    "action_index": index,
                    "language": self.language,
                    #: 仅 PlanAction 有。见模块文档——不作指令，留作模式 B 语料
                    **({"plan_element": plan_element} if plan_element else {}),
                },
            )

    @staticmethod
    def _parse_action(action: dict) -> tuple[ActionType, Point | None, BBox | None] | None:
        kind = action.get("action_type")

        if kind == "MouseAction":
            subtype = action.get("mouse_action_type", "")
            mapped = _MOUSE_ACTIONS.get(subtype)
            if mapped is None:
                return None
            return (
                mapped,
                _point_of(action.get("mouse_position")),
                _bbox_of(action.get("clickable_area")),
            )

        if kind == "KeyboardAction":
            subtype = action.get("keyboard_action_type", "")
            mapped = _KEYBOARD_ACTIONS.get(subtype)
            return (mapped, None, None) if mapped else None

        if kind == "WaitAction":
            return ActionType.WAIT, None, None

        # PlanAction / EvaluateSubTaskAction：不是键鼠动作，但确实是数据集
        # 标注的一部分（ScreenAgent 标的是"规划-执行-评估"三段循环）。
        # 保留成 OTHER，动作类型分布那张图才如实反映这个结构——这恰好也是
        # M4 Reflector 的设计参考。
        if kind in ("PlanAction", "EvaluateSubTaskAction"):
            return ActionType.OTHER, None, None

        return None
