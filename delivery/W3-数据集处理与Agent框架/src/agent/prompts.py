"""提示词模板库 —— YAML 外置、版本化、可消融。

## 为什么外置成 YAML

M3 任务 3 要做提示词消融：三个版本（零样本基线 / +角色设定+few-shot /
+CoT 显式推理）各跑 45 次，比成功率。**如果提示词写死在代码里，"换一个
版本"就意味着改代码重跑**——那样的对比不成立，因为你没法保证除了提示词
之外别的都没变。

外置之后，消融是 `--prompt-version p2` 一个参数的事。

## 动作说明必须从 ACTION_SPECS 生成，不能手写在 YAML 里

M1 的 `control/actions.py` 开头立了规矩：

> JSON Schema 与参数校验都从这张表生成，保证"给模型看的"和"实际校验的"
> 永远是同一份定义。

如果在 YAML 里手抄一份动作列表，改了某个动作的参数名，YAML 不会跟着变。
模型照着过时的说明输出，然后在 `Action.validate` 那里被拒——而且报错
指向的是模型不听话，不是提示词过期。这种 bug 极难查。

所以 YAML 只放**叙述性内容**（角色设定、原则、few-shot 示例），动作清单
由 `render_action_reference()` 从 `ACTION_SPECS` 现场生成后注入。

## few-shot 为什么是必需项而不是优化项

M2 设计思路里那句话说得很直白：

> 开源模型的规划能力弱于闭源前沿模型，以下两条不是优化项，是能不能跑
> 起来的前提。

其中第二条就是"提示词内置 few-shot 示例"。因此模板格式里 `few_shot` 是
一等公民，不是可选附加。
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from control.actions import ACTION_SPECS, CORE_ACTIONS, ActionType

logger = logging.getLogger(__name__)

#: 模板目录。与代码同级而非包内，方便直接编辑与 diff
PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


class PromptError(RuntimeError):
    """模板缺失或格式不对。"""


@dataclass
class FewShotExample:
    """一组"输入 → 正确输出"示例。

    ``input`` 是**截图的文字描述**而非真实图片：附带真实图片会让每次
    请求多花几千 token，而 few-shot 在这里的作用是**示范输出格式与决策
    风格**，文字描述足够。真实的视觉信息由当前这一步的截图提供。
    """

    input: str
    output: str
    note: str = ""


@dataclass
class PromptTemplate:
    """一个版本化的提示词模板。"""

    name: str
    version: str = "v1"
    description: str = ""
    system: str = ""
    user_template: str = ""
    few_shot: list[FewShotExample] = field(default_factory=list)
    #: 该版本启用了哪些特性，消融报告里按这个分组
    features: list[str] = field(default_factory=list)
    source_path: str = ""

    def render_system(self, allowed_actions: Sequence[str] | None = None, **kwargs: Any) -> str:
        """渲染系统提示。

        ``{action_reference}`` 是保留占位符，自动注入由 `ACTION_SPECS`
        生成的动作清单——见模块文档"不能手写"那一节。

        ``allowed_actions`` 限定只列这个模型真的会的动作，
        理由见 `render_action_reference`。
        """
        values = {
            "action_reference": render_action_reference(allowed=allowed_actions),
            **kwargs,
        }
        text = _safe_format(self.system, values, where=f"{self.name}.system")
        return _drop_rules_naming(text, allowed_actions)

    def render_user(self, **kwargs: Any) -> str:
        return _safe_format(self.user_template, kwargs, where=f"{self.name}.user_template")

    def few_shot_pairs(self, allowed_actions: Sequence[str] | None = None) -> list[dict]:
        """转成 `OpenAICompatBackend` 认的形状。

        ``allowed_actions`` 非空时，**丢掉输出用了不可用动作的示例**。

        不过滤的话，同一份提示词会一边说「你只有这四个动作」，一边给出
        一条 `{"action": "type", ...}` 的示例——而 2026-08-25 的实测表明
        **3B 更听示例的**：`allowed_actions` 的硬约束段明说了「不要拆成
        点开始→输入→回车」，规划器照样这么拆，因为它的 few-shot 就是
        那么写的。

        示例赢过约束，所以示例本身必须对。
        """
        kept = self.few_shot
        if allowed_actions:
            excluded = {t.value for t in CORE_ACTIONS} - {str(n) for n in allowed_actions}
            marks = tuple(f'"action": "{name}"' for name in excluded)
            kept = [e for e in kept if not any(m in e.output for m in marks)]
        return [{"input": e.input, "output": e.output} for e in kept]

    def as_dict(self) -> dict:
        """进轨迹日志。M3 消融要能追溯每条轨迹用的是哪个提示词版本。"""
        return {
            "name": self.name,
            "version": self.version,
            "features": list(self.features),
            "few_shot_count": len(self.few_shot),
        }


def _safe_format(template: str, values: dict, where: str) -> str:
    """按 ``{name}`` 占位符替换。

    不用 `str.format`：提示词里大量出现 JSON 示例（``{"action": ...}``），
    `format` 会把那些花括号当占位符，抛 KeyError 或吃掉内容。**而 JSON
    示例恰恰是这个项目提示词里最重要的部分**，不能要求写模板的人把每个
    花括号都转义成 ``{{``——那样模板会变得没法读。

    因此改成只替换**明确声明过的键**，其余花括号原样保留。
    """
    result = template
    for key, value in values.items():
        result = result.replace("{" + key + "}", str(value))

    leftover = [k for k in values if "{" + k + "}" in result]
    if leftover:  # pragma: no cover - 替换后不该还有残留
        logger.warning("%s 中的占位符 %s 未被替换", where, leftover)
    return result


# ---------------------------------------------------------------------- #
# 动作清单
# ---------------------------------------------------------------------- #


#: 列表项的开头：`- `、`1. `、`* ` 等。用来判断一条规则从哪儿开始。
_ITEM_START = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s")


def _blocks(lines: list[str]) -> list[list[str]]:
    """把行按「列表项 + 它的续行」分组。

    续行的判据是**缩进比列表项本身深**。模板里长规则就是这么写的：

        2. **看不清就先观察。** 如果界面还在加载、菜单还在展开，用
           `wait` 等一下，不要对着中间态乱点。

    不成组就没法整条删——见 `_drop_rules_naming` 的事故记录。
    """
    blocks: list[list[str]] = []
    for line in lines:
        starts_item = bool(_ITEM_START.match(line))
        indent = len(line) - len(line.lstrip())
        if blocks and not starts_item and line.strip():
            head = blocks[-1][0]
            if _ITEM_START.match(head) and indent > len(head) - len(head.lstrip()):
                blocks[-1].append(line)
                continue
        blocks.append([line])
    return blocks


def _drop_rules_naming(text: str, allowed: Sequence[str] | None) -> str:
    """删掉正文里点名了**不可用动作**的规则——整条删，连同它的续行。

    ## 为什么光过滤动作清单不够

    模板的「使用规则」那几条是手写的，里面直接点了动作名：

        3. **中文输入用 `type`。** 它内部走剪贴板，可以正确输入中文。

    `{action_reference}` 过滤之后 `type` 不在动作清单里了，**这一行还在**——
    于是同一份提示词一边说「你只有这四个动作」，一边说「中文输入用 type」。
    这正是本项目一直在防的那种自相矛盾（见模块文档「不能手写」那一节）。

    ## 为什么按「条」而不是按「行」——一次真实事故

    初版按行删。而模板里的规则 2 是**两行**：

        2. **看不清就先观察。** 如果界面还在加载、菜单还在展开，用
           `wait` 等一下，不要对着中间态乱点。

    只有第二行带 `wait`，于是删完剩下 **「…菜单还在展开，用」**——
    **一句话被砍成半截。**

    2026-08-25 实测：`close_app` 从 4/4 掉到 0/2，两轮都在第一次调用上
    `parse_error`。这个函数的初版文档里写着「改写留下的半句话比整行删掉
    更危险」，而它自己造出了一模一样的东西。

    n=2 证不了因果，但半截句子必须修——**提示词里不该有语法不完整的指令**。

    ## 删整条，不改写，不renumber

    编号会出现空档（1、2、4）。这是有意的：**看得出少了一条，比看不出强。**
    """
    if not allowed:
        return text
    excluded = {t.value for t in CORE_ACTIONS} - {str(name) for name in allowed}
    if not excluded:
        return text
    marks = tuple(f"`{name}`" for name in excluded)
    kept = [
        line
        for block in _blocks(text.splitlines())
        if not any(m in line for line in block for m in marks)
        for line in block
    ]
    return chr(10).join(kept)


def render_action_reference(only_core: bool = True, allowed: Sequence[str] | None = None) -> str:
    """从 `ACTION_SPECS` 生成给模型看的动作说明。

    与 `control.actions.to_tool_schemas` 同源，因此动作参数一改，提示词
    自动跟着变——不会出现"提示词说有这个参数、校验说没有"的错位。

    ## `allowed`：只列这个模型真的会的动作

    2026-08-25 查出的一处结构性错位：微调模型的训练集里
    `type` / `key` / `done` **各 0 条**（ScreenAgent 的键盘动作因不带坐标
    被整批过滤），而提示词照样告诉它「你可以 `type`」。

    后果是五个基础任务里四个需要打字，模型**连掷骰子的机会都没有**——
    而系统里没有任何一处发现这件事，25 轮跑完只看到一片 0/5。

    给 `allowed` 之后，提示词只列它真的会的动作。**这不会让模型变强，
    但会让它和规划器停止假装能做做不到的事。**

    传空或 None 表示不限制（默认行为，与从前一致）。
    """
    types = sorted(CORE_ACTIONS, key=lambda t: t.value) if only_core else list(ActionType)
    if allowed:
        wanted = {str(name) for name in allowed}
        kept = [t for t in types if t.value in wanted]
        unknown = wanted - {t.value for t in types}
        if unknown:
            # **不静默忽略。** 拼错一个动作名就等于悄悄放宽了限制，
            # 而那正是这个参数要防的事。
            raise PromptError(
                f"allowed 里有不存在的动作：{sorted(unknown)}；"
                f"可用的是 {sorted(t.value for t in types)}"
            )
        types = kept

    lines: list[str] = []
    for action_type in types:
        description, specs = ACTION_SPECS[action_type]
        if not specs:
            lines.append(f"- `{action_type.value}` —— {description}（无参数）")
            continue

        parts = []
        for spec in specs:
            marker = "" if spec.required else "，可选"
            enum_hint = f"，取值 {'/'.join(spec.enum)}" if spec.enum else ""
            parts.append(f"`{spec.name}`({spec.json_type}{marker}{enum_hint})")
        lines.append(f"- `{action_type.value}` —— {description}。参数：{', '.join(parts)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------- #
# 加载
# ---------------------------------------------------------------------- #


def load_template(name: str, directory: Path | str | None = None) -> PromptTemplate:
    """按名字加载模板。``name`` 形如 ``executor_v1``，对应 ``executor_v1.yaml``。"""
    directory = Path(directory) if directory else PROMPT_DIR
    path = directory / f"{name}.yaml"
    if not path.exists():
        available = ", ".join(sorted(p.stem for p in directory.glob("*.yaml"))) or "（无）"
        raise PromptError(f"找不到提示词模板 {path}。可用：{available}")

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - 依赖缺失
        raise PromptError("缺少 pyyaml，请先 pip install pyyaml") from exc

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise PromptError(f"{path} 的顶层必须是映射")

    examples = []
    for index, raw in enumerate(data.get("few_shot") or []):
        if not isinstance(raw, dict) or "input" not in raw or "output" not in raw:
            raise PromptError(f"{path} 的 few_shot[{index}] 必须同时含 input 与 output")
        examples.append(
            FewShotExample(
                input=str(raw["input"]).strip(),
                output=str(raw["output"]).strip(),
                note=str(raw.get("note", "")).strip(),
            )
        )

    return PromptTemplate(
        name=str(data.get("name", name)),
        version=str(data.get("version", "v1")),
        description=str(data.get("description", "")),
        system=str(data.get("system", "")).strip(),
        user_template=str(data.get("user_template", "")).strip(),
        few_shot=examples,
        features=[str(f) for f in (data.get("features") or [])],
        source_path=str(path),
    )


def list_templates(directory: Path | str | None = None) -> list[str]:
    """列出可用模板名。CLI 的 `config --show` 与消融脚本用。"""
    directory = Path(directory) if directory else PROMPT_DIR
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.yaml"))
