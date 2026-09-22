"""动作生成评测 —— 大纲 W5「对比微调前后模型在 GUI 任务理解与动作生成上的效果」。

## 判什么

一条样本给「截图 + 任务目标」，模型输出「动作类型 + 坐标」。判两件事：

1. **动作类型对不对**（`left_click` / `double_click` / `mouse_move`）
2. **坐标偏多远**

## 为什么不设单一的「成功率」

动作生成不像 grounding 那样有一个干净的判据。grounding 有真值 bbox，
点落进去就是对；而这里只有一个真值坐标点，**「差多少算对」是人定的**。

所以不报单一数字，报三组：

- **动作类型准确率** —— 无歧义
- **坐标误差的中位数** —— 无阈值，直接看分布
- **联合命中率 @ 半径 r** —— 动作类型对且坐标在 r 内，**r 是显式参数**

一个能随阈值变号的指标不该被当成结论。M2 的稳定性测试栽过这个跟头：
句柄斜率 +9.9 而阈值 10.0，差 0.1 就翻转结论。所以这里把阈值摆在明面上，
并同时报几个不同的 r。

## 按动作类型分开报

训练集里 `left_click` 占 75.9%。**只报总体准确率会被它主导**——
模型全输出 `left_click` 就能拿 76%，而那说明不了任何事。

## 微调前后怎么比

    # 微调前（base 模型零样本）
    python -m eval.action --local Qwen/Qwen2.5-VL-3B-Instruct --tag before

    # 微调后
    python -m eval.action --local Qwen/Qwen2.5-VL-3B-Instruct \\
        --adapter finetune/outputs/<run>/adapter --tag after

    # 汇总对比
    python -m eval.action --report

**两次必须用同一套提示词、同一个坐标空间、同一个验证集。**
微调前的基线必须是自己跑出来的实测值，不能引用官方模型卡数字——
拿官方数字当基线、自测数字当结果，是无效对比。

也支持 API 后端做参照：

    python -m eval.action --provider dashscope --tag qwen-api
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

#: 结果落点，一个模型一个文件。
RESULT_DIR = Path("docs/m3-action")

#: 默认验证集，由 `python -m finetune.dataset` 生成。
VAL_FILE = Path("finetune/data/val.jsonl")

#: 联合命中的半径，归一化坐标单位（空间是 1000×1000）。
#: 50 ≈ 屏幕宽度的 5%，在 1920 宽的屏上约 96px——大致是一个按钮的尺度。
DEFAULT_RADII = (25, 50, 100)

#: 系统提示词里给出的动作空间。输出这之外的动作即为不合规。
#: 输出里允许出现的动作。**2026-08-26 从 4 个扩到 8 个**——训练集放宽后
#: 模型会输出 `type` / `key` / `wait` / `scroll`，而它们此前不在这个集合里，
#: 于是每一条都被 `format_compliant` 判为不合规。
#:
#: 实测：`--limit 20` 那批格式合规率报 70%，正好等于 20 条里坐标类的 14 条
#: ——**非坐标动作的输出被全部判成格式错误**，而它们其实完全合规。
ALLOWED_ACTIONS = frozenset(
    {
        "left_click",
        "double_click",
        "right_click",
        "mouse_move",
        "type",
        "key",
        "wait",
        "scroll",
    }
)

#: 涉及坐标的动作。只有它们才要求输出里有整数 x/y。
_NEEDS_XY = frozenset({"left_click", "double_click", "right_click", "mouse_move"})


def format_compliant(raw: str) -> bool:
    """输出是否合规：合法 JSON + 动作在允许集合内 + 带整数 x/y。

    **要和「动作类型对不对」分开统计。** 基座模型的实测输出长这样：

        {"action": "left_click", "x": 682, 345}}      少了 "y" 键
        {"action": "left_click", "x": [527, 134]}     坐标塞进数组
        {"action": "type", "text": "Hello World!"}    动作空间里没有的动作

    微调之后准确率一定会涨，而涨的那部分里有多少是「学会了格式」、
    多少是「学会了点哪」——不分开就说不清。报告里写「动作准确率从 50%
    提到 X%」，读的人会以为是空间能力提升，实际可能大半是 JSON 格式。

    解析器有兜底（能从畸形 JSON 里抠出坐标），所以**合规率低不等于
    分数低**——这两件事必须分别报。
    """
    try:
        payload = json.loads((raw or "").strip())
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(payload, dict):
        return False
    # `done` 是完成信号不是动作，输出里没有 `action` 字段。
    # 它的合规形态就是 `{"done": true}`——不单独放行的话，
    # 验证集里占 41.6% 的那一类会被全部判成格式错误。
    action = payload.get("action")
    if action is None:
        return payload.get("done") is True
    if action not in ALLOWED_ACTIONS:
        return False
    # **只有涉及坐标的动作才要求 x/y。** 无条件要求的话，
    # `{"action": "type", "text": "你好"}` 这种完全合规的输出会被判错。
    if action in _NEEDS_XY:
        return isinstance(payload.get("x"), int) and isinstance(payload.get("y"), int)
    return True


def parroted(raw: str, few_shot: list) -> bool:
    """输出是否逐字照抄了某条 few-shot 示例。

    只做**精确匹配**（去空白后），不做模糊匹配：模糊匹配会把「学到了
    示例的格式」误判成「抄了示例的答案」，而前者正是 few-shot 该起的作用。
    宁可漏报，不可误报——这个数字要进报告。
    """
    if not few_shot:
        return False
    answer = (raw or "").strip()
    return any(answer == str(e.get("output", "")).strip() for e in few_shot)


@dataclass
class Prediction:
    sample_id: str
    instruction: str
    truth_action: str = ""
    #: 真值坐标。**只有涉及坐标的动作才有**——2026-08-26 训练集放宽后，
    #: 验证集里 `type` / `key` / `done` / `wait` / `scroll` 都没有坐标。
    #: 此前这里无条件读 `row["point_norm"]`，新数据一进来就 KeyError。
    truth_xy: list = field(default_factory=list)
    pred_action: str = ""
    pred_xy: list = field(default_factory=list)
    #: 坐标之外的关键参数。`type` 看 text，`key` 看 keys，`scroll` 看 direction。
    #: **不看这个的话，「知道要打字」和「打对了字」是一回事**——而后者才是
    #: 本轮补 216 条 type 样本要解决的问题。
    truth_params: dict = field(default_factory=dict)
    pred_params: dict = field(default_factory=dict)
    #: 关键参数是否一致。不涉及参数的动作恒为 True。
    params_ok: bool = False
    action_ok: bool = False
    #: 输出是否合规：能解析成 JSON、动作在允许集合内、带 x/y 两个整数。
    #: **与 action_ok 分开记**，理由见 summarize 的说明。
    format_ok: bool = False
    distance: float = -1.0
    error: str = ""
    latency_ms: float = 0.0
    raw: str = ""
    #: 输出是否**逐字照抄了某条 few-shot 示例**。
    #:
    #: 2026-08-24 实测撞见的：3B 会把示例的答案原样吐出来，坐标是示例里的
    #: 常数。这种输出**格式 100% 合规、动作类型可能也对**，只有坐标是错的
    #: ——`format_ok` 和 `action_ok` 一个都拦不住它。
    #: 提示词消融必须单独量这一项，否则 few-shot 看起来只有好处。
    parroted: bool = False


#: 每个动作除坐标外，**哪个参数决定成败**。
#:
#: 2026-08-26 训练集从 3 种动作放宽到 8 种后加的。此前所有动作都涉及坐标，
#: 「命中」只需比坐标；现在过半样本没有坐标，得另有判据。
#:
#: 选哪个参数是有取舍的：
#:
#:   type    看 text        打字打错了字，动作类型再对也没用
#:   key     看 keys        按错键同理
#:   scroll  看 direction   方向错了就是反向滚
#:   wait    不看           秒数差一点不影响任务成败
#:   done    不看           它没有参数，发出来就算对
#:   点击类   不看           它们的成败由坐标距离判定
#: 值是 (真值里的键名, 模型输出里的键名)。**两边不一样，这是坑。**
#:
#: `val.jsonl` 的 `params` 用装载器的词表（`key` 单数，来自 ScreenAgent 的
#: `keyboard_key`）；模型输出经 `llm.parsing` 归一化后用的是执行器的词表
#: （`keys` 复数，`PARAM_ALIASES` 把 `key` 映射过去）。
#:
#: 混用的后果实测过：5 条 `key` 样本类型准确率 100%、命中率 0%——
#: 真值侧按 `keys` 去取，取到空，于是永远和预测对不上。
KEY_PARAM = {
    "type": ("text", "text"),
    "key": ("key", "keys"),
    "scroll": ("direction", "direction"),
}

#: 涉及坐标的动作。与 `data.schema.ACTION_REQUIREMENTS` 里标了 `"point"` 的一致。
COORD_ACTIONS = frozenset({"left_click", "double_click", "right_click", "mouse_move"})


def key_params(action: str, params: dict, *, predicted: bool = False) -> dict:
    """取出该动作的关键参数，归一化后用于比对。

    `predicted=True` 时按**模型输出**的词表取键，否则按**真值**的词表取
    （两者不同，见 `KEY_PARAM`）。返回时统一用一个中性键名，
    这样真值与预测才能直接相等比较。

    **归一化是必需的**：模型可能输出 `"Enter"` 而真值是 `"enter"`，
    或者文本首尾多个空格。不归一化的话这类差异会被记成错误，
    而它们在执行时其实等价（`control.executor` 会 `.lower()` 再拆分）。
    """
    pair = KEY_PARAM.get(action)
    if not pair:
        return {}
    name = pair[1] if predicted else pair[0]
    value = params.get(name)
    if value is None:
        return {}
    return {"value": str(value).strip().lower()}


def parse_params(raw: str) -> dict:
    """从模型输出里抠出参数。解析不出来返回空 dict。

    走 `llm.parsing.parse_action_payload` —— **与推理时同一套解析**。
    评测自己写一套的话，量到的就不是模型在真实回路里的表现。
    """
    try:
        from llm.parsing import parse_action_payload

        return parse_action_payload(raw).get("params") or {}
    except Exception:  # noqa: BLE001 —— 解析失败等同于没有参数
        return {}


def is_hit(record, radius: float) -> bool:
    """这条样本算不算命中。

    **判据随动作类型变，因为「做对了」的含义本来就不同：**

        点击类   动作类型对 **且** 坐标落在半径内
        其他     动作类型对 **且** 关键参数一致

    统一用坐标判据的话，`type` / `done` 这些没有坐标的样本会全部记成失败；
    统一用类型判据的话，点击到屏幕另一头也算对。两种都不成立。
    """
    if not record.action_ok:
        return False
    if record.truth_action in COORD_ACTIONS:
        return 0 <= record.distance <= radius
    return record.params_ok


def load_val(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"找不到验证集 {path}。先跑 python -m finetune.dataset")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _done_ids(path: Path) -> set[str]:
    """已完成的 sample_id。坏行跳过——断电时最后一行天然可能残缺。"""
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["sample_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _screenshot_of(record: dict, scale: float = 1.0):
    """把验证集的图读成 Screenshot。``scale`` 与本地路径的 `downscale` 同义。

    **API 路径也必须吃这个参数。** 少了它，`--image-scale` 在 `--provider`
    下会被静默忽略，四档消融跑出四组一模一样的数——而那看起来正是一个
    干净的「分辨率无影响」结论。
    """
    import cv2
    import numpy as np

    from perception.capture import Screenshot
    from perception.types import BBox

    # np.fromfile 而不是 cv2.imread：后者在中文路径上会静默返回 None
    buffer = np.fromfile(record["image"], dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"读不出图片：{record['image']}")
    if scale < 1.0:
        height, width = image.shape[:2]
        image = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    height, width = image.shape[:2]
    return Screenshot(image=image, region=BBox(0, 0, width, height), engine="dataset")


def downscale(image, scale: float):
    """把图缩到 ``scale`` 倍，供分辨率消融用。``scale >= 1.0`` 时原样返回。

    ## 为什么只降不升

    ScreenAgent 的截图**原生就是 1024×768**——训练集 543 条、验证集 142 条
    无一例外。放大它不产生任何细节，只会把插值出来的模糊像素喂给模型，
    测出来的是「模型怕不怕插值」，不是「模型吃不吃分辨率」。

    ## 真值不用跟着变

    真值坐标存的是**归一化的 1000 空间**（`point_norm`），与图片像素尺寸
    无关；模型也被要求在 1000 空间作答（`SessionConfig.coordinate_space`）。
    所以缩图这件事**不动真值、不动输出口径**，差值干净地归给分辨率。

    这是能在验证集上做这个消融的前提。若真值存的是像素坐标，
    缩图就必须同步缩真值，任何一处忘了同步都会让整组数据静默地错掉。

    ## 用 LANCZOS 而不是最近邻

    降采样要的是「看起来像一张低分辨率截图」，不是「丢掉四分之三的像素」。
    最近邻会产生真实缩放不会有的锯齿，那是另一种伪影。
    """
    if scale >= 1.0:
        return image
    from PIL import Image as _Image

    width, height = image.size
    return image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        _Image.LANCZOS,
    )


def spread_by_session(rows: list) -> list:
    """把样本按 session 轮流取，供 `--limit` 用。

    ## 为什么不能直接切前 N 条

    `val.jsonl` **是按 session_id 排序的**——同一个 session 的十几条样本
    连在一起。切前 20 条拿到的不是 20 个样本，是 **6 个 session**。

    2026-08-25 的提示词消融就栽在这上面。`--limit 20` 那批：

        P2 格式合规   前 20 条 5/20 = 25%     后 122 条 2/122 = 1.6%
        P3 格式合规   前 20 条 7/20 = 35%     后 122 条 2/122 = 1.6%

    前 20 条里恰好有一两个 session 的界面模型能应付，而它们占了那批的
    三分之一，于是「few-shot 的效果」被放大了约 15 倍——**还配上了
    一个 p=0.047 的显著性**。跑满 142 条才发现真值是 4.9%。

    探路样本的用途正是「先看个大概」，它给出系统性偏高的数就比不跑更糟。

    ## 轮流取而不是随机打散

    随机能降低偏差但不保证覆盖；轮流取保证 N 条来自 min(N, session 数)
    个不同 session——**N=20 就是 20 个不同 session**，这是能拿到的最好
    覆盖。同时它是确定性的，同一份 val 跑两次选中的是同一批。

    ## 2026-08-26：光按 session 铺开还不够

    上面那个修复只解决了「不同 session」，没解决「session 内部的位置」。
    实现是 `buckets[key].pop(0)`——第一轮取每个 session 的**第 0 条**，
    而验证集有 43 个 session，`--limit 20` 就只会取到第 0 条。

    **动作类型和位置强相关**：`done` 是一段动作序列的最后一个，永远不在
    第 0 位。实测 `--limit 20` 抽出来的 20 条里 `done` 是 **0 条**，
    而全集里它占 **41.6%**（156/375）——随机抽中 0 条的概率约十万分之三。

    于是探路样本再一次给出了系统性偏差的数：它完全没测到占四成的那类动作。

    所以现在**先按动作类型分层**，层内再按 session 轮流取。前者保证
    动作分布与全集一致，后者保证同一动作的样本来自不同 session。
    """
    from collections import OrderedDict

    def _round_robin(subset: list) -> list:
        buckets: OrderedDict[str, list] = OrderedDict()
        for row in subset:
            buckets.setdefault(row.get("session_id", ""), []).append(row)
        out = []
        while buckets:
            for key in list(buckets):
                out.append(buckets[key].pop(0))
                if not buckets[key]:
                    del buckets[key]
        return out

    strata: OrderedDict[str, list] = OrderedDict()
    for row in rows:
        strata.setdefault(row.get("action", ""), []).append(row)

    # 各层按比例交错：给每条样本一个 (层内序号 + 0.5) / 层大小 的位置，
    # 再按这个位置排序。**这样取前 N 条，各动作的占比严格接近全集。**
    #
    # 不用「每轮各层取 k 条」那种写法——`k = max(1, ...)` 会让小层每轮
    # 至少取 1 条，于是 `wait`（占 2.7%）在前 20 条里能拿到 2 条，
    # 过采样 4 倍。**那仍然是系统性偏差，只是方向反了**，
    # 而这个函数存在的全部理由就是不让探路样本给出偏差的数。
    ranked: list[tuple[float, int, dict]] = []
    for name, subset in strata.items():
        ordered = _round_robin(subset)
        size = len(ordered)
        for i, row in enumerate(ordered):
            ranked.append(((i + 0.5) / size, len(strata) - list(strata).index(name), row))
    # 位置相同时按层的大小排，让大层稍占先——否则小类会挤在最前面
    ranked.sort(key=lambda t: (t[0], -t[1]))
    return [row for _, _, row in ranked]


def evaluate(
    val_path: Path,
    tag: str,
    provider: str = "",
    model: str = "",
    local_model: str = "",
    adapter: str = "",
    limit: int = 0,
    resume: bool = True,
    load_4bit: bool = False,
    prompt: str = "",
    image_scale: float = 1.0,
) -> Path:
    """跑一档评测，逐条追加到 JSONL。

    ``prompt`` 给模板名（如 `executor_v1`）时用那套提示词与 few-shot；
    留空则用训练时的 `SYSTEM_PROMPT`（微调前后对比的默认口径）。
    提示词消融靠这个参数切换三档。
    """
    from finetune.train_lora import SYSTEM_PROMPT

    few_shot: list = []
    if prompt:
        from agent.prompts import load_template
        from agent.session import SessionConfig

        template = load_template(prompt)
        # **宽高用坐标系尺寸，不是图片尺寸。** 与 Agent 运行时同一口径，
        # 否则消融测的提示词和实际跑任务用的不是同一份。
        width, height = SessionConfig().coordinate_space
        system = template.render_system(width=width, height=height)
        few_shot = template.few_shot_pairs()
    else:
        system = SYSTEM_PROMPT

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / f"{tag}.jsonl"

    rows = load_val(val_path)
    if not resume and out.exists():
        # **`--no-resume` 必须清空文件，不能只是忽略已完成记录。**
        # 只忽略的话新结果会追加在旧结果后面，而 `summarize` 读整个文件
        # ——实测汇总报「样本 40」，前 20 条是用**改动前的代码**跑的，
        # 两批混在一起，所有比例都不可用。
        backup = out.with_suffix(".jsonl.bak")
        out.replace(backup)
        print(f"  --no-resume：旧结果已移到 {backup.name}，本次从零开始")
    done = _done_ids(out) if resume else set()
    todo = [r for r in rows if r["sample_id"] not in done]
    if limit:
        todo = spread_by_session(todo)[:limit]

    predict, closer, label = _build_predictor(
        provider, model, local_model, adapter, system, load_4bit, few_shot, image_scale
    )

    print("=" * 70)
    print("动作生成评测")
    print("=" * 70)
    print(f"  标签      {tag}")
    print(f"  模型      {label}")
    print(f"  提示词    {prompt or '训练时的 SYSTEM_PROMPT'}    few-shot {len(few_shot)} 条")
    print(f"  验证集    {val_path}    共 {len(rows)}，已完成 {len(done)}，本次 {len(todo)}")
    if image_scale < 1.0:
        print(f"  送图缩放  ×{image_scale}    （分辨率消融，真值是归一化坐标，不随缩放变）")
    print(f"  结果      {out}（逐条追加，可断点续跑）")
    print()
    if not todo:
        print("  没有待跑样本。加 --no-resume 可重跑全部。")
        return out

    hits = 0
    try:
        with out.open("a", encoding="utf-8") as handle:
            for index, row in enumerate(todo, start=1):
                record = Prediction(
                    sample_id=row["sample_id"],
                    instruction=row["instruction"][:120],
                    truth_action=row["action"],
                    # **用 get 而不是下标。** 放宽后的验证集里过半样本没有坐标。
                    truth_xy=list(row.get("point_norm") or []),
                    truth_params=key_params(row["action"], row.get("params") or {}),
                )
                started = time.perf_counter()
                try:
                    raw, action, point = predict(row)
                    record.raw = raw[:200]
                    record.pred_action = action
                    if point is not None:
                        record.pred_xy = [point.x, point.y]
                        if record.truth_xy:
                            record.distance = round(math.dist(record.truth_xy, record.pred_xy), 2)
                    record.pred_params = key_params(action, parse_params(raw), predicted=True)
                    record.action_ok = action == row["action"]
                    record.params_ok = record.pred_params == record.truth_params
                    record.format_ok = format_compliant(raw)
                    record.parroted = parroted(raw, few_shot)
                except Exception as exc:  # noqa: BLE001 —— 单样本失败不中断整档
                    record.error = f"{type(exc).__name__}: {exc}"[:200]
                record.latency_ms = round((time.perf_counter() - started) * 1000, 1)

                if is_hit(record, DEFAULT_RADII[1]):
                    hits += 1
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
                handle.flush()

                if index % 10 == 0 or index == len(todo):
                    print(
                        f"  [{index}/{len(todo)}] 联合命中@50 {hits}/{index} = {hits / index:.1%}"
                    )
    except KeyboardInterrupt:
        print("\n  已中断。已完成的结果都在文件里，重跑会自动续上。")
    finally:
        closer()

    return out


def _build_predictor(
    provider: str,
    model: str,
    local_model: str,
    adapter: str,
    system: str,
    load_4bit: bool = False,
    few_shot: list | None = None,
    image_scale: float = 1.0,
):
    """返回 (predict, close, label)。

    **API 与本地两条路共用同一个提示词和同一套解析。** 分成两套的话，
    「微调前 vs 微调后」的差值里会混进提示词与解析差异，那就不是模型的差值了。
    """
    from grounding.local_vlm import parse_point

    def extract(raw: str):
        """从模型输出里抠出 (动作名, 坐标)。

        **`done` 不是动作，是完成信号**，输出里没有 `action` 字段
        （`{"done": true}`）。训练集里它占 41.6%，认不出来的话这一大块
        会全部记成「动作类型错」，而模型其实答对了。
        """
        action = ""
        try:
            payload = json.loads(raw.strip())
            if isinstance(payload, dict):
                action = str(payload.get("action", "")).strip()
                if not action and payload.get("done"):
                    action = "done"
        except (json.JSONDecodeError, TypeError):
            if '"done"' in raw and "true" in raw.lower():
                action = "done"
            else:
                for name in ("double_click", "right_click", "left_click", "mouse_move"):
                    if name in raw:
                        action = name
                        break
        return action, parse_point(raw)

    if local_model:
        import torch
        from PIL import Image
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(local_model)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        kwargs = {"dtype": dtype, "device_map": "auto"}
        if load_4bit:
            # **与训练时的加载方式对齐。** 训练用的是 4-bit NF4，评测若用
            # bf16，权重光是 6.2GB 就塞不进 8GB 卡，device_map="auto" 会把
            # 一部分层卸到内存，每次前向都要在 CPU↔GPU 之间搬数据——慢好几倍。
            #
            # 数值上两者不完全等价，所以**同一组对比里必须前后一致**：
            # before 用 bf16、after 用 4-bit 的话，差值里就混进了量化的影响。
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
        net = Qwen2_5_VLForConditionalGeneration.from_pretrained(local_model, **kwargs)
        if adapter:
            from peft import PeftModel

            net = PeftModel.from_pretrained(net, adapter)
        net.eval()

        def predict(row: dict):
            image = downscale(Image.open(row["image"]).convert("RGB"), image_scale)
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system}]},
            ]
            # few-shot 的 input 是截图的**文字描述**而非真图（见
            # executor_v1.yaml 的说明），所以这里全是纯文本，不进 images。
            # 顺序必须是 系统 → few-shot → 当前样本，与 Agent 运行时一致。
            for example in few_shot or []:
                messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": str(example.get("input", ""))}],
                    }
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": str(example.get("output", ""))}],
                    }
                )
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": row["instruction"]},
                    ],
                }
            )
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(text=[text], images=[image], return_tensors="pt").to(net.device)
            with torch.no_grad():
                # 动作生成是确定性任务，不采样
                generated = net.generate(**inputs, max_new_tokens=64, do_sample=False)
            raw = processor.decode(
                generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            action, point = extract(raw)
            return raw, action, point

        return predict, lambda: None, f"本地 {local_model}" + (f" + {adapter}" if adapter else "")

    from llm.openai_compat import OpenAICompatBackend
    from llm.providers import load_dotenv_if_present, resolve

    load_dotenv_if_present()
    config = resolve(provider or "dashscope", model=model or None)
    backend = OpenAICompatBackend(
        config=config, system_prompt=system, few_shot=list(few_shot or [])
    )

    def predict(row: dict):
        from llm.base import LLMBackendError

        try:
            intent = backend.predict_action(row["instruction"], _screenshot_of(row, image_scale))
        except LLMBackendError as exc:
            # **解析失败是模型的失败，不是调用的失败。**
            #
            # 本地路径遇到解析不出来的输出时不抛异常——`extract()` 静默
            # 返回 None，那一条留在分母里、记 `format_ok=False`。API 路径
            # 原来是抛出去的，于是同一种失败在两条路上被记成了两件事：
            #
            #     本地：留在分母里，拉低格式合规率
            #     API ：从分母里剔掉，格式合规率反而变好看
            #
            # 2026-08-25 的 8B 评测上，这让格式合规率从 38.7% 变成 45.8%、
            # 联合命中从 30.3% 变成 35.8%——而它正要跟本地的 142/142 比。
            #
            # 只吞 `parse_error`。网络超时、限流、余额不足仍然抛出去记成
            # error：那些**确实**不是模型的失败，剔出分母才对。
            if exc.kind != "parse_error":
                raise
            return exc.raw, "", None
        raw = intent.raw_text or ""
        action = intent.action_type or ""
        point = None
        x, y = intent.params.get("x"), intent.params.get("y")
        if x is not None and y is not None:
            from perception.types import Point

            point = Point(int(x), int(y))
        if point is None:
            _, point = extract(raw)
        return raw, action, point

    return predict, backend.close, f"{config.provider.label} / {config.model}"


def summarize(path: Path, radii: tuple = DEFAULT_RADII) -> dict:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    valid = [r for r in rows if not r["error"]]
    with_xy = [r for r in valid if r["distance"] >= 0]
    distances = sorted(r["distance"] for r in with_xy)

    stats: dict = {
        "tag": path.stem,
        "total": len(rows),
        "errors": len(rows) - len(valid),
        "no_coords": len(valid) - len(with_xy),
        "action_acc": (sum(r["action_ok"] for r in valid) / len(valid)) if valid else 0.0,
        "format_acc": (sum(r.get("format_ok", False) for r in valid) / len(valid))
        if valid
        else 0.0,
        "median_distance": distances[len(distances) // 2] if distances else -1.0,
        # **照抄率要单独报。** 逐字抄 few-shot 的输出格式 100% 合规、
        # 动作类型也可能对，只有坐标是示例里的常数——format_acc 和
        # action_acc 一个都拦不住它。不单独量，few-shot 看起来只有好处。
        "parrot_rate": (sum(r.get("parroted", False) for r in valid) / len(valid))
        if valid
        else 0.0,
        "joint": {},
        "by_action": {},
        "median_latency_ms": 0.0,
    }
    # **两套口径都要报，因为它们回答的不是同一个问题。**
    #
    #   joint        坐标类样本的命中率。分母只含带坐标的样本，
    #                与 2026-08-26 之前的数字**同口径**，可以纵向比。
    #   overall_hit  全部样本的命中率（坐标类看距离，其余看关键参数）。
    #                这是放宽后的新指标，**不能和旧数并列**。
    #
    # 混成一个数的话，`type` / `done` 这些没有坐标的样本会把分母撑大而
    # 分子不动，看起来像是模型退步了——而那纯粹是口径变了。
    coord_rows = [r for r in valid if r["truth_action"] in COORD_ACTIONS]
    for radius in radii:
        hit = sum(1 for r in coord_rows if r["action_ok"] and 0 <= r["distance"] <= radius)
        stats["joint"][str(radius)] = hit / len(coord_rows) if coord_rows else 0.0
    stats["coord_n"] = len(coord_rows)

    def _hit(r: dict, radius: float) -> bool:
        if not r["action_ok"]:
            return False
        if r["truth_action"] in COORD_ACTIONS:
            return 0 <= r["distance"] <= radius
        return r.get("params_ok", False)

    stats["overall_hit"] = sum(1 for r in valid if _hit(r, radii[1])) / len(valid) if valid else 0.0

    latencies = sorted(r["latency_ms"] for r in rows if r["latency_ms"])
    if latencies:
        stats["median_latency_ms"] = latencies[len(latencies) // 2]

    buckets: dict = defaultdict(lambda: {"n": 0, "action_ok": 0, "hit": 0, "dists": []})
    for row in valid:
        bucket = buckets[row["truth_action"]]
        bucket["n"] += 1
        bucket["action_ok"] += row["action_ok"]
        bucket["hit"] += _hit(row, radii[1])
        if row["distance"] >= 0:
            bucket["dists"].append(row["distance"])
    for name, bucket in buckets.items():
        dists = sorted(bucket["dists"])
        stats["by_action"][name] = {
            "n": bucket["n"],
            "action_acc": bucket["action_ok"] / bucket["n"],
            # **命中率与类型准确率必须分开看。** 对 `type` 来说，
            # 「知道该打字」和「打对了字」是两回事——前者是 action_acc，
            # 后者才是 hit_rate，而本轮补 216 条 type 样本要解决的是后者。
            "hit_rate": bucket["hit"] / bucket["n"],
            "median_distance": dists[len(dists) // 2] if dists else -1.0,
        }

    # **`done` 误报率单独算。** 该继续时喊停 = 任务直接失败，比漏报更糟。
    # 而它在训练集里占 41.6%，模型只要学会多喊 done，总体命中率就会涨——
    # 不单独量这一项，那种退步在汇总数字上完全看不出来。
    not_done = [r for r in valid if r["truth_action"] != "done"]
    if not_done:
        stats["done_false_positive"] = sum(1 for r in not_done if r["pred_action"] == "done") / len(
            not_done
        )
    return stats


def print_summary(stats: dict) -> None:
    print()
    print("=" * 70)
    print(f"汇总 {stats['tag']}")
    print("=" * 70)
    print(f"  样本 {stats['total']}    调用失败 {stats['errors']}    没给坐标 {stats['no_coords']}")
    print(f"  **输出格式合规率 {stats['format_acc']:.1%}**（合法 JSON + 动作在集合内 + 整数 x/y）")
    print(f"  **动作类型准确率 {stats['action_acc']:.1%}**")
    print(f"  坐标误差中位数   {stats['median_distance']:.1f}（归一化 1000 空间）")
    parrot = stats.get("parrot_rate", 0.0)
    if parrot:
        print(f"  **逐字照抄 few-shot 示例 {parrot:.1%}** —— 这些输出格式合规、")
        print("  坐标却是示例里的常数。格式与动作类型两个指标都拦不住它。")
    coord_n = stats.get("coord_n")
    label = f"，分母 {coord_n} 条坐标类样本" if coord_n is not None else ""
    print(f"  联合命中（动作类型对 且 坐标在半径内{label}）：")
    for radius, value in stats["joint"].items():
        print(f"    r={radius:<5}{value:>7.1%}")
    if "overall_hit" in stats:
        print(f"  **全量命中率 {stats['overall_hit']:.1%}**（坐标类看距离，其余看关键参数）")
        print("  ↑ 放宽后的新指标，**与 2026-08-26 之前的数字口径不同，不可并列**。")
        print("    上面那组 joint 才是同口径的那一列。")
    fp = stats.get("done_false_positive")
    if fp is not None:
        print(f"  **done 误报率 {fp:.1%}** —— 真值不是 done 却输出了 done。")
        print("  该继续时喊停 = 任务直接失败，比漏报更糟。训练集里 done 占 41.6%，")
        print("  模型多喊几次总体命中率就会涨，不单独量这一项就看不出那是退步。")
    print(f"  中位延迟 {stats['median_latency_ms']:.0f}ms")
    if stats["by_action"]:
        print(
            f"\n  {'真值动作':<16}{'样本':>6}{'类型准确率':>12}{'命中率':>10}{'坐标误差中位数':>16}"
        )
        for name, v in sorted(stats["by_action"].items(), key=lambda kv: -kv[1]["n"]):
            dist = f"{v['median_distance']:.1f}" if v["median_distance"] >= 0 else "—"
            print(
                f"  {name:<16}{v['n']:>6}{v['action_acc']:>12.1%}"
                f"{v.get('hit_rate', 0):>10.1%}{dist:>16}"
            )
        print("\n  **按动作类型分开看，且类型准确率与命中率要分开。**")
        print("  验证集里 done 占 41.6%，只报总体会被它主导；而对 type 来说，")
        print("  「知道该打字」（类型准确率）和「打对了字」（命中率）是两回事。")


def build_parser() -> argparse.ArgumentParser:
    """抽成函数是为了让测试能检查参数表。

    曾经在 `serve_local_model.py` 上栽过：`main()` 读 `args.max_new_tokens`
    而那个参数**从没注册过**——权重加载完几十秒之后才 AttributeError。
    参数表拿不到就测不了这种漏接。
    """
    parser = argparse.ArgumentParser(description="动作生成评测（大纲 W5 微调前后对比）")
    parser.add_argument("--val", default=str(VAL_FILE))
    parser.add_argument("--tag", default="", help="这一档的名字，作为结果文件名")
    parser.add_argument("--provider", default="", help="走 API 后端，如 dashscope")
    parser.add_argument("--model", default="", help="覆盖平台默认模型")
    parser.add_argument(
        "--prompt",
        default="",
        help="提示词模板名（executor_v0 / v1 / v2）。留空用训练时的 SYSTEM_PROMPT",
    )
    parser.add_argument(
        "--image-scale",
        type=float,
        default=1.0,
        help="送图前先缩到这个倍数（分辨率消融用）。>=1.0 表示原样，只降不升",
    )
    parser.add_argument("--local", default="", help="走本地模型，给 HF 模型名或路径")
    parser.add_argument("--adapter", default="", help="LoRA adapter 目录，微调后评测用")
    parser.add_argument(
        "--load-4bit",
        action="store_true",
        help="按 4-bit NF4 加载，与训练时一致。8GB 卡上能全放进显存，快好几倍。"
        "**同一组对比里前后必须一致**",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--report", action="store_true", help="只汇总已有结果，不调模型")
    return parser


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args()

    if args.report:
        files = sorted(RESULT_DIR.glob("*.jsonl"))
        if not files:
            print(f"{RESULT_DIR} 下没有结果文件。")
            return 1
        for path in files:
            print_summary(summarize(path))
        if len(files) >= 2:
            print()
            print("=" * 70)
            print("对比")
            print("=" * 70)
            rows = [summarize(p) for p in files]
            print(
                f"  {'档':<20}{'格式合规':>10}{'类型准确率':>12}{'误差中位数':>12}{'联合@50':>10}"
            )
            for r in rows:
                print(
                    f"  {r['tag']:<20}{r.get('format_acc', 0):>10.1%}"
                    f"{r['action_acc']:>12.1%}"
                    f"{r['median_distance']:>12.1f}{r['joint'].get('50', 0):>10.1%}"
                )
            print()
            print("  **格式合规与类型准确率要分开看。** 微调涨的那部分里，")
            print("  有多少是「学会了格式」、多少是「学会了点哪」，只有这两列能分辨。")
        return 0

    if not args.provider and not args.local:
        raise SystemExit("要么 --provider（API），要么 --local（本地模型）")
    tag = args.tag or (args.provider or Path(args.local).name)

    out = evaluate(
        val_path=Path(args.val),
        tag=tag,
        provider=args.provider,
        model=args.model,
        local_model=args.local,
        adapter=args.adapter,
        limit=args.limit,
        resume=not args.no_resume,
        load_4bit=args.load_4bit,
        prompt=args.prompt,
        image_scale=args.image_scale,
    )
    print_summary(summarize(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
