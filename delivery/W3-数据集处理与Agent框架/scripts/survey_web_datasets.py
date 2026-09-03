"""Mind2Web 与 WebArena 抽样查看 —— M2 任务 1 的"仅抽样"部分。

大纲要求这两个数据集**只抽样查看，不深度处理**，并在报告中说明理由。
本脚本抽样并把理由落成可复核的证据，而不是留一句"它们是网页数据"的断言。

## 抽样要回答的唯一问题

**它们能不能给 M3 的 grounding 训练提供监督信号？**

grounding 训练需要的是 `(截图, 元素描述) → 像素坐标`。所以抽样只查三件事：
有没有截图、有没有像素坐标、动作标在什么东西上。查完就收工。

不下载全量：Mind2Web 一个分片就足以看清标注结构，WebArena 的任务配置
文件本身只有几 MB。

## 用法

    python scripts/survey_web_datasets.py
"""

from __future__ import annotations

import collections
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "src")
)  # 交付包把各个包放在 src/ 下

MIND2WEB_SHARD = (
    "https://huggingface.co/datasets/osunlp/Mind2Web/resolve/main/data/train/train_0.json"
)
WEBARENA_TASKS = (
    "https://raw.githubusercontent.com/web-arena-x/webarena/main/config_files/test.raw.json"
)
OUT = Path("docs/网页数据集抽样查看.md")

#: 像素坐标字段的候选名。查的是"有没有"，所以宁可宽一点
_COORD_HINTS = ("bbox", "box", "coord", "point", "position", "rect", "x1", "screenshot")


def _fetch(url: str, timeout: int = 120):
    with urllib.request.urlopen(url, timeout=timeout) as handle:
        return json.loads(handle.read())


def _coordinate_fields(payload: dict) -> list[str]:
    return [k for k in payload if any(h in k.lower() for h in _COORD_HINTS)]


def survey_mind2web() -> dict:
    tasks = _fetch(MIND2WEB_SHARD)
    action = (tasks[0].get("actions") or [{}])[0]
    operations = collections.Counter(
        (a.get("operation") or {}).get("op") for task in tasks for a in (task.get("actions") or [])
    )
    return {
        "抽样规模": f"train_0 分片共 {len(tasks)} 条任务",
        "任务字段": list(tasks[0]),
        "动作字段": list(action),
        "疑似坐标 / 截图字段": _coordinate_fields(action) or ["无"],
        "动作类型": dict(operations.most_common()),
        "领域分布": dict(collections.Counter(t.get("domain") for t in tasks).most_common()),
        "示例任务": tasks[0].get("confirmed_task", ""),
    }


def survey_webarena() -> dict:
    tasks = _fetch(WEBARENA_TASKS)
    return {
        "抽样规模": f"全量任务配置共 {len(tasks)} 条",
        "任务字段": list(tasks[0]),
        "疑似坐标 / 截图字段": _coordinate_fields(tasks[0]) or ["无"],
        "站点分布": {
            "+".join(k): v
            for k, v in collections.Counter(tuple(t.get("sites") or []) for t in tasks).most_common(
                8
            )
        },
        "示例任务": tasks[0].get("intent", ""),
    }


def _render(mind2web: dict, webarena: dict) -> str:
    lines: list[str] = []
    add = lines.append

    add("# Mind2Web 与 WebArena 抽样查看")
    add("")
    add("> 由 `python scripts/survey_web_datasets.py` 生成。对应 M2 任务 1")
    add("> 中「仅抽样查看」的两个数据集。**本文件是生成物，改它没有意义，改脚本。**")
    add("")

    add("## 结论")
    add("")
    add("**两者都不深度处理，不进统一 schema。**")
    add("")
    add("理由不是「它们是网页数据」——那只是场景不对齐。真正的原因是**两者都")
    add("不提供像素坐标**，而这正是 M3 的 grounding 训练唯一需要的监督信号：")
    add("")
    add("| 数据集 | 动作标在什么上 | 有截图 | 有像素坐标 |")
    add("|---|---|---|---|")
    add(
        "| Mind2Web | **DOM 节点**（`pos_candidates` 从 `cleaned_html` 里挑） | HF 发行版没有 | **否** |"
    )
    add("| WebArena | 什么也没标——它是**给活的沙箱环境用的任务定义** | 否 | **否** |")
    add("")
    add("Mind2Web 那一格要说准确些：HuggingFace 上的 `osunlp/Mind2Web` 只有 HTML，")
    add("**截图在另行分发的 raw_dump 里**（需用 Globus 下载，ScreenAgent 的仓库")
    add("就是这么取的）。但即便取到截图，标注给的仍是 DOM 节点而不是像素框——")
    add("要拿它做 grounding，得先把 DOM 节点渲染回像素坐标，那是另一个工程，")
    add("且结果依赖渲染时的浏览器与视口尺寸。**不值得在 M3 那一周做。**")
    add("")
    add("换句话说，就算不介意平台不对齐，也没法直接从中构造出一条 grounding 训练样本。")
    add("两者的价值在方法论与提示词组织上（任务如何表述、成功如何判定），")
    add("这部分参考写进技术报告的「相关工作」章节。")
    add("")

    for title, payload in (("Mind2Web", mind2web), ("WebArena", webarena)):
        add(f"## {title}")
        add("")
        for key, value in payload.items():
            if isinstance(value, dict):
                rendered = "，".join(f"`{k}` {v}" for k, v in value.items())
            elif isinstance(value, list):
                rendered = "、".join(f"`{v}`" for v in value)
            else:
                rendered = str(value)
            add(f"- **{key}**：{rendered}")
        add("")

    add("## 一条附带发现")
    add("")
    add("WebArena 的 `eval` 字段给出了**程序化的成功判定**（检查 URL、页面文本、")
    add("或调用站点 API 核对状态）。M2 任务 8 要求「成功判定必须程序化、")
    add("不用人工目测」，这套做法可以直接借鉴——这是抽样查看真正的收获。")
    add("")

    return "\n".join(lines) + "\n"


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    print("[抽样] Mind2Web …")
    mind2web = survey_mind2web()
    print(f"        {mind2web['抽样规模']}，坐标字段：{mind2web['疑似坐标 / 截图字段']}")

    print("[抽样] WebArena …")
    webarena = survey_webarena()
    print(f"        {webarena['抽样规模']}，坐标字段：{webarena['疑似坐标 / 截图字段']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(_render(mind2web, webarena), encoding="utf-8", newline="\n")
    print(f"\n已写入 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
