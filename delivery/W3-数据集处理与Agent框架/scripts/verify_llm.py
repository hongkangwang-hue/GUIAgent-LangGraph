"""多模态 API 连通性验证 —— 补上 M1 D1 的那条欠账。

## 这个脚本对应哪条要求

《开发环境配置文档》第 246 行：

> **第 1 周 D1 必须跑通**（排在写任何代码之前，这是整条技术路线的能力
> 前提）：用真实桌面截图测候选 API，**同时验两件事**——模型能否看懂截图
> 并输出可解析的结构化结果，以及 `ChatOpenAI` + 平台 OpenAI 兼容
> `base_url` 能否正确传图。至少测两家。

两件事必须一起验。只验"模型聪明不聪明"而不验传图通道，等到 M2 主链路
每步都要传图时才发现构造方式不对，那时改动面就大了。

## 用法

    python scripts/verify_llm.py                    # 测所有已配 key 的平台
    python scripts/verify_llm.py --provider zhipu   # 只测一家
    python scripts/verify_llm.py --image a.png      # 用指定图片而非现场截屏
    python scripts/verify_llm.py --repeat 3         # 每家跑 3 次看稳定性

**这个脚本会真的花钱。** 每次调用都是一次计费请求，因此它不进 CI，
默认每家只跑一次。

## 输出

结果写进 ``docs/m1-d1-api-verification.md``，那是可以直接交给导师的证据：
哪家通了、原始输出长什么样、解析成不成功、花了多少 token。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "src")
)  # 交付包把各个包放在 src/ 下

# Windows 控制台默认 cp936（GBK），而**模型输出里带 emoji 是常态**——
# 实测百炼回过 "OK! 😊"，U+1F60A 不在 GBK 码表里，print 直接抛
# UnicodeEncodeError。本脚本的全部价值就是把模型的真实输出打给人看，
# 它不能因为对方多打了个表情就崩掉。
#
# errors="replace" 而不是 "ignore"：编不出的字符要留个占位，让人知道
# 那里原本有东西，而不是无声消失。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from llm.base import LLMBackendError  # noqa: E402
from llm.openai_compat import OpenAICompatBackend  # noqa: E402
from llm.parsing import OutputParseError, parse_action_payload  # noqa: E402
from llm.providers import (  # noqa: E402
    PROVIDERS,
    ProviderNotConfigured,
    available_providers,
    load_dotenv_if_present,
    resolve,
)

#: M1 D1 的验收证据。**只有跑全部已配置平台时才写它。**
#:
#: 这个文件被覆盖过两次：`--provider zhipu` 单测一家，报告就被重写成
#: 「通过 1/1 家」，而 M1 D1 要求的是「至少测通两家」——交付物在无人
#: 察觉的情况下从达标变成不达标，靠 `git checkout` 才捞回来。
#:
#: 评测产物是跑一次贵一次的东西，默认行为不该是覆盖。
REPORT = Path("docs/m1-d1-api-verification.md")
#: 单平台补测的落点，按 `时间戳-平台.md` 另存，不碰 REPORT。
RUNS = Path("docs/m1-runs")

#: 当前正在测的截图。_call_with_retry 从这里取，避免多传一层参数
screenshot_holder: list = [None]

#: 给模型的验证任务。
#:
#: 刻意选一个**能程序化判对错**的问题：让它指出一个屏幕上必然存在、
#: 位置又大致可知的元素。问"你看到了什么"没法判分——模型胡诌一段也
#: 像模像样，而这个脚本的意义正是拿到可判定的证据。
PROBE_INSTRUCTION = (
    "这是一张桌面截图。请找到屏幕最上方的窗口标题栏或菜单栏，给出一个可以点击它的坐标。"
)

SYSTEM_PROMPT = """你是一个桌面 GUI 智能体。你会看到一张屏幕截图，需要决定下一步动作。

只输出一个 JSON 对象，不要任何解释文字，格式如下：

{"action": "动作名", "x": 横坐标, "y": 纵坐标, "thinking": "你的判断理由"}

可用动作：left_click / right_click / double_click / mouse_move / scroll / key / type / wait
坐标必须落在图片范围内，原点在左上角。"""


def grab_screenshot(path: str | None):
    """拿一张截图：优先用指定图片，否则现场截屏。"""
    import cv2
    import numpy as np

    from perception.capture import ScreenCapturer, Screenshot
    from perception.types import BBox

    if path:
        buffer = np.fromfile(path, dtype=np.uint8)  # 兼容中文路径
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"读不出图片：{path}")
        height, width = image.shape[:2]
        return Screenshot(image=image, region=BBox(0, 0, width, height), engine="file")

    with ScreenCapturer() as capturer:
        return capturer.capture(fresh=True)


def probe(provider_key: str, screenshot, repeat: int) -> dict:
    """测一家平台，返回可直接进报告的记录。"""
    record: dict = {"provider": provider_key, "runs": []}

    try:
        config = resolve(provider_key)
    except ProviderNotConfigured as exc:
        record["status"] = "未配置"
        record["error"] = str(exc)
        return record

    record["config"] = config.masked()
    backend = OpenAICompatBackend(config=config, system_prompt=SYSTEM_PROMPT)
    screenshot_holder[0] = screenshot

    for index in range(repeat):
        run: dict = {"index": index + 1}
        start = time.perf_counter()
        try:
            intent = _call_with_retry(backend, run)
        except LLMBackendError as exc:
            run.update(ok=False, error=str(exc), kind=exc.kind, retryable=exc.retryable)
            # 解析失败时原始输出仍有价值 —— 那正是判断"模型能不能用"的依据
            run["raw"] = getattr(exc.__cause__, "raw", "") if exc.__cause__ else ""
        else:
            run.update(
                ok=True,
                action=intent.action_type,
                params=intent.params,
                thinking=intent.thinking[:200],
                raw=intent.raw_text[:600],
                tokens=intent.usage.as_dict(),
                cost_cny=round(intent.cost_cny, 6),
                request_id=intent.request_id,
            )
            run["parsable"] = _reparse_ok(intent.raw_text)
            run["in_bounds"] = _in_bounds(intent.params, backend.image_size)
        run["latency_ms"] = round((time.perf_counter() - start) * 1000.0, 1)
        record["runs"].append(run)

        print(f"  [{index + 1}/{repeat}] {_one_line(run)}")

    record["image_meta"] = backend.last_image_meta
    record["cost"] = backend.get_cost().as_dict()
    record["status"] = _verdict(record["runs"])
    record["throttled"] = sum(1 for run in record["runs"] if run.get("retries"))
    backend.close()
    return record


def _verdict(runs: list[dict]) -> str:
    """判定这家平台是否通过。

    **限流不算能力失败。** 本脚本要回答的是"传图通道通不通、模型能不能
    输出可解析的结构化结果"，而 429 说明的是平台当前有多忙——免费档模型
    （glm-4.6v-flash 一类）被限流是常态，把它和"模型看不懂图"混成一个
    "失败"，会让 M1 D1 的结论失真。

    因此：只要有成功的调用、且没有非限流的失败，就算通过；限流次数
    单独记一列，它是 M3 选型时的吞吐参考，不是能力判据。
    """
    ok = [run for run in runs if run.get("ok")]
    if not ok:
        return "失败"
    hard_failures = [run for run in runs if not run.get("ok") and run.get("kind") != "transient"]
    if hard_failures:
        return "部分通过"
    return "通过" if len(ok) == len(runs) else "通过（有限流）"


#: 限流后的退避秒数。免费档模型（如 glm-4.6v-flash）被限流是常态，
#: 实测智谱三次里两次返回 429「该模型当前访问量过大」。不退避重试的话，
#: 这个脚本对免费模型基本得不到结论，而它的用途恰恰是给出结论。
BACKOFF_SECONDS = (5, 15, 30)


def _call_with_retry(backend, run: dict):
    """可重试的错误就退避后再试，不可重试的立刻抛。

    重试次数记进 run，报告里要体现——"重试两次才成功"和"一次就成功"
    是不同的稳定性，M3 选型时这个差别有意义。
    """
    last: LLMBackendError | None = None
    for attempt, delay in enumerate((0, *BACKOFF_SECONDS)):
        if delay:
            print(f"      限流，{delay}s 后重试…")
            time.sleep(delay)
        try:
            intent = backend.predict_action(PROBE_INSTRUCTION, screenshot_holder[0])
        except LLMBackendError as exc:
            last = exc
            if not exc.retryable:
                raise
            run["retries"] = attempt + 1
            continue
        return intent
    raise last


def _reparse_ok(raw: str) -> bool:
    try:
        parse_action_payload(raw)
        return True
    except OutputParseError:
        return False


def _in_bounds(params: dict, size) -> bool | None:
    """坐标是否落在画布内。

    这是**可程序化判定**的那一半：模型看没看懂图不好说，但它给的坐标在
    不在范围内是硬事实。越界说明它没建立起画布尺寸的概念，模式 A 直接
    受影响。
    """
    x, y = params.get("x"), params.get("y")
    if x is None or y is None:
        return None
    return 0 <= x < size[0] and 0 <= y < size[1]


def _one_line(run: dict) -> str:
    if not run.get("ok"):
        return f"失败（{run.get('kind', '?')}）：{str(run.get('error', ''))[:90]}"
    bounds = {True: "范围内", False: "越界", None: "无坐标"}[run.get("in_bounds")]
    return (
        f"{run['action']} {run.get('params')} | {bounds} | "
        f"{run['latency_ms']:.0f}ms | {run['tokens']['total_tokens']} tokens"
    )


# ---------------------------------------------------------------------- #


def write_report(records: list[dict], image_note: str, full_run: bool = True) -> Path:
    lines = [
        "# M1 D1：多模态 API 连通性验证",
        "",
        f"> 采集时间：{datetime.now():%Y-%m-%d %H:%M}",
        f"> 测试图片：{image_note}",
        "",
        "验证两件事："
        "① 模型能否看懂真实桌面截图并输出可解析的结构化结果；"
        "② `ChatOpenAI` + 平台 OpenAI 兼容 `base_url` 能否正确传图。",
        "",
        "## 结果汇总",
        "",
        "| 平台 | 模型 | 状态 | 可解析 | 坐标在范围内 | 平均延迟 | 平均 tokens | 限流重试 |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for record in records:
        runs = [r for r in record.get("runs", []) if r.get("ok")]
        config = record.get("config", {})
        if runs:
            latency = sum(r["latency_ms"] for r in runs) / len(runs)
            tokens = sum(r["tokens"]["total_tokens"] for r in runs) / len(runs)
            parsable = f"{sum(1 for r in runs if r.get('parsable'))}/{len(record['runs'])}"
            bounded = f"{sum(1 for r in runs if r.get('in_bounds'))}/{len(record['runs'])}"
            latency_text, tokens_text = f"{latency:.0f}ms", f"{tokens:.0f}"
        else:
            parsable = bounded = latency_text = tokens_text = "—"
        lines.append(
            f"| {PROVIDERS[record['provider']].label} | {config.get('model', '—')} | "
            f"{record['status']} | {parsable} | {bounded} | {latency_text} | {tokens_text} | "
            f"{record.get('throttled', 0)} 次 |"
        )

    lines += ["", "## 逐家详情", ""]
    for record in records:
        provider = PROVIDERS[record["provider"]]
        lines += [f"### {provider.label}", ""]
        if record["status"] == "未配置":
            lines += [f"未配置 API key，跳过。（{record.get('error', '')}）", ""]
            continue

        config = record.get("config", {})
        lines += [
            f"- 模型：`{config.get('model')}`",
            f"- 端点：`{config.get('base_url')}`",
            f"- 送出图片：{record.get('image_meta', {})}",
            f"- 累计用量：{record.get('cost', {})}",
            "",
        ]
        if provider.notes:
            lines += [f"> 已知限制：{provider.notes}", ""]

        for run in record["runs"]:
            lines += [f"**第 {run['index']} 次** — {_one_line(run)}", ""]
            if run.get("raw"):
                lines += ["```", run["raw"], "```", ""]

    lines += [
        "## 结论",
        "",
        "此文件由 `python scripts/verify_llm.py` 生成，是 M1 D1 验收项的证据。",
        "M2 的 `OpenAICompatBackend` 使用的正是这里验证过的消息构造方式，",
        "三家平台共用同一实现，M3 横评时只切换配置。",
        "",
    ]

    text = "\n".join(lines)
    if full_run:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(text, encoding="utf-8")
        return REPORT

    # 单平台补测另存，不动 M1 D1 的验收证据
    RUNS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    scope = "-".join(r["provider"] for r in records)
    path = RUNS / f"{stamp}-{scope}.md"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="多模态 API 连通性验证（会真实计费）")
    parser.add_argument("--provider", action="append", help="只测指定平台，可重复")
    parser.add_argument("--image", help="用指定图片代替现场截屏")
    parser.add_argument("--repeat", type=int, default=1, help="每家跑几次")
    parser.add_argument("--json", action="store_true", help="额外输出原始 JSON")
    args = parser.parse_args()

    load_dotenv_if_present()

    configured = [key for key, ok in available_providers() if ok]
    targets = args.provider or configured
    if not targets:
        print("没有任何平台配置了 API key。请复制 .env.example 为 .env 并填入。")
        print("已注册的平台：" + "、".join(sorted(PROVIDERS)))
        return 1

    screenshot = grab_screenshot(args.image)
    image_note = args.image or f"现场截屏 {screenshot.width}×{screenshot.height}"
    print(f"测试图片：{image_note}")
    print(f"待测平台：{'、'.join(targets)}（每家 {args.repeat} 次，会产生真实费用）\n")

    records = []
    for key in targets:
        print(f"{PROVIDERS[key].label if key in PROVIDERS else key}：")
        records.append(probe(key, screenshot, args.repeat))
        print()

    # 只有把所有已配置的平台都跑了，才算一次可以覆盖验收证据的完整验证
    full_run = set(targets) >= set(configured)
    path = write_report(records, image_note, full_run=full_run)
    print(f"报告已写入：{path}")
    if not full_run:
        print(f"  （单平台补测，未覆盖 {REPORT}——那是 M1 D1 的验收证据）")

    if args.json:
        print(json.dumps(records, ensure_ascii=False, indent=2, default=str))

    passed = [r for r in records if r["status"].startswith("通过")]
    print(f"\n通过 {len(passed)}/{len(records)} 家。")
    # 「至少测通两家」是对**完整验证**说的。单平台补测本来就只跑一家，
    # 在那里提醒「当前不足」纯属噪声，还会让人以为验收退化了。
    if full_run and len(passed) < 2:
        print("M1 D1 要求至少测通两家，当前不足。")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
