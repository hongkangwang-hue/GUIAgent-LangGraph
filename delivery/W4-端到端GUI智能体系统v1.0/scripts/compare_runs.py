"""把多次基础任务实测汇总成一张对比表。

## 为什么要有这个脚本

在线 / 离线-基座 / 离线-微调三组各跑 25 轮，产出三份存档。手工从三份
JSON 里抄数字到报告里，**抄错一个不会有任何提示**——而这正是本项目
反复栽跟头的那类错误（M1 的召回率 10%→100%、OCR 86.7%→100%、
模式 B 完整率 85.7%→100%，全都是"看起来对、实际错"）。

所以口径写成代码：

- **无效轮与剔除轮不进分子也不进分母。** 起点没建立、或人为干预过的
  轮次既不是成功也不是失败。当失败会低估能力，当成功会高估。
- **一组可以由多份存档合并。** 在线组就是两份：25 轮全量 + 5 轮补跑
  （第一次跑时消息程序没开，5 轮全部作废）。
- **成本区分"确认的 0"与"未知"。** 离线版的 API 费用是实打实的 0；
  在线组没配单价时也显示 0，但那是"不知道"。两者都是 0.0000，
  混为一谈就会写出"离线版比在线版便宜"这种看似有据实则无据的话。

## 用法

    # 自动发现 docs/m2-runs/ 下的存档，按 tag 分组
    python scripts/compare_runs.py

    # 显式指定哪些 tag 归到哪一组（在线组由两个 tag 合并）
    python scripts/compare_runs.py \\
        --arm "在线=online,online-send" \\
        --arm "离线-基座=base" \\
        --arm "离线-微调=lora"

    # 输出可直接粘进报告的 markdown
    python scripts/compare_runs.py --markdown > /tmp/table.md

## 不做的事

**不下结论。** 脚本只把数字摆出来，"为什么离线是 0%"要靠轨迹回放去看。
一个自动生成的因果解释比没有解释更危险。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "src")
)  # 交付包把各个包放在 src/ 下

RUNS = Path("docs/m2-runs")

#: 失败形态。键是 `loop_status`，值是人话。
#:
#: 取值来自 `agent/session.py`：任务跑完是 `completed`，某个子任务没做成
#: 是 `subtask_failed`，任务拆解就失败了是 `plan_failed`，被急停或熔断
#: 打断是 `aborted`。空串是**会话根本没启动**（起点检查没过）。
STATUS_LABEL = {
    "completed": "跑完但判定未通过",
    "subtask_failed": "子任务失败",
    "plan_failed": "任务拆解失败",
    "aborted": "中止（急停/熔断）",
    "": "未启动",
}


@dataclass
class Arm:
    """一组实验。"""

    label: str
    tags: list[str] = field(default_factory=list)
    records: list[dict] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    backend: str = ""
    offline: bool | None = None

    # ---- 分母口径都收在这里，别在别处再算一遍 ---- #

    @property
    def valid(self) -> list[dict]:
        """进成功率统计的轮次。

        **无效轮（起点未建立）与剔除轮（人为干预）都不算。** 它们既不是
        成功也不是失败——那是环境的失败，混进 Agent 的成功率里两个数字
        都不可信。
        """
        return [
            r
            for r in self.records
            if r.get("precondition_ok", True) and not r.get("excluded", False)
        ]

    @property
    def invalid(self) -> list[dict]:
        return [r for r in self.records if not r.get("precondition_ok", True)]

    @property
    def dropped(self) -> list[dict]:
        return [
            r for r in self.records if r.get("precondition_ok", True) and r.get("excluded", False)
        ]

    @property
    def ok(self) -> int:
        return sum(1 for r in self.valid if r.get("verified"))

    def rate(self) -> float | None:
        """成功率。一轮有效的都没有时返回 None —— **不返回 0.0**。

        「0% 成功」和「没有数据」是两件完全不同的事，用同一个 0.0 表示，
        报告里就分不出来了。
        """
        return self.ok / len(self.valid) if self.valid else None


def load_archives(directory: Path) -> list[tuple[Path, dict]]:
    out = []
    for path in sorted(directory.glob("*.json")):
        try:
            out.append((path, json.loads(path.read_text(encoding="utf-8"))))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[跳过] {path.name}：{exc}", file=sys.stderr)
    return out


def build_arms(
    archives: list[tuple[Path, dict]],
    mapping: dict[str, list[str]] | None,
    skip: list[str] | None = None,
) -> list[Arm]:
    """把存档按 tag 归入各组。

    ``mapping`` 形如 ``{"在线": ["online", "online-send"]}``。不给时按
    每个 tag 自成一组——**没有 tag 的老存档会被跳过并明确说出来**，
    因为分不清它属于哪一组，猜一个比漏掉更糟。
    """
    if mapping is None:
        tags = {a.get("tag", "") for _, a in archives if a.get("tag")}
        mapping = {t: [t] for t in sorted(tags)}

    by_tag: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    untagged: list[str] = []
    for path, archive in archives:
        # **整份作废的运行要能整份剔掉。**
        #
        # 2026-08-24 的离线-基座组跑了两次，第一次因为客户端超时 90s 而
        # 生成需要 140s 全废（25 轮里 15 轮连规划都没出来）。两次的 tag
        # 都是 `base`，按 tag 合并就成了 50 轮，而那 15 个「任务拆解失败」
        # 全部来自工具缺陷，不是模型能力——**混进去整张表就不可信了**。
        #
        # 这不是 `excluded`（那是逐轮的人为干预），是整次运行的环境无效。
        # 存档留着当缺陷证据，只是不进对比。
        if skip and any(pattern in path.name for pattern in skip):
            print(f"[排除] {path.name}", file=sys.stderr)
            continue
        tag = archive.get("tag", "")
        if not tag:
            untagged.append(path.name)
            continue
        by_tag[tag].append((path, archive))

    if untagged:
        print(
            f"[提示] {len(untagged)} 份存档没有 tag 字段（生成于加 --tag 之前），已跳过：",
            ", ".join(untagged[:4]) + ("…" if len(untagged) > 4 else ""),
            file=sys.stderr,
        )

    arms = []
    for label, tags in mapping.items():
        arm = Arm(label=label, tags=list(tags))
        for tag in tags:
            for path, archive in by_tag.get(tag, []):
                arm.records.extend(archive.get("records", []))
                arm.sources.append(path.name)
                arm.backend = arm.backend or archive.get("backend", "")
                if arm.offline is None:
                    arm.offline = archive.get("offline")
        if arm.records:
            arms.append(arm)
    return arms


def per_task(arm: Arm) -> dict[str, tuple[int, int]]:
    """每个任务的 (成功数, 有效轮数)。"""
    table: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for record in arm.valid:
        entry = table[record.get("title") or record.get("task", "?")]
        entry[1] += 1
        entry[0] += bool(record.get("verified"))
    return {k: (v[0], v[1]) for k, v in table.items()}


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def step_latency(arm: Arm) -> dict[str, float | None]:
    """每步平均延迟，按四段分解（M2 验收标准 3 的口径）。

    **除以总步数而不是总轮数。** 一轮 12 步和一轮 2 步不可比，而离线组
    恰恰因为卡循环把步数撑满了——按轮平均会把「每步慢」和「步数多」
    这两件事混成一个数。
    """
    total = defaultdict(float)
    steps = 0
    for record in arm.valid:
        steps += record.get("steps", 0)
        for key, value in (record.get("latency") or {}).items():
            total[key] += float(value or 0.0)
    if not steps:
        return dict.fromkeys(("api_ms", "grounding_ms", "execute_ms", "screenshot_ms"))
    return {k: total[k] / steps for k in ("api_ms", "grounding_ms", "execute_ms", "screenshot_ms")}


def failure_modes(arm: Arm) -> dict[str, int]:
    """失败轮次按 loop_status 归类。只统计判定未通过的。"""
    modes: dict[str, int] = defaultdict(int)
    for record in arm.valid:
        if record.get("verified"):
            continue
        modes[STATUS_LABEL.get(record.get("loop_status", ""), record.get("loop_status", "?"))] += 1
    return dict(modes)


def cost_note(arm: Arm) -> str:
    """成本一栏。**区分「确认的 0」与「未知」。**"""
    total = sum(float(r.get("cost_cny") or 0.0) for r in arm.records)
    if arm.offline:
        return "0（确认值，本地推理无 API 计费）"
    if total == 0.0:
        return "未知（未配单价，非 0 元）"
    return f"{total:.4f} 元 / {len(arm.records)} 轮"


def render(arms: list[Arm], markdown: bool) -> None:
    def line(cells: list[str]) -> str:
        return f"| {' | '.join(cells)} |" if markdown else "  ".join(c.ljust(22) for c in cells)

    print("# 在线 vs 离线：基础任务实测对比\n" if markdown else "=" * 78)
    if not markdown:
        print("在线 vs 离线：基础任务实测对比")
        print("=" * 78)

    # ---- 数据来源，先摆出来 ---- #
    print("\n## 数据来源\n" if markdown else "\n数据来源")
    for arm in arms:
        print(
            f"- **{arm.label}**：{arm.backend or '（未记录后端）'}"
            if markdown
            else f"  {arm.label}：{arm.backend}"
        )
        print(
            f"  - 存档 {', '.join(arm.sources)}"
            if markdown
            else f"    存档 {', '.join(arm.sources)}"
        )

    # ---- 逐任务 ---- #
    titles: list[str] = []
    for arm in arms:
        for title in per_task(arm):
            if title not in titles:
                titles.append(title)

    print("\n## 逐任务成功率\n" if markdown else "\n逐任务成功率")
    if markdown:
        print(line(["任务", *(a.label for a in arms)]))
        print(line(["---"] * (len(arms) + 1)))
    tables = [per_task(a) for a in arms]
    for title in titles:
        cells = [title]
        for table in tables:
            hit, total = table.get(title, (0, 0))
            cells.append(f"{hit}/{total}" if total else "—")
        print(line(cells))

    # ---- 总体 ---- #
    print("\n## 总体\n" if markdown else "\n总体")
    if markdown:
        print(line(["指标", *(a.label for a in arms)]))
        print(line(["---"] * (len(arms) + 1)))

    rows: list[list[str]] = [
        ["有效轮数"],
        ["成功率"],
        ["平均步数"],
        ["平均每轮耗时"],
        ["每步 API 延迟"],
        ["数据边界"],
        ["API 费用"],
    ]
    for arm in arms:
        rate = arm.rate()
        avg_steps = mean([r.get("steps", 0) for r in arm.valid])
        avg_secs = mean([float(r.get("duration_s") or 0.0) for r in arm.valid])
        api = step_latency(arm)["api_ms"]
        rows[0].append(str(len(arm.valid)))
        rows[1].append(
            f"{arm.ok}/{len(arm.valid)} = {rate:.0%}" if rate is not None else "无有效轮"
        )
        rows[2].append(f"{avg_steps:.1f}" if avg_steps is not None else "—")
        rows[3].append(f"{avg_secs:.1f}s" if avg_secs is not None else "—")
        rows[4].append(f"{api:.0f}ms" if api else "—")
        rows[5].append("截图不出本机" if arm.offline else "截图上传平台")
        rows[6].append(cost_note(arm))
    for row in rows:
        print(line(row))

    # ---- 失败形态 ---- #
    print("\n## 失败形态\n" if markdown else "\n失败形态")
    modes: list[str] = []
    for arm in arms:
        for mode in failure_modes(arm):
            if mode not in modes:
                modes.append(mode)
    if markdown:
        print(line(["形态", *(a.label for a in arms)]))
        print(line(["---"] * (len(arms) + 1)))
    for mode in modes:
        print(line([mode, *(str(failure_modes(a).get(mode, 0)) for a in arms)]))

    # ---- 必须说明的 ---- #
    print("\n## 数据说明\n" if markdown else "\n数据说明")
    for arm in arms:
        if arm.invalid:
            print(f"- {arm.label}：**{len(arm.invalid)} 轮无效**（起点未建立），已排除出分子分母")
            for record in arm.invalid[:3]:
                print(
                    f"  - {record.get('title')} 第{record.get('attempt')}次：{(record.get('precondition_detail') or '')[:80]}"
                )
        if arm.dropped:
            print(f"- {arm.label}：**{len(arm.dropped)} 轮事后剔除**（人为干预），已排除")
            for record in arm.dropped:
                print(
                    f"  - {record.get('title')} 第{record.get('attempt')}次：{record.get('exclusion_reason', '')}"
                )

    gaps = [
        (a, [r for r in a.valid if r.get("model_said_done") and not r.get("verified")])
        for a in arms
    ]
    for arm, gap in gaps:
        if gap:
            print(
                f"- {arm.label}：**{len(gap)} 轮模型自报完成但判定未通过** —— 模型的自我判断不可信到什么程度，只有程序化判定量得出来"
            )


def parse_arms(values: list[str]) -> dict[str, list[str]] | None:
    """``--arm "在线=online,online-send"`` → ``{"在线": ["online", "online-send"]}``"""
    if not values:
        return None
    mapping: dict[str, list[str]] = {}
    for raw in values:
        label, _, tags = raw.partition("=")
        if not label.strip() or not tags.strip():
            raise SystemExit(f"--arm 格式应为 '名字=tag1,tag2'，收到 {raw!r}")
        mapping[label.strip()] = [t.strip() for t in tags.split(",") if t.strip()]
    return mapping


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="汇总多次基础任务实测，产出对比表")
    parser.add_argument("--runs", default=str(RUNS), help="存档目录")
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="名字=tag1,tag2",
        help="把哪些 tag 归为一组。可重复。不给时每个 tag 自成一组",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        metavar="文件名片段",
        help="整份排除某次运行（按文件名匹配）。用于因工具缺陷而整体作废的运行。可重复",
    )
    parser.add_argument("--markdown", action="store_true", help="输出 markdown 表格")
    return parser


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args()
    archives = load_archives(Path(args.runs))
    if not archives:
        raise SystemExit(f"{args.runs} 下没有存档")

    arms = build_arms(archives, parse_arms(args.arm), skip=args.skip)
    if not arms:
        raise SystemExit("没有匹配到任何一组。用 --arm 指定 tag，或先跑一次带 --tag 的实测")
    render(arms, markdown=args.markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
