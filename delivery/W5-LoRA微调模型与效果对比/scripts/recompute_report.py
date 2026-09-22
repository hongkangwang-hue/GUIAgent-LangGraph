"""复算《微调效果对比分析报告》里的每一个数字 —— **交付包专用，源仓库没有这个文件**。

    python scripts/recompute_report.py          # 在交付包根目录执行

只用标准库，不需要 GPU、不需要 torch、不需要模型权重。
输出按报告的节号排列，每一行都能在报告里找到对应的数。

## 为什么要单独写这个脚本，而不是只让读者跑 `eval.action --report`

报告里的数字有三类来源，`--report` 只覆盖第一类：

1. **主结果（§5.1）**：`eval.action.summarize()` 直接算出，与 `--report` 同一个函数。
2. **对照实验（§5.2、§5.3）**：需要跨两份结果文件按 `sample_id` 配对，
   `--report` 不做配对。源仓库里这些数是临时分析脚本算的，
   而那份文档自己记录过两次分析脚本出错（漏改 `point_norm`、
   把哨兵值 `-1.0` 当成有效距离），所以这里用正确的过滤条件重写一遍。
3. **附录 A（旧版验证集）**：`旧口径/api-8b.jsonl` 是在分母缺陷修复**之前**跑的，
   文件里仍有 22 行「解析失败被记成调用失败」。直接 `summarize()` 会把它们
   剔出分母，得到 35.8% 而不是报告里的 30.3%。这里显式把它们放回分母。
   **不修正的话，读者在包内复算会得到一个与报告矛盾的数。**

## 两条必须守的字段约定

- `distance == -1.0` 是「这条没有坐标」的**哨兵值**，统计距离时必须用
  `distance >= 0` 过滤，不能用 `isinstance(distance, float)`。
- 分位数用 `statistics.quantiles(..., method="inclusive")`，
  与 numpy 的默认线性插值一致；默认的 `exclusive` 会让 P25/P75 差几个像素。
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eval.action import summarize  # noqa: E402

ACTION = ROOT / "docs" / "m3-action"
RUNS = ROOT / "docs" / "m2-runs"
DATA = ROOT / "finetune" / "data"


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def by_id(path: Path) -> dict[str, dict]:
    return {r["sample_id"]: r for r in read_jsonl(path)}


def has_coord(r: dict) -> bool:
    return r["distance"] >= 0  # -1.0 是哨兵值


def pct(x: float) -> str:
    return f"{x:.1%}"


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """2×2 列联表 [[a, b], [c, d]] 的 Fisher 精确检验双侧 p 值。"""
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d
    denom = math.comb(n, c1)

    def p(x: int) -> float:
        return math.comb(r1, x) * math.comb(r2, c1 - x) / denom

    observed = p(a)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= observed * (1 + 1e-7)))


# ---------------------------------------------------------------------- #
# §3.2 数据集构建
# ---------------------------------------------------------------------- #
def section_data() -> None:
    banner("§3.2 数据集构建（表 1、表 2）")
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    train, val = read_jsonl(DATA / "train.jsonl"), read_jsonl(DATA / "val.jsonl")
    print(f"  划分指纹 {meta['fingerprint']}   种子 {meta['seed']}   切分单位 {meta['group_by']}")
    print(f"  样本     train {len(train)}   val {len(val)}")
    print(f"  会话     train {len({r['session_id'] for r in train})}   "
          f"val {len({r['session_id'] for r in val})}   "
          f"交集 {len({r['session_id'] for r in train} & {r['session_id'] for r in val})}")
    print(f"  按划分应有 train {meta['expected']['train']} / val {meta['expected']['val']}，"
          f"丢弃 train {meta['dropped']['train']} / val {meta['dropped']['val']}")

    dist = Counter(r["action"] for r in train + val)
    total = sum(dist.values())
    print(f"  动作分布（train+val，共 {total}）：")
    for action, n in dist.most_common():
        print(f"    {action:<14}{n:>6}  {n / total:6.1%}")
    assert dict(dist) == meta["action_dist"], "meta.json 的动作分布与实际文件不符"

    res = Counter(tuple(r["resolution"]) for r in train + val)
    print(f"  截图分辨率：{dict(res)}")
    langs = sum(any("一" <= ch <= "鿿" for ch in r["instruction"]) for r in train + val)
    print(f"  指令含中文的样本 {langs}/{total}")

    train_dist = Counter(r["action"] for r in train)
    tkd = train_dist["type"] + train_dist["key"] + train_dist["done"]
    print(f"  训练集中 type+key+done 占 {tkd}/{len(train)} = {tkd / len(train):.1%}")

    # 新旧两份冻结划分的会话交叉：旧模型的数字为什么不能与新验证集并列
    prereq = ROOT / "docs" / "m3-prereq"
    old = json.loads((prereq / "split-screenagent-desktop-grounding-旧.json").read_text(encoding="utf-8"))
    new = json.loads((prereq / "split-screenagent-desktop.json").read_text(encoding="utf-8"))
    o_t, o_v, n_t, n_v = (set(s[k]) for s in (old, new) for k in ("train_groups", "val_groups"))
    print(f"  旧划分 指纹 {old['fingerprint']}  样本池 {old['pool_size']}（{old['train_size']} / {old['val_size']}）")
    print(f"  新划分 指纹 {new['fingerprint']}  样本池 {new['pool_size']}（{new['train_size']} / {new['val_size']}）")
    print(f"  会话交叉：旧训练∩新验证 {len(o_t & n_v)}/{len(n_v)}   "
          f"旧验证∩新训练 {len(o_v & n_t)}   旧验证∩新验证 {len(o_v & n_v)}")

    # val-4x3-coords.jsonl 必须是 val.jsonl 里坐标类样本的逐字子集
    val_map = {r["sample_id"]: r for r in val}
    coords = read_jsonl(DATA / "val-4x3-coords.jsonl")
    coord_truth = [r for r in val if r.get("point_norm")]
    same = all(val_map.get(r["sample_id"]) == r for r in coords)
    print(f"  val-4x3-coords.jsonl {len(coords)} 条，val 中坐标类 {len(coord_truth)} 条，"
          f"逐字相同：{same}")


# ---------------------------------------------------------------------- #
# §5.1 RQ1 主结果
# ---------------------------------------------------------------------- #
def section_main() -> None:
    banner("§5.1 RQ1 微调前后对比（表 5、表 6；末尾为 §4 跨硬件验证）")
    before = summarize(ACTION / "before-fixed.jsonl")
    after = summarize(ACTION / "after-fixed.jsonl")
    rows = [
        ("样本", before["total"], after["total"], "d"),
        ("坐标类样本（联合命中分母）", before["coord_n"], after["coord_n"], "d"),
        ("输出格式合规率", before["format_acc"], after["format_acc"], "p"),
        ("动作类型准确率", before["action_acc"], after["action_acc"], "p"),
        ("坐标误差中位数", before["median_distance"], after["median_distance"], "f"),
        ("联合命中 r=25", before["joint"]["25"], after["joint"]["25"], "p"),
        ("联合命中 r=50", before["joint"]["50"], after["joint"]["50"], "p"),
        ("联合命中 r=100", before["joint"]["100"], after["joint"]["100"], "p"),
        ("全量命中率", before["overall_hit"], after["overall_hit"], "p"),
        ("done 误报率", before["done_false_positive"], after["done_false_positive"], "p"),
        ("中位延迟 ms", before["median_latency_ms"], after["median_latency_ms"], "f"),
    ]
    for name, b, a, kind in rows:
        fmt = {"d": lambda v: f"{v}", "p": pct, "f": lambda v: f"{v:.1f}"}[kind]
        print(f"  {name:<22}{fmt(b):>10}{fmt(a):>10}")

    print("\n  按真值动作（类型准确率 / 命中率，before → after）")
    for action, a_row in sorted(after["by_action"].items(), key=lambda kv: -kv[1]["n"]):
        b_row = before["by_action"][action]
        print(f"    {action:<14}n={a_row['n']:>4}   "
              f"{pct(b_row['action_acc']):>6} → {pct(a_row['action_acc']):>6}   "
              f"{pct(b_row['hit_rate']):>6} → {pct(a_row['hit_rate']):>6}")

    # 派生差值必须用原始值算，不能拿四舍五入后的百分数相减
    print("\n  变化量（按原始值计算）")
    for name, key in (("格式合规率", "format_acc"), ("动作类型准确率", "action_acc"),
                      ("全量命中率", "overall_hit"), ("done 误报率", "done_false_positive")):
        print(f"    {name:<14}{(after[key] - before[key]) * 100:+.1f}pp")
    for r in ("25", "50", "100"):
        print(f"    联合命中 r={r:<6}{(after['joint'][r] - before['joint'][r]) * 100:+.1f}pp")
    drop = before["median_distance"] - after["median_distance"]
    print(f"    坐标误差中位数  -{drop:.1f}（降 {drop / before['median_distance']:.0%}）")

    # done 误报在全部动作类型错误中的占比，以及修好它的上界
    rows = read_jsonl(ACTION / "after-fixed.jsonl")
    wrong = [r for r in rows if not r["action_ok"]]
    fp_done = [r for r in wrong if r["pred_action"] == "done" and r["truth_action"] != "done"]
    right = len(rows) - len(wrong)
    non_done = [r for r in rows if r["truth_action"] != "done"]
    fp_all = [r for r in non_done if r["pred_action"] == "done"]
    print(f"\n  done 误报率分母：真值非 done 的样本 {len(non_done)} 条，其中报 done {len(fp_all)} 条 = "
          f"{len(fp_all) / len(non_done):.1%}")
    print(f"\n  动作类型错误 {len(wrong)} 条，其中「不该停却报 done」{len(fp_done)} 条 = "
          f"{len(fp_done) / len(wrong):.1%}")
    print(f"  若这 {len(fp_done)} 条全部改对，动作类型准确率上界 "
          f"{right / len(rows):.1%} → {(right + len(fp_done)) / len(rows):.1%}")
    train_done = sum(r["action"] == "done" for r in read_jsonl(DATA / "train.jsonl"))
    print(f"  训练集中 done 占 {train_done}/1783 = {train_done / 1783:.1%}")

    # 本机 8GB 卡与服务器 4090 D 的逐字节对照
    cross = read_jsonl(ACTION / "交叉验证" / "after-fixed-本机18条交叉验证.jsonl")
    ref = by_id(ACTION / "after-fixed.jsonl")
    raw_same = sum(ref[r["sample_id"]]["raw"] == r["raw"] for r in cross)
    act_same = sum(ref[r["sample_id"]]["pred_action"] == r["pred_action"] for r in cross)
    xy_same = sum(ref[r["sample_id"]]["pred_xy"] == r["pred_xy"] for r in cross)
    print(f"\n  硬件交叉验证：原始输出一致 {raw_same}/{len(cross)}   "
          f"动作一致 {act_same}/{len(cross)}   坐标一致 {xy_same}/{len(cross)}")


# ---------------------------------------------------------------------- #
# §5.2 RQ2 提示词工程
# ---------------------------------------------------------------------- #
def section_prompt() -> None:
    banner("§5.2 RQ2 提示词对照（表 7，微调模型，99 条坐标类样本）")
    base = [r for r in read_jsonl(ACTION / "after-fixed.jsonl") if r["truth_xy"]]
    v0 = read_jsonl(ACTION / "对照实验" / "after-executor-v0.jsonl")
    assert {r["sample_id"] for r in base} == {r["sample_id"] for r in v0}, "两档样本集不同"
    print(f"  {'':<14}{'n':>4}{'格式':>7}{'类型准确':>9}{'答done':>8}{'有坐标':>7}{'误差中位':>9}{'r=50内':>8}")
    for label, rows in (("训练提示词", base), ("executor_v0", v0)):
        n = len(rows)
        d = sorted(r["distance"] for r in rows if has_coord(r))
        print(f"  {label:<14}{n:>4}"
              f"{sum(r['format_ok'] for r in rows) / n:>7.0%}"
              f"{sum(r['action_ok'] for r in rows) / n:>9.0%}"
              f"{sum(r['pred_action'] == 'done' for r in rows) / n:>8.0%}"
              f"{len(d):>7}{statistics.median(d):>9.0f}"
              f"{sum(x <= 50 for x in d) / len(d):>8.0%}")


# ---------------------------------------------------------------------- #
# §5.3 RQ3 缩放强度
# ---------------------------------------------------------------------- #
def section_scale() -> None:
    banner("§5.3 RQ3 缩放强度（表 8、表 9，严格配对：同一样本、真值为坐标、两边都给了坐标）")
    base = by_id(ACTION / "after-fixed.jsonl")
    up = by_id(ACTION / "对照实验" / "after-upscaled.jsonl")
    ids = [i for i in up if i in base and base[i]["truth_xy"] and has_coord(base[i]) and has_coord(up[i])]
    print(f"  配对样本 n = {len(ids)}")
    print(f"  {'':<18}{'P25':>6}{'中位':>6}{'P75':>6}{'r=25':>7}{'r=50':>7}{'r=100':>7}{'类型对':>9}")
    for label, src in (("原图 1024×768", base), ("放大 1664×1248", up)):
        d = sorted(src[i]["distance"] for i in ids)
        q1, med, q3 = statistics.quantiles(d, n=4, method="inclusive")
        within = [sum(x <= r for x in d) / len(d) for r in (25, 50, 100)]
        ok = sum(src[i]["action_ok"] for i in ids)
        print(f"  {label:<18}{q1:>6.0f}{med:>6.0f}{q3:>6.0f}"
              f"{within[0]:>7.0%}{within[1]:>7.0%}{within[2]:>7.0%}{ok:>6}/{len(ids)}")
    med_base = statistics.median(base[i]["distance"] for i in ids)
    med_up = statistics.median(up[i]["distance"] for i in ids)
    print(f"  误差中位数扩大 {med_up / med_base:.1f} 倍")
    worse = sum(up[i]["distance"] > base[i]["distance"] for i in ids)
    print(f"  逐条比较：放大后更差 {worse}/{len(ids)} = {worse / len(ids):.0%}")

    max_pixels = 896 * 896
    print(f"\n  max_pixels = 896² = {max_pixels:,}")
    for label, w, h in (("训练 1024×768", 1024, 768), ("放大 1664×1248", 1664, 1248),
                        ("客机 1920×1080", 1920, 1080)):
        scale = min(1.0, math.sqrt(max_pixels / (w * h)))
        print(f"    {label:<16}{w * h:>11,} px   缩放系数 {scale:.4f}")


# ---------------------------------------------------------------------- #
# §5.4 RQ4 端到端
# ---------------------------------------------------------------------- #
def section_e2e() -> None:
    banner("§5.4 RQ4 端到端（表 10，客机，微调 3B，reflector 关，executor_v0）")
    files = sorted(RUNS.glob("*.json"))
    for path in files:
        run = json.loads(path.read_text(encoding="utf-8"))
        recs = run["records"]
        valid = [r for r in recs if not r.get("excluded")]
        per = Counter(r["task"] for r in valid if r["verified"])
        tasks = list(dict.fromkeys(r["task"] for r in recs))
        subtasks = sum(r.get("subtasks") or 0 for r in valid)
        empty = sum(r.get("empty_done") or 0 for r in valid)
        print(f"\n  {path.name}")
        print(f"    分辨率 {run['screen'].get('resolution')}   快照 {run.get('guest_snapshot')}   "
              f"reflector {run.get('reflector')}   模板 {run.get('executor_template')}")
        print(f"    程序化判定成功 {sum(per.values())}/{len(valid)}   "
              + "  ".join(f"{t}:{per[t]}/{sum(r['task'] == t for r in valid)}" for t in tasks))
        print(f"    模型自报完成 {sum(r['model_said_done'] for r in valid)}/{len(valid)}   "
              f"总步数 {sum(r['steps'] for r in valid)}   "
              f"子任务 {subtasks}   空 done 子任务 {empty} = {empty / subtasks:.1%}")


# ---------------------------------------------------------------------- #
# 附录 A 旧版验证集
# ---------------------------------------------------------------------- #
def summarize_fixing_denominator(path: Path) -> dict:
    """把「解析失败被记成调用失败」的行放回分母，按格式失败计。

    只放回 `模型输出无法解析` 这一类；网络超时、限流等真正的调用失败不放回。
    """
    rows = read_jsonl(path)
    parse_fail = [r for r in rows if "无法解析" in r.get("error", "")]
    other_err = [r for r in rows if r.get("error") and r not in parse_fail]
    kept = [r for r in rows if not r.get("error")]
    n = len(kept) + len(parse_fail)
    coord_rows = [r for r in kept + parse_fail if r["truth_xy"]]
    joint = {
        radius: sum(r["action_ok"] and has_coord(r) and r["distance"] <= radius for r in kept)
        / len(coord_rows)
        for radius in (25, 50, 100)
    }
    return {
        "n": n,
        "parse_fail": len(parse_fail),
        "other_err": len(other_err),
        "format_acc": sum(r["format_ok"] for r in kept) / n,
        "action_acc": sum(r["action_ok"] for r in kept) / n,
        "joint": joint,
        "naive": summarize(path),
    }


def section_legacy() -> None:
    banner("附录 A 旧版验证集（表 A1–A3；142 条、仅点击类、英文总目标指令）—— 不可与正文并列")
    legacy = ACTION / "旧口径"
    print(f"  {'档':<16}{'n':>5}{'格式合规':>9}{'类型准确':>9}{'误差中位':>9}{'联合@50':>9}{'中位延迟':>9}")
    for tag in ("before", "after", "before-4bit", "after-4bit",
                "executor_v0", "executor_v1", "executor_v2"):
        s = summarize(legacy / f"{tag}.jsonl")
        print(f"  {tag:<16}{s['total']:>5}{s['format_acc']:>9.1%}{s['action_acc']:>9.1%}"
              f"{s['median_distance']:>9.1f}{s['joint']['50']:>9.1%}{s['median_latency_ms']:>9.0f}")

    b16, a16 = summarize(legacy / "before.jsonl"), summarize(legacy / "after.jsonl")
    b4, a4 = summarize(legacy / "before-4bit.jsonl"), summarize(legacy / "after-4bit.jsonl")
    base_dmg = (b16["action_acc"] - b4["action_acc"]) * 100
    ft_dmg = (a16["action_acc"] - a4["action_acc"]) * 100
    print(f"\n  量化伤害（动作类型准确率 bf16 → 4-bit）：基座 -{base_dmg:.1f}pp   "
          f"微调 -{ft_dmg:.1f}pp   相差 {base_dmg / ft_dmg:.2f} 倍")
    print(f"  微调带来的动作类型提升：bf16 +{(a16['action_acc'] - b16['action_acc']) * 100:.1f}pp   "
          f"4-bit +{(a4['action_acc'] - b4['action_acc']) * 100:.1f}pp")
    print(f"  4-bit 下微调前后延迟比 {a4['median_latency_ms'] / b4['median_latency_ms']:.2f} 倍"
          f"（{b4['median_latency_ms']:.0f} → {a4['median_latency_ms']:.0f} ms）")

    v = {t: read_jsonl(legacy / f"executor_{t}.jsonl") for t in ("v0", "v1", "v2")}
    fmt = {t: sum(r["format_ok"] for r in rows) for t, rows in v.items()}
    print(f"\n  提示词消融（基座 3B）格式合规条数 v0 {fmt['v0']}/142  v1 {fmt['v1']}/142  v2 {fmt['v2']}/142")
    print(f"    few-shot v0→v1   Fisher p = {fisher_two_sided(fmt['v0'], 142 - fmt['v0'], fmt['v1'], 142 - fmt['v1']):.4f}")
    print(f"    CoT      v1→v2   Fisher p = {fisher_two_sided(fmt['v1'], 142 - fmt['v1'], fmt['v2'], 142 - fmt['v2']):.3f}")

    s8 = summarize_fixing_denominator(legacy / "api-8b.jsonl")
    naive = s8["naive"]
    print(f"\n  在线 8B（api-8b.jsonl，修正分母：解析失败 {s8['parse_fail']} 行放回，其他调用失败 {s8['other_err']} 行）")
    print(f"    {'':<14}{'直接 summarize':>16}{'修正后（分母 ' + str(s8['n']) + '）':>18}")
    print(f"    {'格式合规率':<14}{naive['format_acc']:>16.1%}{s8['format_acc']:>18.1%}")
    print(f"    {'动作类型准确率':<12}{naive['action_acc']:>16.1%}{s8['action_acc']:>18.1%}")
    print(f"    {'联合命中@50':<13}{naive['joint']['50']:>16.1%}{s8['joint'][50]:>18.1%}")
    print(f"    同口径 4-bit 微调 3B 联合命中@50 {a4['joint']['50']:.1%}   "
          f"差 {(s8['joint'][50] - a4['joint']['50']) * 100:.1f}pp")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import os

    os.chdir(ROOT)
    section_data()
    section_main()
    section_prompt()
    section_scale()
    section_e2e()
    section_legacy()
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
