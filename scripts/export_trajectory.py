"""把客机里的轨迹导出到宿主机 —— **带扫描，且扫描的边界要写清楚**。

## 这个脚本存在的唯一理由是一条硬规矩

项目规定「轨迹截图不出客机」，理由是真实事故：Agent 运行时截到过 `.env`
明文，里面是 API Key。所以平时只导出 `meta.json` + `steps.jsonl`。

而蒸馏训练必须要截图。**规矩不能绕过，只能带着条件放行**，条件有三层：

1. 采集必须跑在**干净快照**里：没有 `.env`、不登录真实账号、不同步 OneDrive
2. 导出时对**文本**做正则扫描，命中即拒绝导出（不是警告）
3. 导出后**人工抽查截图**，抽查结果写进报告

## 第 2 层扫不到什么，必须说清楚

正则扫的是 `steps.jsonl` / `meta.json` 里的文字。**截图是图像，扫不到里面
的内容**——屏幕上摆着的密码框、登录页、文件内容，这个脚本一个都发现不了。

`control/sentinel.py` 那个哨兵也帮不上：它查的是窗口标题和进程名，运行期
拦截用，不负责事后看图。

所以第 1 层和第 3 层不是补充，是**主力**。把这一点写进 EXPORT-REPORT.md，
是为了让看报告的人知道这份数据的保证到哪为止。

## 用法（在客机上跑）

    python scripts/export_trajectory.py --root outputs/trajectories --out D:/export
    python scripts/export_trajectory.py --root outputs/trajectories --out D:/export --no-frames
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

#: 文本里命中即**拒绝导出**的模式。宁可误杀：漏一条轨迹只是少几条样本，
#: 漏一个密钥是不可撤销的。
SENSITIVE_PATTERNS: tuple[tuple[str, str, str], ...] = (
    ("api_key", r"sk-[A-Za-z0-9]{16,}|(?i:api[_-]?key)\s*[=:]\s*\S{8,}", "疑似 API Key"),
    ("bearer", r"(?i:bearer)\s+[A-Za-z0-9._\-]{16,}", "疑似访问令牌"),
    ("password", r"(?i:password|passwd|密码)\s*[=:：]\s*\S{4,}", "疑似密码"),
    (
        "dotenv",
        r"(?i:LLM_PROVIDER|DASHSCOPE_API_KEY|ZHIPU_API_KEY|NVIDIA_API_KEY)\s*=",
        "疑似 .env 内容",
    ),
    ("email", r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", "疑似邮箱地址"),
    ("login", r"(?i:登录|sign\s?in|账户|帐户)\s*(?i:密码|password)", "疑似登录界面文本"),
)

_COMPILED = tuple(
    (name, re.compile(pattern), reason) for name, pattern, reason in SENSITIVE_PATTERNS
)

#: 扫描哪些文件。**截图不在此列，因为正则扫不了图像。**
SCANNED_SUFFIXES = frozenset({".jsonl", ".json", ".txt", ".md", ".log"})


@dataclass
class Hit:
    file: str
    rule: str
    reason: str
    #: 命中片段做掩码后的样子。**不记原文**——报告本身会被传阅
    masked: str


@dataclass
class ExportReport:
    trajectory: str
    exported: bool
    frames: int = 0
    steps: int = 0
    scanned_files: int = 0
    hits: list[Hit] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "trajectory": self.trajectory,
            "exported": self.exported,
            "frames": self.frames,
            "steps": self.steps,
            "scanned_files": self.scanned_files,
            "hits": [h.__dict__ for h in self.hits],
            "note": self.note,
        }


def _console() -> None:
    """客机控制台是 GBK，直接 print 出 ⚠ / ✅ 会抛 UnicodeEncodeError 把脚本打断。

    **这不是显示问题，是崩溃**——演练时就是在这一行挂掉的。
    与 `scripts/run_basic_tasks.py` 同一处理。
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def mask(text: str, keep: int = 4) -> str:
    """只留头尾各几个字符。命中片段本身也可能是密钥。"""
    text = text.replace("\n", " ").strip()
    if len(text) <= keep * 2:
        return "*" * len(text)
    return f"{text[:keep]}{'*' * 6}{text[-keep:]}"


def scan_text(path: Path) -> list[Hit]:
    """扫一个文本文件。读不出来按「扫不了」处理，返回一条命中拦下它。

    读不出来就放行是危险的默认值：一个二进制垃圾文件可能正好是被改名的
    dump。拒绝比放行安全。
    """
    try:
        content = path.read_text(encoding="utf-8", errors="strict")
    except (UnicodeDecodeError, OSError) as exc:
        return [Hit(path.name, "unreadable", f"文件读不出来，无法扫描：{exc}", "")]

    hits = []
    for rule, pattern, reason in _COMPILED:
        for match in pattern.finditer(content):
            hits.append(Hit(path.name, rule, reason, mask(match.group(0))))
            break  # 同一规则记一条就够，报告不需要列出全部
    return hits


def scan_trajectory(traj_dir: Path) -> tuple[list[Hit], int]:
    """扫一条轨迹里所有**文本**文件。返回 (命中, 扫了几个文件)。"""
    hits: list[Hit] = []
    scanned = 0
    for path in sorted(traj_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SCANNED_SUFFIXES:
            continue
        scanned += 1
        hits.extend(scan_text(path))
    return hits, scanned


def export_one(traj_dir: Path, out_dir: Path, include_frames: bool = True) -> ExportReport:
    """导出一条轨迹。**命中敏感模式就不导出**，只返回报告。"""
    report = ExportReport(trajectory=traj_dir.name, exported=False)

    steps_file = traj_dir / "steps.jsonl"
    if not steps_file.exists():
        report.note = "没有 steps.jsonl，跳过"
        return report
    report.steps = sum(
        1 for line in steps_file.read_text(encoding="utf-8").splitlines() if line.strip()
    )

    hits, scanned = scan_trajectory(traj_dir)
    report.hits = hits
    report.scanned_files = scanned
    if hits:
        report.note = "文本扫描命中，**拒绝导出**"
        return report

    target = out_dir / traj_dir.name
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    for name in ("steps.jsonl", "meta.json"):
        if (traj_dir / name).exists():
            shutil.copy2(traj_dir / name, target / name)

    if include_frames:
        frames_src = traj_dir / "frames"
        if frames_src.is_dir():
            shutil.copytree(frames_src, target / "frames")
            report.frames = sum(1 for _ in (target / "frames").glob("*.png"))

    report.exported = True
    report.note = "已导出" if include_frames else "已导出（不含截图）"
    return report


def write_report(reports: list[ExportReport], out_dir: Path, include_frames: bool) -> Path:
    """导出报告。**扫描的能力边界必须写在里面。**"""
    total_frames = sum(r.frames for r in reports)
    ok = [r for r in reports if r.exported]
    blocked = [r for r in reports if not r.exported]

    lines = [
        "# 轨迹导出报告",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"轨迹 {len(reports)} 条，导出 {len(ok)} 条，拦下 {len(blocked)} 条，截图 {total_frames} 张",
        "",
        "## 这份扫描保证了什么，没保证什么",
        "",
        "| | 能不能查出来 |",
        "|---|---|",
        "| `steps.jsonl` / `meta.json` 里的密钥、邮箱、密码、`.env` 片段 | **能**，命中即拒绝导出 |",
        "| **截图里的内容**（屏幕上的密码框、登录页、文件正文） | **不能** —— 正则扫不了图像 |",
        "",
        "所以这份数据的保证来自另外两条，**不是来自这个脚本**：",
        "",
        "1. 采集跑在干净快照里：无 `.env`、不登录真实账号、不同步个人文件",
        "2. 导出后人工抽查截图（抽查结果请手动补在本文件末尾）",
        "",
        "## 逐条结果",
        "",
        "| 轨迹 | 步数 | 截图 | 扫描文件 | 结果 |",
        "|---|---:|---:|---:|---|",
    ]
    for r in reports:
        lines.append(
            f"| {r.trajectory} | {r.steps} | {r.frames} | {r.scanned_files} | "
            f"{'✅ ' if r.exported else '⛔ '}{r.note} |"
        )

    if blocked:
        lines += ["", "## 被拦下的轨迹", ""]
        for r in blocked:
            for hit in r.hits:
                lines.append(
                    f"- `{r.trajectory}` / {hit.file}：{hit.reason}（{hit.rule}）`{hit.masked}`"
                )

    lines += [
        "",
        "## 人工抽查（**请填写**）",
        "",
        "- 抽查了几张截图：",
        "- 抽查方式（随机 / 按任务各抽几张）：",
        "- 有没有看到敏感内容：",
        "- 抽查人与日期：",
        "",
    ]

    # **全部被拦下时目录还不存在**（export_one 只在放行后才建目录）。
    # 而那正是最需要看报告的时候：报告要说清楚拦了什么、为什么。
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / "EXPORT-REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "export-report.json").write_text(
        json.dumps(
            {"include_frames": include_frames, "reports": [r.as_dict() for r in reports]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def main() -> int:
    _console()
    parser = argparse.ArgumentParser(description="导出轨迹（带敏感信息扫描）")
    parser.add_argument("--root", default="outputs/trajectories", help="轨迹根目录")
    parser.add_argument("--out", required=True, help="导出到哪")
    parser.add_argument("--only", action="append", default=[], help="只导出指定轨迹 id，可多次给")
    parser.add_argument("--no-frames", action="store_true", help="不带截图（平时的默认做法）")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"轨迹目录不存在：{root}")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    include_frames = not args.no_frames

    traj_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if args.only:
        wanted = set(args.only)
        traj_dirs = [d for d in traj_dirs if d.name in wanted]

    print("=" * 62)
    print("轨迹导出")
    print("=" * 62)
    print(f"  轨迹 {len(traj_dirs)} 条    带截图：{'是' if include_frames else '否'}")
    if include_frames:
        print("  ⚠ 带截图导出。请确认本机是**采集专用的干净快照**：")
        print("    无 .env、未登录真实账号、未同步个人文件。")

    reports = [export_one(d, out_dir, include_frames) for d in traj_dirs]
    for r in reports:
        mark = "✅" if r.exported else "⛔"
        print(f"  {mark} {r.trajectory:<28}{r.steps:>4} 步  {r.frames:>4} 图  {r.note}")

    path = write_report(reports, out_dir, include_frames)
    blocked = sum(1 for r in reports if not r.exported)
    print(f"\n  报告 {path}")
    if blocked:
        print(f"  ⛔ {blocked} 条被拦下，**没有导出**。先处理命中项，再重跑。")
    print("  ⚠ 报告末尾的人工抽查一节需要手动填写——正则扫不了截图内容。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
