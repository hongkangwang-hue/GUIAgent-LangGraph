"""在客机内预置测试环境 —— 跑基础任务前执行一次。

建 `C:/agent-test/` 并放入固定内容的测试文件。**内容固定是刻意的**：
任务判据会检查文件是否被正确打开，内容每次都一样，结果才可比。

也顺带检查任务清单里点名的程序是否存在（记事本 / Edge / 计算器），
缺哪个现在就知道，而不是跑到第 3 个任务才发现。

    python tasks/setup_env.py
    python tasks/setup_env.py --check   # 只检查不写入
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

TEST_DIR = Path("C:/agent-test") if sys.platform == "win32" else Path("/tmp/agent-test")

TEST_FILE_NAME = "测试文档.txt"
TEST_FILE_BODY = """这是 GUI Agent 项目的测试文档。

本文件由 tasks/setup_env.py 生成，内容固定不变——
基础任务「打开指定文件」的判定依赖窗口标题包含「测试文档」，
而每轮评测都必须从同一个起点开始，内容变了结果就不可比。

测试行 1：Hello World
测试行 2：中文内容测试
测试行 3：1234567890
"""

# ===================================================================== #
# M5 任务集（tasks/desktop_20.yaml）新增的固定文件
#
# **内容必须固定**，判据直接引用其中的字符串。改了内容就要同步改判据，
# 否则任务会以「文件不含 X」的形式静默失败，看起来却像是 Agent 做错了。
# ===================================================================== #

#: `copy_paste_text` 的源。判据查目标文件是否含这一行。
SOURCE_BODY = """需要被复制的内容

这一行是 copy_paste_text 任务的判据依据。
"""

#: `rename_file` 的对象。内容无所谓，存在与否才是判据。
RENAME_BODY = "这个文件会被重命名为 已改名.txt。\n"

#: `delete_file` 的对象。**任务集里唯一的破坏性操作，限定在测试目录内。**
TEMP_BODY = "这个文件会被删除。删错了重跑 setup_env.py 即可恢复。\n"

#: `scroll_document` 用。**必须一屏放不下**，否则测不到滚动。
LONG_DOC_BODY = (
    "这是一个长文档，用于测试滚动到末尾的能力。\n\n"
    + "".join(f"第 {i} 行：填充内容，使文档超出一屏。\n" for i in range(1, 201))
    + "\n这是最后一行\n"
)

#: 任务运行时**生成**的文件。它们不该在起点存在——`--clean` 负责删掉。
GENERATED = ("笔记.txt", "目标文件.txt", "新建文件夹")


def _console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def check_programs() -> list[tuple[str, bool, str]]:
    """任务清单点名的程序在不在。"""
    checks = []
    if sys.platform == "win32":
        candidates = [
            ("记事本", "notepad.exe"),
            ("计算器", "calc.exe"),
        ]
        for label, exe in candidates:
            found = shutil.which(exe)
            checks.append((label, bool(found), found or "未找到"))

        edge = Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")
        edge_alt = Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe")
        hit = edge if edge.exists() else (edge_alt if edge_alt.exists() else None)
        checks.append(("Microsoft Edge", bool(hit), str(hit) if hit else "未找到"))
    else:
        checks.append(("（非 Windows，跳过程序检查）", True, ""))
    return checks


def check_deps() -> list[tuple[str, bool, str]]:
    """判定器依赖的库。缺了会让成功判定整体失效，必须提前发现。"""
    out = []
    for module, why in (("psutil", "进程判据"), ("uiautomation", "窗口标题判据")):
        try:
            __import__(module)
            out.append((module, True, why))
        except ImportError:
            out.append((module, False, f"{why}将无法使用 —— pip install {module}"))
    return out


def clean_generated() -> int:
    """删掉任务运行时生成的文件与目录。返回删掉的数量。

    **这不是清理垃圾，是重建起点。** `write_note` 的判据是「笔记.txt 含
    某句话」，那么起点就必须是「笔记.txt 不存在」——上一轮留下的文件会让
    precondition 判定起点未建立，整轮记为无效。
    """
    removed = 0
    for name in GENERATED:
        target = TEST_DIR / name
        if not target.exists():
            continue
        try:
            shutil.rmtree(target) if target.is_dir() else target.unlink()
            removed += 1
        except OSError as exc:
            print(f"  [警告] 删不掉 {target}：{exc}")
    return removed


def main() -> int:
    _console()
    parser = argparse.ArgumentParser(description="预置基础任务的测试环境")
    parser.add_argument("--check", action="store_true", help="只检查，不写入文件")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="只删掉任务生成物（笔记.txt 等），不重写固定文件。供任务 reset 调用",
    )
    args = parser.parse_args()

    if args.clean:
        print(f"已清理 {clean_generated()} 项生成物：{'、'.join(GENERATED)}")
        return 0

    print("=" * 62)
    print("基础任务测试环境")
    print("=" * 62)

    test_file = TEST_DIR / TEST_FILE_NAME
    if args.check:
        print(f"  测试目录  {TEST_DIR}  {'存在' if TEST_DIR.exists() else '**不存在**'}")
        print(f"  测试文件  {test_file.name}  {'存在' if test_file.exists() else '**不存在**'}")
    else:
        TEST_DIR.mkdir(parents=True, exist_ok=True)
        test_file.write_text(TEST_FILE_BODY, encoding="utf-8", newline="\r\n")
        (TEST_DIR / "messages.log").write_text("", encoding="utf-8")
        print(f"  已建 {TEST_DIR}")
        print(f"  已写 {test_file.name}（{len(TEST_FILE_BODY)} 字符）")
        print("  已清空 messages.log")

        # M5 任务集的固定文件。**每次都重写**——上一轮任务可能改过它们
        # （append_line 会往测试文档里加内容），不重置的话下一轮的起点
        # 就不是同一个了。
        for name, body in (
            ("源文件.txt", SOURCE_BODY),
            ("待改名.txt", RENAME_BODY),
            ("待删除.txt", TEMP_BODY),
            ("长文档.txt", LONG_DOC_BODY),
        ):
            (TEST_DIR / name).write_text(body, encoding="utf-8", newline="\r\n")
            print(f"  已写 {name}")

        # 生成物必须删掉。留着的话它们的 precondition
        # （`should_exist: false`）会判定起点未建立，整轮记为无效，
        # 而无效轮不进分母——样本量会悄悄变少。
        print(f"  已清理生成物 {clean_generated()} 项")

    print("\n  程序：")
    ok = True
    for label, found, detail in check_programs():
        print(f"    {'[OK]' if found else '[缺]'} {label:<16} {detail}")
        ok = ok and found

    print("\n  判定器依赖：")
    for module, found, why in check_deps():
        print(f"    {'[OK]' if found else '[缺]'} {module:<16} {why}")
        ok = ok and found

    print()
    if ok:
        print("  环境就绪。下一步：")
        print("    python tasks/mock_messenger.py     # 另开一个窗口，保持运行")
        print("    python scripts/run_basic_tasks.py --execute")
    else:
        print("  **有缺项，先补齐再跑任务**——缺判定器依赖会让成功判定整体失效，")
        print("  而那会让整批评测数据作废。")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
