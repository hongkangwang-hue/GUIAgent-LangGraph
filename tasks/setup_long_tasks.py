"""长任务测试集的环境准备 —— M4 交付物 4。

## 为什么每轮都重写，而不是建好就不管

`long_append_and_save` 会往底稿里追加内容，`long_replace_word` 会改掉文中的词。
**跑完一轮，文件就不是原来那份了。** 第二轮的起点与第一轮不同，5 轮的数据
就不可比——这正是 `basic_tasks.yaml` 文件头记的那个教训：
reset 必须恢复到同一个起点，不只是清进程。

`long_write_three_lines` 的产物 `三行.txt` 要**删掉**：它的起点检查是
「这个文件不存在」，留着的话 Agent 什么都不做，判据也会打勾。

用法：

    python tasks/setup_long_tasks.py           # 重置
    python tasks/setup_long_tasks.py --check   # 只看状态
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TEST_DIR = Path("C:/agent-test") if sys.platform == "win32" else Path("/tmp/agent-test")

#: 每轮重写的底稿。内容固定，且**必须含「底稿原有内容」**——
#: `long_append_and_save` 用它验证 Agent 是「追加」而不是「清空重写」。
FIXTURES: dict[str, str] = {
    "长任务底稿.txt": "底稿原有内容\r\n第二段\r\n第三段\r\n",
    "长任务替换.txt": "这里有一个旧词\r\n这一行也有旧词\r\n最后一行还是旧词\r\n",
}

#: 任务的产物，每轮**必须删掉**。留着会让起点检查直接打勾。
GENERATED = ("三行.txt",)


def _console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def reset() -> tuple[int, int]:
    """重写底稿、删掉产物。返回 (重写几个, 删掉几个)。"""
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    for name, body in FIXTURES.items():
        (TEST_DIR / name).write_text(body, encoding="utf-8", newline="")
    removed = 0
    for name in GENERATED:
        path = TEST_DIR / name
        if path.exists():
            path.unlink()
            removed += 1
    return len(FIXTURES), removed


def main() -> int:
    _console()
    parser = argparse.ArgumentParser(description="长任务测试集环境准备")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    print("=" * 62)
    print("长任务测试环境")
    print("=" * 62)

    if args.check:
        ok = True
        for name, body in FIXTURES.items():
            path = TEST_DIR / name
            same = path.exists() and path.read_text(encoding="utf-8") == body
            print(f"  {name}  {'内容正确' if same else '**缺失或被改过**'}")
            ok = ok and same
        for name in GENERATED:
            exists = (TEST_DIR / name).exists()
            print(f"  {name}  {'**还在，起点不干净**' if exists else '已清理'}")
            ok = ok and not exists
        return 0 if ok else 1

    written, removed = reset()
    print(f"  重写底稿 {written} 个，清理产物 {removed} 个")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
