"""蒸馏采集专用的测试文件 —— **与评测环境分开**。

## 为什么不复用 `setup_env.py` 建的那些文件

`setup_env.py` 建的 `测试文档.txt` / `源文件.txt` 等，是 `basic_tasks.yaml`
与 `desktop_20.yaml` 两个**评测任务集**的操作对象。采集数据如果也用它们，
训练时模型就见过了评测任务的目标文件——那是测试泄漏，训完的成绩不可信。

所以采集另起一套文件，文件名带 `采集` 前缀，评测任务一个都不碰。

## 内容固定

同 `setup_env.py` 的理由：内容每轮都一样，各轮之间才可比。

用法：

    python tasks/setup_collect.py           # 建文件
    python tasks/setup_collect.py --check   # 只看在不在
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TEST_DIR = Path("C:/agent-test") if sys.platform == "win32" else Path("/tmp/agent-test")

#: 采集专用文件。**名字不得与 setup_env.py 的任何文件相同**，
#: `tests/test_collect_variants.py` 钉住这一条。
COLLECT_FILES: dict[str, str] = {
    "采集文档A.txt": "采集样本 A\r\n第一行\r\n第二行\r\n第三行\r\n",
    "采集文档B.txt": "采集样本 B\r\n内容与 A 不同，避免模型靠记忆答题\r\n",
    "采集文档C.txt": "采集样本 C\r\n用于第三组采集变体\r\n",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="建采集专用测试文件")
    parser.add_argument("--check", action="store_true", help="只检查，不写入")
    args = parser.parse_args()

    print("=" * 62)
    print("蒸馏采集环境")
    print("=" * 62)
    print(f"  目录 {TEST_DIR}")

    if args.check:
        missing = [name for name in COLLECT_FILES if not (TEST_DIR / name).exists()]
        for name in COLLECT_FILES:
            print(f"  {name}  {'存在' if (TEST_DIR / name).exists() else '**不存在**'}")
        if missing:
            print(f"\n  缺 {len(missing)} 个，跑 python tasks/setup_collect.py 建。")
            return 1
        return 0

    TEST_DIR.mkdir(parents=True, exist_ok=True)
    for name, body in COLLECT_FILES.items():
        (TEST_DIR / name).write_text(body, encoding="utf-8", newline="")
        print(f"  已写 {name}")
    print("\n  完成。采集任务用 tasks/collect_variants.yaml。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
