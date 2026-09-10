"""等某个进程起来，起来了就立刻返回。

## 为什么不用 `timeout /t 3`

「关闭应用」的 reset 是：杀掉计算器 → `start calc` → 等 3 秒 → 起点检查。

计算器是 UWP 应用，**冷启动经常超过 3 秒**——尤其是刚被 `taskkill` 干掉
之后的第一次。2026-08-25 的离线-微调组实测：25 轮里第 1 轮起点未建立，
后面 4 轮都过了，正是冷启动慢、之后系统缓存变热的形态。

固定等待有两头不讨好的毛病：

- 等短了：偶发地把好轮次判成无效，而**无效轮是不进分母的**，
  于是分母悄悄从 5 变成 4
- 等长了：25 轮每轮多等几秒，白白拖长挂机时间

轮询两头都占：起来了立刻走，没起来才多等。

## 用法

    python tasks/wait_for_process.py CalculatorApp.exe
    python tasks/wait_for_process.py CalculatorApp.exe --timeout 15

**超时不报错**（返回码仍是 0）：真的没起来，让**起点检查**去判——
它给的信息比这里准确得多（进程列表、找到几个），而且那才是它的职责。
这个脚本只负责「别过早往下走」。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "src")
)  # 交付包把各个包放在 src/ 下

#: 默认最多等多久。UWP 冷启动实测在 3-8 秒之间，12 秒留足余量。
DEFAULT_TIMEOUT_S = 12.0

#: 每次查询之间隔多久。0.3 秒足够快，又不至于把 CPU 打满。
POLL_INTERVAL_S = 0.3


def wait(name: str, timeout: float = DEFAULT_TIMEOUT_S) -> tuple[bool, float]:
    """等到进程出现，返回 (是否等到, 实际等了多久)。"""
    from core.verify import check_process

    started = time.monotonic()
    while True:
        if check_process(name, should_run=True).passed:
            return True, time.monotonic() - started
        if time.monotonic() - started >= timeout:
            return False, time.monotonic() - started
        time.sleep(POLL_INTERVAL_S)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="等某个进程起来")
    parser.add_argument("name", help="进程名，如 CalculatorApp.exe")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    return parser


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args()
    found, waited = wait(args.name, args.timeout)
    if found:
        print(f"{args.name} 已就绪（等了 {waited:.1f}s）")
    else:
        # 不返回非零：reset 的失败不该中断整轮，起点检查会给出更准确的判断
        print(f"{args.name} 等了 {waited:.1f}s 仍未出现——交给起点检查判定")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
