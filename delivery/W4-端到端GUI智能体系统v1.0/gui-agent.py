"""命令行入口 —— **交付包专用，源仓库里没有这个文件**。

源仓库把各个包放在根目录，`python -m cli` 直接可用。交付包把包放在 `src/`
下，`python -m cli` 就找不到了（`pyproject.toml` 里的 `pythonpath = ["src"]`
只对 pytest 生效，对 `python -m` 不生效）。

这个文件把 `src` 加进 `sys.path` 再转交给 `cli.main`，让交付包里的命令行
用法与源仓库保持一致：

    python gui-agent.py --help
    python gui-agent.py run "打开记事本" --provider dashscope
    python gui-agent.py run "打开记事本" --provider dashscope --execute
    python gui-agent.py replay <trajectory-id>
    python gui-agent.py config --show

**`--execute` 会真的操作鼠标键盘。** 本项目全局约束要求一切键鼠执行只发生
在隔离虚拟机客机内，宿主机不受控制。不加 `--execute` 时只演练：走完整链路
（截图、问模型、解析动作、记轨迹）但不发出任何键鼠事件。

见 README §三 偏差 4。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cli.main import app  # noqa: E402

if __name__ == "__main__":
    app()
