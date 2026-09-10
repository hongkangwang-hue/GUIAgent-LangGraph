"""命令行界面。

五个命令对应 M2 任务 7：``run`` / ``run --task-file`` / ``replay`` /
``label`` / ``config --show``。

入口**惰性导入**：``import cli`` 不该把 typer、rich、langchain 全拉起来。
用 ``python -m cli`` 或安装后的 ``gui-agent`` 命令启动。
"""

from __future__ import annotations

__all__ = ["app", "main"]


def __getattr__(name: str):
    if name in __all__:
        from cli import main as _main

        return getattr(_main, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
