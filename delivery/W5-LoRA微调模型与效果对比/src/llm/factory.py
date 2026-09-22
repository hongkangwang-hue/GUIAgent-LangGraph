"""后端工厂 —— 一个 `--provider` 开关切在线/离线。

## 为什么需要这个文件

M2 的验收标准第 8 条是「新增一个模型后端不需要改动 Agent 层与 Loop 层」。
这条一直是满足的：Agent 层只认 `LLMBackend`。

但**脚本层**没被这条覆盖。审计发现有七处直接写死了
`OpenAICompatBackend(...)`：

    scripts/run_basic_tasks.py   scripts/stability_run.py   cli/main.py
    eval/action.py   eval/grounding.py   scripts/verify_llm.py
    scripts/verify_mode_b.py

于是「Agent 层无感知」在实践中变成了「只要想换后端，就得改七个文件」。
离线版要跑同一套基础任务做对比，绕不开这一层。

工厂把「provider 名 → 后端实例」这件事收成一份。加平台照旧只需在
`llm.providers.PROVIDERS` 里加一条记录；加**一类**后端（比如本地推理）
才动这里。

## 在线与离线共用同一条代码路径

这是对比实验能成立的前提。两边只在「请求发到哪」这一点上不同，
提示词构造、坐标空间、动作解析、用量记账全都一样——不一样的话，
跑出来的差值分不清是模型带来的还是实现带来的。

## 用法

    from llm.factory import add_backend_args, build_backend, describe

    parser = argparse.ArgumentParser()
    add_backend_args(parser)
    args = parser.parse_args()

    backend = build_backend(args)
    print(describe(backend))

    # 在线：--provider dashscope
    # 离线（进程内，宿主机上）：--provider local
    # 离线（走 HTTP，客机上）：--provider selfhost --base-url http://<宿主机IP>:8000/v1
"""

from __future__ import annotations

import argparse
from typing import Any

from llm.base import LLMBackend

#: 走本地推理的 provider 名。其余名字都去 `llm.providers` 里查。
LOCAL_PROVIDER = "local"

#: 连自建本地服务时的超时（秒）。理由见 `build_backend`。
LOCAL_TIMEOUT_S = 180.0


def add_backend_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """挂上选后端要用的那几个参数。

    所有脚本共用同一组参数名，是为了让「在线跑一遍、离线跑一遍」这件事
    在命令行上就是换一个词，不用记每个脚本各自的写法。
    """
    group = parser.add_argument_group("模型后端")
    group.add_argument(
        "--provider",
        default="",
        help=f"模型平台；{LOCAL_PROVIDER} 表示本地离线推理。留空读 .env 的 LLM_PROVIDER",
    )
    group.add_argument("--model", default="", help="覆盖平台默认模型 / 本地模型路径")
    group.add_argument(
        "--base-url",
        default="",
        help="覆盖平台端点。自建服务用，例如 http://192.168.112.1:8000/v1",
    )
    group.add_argument(
        "--adapter",
        default="",
        help="QLoRA adapter 目录，仅本地后端有效。给了就是「离线-微调」那一档",
    )
    group.add_argument(
        "--max-pixels",
        type=int,
        default=0,
        help="本地后端的视觉 token 上限。默认与训练时一致（896×896）",
    )
    group.add_argument(
        "--no-4bit",
        action="store_true",
        help="本地后端不做 4-bit 量化。显存够才用——训练用的是 4-bit，改了就不同构",
    )
    group.add_argument(
        "--pin-prompt",
        action="store_true",
        help="本地后端强制使用训练时的提示词（分布内对照组），忽略 Session 注入的那套",
    )
    return parser


def build_backend(args: Any = None, **overrides) -> LLMBackend:
    """按参数造一个后端。

    ``args`` 是 argparse 的结果（或任意有同名属性的对象），``overrides``
    可以逐项覆盖。两者都不给时读 .env 的 `LLM_PROVIDER`。
    """
    from llm.providers import load_dotenv_if_present

    def pick(name: str, default=None):
        if name in overrides:
            return overrides[name]
        return getattr(args, name, default) if args is not None else default

    provider = str(pick("provider", "") or "").strip().lower()
    model = str(pick("model", "") or "").strip()

    # .env 先读，否则 LLM_PROVIDER=local 这种配置法读不到
    load_dotenv_if_present()

    if not provider:
        import os

        provider = os.getenv("LLM_PROVIDER", "").strip().lower()

    if provider == LOCAL_PROVIDER:
        from llm.qwen_vl_local import DEFAULT_MODEL, MAX_PIXELS, QwenVLLocalBackend

        return QwenVLLocalBackend(
            model=model or DEFAULT_MODEL,
            adapter=str(pick("adapter", "") or "").strip(),
            load_4bit=not bool(pick("no_4bit", False)),
            max_pixels=int(pick("max_pixels", 0) or MAX_PIXELS),
            pin_prompt=bool(pick("pin_prompt", False)),
        )

    from llm.openai_compat import OpenAICompatBackend
    from llm.providers import resolve

    config = resolve(
        provider or None,
        model=model or None,
        base_url=str(pick("base_url", "") or "").strip() or None,
    )
    # **本地服务要更长的超时。** 默认 90 秒是按云端 API 的速度定的
    # （50-100 tok/s），而本机 3B 4-bit 实测只有 7.3 tok/s——差一个数量级。
    # 2026-08-24 的离线-基座组 0/25 就栽在这：调用没失败，是**客户端等不及
    # 先掐了**，报出来的是 `Request timed out`，看着像网络问题，其实是
    # 拿云端的耐心去等本地的算力。
    #
    # 服务端已把生成长度压到 384（最坏 ~53 秒），这里再留一倍余量。
    timeout = LOCAL_TIMEOUT_S if config.provider.weights_local else None
    return OpenAICompatBackend(config, **({"timeout": timeout} if timeout else {}))


def describe(backend: LLMBackend) -> str:
    """一行人话，进脚本的抬头与报告。

    **在线还是离线要一眼看得出来**：两组结果放在一起时，光看模型名分不清
    ——`qwen3-vl-8b` 和 `Qwen2.5-VL-3B` 都像是「Qwen」。
    """
    from llm.qwen_vl_local import QwenVLLocalBackend

    if isinstance(backend, QwenVLLocalBackend):
        parts = [f"离线 / 本地 {backend.model}"]
        if backend.adapter:
            parts.append(f"+ adapter {backend.adapter}")
        parts.append("4-bit NF4" if backend.load_4bit else "未量化")
        if backend.pin_prompt:
            parts.append("训练提示词（分布内）")
        return "  ".join(parts)

    config = getattr(backend, "config", None)
    provider = getattr(config, "provider", None)
    label = getattr(provider, "label", backend.name)
    if getattr(provider, "weights_local", False):
        # 自建服务：走的是 HTTP，但权重在本机 GPU 上
        return f"离线 / {label} / {backend.model} @ {getattr(config, 'base_url', '')}"
    return f"在线 / {label} / {backend.model}"


def is_offline(backend: LLMBackend) -> bool:
    """这个后端跑起来，数据会不会出本机。

    报告里「数据边界」那一栏、存档文件名的后缀、以及「要不要覆盖 M2 的
    在线交付数据」这三处判断，全靠它。

    ## 两条路都要认

    离线版有两种接法，**判据不能只认其中一种**：

    | 接法 | 后端类型 | 场景 |
    |---|---|---|
    | 进程内 | `QwenVLLocalBackend` | 宿主机上批量评测（`eval/action.py --local`） |
    | 走 HTTP | `OpenAICompatBackend` + `selfhost` | 客机跑任务（M0 定的拓扑） |

    第二种最容易漏：它和连百炼用的是**同一个后端类**，光看类型分不出来。
    漏了的后果很具体——客机跑完一整轮离线实验，存档标成 `-online`，
    还顺手把 M2 的在线交付数据覆盖了。

    所以类型只是其中一条判据，另一条绑在 `Provider.weights_local` 这个
    事实上。**都不看名字**：名字可以被 `--model` 改花。
    """
    from llm.qwen_vl_local import QwenVLLocalBackend

    if isinstance(backend, QwenVLLocalBackend):
        return True
    provider = getattr(getattr(backend, "config", None), "provider", None)
    return bool(getattr(provider, "weights_local", False))
