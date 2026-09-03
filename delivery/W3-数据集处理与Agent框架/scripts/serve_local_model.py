"""把本地模型起成 OpenAI 兼容端点 —— 宿主机跑权重，客机 HTTP 调用。

## 为什么需要它

M0 定的拓扑（《硬件与部署环境》一节）：

> **执行位置：Agent 整体运行在客机内**（感知、规划调用、坐标换算、键鼠
> 执行全在客机）。本地模型仍跑在宿主机 / 服务器上，客机经 HTTP 调用其
> OpenAI 兼容端点——**权重在本地硬件上运行，这仍是「本地部署」，只是换了
> 传输方式**。

原因是硬的：GPU 在宿主机，客机没有直通（M0：「GPU 无需给虚拟机直通」）。
而全局约束 1 要求键鼠执行只发生在客机内。两条摆在一起，模型和 Agent 就
必须分处两台机器，中间只能走网络。

**这条 HTTP 不出本机**：宿主机与客机在同一台物理机上，走的是 VMware 的
虚拟网卡（本项目为 NAT / VMnet8）。截图从客机发到宿主机的显存里，再没去过别处。这正是老师说的
「对数据保密比较高的部署环境」里模型服务的标准拓扑——模型在保密边界内，
客户端连它，而不是把数据发给外面。

## 为什么不用 vLLM

大纲技术栈点名了 vLLM，`docs/服务器环境搭建说明.md` 也写了它的起法。
但那份文档针对的是**短租的 Linux 服务器**，而这里是本机：

**vLLM 在 Windows 上没有原生支持**，要么装 WSL2 再做 GPU 直通，要么
放弃。为一个只服务单个客机、并发恒为 1 的场景引入 WSL，代价远大于收益。

所以这里用 `QwenVLLocalBackend.generate()` 起一个最小的兼容层。它与
vLLM 暴露的是同一个协议，客机侧的 `OpenAICompatBackend` 一行都不用改
——**哪天换成 vLLM 或换成真服务器，客机侧无感知**。

## 起法

宿主机（有 GPU 的那台）：

    # 离线-基座
    python scripts/serve_local_model.py

    # 离线-微调
    python scripts/serve_local_model.py --adapter finetune/outputs/20260824-170900/adapter

    # 让客机能连上（默认只听 127.0.0.1）
    python scripts/serve_local_model.py --host 0.0.0.0

客机 `.env`：

    LLM_PROVIDER=selfhost
    SELFHOST_BASE_URL=http://<宿主机在 VMnet8 上的 IP>:8000/v1
    SELFHOST_API_KEY=local            # 本服务不校验，填任意非空值

然后照常跑：

    python scripts/run_basic_tasks.py --execute --repeats 5

## 走 HTTP 与进程内直接调，图片不是同一张

`QwenVLLocalBackend` 有两种用法，喂给模型的图片经过的处理**不一样**：

| 路径 | 谁在用 | 图片怎么到模型 |
|---|---|---|
| 进程内 | `eval/action.py --local` | PIL 缩放后直接进 processor，无有损压缩 |
| 走 HTTP | 客机跑任务（本服务） | `encode_screenshot` 压成 JPEG q85 → base64 |

**这不是缺陷，恰恰是对比要的。** 在线版走的也是 `encode_screenshot`，
同样是 JPEG q85。所以做「在线 vs 离线」时，HTTP 这条路比进程内那条更
可比——两边的图片预处理完全一致，差的只有模型。

代价是：进程内测出来的数（比如验证集上的动作类型准确率 85.2%）与客机跑
任务测出来的数**不能直接互推**，中间隔着一次 JPEG 压缩。报告里两组数
要分开标明来路。

## 安全边界

**默认只监听 127.0.0.1。** 要给客机用才需要 `--host 0.0.0.0`，而那会让
同网段的任何机器都能调用这个端点。本服务**不做鉴权**（`Authorization`
头收下就扔），所以：

- 只在 VMware 的虚拟网段（VMnet8 / VMnet1）上开
- 不要在公共 Wi-Fi 或办公网上开着 `--host 0.0.0.0` 不管

这不是能力不足，是范围选择：本项目的威胁模型里没有"同网段的敌人"，
加一套鉴权只会让客机配置多一个出错的地方。但**默认值必须是安全的那个**。
"""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import logging
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "src")
)  # 交付包把各个包放在 src/ 下

logger = logging.getLogger("serve_local_model")

#: 一次只让一个请求进 GPU。
#:
#: uvicorn 会把同步端点丢进线程池，多个请求于是能真并发。而这里只有一张
#: 卡、一份权重：两个 `generate()` 同时跑，轻则显存翻倍 OOM，重则拿到
#: 交错的输出。并发恒为 1 是这个场景的事实，用锁把它变成约束。
_GPU_LOCK = threading.Lock()

#: 生成长度上限，**服务端强制**，不管客户端要多少。
#:
#: 2026-08-24 实测：本机 3B 4-bit 持续 **7.3 tok/s**。而
#: `OpenAICompatBackend` 默认发 `max_tokens=1024`（对云端 API 是合理的，
#: 那边 50-100 tok/s，1024 也就十几秒），到本地就是 **140 秒**，
#: 而客户端超时是 90 秒——**模型只要没提前收口，必然超时**。
#:
#: 离线-基座组第一次跑因此 0/25：三个任务连规划都没出来（步数 0.0），
#: 因为规划那一次调用就超时了。
#:
#: 384 的依据也是实测：一条动作 JSON 约 40 token，一份任务拆解约 150。
#: 384 留了一倍余量，最坏耗时 384/7.3 ≈ 53 秒，稳稳在 90 秒之内。
#: **正常输出根本碰不到这个上限**，它只截断跑飞的生成——而跑飞的生成
#: 本来也会超时失败，早点失败还便宜些。
DEFAULT_MAX_NEW_TOKENS = 384


def _decode_data_url(url: str):
    """``data:image/jpeg;base64,...`` → PIL Image。

    只认 data URL，**不认 http(s) 链接**：认了就等于让本服务去访问外网，
    而这个服务存在的全部理由是数据不出本机。
    """
    from PIL import Image

    if not url.startswith("data:"):
        raise ValueError("只接受内联的 data URL 图片；本服务不会去下载外部链接")
    _, _, payload = url.partition(",")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"图片 base64 解不开：{exc}") from exc
    return Image.open(io.BytesIO(raw)).convert("RGB")


def to_hf_messages(messages: list[dict]) -> tuple[list[dict], list]:
    """OpenAI 格式的 messages → (HF chat 格式, 图片列表)。

    两边的差别只在图片怎么带：OpenAI 用
    ``{"type": "image_url", "image_url": {"url": "data:..."}}``，
    HF 的 processor 用 ``{"type": "image"}`` 占位 + 一个平行的 images 列表。

    **顺序必须严格保持。** processor 按占位符出现的先后去 images 里取，
    错开一位模型看到的就是上一帧——不报错，只是每步都基于过期画面决策。
    """
    hf_messages: list[dict] = []
    images: list = []

    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")

        if isinstance(content, str):
            hf_messages.append({"role": role, "content": [{"type": "text", "text": content}]})
            continue

        parts: list[dict] = []
        for part in content or []:
            kind = part.get("type")
            if kind == "text":
                parts.append({"type": "text", "text": part.get("text", "")})
            elif kind == "image_url":
                image = _decode_data_url((part.get("image_url") or {}).get("url", ""))
                images.append(image)
                parts.append({"type": "image", "image": image})
            else:
                raise ValueError(f"不支持的内容类型 {kind!r}")
        hf_messages.append({"role": role, "content": parts})

    return hf_messages, images


def build_app(backend, cap: int = DEFAULT_MAX_NEW_TOKENS):
    """造 FastAPI 应用。后端从外面传进来，测试时可以塞假的。

    ``cap`` 是生成长度的硬上限，见 `DEFAULT_MAX_NEW_TOKENS`。
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse

    app = FastAPI(title="GUIAgent 本地模型服务", version="1.0")

    @app.get("/health")
    def health():
        """客机连不上时先打这个，把「服务没起」和「模型没加载」分开。"""
        return {
            "status": "ok",
            "model": backend.model,
            "adapter": backend.adapter or None,
            "loaded": backend._model is not None,
            "max_new_tokens": cap,
        }

    @app.get("/v1/models")
    def models():
        """有些客户端启动时会探这个端点，缺了会报一个很难懂的错。"""
        return {
            "object": "list",
            "data": [{"id": backend.model, "object": "model", "owned_by": "local"}],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(body: dict):
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="messages 不能为空")
        if body.get("stream"):
            # 明确拒绝而不是假装支持：客户端等一个 SSE 流却收到一个 JSON，
            # 报出来的错会指向解析层，查半天才发现是这里没实现
            raise HTTPException(status_code=400, detail="本服务不支持 stream=true")

        try:
            hf_messages, images = to_hf_messages(messages)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # **上限由服务端说了算。** 客户端的 max_tokens 只能往小了要，
        # 不能往大了要——它不知道这台机器每秒能吐几个 token。
        asked = int(body.get("max_tokens") or 0)
        max_tokens = min(asked, cap) if asked > 0 else cap
        started = time.perf_counter()
        with _GPU_LOCK:
            try:
                text, usage, latency_ms = backend.generate(
                    hf_messages, images, max_new_tokens=max_tokens
                )
            except Exception as exc:  # noqa: BLE001 - 后端可能抛任何东西
                logger.exception("推理失败")
                return JSONResponse(
                    status_code=500,
                    content={"error": {"message": str(exc), "type": type(exc).__name__}},
                )
        waited_ms = (time.perf_counter() - started) * 1000.0

        logger.info(
            "完成：图 %d 张，in=%d out=%d，推理 %.0fms（含排队 %.0fms）",
            len(images),
            usage.prompt_tokens,
            usage.completion_tokens,
            latency_ms,
            waited_ms,
        )
        return {
            "id": f"local-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": backend.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            },
        }

    return app


def build_parser() -> argparse.ArgumentParser:
    """命令行参数。

    **抽成独立函数是为了能测。** 上一版把 `--max-new-tokens` 漏掉了，而
    `main()` 里已经在用 `args.max_new_tokens`——单测全绿（736 条），因为
    没有一条测试会去调 `main()`，于是这个 `AttributeError` 一路跑到了
    用户的服务启动那一刻，还是在权重都装完之后才炸。
    """
    parser = argparse.ArgumentParser(description="把本地 Qwen2.5-VL 起成 OpenAI 兼容端点")
    parser.add_argument("--model", default="", help="模型名或本地路径")
    parser.add_argument("--adapter", default="", help="QLoRA adapter 目录，给了就是微调版")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="监听地址。默认只本机可连；要给客机用才改 0.0.0.0（本服务不鉴权）",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-4bit", action="store_true", help="不做 4-bit 量化")
    parser.add_argument(
        "--preload",
        action="store_true",
        help="启动时装权重并跑一次热身推理（含图），而不是等第一个请求。挂机跑评测时用",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"生成长度硬上限，服务端强制（默认 {DEFAULT_MAX_NEW_TOKENS}）",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="彻底不联网：设 HF_HUB_OFFLINE，连启动时校验缓存的那几个请求也不发",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    # **必须在 import transformers 之前设。** huggingface_hub 在导入时就
    # 读这两个变量，之后再改不生效。本模块把 transformers 的 import 放在
    # `QwenVLLocalBackend._ensure_model` 里，所以这里设还来得及。
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        import uvicorn
    except ImportError:
        print("缺少 fastapi / uvicorn：pip install fastapi uvicorn", file=sys.stderr)
        return 1

    from llm.factory import describe
    from llm.qwen_vl_local import DEFAULT_MODEL, QwenVLLocalBackend

    backend = QwenVLLocalBackend(
        model=args.model or DEFAULT_MODEL,
        adapter=args.adapter,
        load_4bit=not args.no_4bit,
    )

    print("=" * 74)
    print("本地模型服务")
    print("=" * 74)
    print(f"  后端      {describe(backend)}")
    print(f"  监听      http://{args.host}:{args.port}/v1")
    if args.host not in ("127.0.0.1", "localhost"):
        print("  ⚠ 本服务不做鉴权。只在 VMware 虚拟网段上开，别挂在公共网络上。")
    print()
    print("  客机 .env：")
    print("    LLM_PROVIDER=selfhost")
    print(f"    SELFHOST_BASE_URL=http://<宿主机内网IP>:{args.port}/v1")
    print("    SELFHOST_API_KEY=local")
    print()

    if args.preload:
        # 挂机跑评测时先把权重装好，**并且跑一次真前向**。
        #
        # 只装权重不够：2026-08-24 实测，装完权重后的**第一次生成**花了
        # 153.7 秒才吐 17 个 token，而之后同样的请求只要 4.8 秒——差 32 倍。
        # 那是 CUDA kernel 首次编译 / cuDNN 自动调优的一次性开销。
        #
        # 不热身的话，这 150 秒会落在第一条任务的第一步上：既污染首轮的
        # 延迟数据，又很可能直接把客户端的 90 秒超时撞爆。
        print("  预加载权重……")
        backend._ensure_model()
        # **热身必须带图。** 纯文本热身只要 0.9 秒，因为它根本没碰视觉塔——
        # 而实测那次 153.7 秒的首次调用是带图的。真正要预热的是视觉编码
        # 那条路径（动态分辨率 + patch 化 + 视觉注意力的 kernel 选择）。
        # 用一张和真实截图同尺寸的图，让它走完整条路。
        from PIL import Image as _Image

        print("  热身推理（首次带图前向要编译 CUDA kernel，会慢）……")
        warm = time.perf_counter()
        backend.generate(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": _Image.new("RGB", (1024, 768), (30, 30, 30))},
                        {"type": "text", "text": "hi"},
                    ],
                }
            ],
            [_Image.new("RGB", (1024, 768), (30, 30, 30))],
            max_new_tokens=8,
        )
        print(f"  就绪（热身 {time.perf_counter() - warm:.1f}s）\n")

    try:
        uvicorn.run(
            build_app(backend, cap=args.max_new_tokens),
            host=args.host,
            port=args.port,
            log_level="warning",
        )
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
