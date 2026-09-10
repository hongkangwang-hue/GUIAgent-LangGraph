"""本地 VLM 定位后端 —— M3 的核心，模式 B 的落地点。

## 它决定微调成果是否承重

M3 的定位是「让微调成果在系统里承重」。一个跑完就归档的微调实验价值有限；
一个被系统真正用起来、并且有明确对照数据的 grounding 模块，才是这个项目
在「模型微调」这项能力上最有说服力的证据。

这个文件就是那个接点。

## 模式 A 与模式 B 的分工

|  | 模式 A（`NativeGrounding`） | 模式 B（本模块） |
|---|---|---|
| 规划模型输出 | `{action, x, y}` | `{action, target_description}` |
| 坐标来源 | 规划模型 | **本地微调模型** |
| 送 API 的截图 | 原分辨率 | 可降分辨率（省 token） |
| 单步模型调用 | 1 次 | 2 次 |

## 对收益的预期要摆正

M2 的 170 步实测里：`grounding` 段耗时合计 **0.00ms**、来源全是 `native`、
**没有一次失败是定位造成的**。失败全在规划层（提前报完成、重复动作、
拆解粒度）和环境层。

模式 B 换的是坐标的**来源**，换不掉规划层的毛病。所以：

- **预期收益在 token 成本，不在成功率**
- 「模式 B 成功率低于模式 A」不视为失败（M3 降级路径 4），作为负面结论
  写进报告即可
- 报告要**主动**给出这个预期与实测的对照，而不是等数据难看再解释

## 三层兜底

    本地模型 → UIA/OCR 文本匹配 → 回落模式 A（用规划模型给的坐标）

每一层都要在 `GroundingResult.source` 里留下痕迹。**分不清来源，
M3 分析「grounding 到底准不准」时就会把兜底救回来的算进模型的功劳里。**

## 坐标空间

本地模型看的是**原分辨率截图**，输出的坐标要经 `CoordinateScaler` 转回
Loop 使用的模型坐标系。这里是 M1 的多坐标系注册能力真正被用上的地方。

**接入时第一件事是实测确认本地模型的输出空间**，跟接任何新后端一样。
微调数据里写的是 [0,1000) 归一化（见 `finetune/dataset.py`），
所以微调后的模型应当输出这个空间——但**「应当」要被实测确认**。

## 当前状态

**接口与兜底逻辑已就位，模型加载部分待 M3 接入。** 现在实例化会在
`_ensure_model()` 处明确报错说明缺什么，而不是静默返回空结果——
静默失败会让人以为「模型定位不准」，而实际是根本没加载。
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from grounding.base import GroundingBackend, GroundingResult
from llm.base import ActionIntent
from perception.capture import Screenshot
from perception.types import Point

logger = logging.getLogger(__name__)

#: 来源标识，进轨迹日志。
SOURCE_LOCAL = "local_vlm"
SOURCE_TEXT_FALLBACK = "uia_ocr_fallback"
SOURCE_NATIVE_FALLBACK = "native_fallback"

#: 微调时用的坐标空间，见 `finetune/dataset.py::COORD_SPACE`。
#: **训练与推理必须一致**，不一致时 loss 正常、分数莫名其妙地低。
MODEL_SPACE = (1000, 1000)

#: 定位提示词。与 `finetune/dataset.py::INSTRUCTION_TEMPLATE` 同构——
#: 训练时怎么问，推理时就怎么问。
PROMPT_TEMPLATE = "在这张界面截图中找到：{description}。返回它的中心坐标。"

_JSON_POINT = re.compile(r'[{"]\s*x["\s:]+(-?\d+)[,\s"]+y["\s:]+(-?\d+)', re.IGNORECASE)
_BARE_PAIR = re.compile(r"\(?\s*(-?\d+)\s*[,，]\s*(-?\d+)\s*\)?")


def parse_point(text: str) -> Point | None:
    """从模型输出里抠出坐标。

    先试 JSON，再试裸的数字对。**两条路都留着**是因为微调后的模型
    未必稳定输出合法 JSON——训练集只有 569 条，格式遵从度是可预期的
    薄弱环节，而一个能解析出坐标的回答不该因为少个花括号被判失败。
    """
    if not text:
        return None
    match = _JSON_POINT.search(text)
    if match:
        return Point(int(match.group(1)), int(match.group(2)))
    try:
        payload = json.loads(text.strip())
        if isinstance(payload, dict) and "x" in payload and "y" in payload:
            return Point(int(payload["x"]), int(payload["y"]))
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    match = _BARE_PAIR.search(text)
    if match:
        return Point(int(match.group(1)), int(match.group(2)))
    return None


class LocalVLMGrounding(GroundingBackend):
    """用本地微调模型把元素描述解析成坐标。

    ``adapter_path`` 指向 LoRA adapter 目录；为空时只加载 base 模型，
    用于跑档 1（零样本基线）。
    """

    name = SOURCE_LOCAL

    def __init__(
        self,
        space_width: int = MODEL_SPACE[0],
        space_height: int = MODEL_SPACE[1],
        base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        adapter_path: str = "",
        max_pixels: int = 896 * 896,
        fallback: GroundingBackend | None = None,
    ) -> None:
        self.space = (space_width, space_height)
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.max_pixels = max_pixels
        #: 本地模型定位失败时的兜底。传 `NativeGrounding` 即为「回落模式 A」。
        self.fallback = fallback
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------ #

    def _ensure_model(self):
        """惰性加载。**常驻显存，不每次重新加载。**

        冷加载可达数十秒，而 grounding 每步都要调用一次。M4 的延迟优化
        里「本地 grounding 模型常驻显存」说的就是这件事。
        """
        if self._model is not None:
            return
        started = time.perf_counter()
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "本地 grounding 需要 transformers 与 torch。"
                "先按 docs/开发环境配置文档.md 装好，再启用模式 B。"
            ) from exc

        self._processor = AutoProcessor.from_pretrained(
            self.base_model, min_pixels=256 * 28 * 28, max_pixels=self.max_pixels
        )
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.base_model,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        if self.adapter_path:
            path = Path(self.adapter_path)
            if not path.exists():
                raise RuntimeError(f"adapter 不存在：{path}。先跑 finetune/train_lora.py")
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, str(path))
        self._model.eval()
        logger.info("本地 grounding 模型加载完成，耗时 %.1fs", time.perf_counter() - started)

    def _infer(self, screenshot: Screenshot, description: str) -> str:
        """跑一次推理，返回原始文本。"""
        import torch
        from PIL import Image

        self._ensure_model()
        # BGR → RGB：OpenCV 与 PIL 的通道序不同，弄反了颜色全错而不报错
        image = Image.fromarray(screenshot.image[:, :, ::-1])
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT_TEMPLATE.format(description=description)},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(text=[text], images=[image], return_tensors="pt")
        inputs = inputs.to(self._model.device)
        with torch.no_grad():
            # 定位是确定性任务，不采样：同一张图同一个描述应当给同一个点
            generated = self._model.generate(**inputs, max_new_tokens=64, do_sample=False)
        trimmed = generated[0][inputs["input_ids"].shape[1] :]
        return self._processor.decode(trimmed, skip_special_tokens=True)

    # ------------------------------------------------------------------ #

    def locate(
        self,
        screenshot: Screenshot,
        target_description: str,
        intent: ActionIntent | None = None,
    ) -> GroundingResult:
        started = time.perf_counter()

        if not target_description:
            # 模式 B 下描述为空说明规划模型没按 schema 输出，
            # **这不是定位失败，是上游格式问题**，要分清
            return self._fail(started, "模型没给 target_description", intent, screenshot)

        try:
            raw = self._infer(screenshot, target_description)
        except Exception as exc:  # noqa: BLE001 —— 定位失败是常态，不抛给 Loop
            logger.warning("本地 grounding 推理失败：%s", exc)
            return self._fail(started, f"{type(exc).__name__}: {exc}"[:200], intent, screenshot)

        point = parse_point(raw)
        latency = round((time.perf_counter() - started) * 1000, 2)
        if point is None:
            return self._fail(started, f"输出里没有坐标：{raw[:120]!r}", intent, screenshot)

        # 越界的点当作没定位到。带着已知非法的坐标继续，最后会被安全白名单
        # 拦下并报成 action_failed，根因（模型输出不合规）就此丢失
        if not (0 <= point.x < self.space[0] and 0 <= point.y < self.space[1]):
            return self._fail(
                started, f"坐标越界 {point.as_tuple()}，空间 {self.space}", intent, screenshot
            )

        return GroundingResult(
            point=point,
            source=SOURCE_LOCAL,
            confidence=1.0,
            latency_ms=latency,
            meta={"raw": raw[:200], "space": list(self.space), "adapter": bool(self.adapter_path)},
        )

    def _fail(
        self,
        started: float,
        reason: str,
        intent: ActionIntent | None,
        screenshot: Screenshot,
    ) -> GroundingResult:
        """本地模型没给出可用坐标时的兜底。

        **兜底成功也要如实记来源。** 记成 `local_vlm` 的话，M3 分析
        「微调模型准不准」时会把兜底救回来的算进模型的功劳里，
        把它的能力估高——而那正是这一周要测的核心数字。
        """
        latency = round((time.perf_counter() - started) * 1000, 2)
        if self.fallback is not None:
            result = self.fallback.locate(screenshot, "", intent)
            if result.found:
                result.source = SOURCE_NATIVE_FALLBACK
                result.latency_ms = latency
                result.meta = dict(result.meta or {})
                result.meta["local_vlm_error"] = reason
                return result
        return GroundingResult(source="none", latency_ms=latency, error=reason)

    def close(self) -> None:
        """释放显存。常驻是为了省冷加载，但进程结束前要还回去。"""
        self._model = None
        self._processor = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


__all__ = [
    "MODEL_SPACE",
    "SOURCE_LOCAL",
    "SOURCE_NATIVE_FALLBACK",
    "SOURCE_TEXT_FALLBACK",
    "LocalVLMGrounding",
    "parse_point",
]
