"""本地 Qwen2.5-VL 后端 —— 离线版 Agent 的规划模型。

## 这个文件解决的是什么问题

老师补充的要求：「根据部署环境的需求，如果对隐私比较高、对数据保密比较
高的情况下，往往会采用离线的情况」——因此除在线版之外还要有一个离线版。

M3 之前，「本地」这件事只做到了一半：

- ✅ 权重在本机（`D:\\hf-cache`），QLoRA 微调全程离线跑完
- ✅ `eval/action.py --local` 在 142 条验证集上零联网跑完
- ✅ 感知（dxcam / OCR / UIA）与执行（pyautogui）本来就全在本机
- ❌ **但 Agent 跑任务时仍然只会走 API**

差的就是这一层：把已经在 `eval/action.py` 里跑通的本地推理，包成
`LLMBackend` 的一个实现。基类只有 `complete()` 一个抽象方法，其余
（动作解析、用量记账、提示词注入）都在基类里，所以这里**只实现
`complete()`**，其他一行都不用抄。

## 离线版与在线版的差别，落到实处是四条

| | 在线版 | 离线版 |
|---|---|---|
| 截图去哪 | 上传到平台服务器 | 不出本机 |
| 模型能力 | qwen3-vl-8b（云端） | Qwen2.5-VL-3B（本机） |
| 单次调用费用 | 按 token 计费 | 0（成本转移到硬件与耗时） |
| 断网 | 不可用 | 可用 |

前两条是取舍关系：**保密性换能力**。这正是要跑同一套基础任务做对比的
理由——不实测就只能说「离线版应该差一些」，实测才能说差多少。

## 三个必须与训练对齐的地方

微调后的 adapter 一旦挂上，「推理时怎么喂」就不再自由，必须复现训练时
模型见到的分布。**这类不一致不会抛异常，只会让分数莫名其妙地低。**

1. **坐标空间**：训练输出归一化到 [0,1000)（`finetune.dataset.COORD_SPACE`），
   而 `SessionConfig.coordinate_space` 恰好也是 (1000, 1000)。这不是巧合
   ——M2 实测出 qwen3-vl 输出的就是这个空间，两边本来就该一致。**因此
   本地后端接进来不需要改任何坐标换算**，`CoordinateScaler` 原样工作。
2. **max_pixels**：训练时 processor 配的是 `896×896`。不配的话，一张
   2560×1600 的桌面截图会产生 2000+ 个视觉 token，既撑爆 8GB 显存，
   也让模型见到的图像分辨率与训练时不是一回事。
3. **量化精度**：训练用 4-bit NF4。评测与推理若用 bf16，权重 6.2GB 塞不
   进 8GB 卡，`device_map="auto"` 会把一部分层卸到内存，每次前向都要在
   CPU↔GPU 之间搬数据。所以 `load_4bit` 默认为 True。

## adapter 与提示词的分布外问题（重要，不要静默掉）

**这一段描述的是 2026-08-26 之前的状态，保留作为记录。**
当时训练提示词只给四个鼠标动作，没有 `type` / `wait` / `done`——不是因为
ScreenAgent 没有，而是**装载层把它们丢了**（只保留带坐标的样本）。
放宽之后 `SYSTEM_PROMPT` 已列全 8 种动作与 `done`，下面这段分布外分析
对新 adapter 不再成立。

而 Agent 跑真实任务时，`agent.session` 注入的是 `executor_v1`，动作空间
大得多，还要求模型会报 `done`。挂着 adapter 去接 `executor_v1` 属于分布外。

**这一段原来写的是「它很可能永远不输出 done、也不会用 type」——实测证明
这个预判过头了**，先记下来：2026-08-24 在 2560×1600 真实桌面截图上各问
四个子任务目标（`--provider local --adapter …`），微调后的模型照样输出了
`{"done": true, "thinking": …}`，也照样带 `thinking` 字段。LoRA 只挂在
投影层上、只训了两个 epoch，基座跟随指令的能力没有被抹掉。

实测到的真实差别是**格式**，与验证集上 2.1%→100% 的合规率一致：

    基座 无 few-shot：{"action": "click", "x": "985", "y": "760", …}   动作名不在允许集合、坐标是字符串
                      {"action": "click", "x": 798, "y": 15}}          多一个右花括号
                      {"action": "key", "keys": "win + b"}
    微调 无 few-shot：4/4 全是 left_click + 整数坐标，无一畸形

所以这里的处理是：

- 默认**照常接受外部注入的提示词**（`pin_prompt=False`），让离线版与在线版
  跑在完全相同的提示词上，差值才是模型的差值；
- 挂了 adapter 时**打一条警告**，说明训练时的动作空间比 executor 窄，
  分布外的部分要按实测说话，不要按推测写进报告；
- 需要跑「分布内」那一组对照时，用 `pin_prompt=True` 强制使用训练时的
  提示词。

把这件事藏起来（比如偷偷替换提示词）会让「微调后端到端成功率」这个数字
无法解释：涨了不知道是模型好还是提示词换了，跌了也说不清。

## few-shot 会让小模型逐字照抄（实测，尚未处理）

同一次探针里还撞见一件事，**它同时影响在线版和离线版**：`executor_v1` 带
三条 few-shot 示例，而 3B 模型会把示例的答案原样吐出来——

    目标「点击任务栏最左边的 Windows 开始按钮」
      带 few-shot：{"action": "left_click", "x": 470, "y": 750, "thinking": "任务栏最左侧的 Windows 徽标就是开始按钮"}
                   ↑ 与 executor_v1.yaml 第一条示例逐字相同，而 y=750 根本不在任务栏上
      去 few-shot：{"action": "left_click", "x": 15, "y": 978}
                   ↑ 左下角，这才是任务栏最左侧

四个目标里，微调版抄了 2 个、基座版抄了 1 个；去掉 few-shot 后两者都是 0。
**照抄的输出格式完全合规、坐标却是示例里的常数**，格式合规率这类指标
一个都拦不住它。

这里只如实记录，不擅自把 few-shot 关掉：`executor_v1` 是 M2 全程用的模板，
M3 的提示词消融（大纲 W5 任务 4）本来就要比 P1 零样本 / P2 few-shot /
P3 CoT 三档——这个现象正是那组对照要量的东西，在这里偷偷改一个默认值，
消融的基线就没了。

## 用法

    from llm.qwen_vl_local import QwenVLLocalBackend

    # 离线-基座
    backend = QwenVLLocalBackend(model="Qwen/Qwen2.5-VL-3B-Instruct")

    # 离线-微调
    backend = QwenVLLocalBackend(
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        adapter="finetune/outputs/20260824-170900/adapter",
    )

    with backend:                      # 退出时卸权重、清显存
        intent = backend.predict_action("点击地址栏", screenshot)

命令行侧见 `llm.factory.build_backend`，脚本用 `--provider local` 切换。
"""

from __future__ import annotations

import gc
import logging
import time
from typing import TYPE_CHECKING

from finetune.train_lora import DEFAULT_MAX_PIXELS
from finetune.train_lora import SYSTEM_PROMPT as TRAIN_SYSTEM_PROMPT
from llm.base import (
    HistoryStep,
    LLMBackend,
    LLMBackendError,
    PriceSheet,
    RawResponse,
    TokenUsage,
)

if TYPE_CHECKING:  # pragma: no cover - 仅供类型检查
    from perception.capture import Screenshot

logger = logging.getLogger(__name__)

#: 本项目实际部署的模型。改这里等于改实验，要同步改报告。
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

#: 视觉 token 的下限。与训练时 processor 的配置一致。
MIN_PIXELS = 256 * 28 * 28

#: 训练时用的上限，从 `finetune.train_lora` 引入而不是抄一份数字过来。
#: 抄一份的话，哪天调了训练超参，这里会安静地留在旧值上。
MAX_PIXELS = DEFAULT_MAX_PIXELS

#: 本地推理的单价：真的是 0，不是「未知」。
#:
#: 这个区分很要紧。基类在 `price is None` 时会把 `CostInfo.priced` 置
#: False，意思是「这个数字不可信，报告里要标注」。而离线版的 API 费用
#: 确实是 0 元，是一个可以写进报告的事实。给一张零单价表，`priced`
#: 保持 True，读报告的人才能分清「0」和「不知道」。
#:
#: 当然离线版不是没有成本，成本转移到了硬件与耗时上——那部分由延迟
#: 数据体现，不该硬折算成钱塞进这里。
LOCAL_PRICE = PriceSheet(model=DEFAULT_MODEL, input_per_1k=0.0, output_per_1k=0.0)


def _executor_action_names() -> frozenset[str]:
    """executor 提示词实际给出的动作名，现场从 ACTION_SPECS 读。

    **不要手抄一份清单。** 本项目已经因为「为旧目标写的东西在目标变了之后
    不报错地活下来」废掉过一轮训练（`SYSTEM_PROMPT` 只列 4 个鼠标动作），
    以及本函数替换掉的那条警告——它把旧 adapter 的 4 个动作一直报到了
    动作空间放宽之后。写死的清单不会报错，只会静默说谎。
    """
    from control.actions import ACTION_SPECS

    return frozenset(getattr(a, "value", str(a)) for a in ACTION_SPECS)


def _trained_action_names() -> frozenset[str]:
    """训练提示词里出现的动作名，现场从 `finetune.train_lora` 的常量读。"""
    import re

    from finetune.train_lora import SYSTEM_PROMPT

    return frozenset(re.findall(r'"action":\s*"(\w+)"', SYSTEM_PROMPT)) | (
        {"done"} if '"done"' in SYSTEM_PROMPT else set()
    )


_EXECUTOR_ACTIONS = _executor_action_names()
_TRAINED_ACTIONS = _trained_action_names()


class QwenVLLocalBackend(LLMBackend):
    """在本机 GPU 上跑 Qwen2.5-VL，可选挂载 QLoRA adapter。

    与 `OpenAICompatBackend` 的唯一区别是「请求发到哪」。消息构造顺序、
    提示词来源、坐标空间、动作解析全部相同——**不同才是问题**，那会让
    在线/离线对比的差值里混进实现差异。
    """

    name = "local"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        adapter: str = "",
        *,
        load_4bit: bool = True,
        max_pixels: int = MAX_PIXELS,
        min_pixels: int = MIN_PIXELS,
        max_new_tokens: int = 128,
        history_images: int = 1,
        system_prompt: str = "",
        few_shot: list[dict] | None = None,
        pin_prompt: bool = False,
        image_size: tuple[int, int] = (1024, 768),
    ) -> None:
        super().__init__(model=model, price=LOCAL_PRICE)
        self.adapter = adapter
        self.load_4bit = load_4bit
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_new_tokens = max_new_tokens
        self.history_images = history_images

        if system_prompt:
            self.system_prompt = system_prompt
        if few_shot:
            self.few_shot = list(few_shot)

        #: 强制使用训练时的提示词，忽略 Session 注入的那套。
        #: 只在需要跑「分布内」对照组时打开，见模块文档。
        self.pin_prompt = pin_prompt
        if pin_prompt:
            self.system_prompt = TRAIN_SYSTEM_PROMPT
            self.user_template = "{instruction}"

        #: 送模型的图片尺寸。`agent.session` 会覆盖它。
        #: 注意这与坐标系尺寸是两回事，见 `SessionConfig.coordinate_space`。
        self.image_size = image_size

        if adapter and not pin_prompt:
            logger.warning(
                "已挂载 adapter %s：训练动作空间 %d 个（%s），"
                "executor 提示词 %d 个，差 %s —— 这几个未训练过，属于分布外。"
                "实测该模型仍会输出 done 与 thinking（见模块文档），但分布外的部分"
                "一律以实测为准，不要按推测写进报告。",
                adapter,
                len(_TRAINED_ACTIONS),
                " / ".join(sorted(_TRAINED_ACTIONS)),
                len(_EXECUTOR_ACTIONS),
                " / ".join(sorted(_EXECUTOR_ACTIONS - _TRAINED_ACTIONS)) or "无",
            )

        #: 权重延迟到第一次调用才加载。构造一个后端只为打印配置时
        #: （`config --show`、参数校验）不该先吃掉 2GB 显存。
        self._model = None
        self._processor = None
        #: 最近一次的图片编码信息，与 `OpenAICompatBackend` 同名同义，
        #: Loop 取它进轨迹时不需要判断面对的是哪种后端
        self.last_image_meta: dict = {}

    # ------------------------------------------------------------------ #
    # 权重加载
    # ------------------------------------------------------------------ #

    def _ensure_model(self):
        """加载权重与 processor，只做一次。

        加载方式与 `eval/action.py` 完全一致——那条路径已经在 142 条
        验证集上跑通过，没有理由另写一套。
        """
        if self._model is not None:
            return self._model, self._processor

        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - 环境问题
            raise LLMBackendError(
                f"本地后端需要 torch 与 transformers：{exc}",
                retryable=False,
                kind="missing_dependency",
            ) from exc

        started = time.perf_counter()
        processor = AutoProcessor.from_pretrained(
            self.model, min_pixels=self.min_pixels, max_pixels=self.max_pixels
        )

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        kwargs: dict = {"dtype": dtype, "device_map": "auto"}
        if self.load_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )

        try:
            net = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model, **kwargs)
            if self.adapter:
                from peft import PeftModel

                net = PeftModel.from_pretrained(net, self.adapter)
        except Exception as exc:  # noqa: BLE001 - 加载失败的形态太多
            raise LLMBackendError(
                f"本地模型加载失败（{self.model}"
                f"{' + ' + self.adapter if self.adapter else ''}）：{exc}",
                retryable=False,
                kind="load_error",
            ) from exc

        net.eval()
        self._model, self._processor = net, processor

        logger.info(
            "本地模型已就绪：%s%s，%s，max_pixels=%d，耗时 %.1fs",
            self.model,
            f" + {self.adapter}" if self.adapter else "",
            "4-bit NF4" if self.load_4bit else str(dtype),
            self.max_pixels,
            time.perf_counter() - started,
        )
        return self._model, self._processor

    # ------------------------------------------------------------------ #
    # 唯一的抽象方法
    # ------------------------------------------------------------------ #

    def generate(
        self,
        messages: list[dict],
        images: list | None = None,
        max_new_tokens: int = 0,
    ) -> tuple[str, TokenUsage, float]:
        """跑一次前向，返回 (文本, 用量, 耗时毫秒)。**不记账、不解析。**

        独立成一个方法，是因为它有两个调用方，而两者拿到的东西不一样：

        - `complete()` —— 自己拼消息，走 `LLMBackend` 那条路
        - `scripts/serve_local_model.py` —— 消息是客机通过 HTTP 送来的，
          已经拼好了，不能再拼一遍

        M0 定的拓扑是「模型跑在宿主机 GPU 上，客机经 HTTP 调用」，所以
        服务端这条路是必需的。不抽出来的话，推理代码就要在项目里出现
        第三份（前两份在 `finetune/train_lora.py` 与 `eval/action.py`）
        ——**而三份代码只要有一份的生成参数写岔，在线/离线对比就废了**。

        记账留给调用方：服务端进程里的用量属于客机那次调用，记在服务端
        的 `CostInfo` 上没有意义。
        """
        import torch

        net, processor = self._ensure_model()

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        try:
            inputs = processor(
                text=[text],
                images=list(images) or None,
                return_tensors="pt",
            ).to(net.device)
        except Exception as exc:  # noqa: BLE001
            raise LLMBackendError(
                f"输入编码失败：{exc}", retryable=False, kind="encode_error"
            ) from exc

        prompt_tokens = int(inputs["input_ids"].shape[1])

        started = time.perf_counter()
        try:
            with torch.inference_mode():
                # 动作决策是确定性任务，不采样。理由与 `OpenAICompatBackend`
                # 把 temperature 设成 0 一样：同一个界面同一个目标应当给出
                # 同一个动作，否则横评每跑一次结论都不同。
                generated = net.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens or self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id
                    or processor.tokenizer.eos_token_id,
                )
        except Exception as exc:  # noqa: BLE001 - torch 可能抛任何东西
            raise _inference_error(exc, self.max_pixels) from exc
        latency_ms = (time.perf_counter() - started) * 1000.0

        new_tokens = generated[0][prompt_tokens:]
        raw_text = processor.decode(new_tokens, skip_special_tokens=True).strip()
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=int(new_tokens.shape[0]),
        )
        return raw_text, usage, latency_ms

    def complete(
        self,
        prompt: str,
        screenshot: Screenshot | None = None,
        history: list[HistoryStep] | None = None,
    ) -> RawResponse:
        messages, images = self.build_messages(prompt, screenshot, history or [])
        raw_text, usage, latency_ms = self.generate(messages, images)
        # 与 API 后端同样在解析之前记账：调用已经发生，算力已经花了。
        # 这里的 token 数没有钱的含义，但它是「离线版每步要处理多少
        # token」的实测值——延迟为什么是那个量级，靠它解释。
        cost = self.record_usage(usage)

        return RawResponse(
            text=raw_text,
            usage=usage,
            cost_cny=cost,
            request_id="",  # 本地推理没有平台侧的请求 ID
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------ #
    # 消息构造
    # ------------------------------------------------------------------ #

    def build_messages(
        self,
        instruction: str,
        screenshot: Screenshot | None,
        history: list[HistoryStep],
    ) -> tuple[list[dict], list]:
        """拼出这一轮的对话，返回 (messages, images)。

        顺序与 `OpenAICompatBackend.build_messages` 一字不差：
        系统提示 → few-shot → 历史 → 当前截图与目标。**顺序不同，
        在线/离线的差值里就混进了上下文结构的差异。**

        图片要单独返回一份列表：HF 的 processor 不从 messages 里取图，
        它按 `apply_chat_template` 生成的占位符顺序去 `images=[...]` 里
        对应。两者顺序错开一位，模型看的就是上一帧。
        """
        messages: list[dict] = []
        images: list = []

        if self.system_prompt:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}
            )

        # few-shot 的 input 是截图的文字描述而非真图（见 executor_v1.yaml
        # 的说明），所以这里全是纯文本，不进 images
        for example in self.few_shot:
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": str(example.get("input", ""))}],
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": str(example.get("output", ""))}],
                }
            )

        self._append_history(messages, images, history)

        text = self._user_prompt(instruction)
        if screenshot is None:
            self.last_image_meta = {}
            messages.append({"role": "user", "content": [{"type": "text", "text": text}]})
            return messages, images

        image = self._prepare_image(screenshot, scale=1.0)
        images.append(image)
        self.last_image_meta = {"width": image.width, "height": image.height, "scale": 1.0}
        messages.append(
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": text}],
            }
        )
        return messages, images

    def _append_history(
        self, messages: list[dict], images: list, history: list[HistoryStep]
    ) -> None:
        """历史步骤。只有最近若干步带图，更早的用文字摘要。

        本地推理没有 token 费用，但**显存与延迟同样是有限的**：每多一张
        1024×768 的图就多几百个视觉 token，3B 模型在 8GB 卡上很快就顶到
        上限。策略与 API 后端保持一致，同时也让两边的上下文长度可比。
        """
        if not history:
            return

        fallback_from = max(0, len(history) - self.history_images)
        for index, step in enumerate(history):
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": step.summary()}]}
            )
            if step.screenshot is None:
                continue
            annotated = step.image_scale != 1.0 or not step.with_image
            wanted = step.with_image if annotated else index >= fallback_from
            if not wanted:
                continue

            image = self._prepare_image(step.screenshot, scale=step.image_scale)
            images.append(image)
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "执行后的界面："},
                    ],
                }
            )

    def _prepare_image(self, screenshot: Screenshot, scale: float = 1.0):
        """`Screenshot` → PIL RGB，并按 `image_size` 与 scale 缩放。

        缩到 `image_size` 而不是直接把原图丢给 processor：processor 的
        `max_pixels` 也会缩，但缩到多大由它按 patch 对齐算，**在线与离线
        就不是同一张图了**。先按同一个尺寸缩，两边看到的内容才一致。
        """
        image = screenshot.to_pil()
        width = max(int(self.image_size[0] * scale), 1)
        height = max(int(self.image_size[1] * scale), 1)
        if (image.width, image.height) != (width, height):
            from PIL import Image as PILImage

            image = image.resize((width, height), PILImage.LANCZOS)
        return image

    def _user_prompt(self, instruction: str) -> str:
        """按 `user_template` 渲染用户消息。

        逐键替换而不是 `str.format`：模板里有 JSON 示例（``{"done": true}``），
        format 会把那些花括号当占位符炸掉。口径与 `OpenAICompatBackend`
        及 `agent.prompts._safe_format` 一致。
        """
        text = self.user_template
        for key, value in (
            ("instruction", instruction),
            # 与 `OpenAICompatBackend` 逐字一致：width/height 是坐标系尺寸
            ("width", self.coordinate_space[0]),
            ("height", self.coordinate_space[1]),
            ("image_width", self.image_size[0]),
            ("image_height", self.image_size[1]),
        ):
            text = text.replace("{" + key + "}", str(value))
        return text

    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """卸权重、清显存。

        基类给 `close()` 留了空实现，注释里写的就是「M3 的本地模型后端才
        会用到」——就是这里。不清的话，同一个进程里连着跑基座与微调两组
        对比时，第二次加载会直接 OOM。
        """
        if self._model is None:
            return
        self._model = None
        self._processor = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass
        logger.info("本地模型已卸载，显存已释放")

    def __repr__(self) -> str:
        tail = f" adapter={self.adapter!r}" if self.adapter else ""
        return f"<QwenVLLocalBackend model={self.model!r}{tail} 4bit={self.load_4bit}>"


def _inference_error(exc: Exception, max_pixels: int) -> LLMBackendError:
    """把 torch 的异常翻成一句能照着做的话。

    M2 的要求是「返回可理解错误」。显存不足是本地后端最常见的失败，而
    `CUDA out of memory` 那一大段堆栈里唯一有用的信息是「该调小
    max_pixels 了」——把它直接说出来，比让人去搜堆栈快得多。

    ``retryable`` 一律为 False：显存不足重试多少次都一样，除非改参数。
    """
    text = str(exc)
    if "out of memory" in text.lower():
        return LLMBackendError(
            f"显存不足（当前 max_pixels={max_pixels}）：{text}\n"
            f"  处理：调小 --max-pixels（如 {640 * 640}），或关掉其他占显存的程序。",
            retryable=False,
            kind="oom",
        )
    return LLMBackendError(f"本地推理失败：{text}", retryable=False, kind="inference_error")
