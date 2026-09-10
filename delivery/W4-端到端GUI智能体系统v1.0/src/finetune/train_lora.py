"""Qwen2.5-VL-3B 的 QLoRA 微调 —— M3 任务 2，同时是前置件② 的载体。

## 这个脚本要证明两件事

1. **链路能跑通**：Transformers 加载 + bitsandbytes 4-bit 量化 + PEFT LoRA
   + Accelerate 训练循环，完整走一遍。这本身就是大纲「掌握模型微调核心
   算法能力」的落点。
2. **微调有没有效**：微调前 vs 微调后在 ScreenAgent 验证集上的差值
   （动作类型准确率 + 坐标误差），对应大纲 W5「对比微调前后模型在
   GUI 任务理解与动作生成上的效果」。

**「微调无显著提升」是一个合法结论**（M3 降级路径 2），只要训练过程
可复现、超参记录完整。实测训练集只有 569 条，远低于大纲定的 3000-4000，
所以这个结论是可预期的——**提前说清楚，比事后找补诚实**。

## 先跑 smoke test

    python -m finetune.train_lora --smoke

100 步、不保存权重，只回答一个问题：**这条链路在这台机器上跑不跑得起来。**
M3 D1 晚间要启动挂机训练，那时候才发现 bitsandbytes 装错了、显存不够、
或者 processor 的图片字段名对不上，就是白白浪费一夜。

前置件②要的就是这一步的输出。

## 显存的大头不是权重

3B 模型 4-bit 量化后权重约 2-3GB，但 Qwen2.5-VL 是动态分辨率——一张
2560×1600 截图会产生 2000+ 个视觉 token，视觉塔与注意力的激活显存可能
超过权重本身。

所以 `--max-pixels` 是必须调的参数，不是可选项。默认值按「最长边 896px」
折算，这是里程碑文档定的。**峰值显存以实测为准，报告里不写「约 3GB」
这类估算。**

## 精度按卡选

- 30 系及以上（Ampere+）：`bf16`
- T4 / V100：只能 `fp16`

选错的表现是 loss 变 NaN，而不是报错。`--precision auto` 会按算力自动选。

## 用法

    python -m finetune.train_lora --smoke                     # 100 步冒烟
    python -m finetune.train_lora                             # 完整训练
    python -m finetune.train_lora --epochs 2 --lr 1e-4
    python -m finetune.train_lora --model Qwen/Qwen2.5-VL-3B-Instruct

## 中断了怎么办

    python -m finetune.train_lora --resume auto

`auto` 自动找 `finetune/outputs/` 下最近一个含 checkpoint 的目录。
也可以显式指定：`--resume finetune/outputs/20260824-150000`。

checkpoint 每 `总步数 // 4` 步存一次（本项目 136 步 → 每 34 步），
所以最多丢 34 步、约 50 分钟。**分几次跑完全可以**——训练是确定性的，
中间停多久都不影响最终结果。

**续跑时超参必须与上次一致**（batch / 梯度累积 / epoch / max_pixels）。
改了的话优化器状态对不上，续出来的模型没有意义。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path("finetune/data")
OUT_DIR = Path("finetune/outputs")

#: 里程碑文档定的 PEFT 配置。改这里等于改实验，要同步改报告。
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
#: 覆盖注意力与 MLP 的全部投影层。只挂 q/v 是更省的做法，但对
#: 「学一个新的输出空间（坐标）」这类任务，MLP 也要动。
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

#: 最长边 896px 折算出的视觉 token 上限。28×28 是 Qwen2.5-VL 的 patch 尺寸。
DEFAULT_MAX_PIXELS = 896 * 896


@dataclass
class TrainStats:
    """训练过程的实测记录。**报告里一律用这里的值，不写估算。**"""

    model: str = ""
    smoke: bool = False
    steps: int = 0
    epochs: float = 0.0
    train_samples: int = 0
    val_samples: int = 0
    precision: str = ""
    max_pixels: int = 0
    peak_vram_mb: float = 0.0
    load_seconds: float = 0.0
    train_seconds: float = 0.0
    trainable_params: int = 0
    total_params: int = 0
    losses: list = field(default_factory=list)
    error: str = ""

    @property
    def trainable_ratio(self) -> float:
        return self.trainable_params / self.total_params if self.total_params else 0.0


def pick_precision(explicit: str = "auto") -> str:
    """按显卡算力选精度。

    **选错不会报错，只会让 loss 变 NaN。** Ampere 之前的卡没有 bf16 的
    硬件支持，跑起来是软件模拟或直接出错。
    """
    if explicit != "auto":
        return explicit
    try:
        import torch

        if not torch.cuda.is_available():
            return "fp32"
        major, _ = torch.cuda.get_device_capability()
        return "bf16" if major >= 8 else "fp16"
    except Exception:  # noqa: BLE001
        return "fp16"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


#: 训练用的系统提示词。**与 `prompts/executor_v1.yaml` 的动作规范同构。**
#:
#: 训练时不给动作空间、推理时给一大段动作说明，模型见到的分布就是两回事。
#: 这里不照抄 executor_v1 的全文（那份还带 few-shot 与子任务约定，对单步
#: 动作生成是噪声），只保留决定输出格式的那部分：动作名与 JSON 结构。
#:
#: 坐标空间与 `finetune.dataset.COORD_SPACE` 一致，不在提示词里另写一套。
#: 训练与评测共用的系统提示词。
#:
#: ## 2026-08-26：动作集从 4 个扩到 8 个 + done
#:
#: 此前这里只列 `left_click / double_click / right_click / mouse_move`
#: 并要求「必须输出 x/y」。那在当时是**对的**——旧训练集只有 3 种动作，
#: 全部带坐标，提示词与数据完全一致。
#:
#: 数据放宽到 8 种动作之后它就不对了，而且**不会报错**：
#:
#:     提示词说   只有 4 个鼠标动作，必须给 x/y
#:     训练目标却是 {"done": true}            占 41.6%
#:                {"action":"type",...}       占 12.3%
#:                {"action":"key",...}        占 15.5%
#:
#: 于是 88 分钟的训练是在**提示词与七成目标直接矛盾**的条件下做的。
#: 实测后果：`done` 8 条全错、误报率 0.0%（提示词从没说过它存在），
#: `type` 类型准确率仅 33%。
#:
#: > 教训：**动作空间是提示词与数据之间的契约。** 只改一边，另一边不会
#: > 报错，只会安静地对抗——而 loss 曲线看起来完全正常。
SYSTEM_PROMPT = """你是一个桌面 GUI 智能体。你会看到一张屏幕截图和一个子任务，
需要判断为完成该子任务，下一步应当执行什么动作。

可用动作与输出格式（只输出一个 JSON 对象，不要任何解释文字）：

{"action": "left_click", "x": 横坐标, "y": 纵坐标}
{"action": "double_click", "x": 横坐标, "y": 纵坐标}
{"action": "right_click", "x": 横坐标, "y": 纵坐标}
{"action": "mouse_move", "x": 横坐标, "y": 纵坐标}
{"action": "type", "text": "要输入的文本"}
{"action": "key", "keys": "enter"}          组合键写成 ctrl+a 这种形式
{"action": "scroll", "direction": "down", "amount": 3}
{"action": "wait", "duration": 1.0}

若该子任务已经完成，输出：

{"done": true}

坐标按 1000×1000 的范围给出，原点在左上角。"""


def build_messages(record: dict) -> list[dict]:
    """一条样本的对话形式。

    **与推理时的构造保持同构。** 训练时如果把图片放在 user 消息里、
    推理时放在 system 里，或者训练时不给动作空间而推理时给，模型见到的
    分布就不是一回事——**而这种错误在 loss 曲线上完全看不出来**，
    只会让评测分数莫名其妙地低。
    """
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": record["image"]},
                {"type": "text", "text": record["instruction"]},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": record["answer"]}]},
    ]


def _last_subsequence_end(haystack: list[int], needle: list[int]) -> int | None:
    """`needle` 在 `haystack` 里最后一次出现的**结束位置**（不含）。找不到返回 None。

    用来定位 `<|im_start|>assistant\\n` —— 它之后就是要训练的答案。
    **取最后一次**：多轮对话里 assistant 可能出现多次，要的是最后那一轮。
    """
    n, m = len(haystack), len(needle)
    if m == 0 or m > n:
        return None
    for i in range(n - m, -1, -1):
        if haystack[i : i + m] == needle:
            return i + m
    return None


def resolve_resume(value: str) -> Path | None:
    """把 `--resume` 的取值变成一个可续跑的运行目录。

    `auto` 时挑 `finetune/outputs/` 下**最近一个含 checkpoint 的目录**——
    按目录名排序即可，因为目录名就是时间戳。

    找不到就返回 None 并说明原因，**不静默当成新训练**：
    以为在续跑而实际从头开始，跑完三小时才发现，比直接报错糟糕得多。
    """
    if value.lower() == "auto":
        # **跳过 smoke 目录。** 实测踩过：`--smoke --max-steps 1` 结束时
        # Trainer 仍会存一个 checkpoint-1（save_steps 设成 10^9 也拦不住
        # 收尾那次保存），于是后来的 `--resume auto` 把一次正式训练续进了
        # 名为 smoke 的目录 —— 训练本身没问题，但产物名字彻底误导。
        candidates = [
            d
            for d in sorted(OUT_DIR.glob("*"), reverse=True)
            if d.is_dir() and d.name != "smoke" and any(d.glob("checkpoint-*"))
        ]
        if not candidates:
            print(f"  [提示] {OUT_DIR} 下没有含 checkpoint 的目录，按新训练开始。")
            return None
        return candidates[0]

    path = Path(value)
    if not path.is_dir():
        raise SystemExit(f"--resume 指向的目录不存在：{path}")
    if not any(path.glob("checkpoint-*")):
        raise SystemExit(f"{path} 下没有 checkpoint-* 子目录，无法续跑。")
    return path


def train(args) -> TrainStats:  # noqa: PLR0915 —— 训练脚本，线性叙事好读
    stats = TrainStats(model=args.model, smoke=args.smoke)

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
        Trainer,
        TrainingArguments,
    )

    train_path = Path(args.data) / "train.jsonl"
    val_path = Path(args.data) / "val.jsonl"
    if not train_path.exists():
        raise SystemExit(f"找不到 {train_path}。先跑 python -m finetune.dataset")

    train_rows = load_jsonl(train_path)
    val_rows = load_jsonl(val_path) if val_path.exists() else []
    if args.smoke:
        # 冒烟只要够跑满 100 步，不需要全量
        train_rows = train_rows[: max(args.max_steps * args.grad_accum, 32)]
        val_rows = val_rows[:8]
    stats.train_samples = len(train_rows)
    stats.val_samples = len(val_rows)

    stats.precision = pick_precision(args.precision)
    stats.max_pixels = args.max_pixels
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(stats.precision, torch.float32)

    print(f"  精度 {stats.precision}    max_pixels {args.max_pixels}")
    print(f"  训练 {stats.train_samples} 条    验证 {stats.val_samples} 条")

    load_started = time.perf_counter()
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    processor = AutoProcessor.from_pretrained(
        args.model, min_pixels=256 * 28 * 28, max_pixels=args.max_pixels
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, quantization_config=quant, dtype=dtype, device_map="auto"
    )
    stats.load_seconds = round(time.perf_counter() - load_started, 1)

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False  # 与 gradient checkpointing 冲突
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=list(TARGET_MODULES),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    stats.trainable_params, stats.total_params = trainable, total
    print(f"  可训练参数 {trainable:,} / {total:,} = {trainable / total:.4%}")

    def collate(batch: list[dict]) -> dict:
        """把一批样本编码成模型输入。

        **只对 assistant 的回答算 loss。** 不 mask 掉 prompt 的话，
        模型会把大部分梯度花在复述指令上，而我们要它学的只有坐标。
        """
        from qwen_vl_utils import process_vision_info

        texts, images = [], []
        for record in batch:
            messages = build_messages(record)
            texts.append(
                processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            )
            image_inputs, _ = process_vision_info(messages)
            images.append(image_inputs)

        inputs = processor(
            text=texts, images=images, return_tensors="pt", padding=True, truncation=False
        )
        labels = inputs["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        # **按 assistant 回合的起始位置掩码，不能按答案长度从末尾倒推。**
        #
        # 2026-08-26 实测出的 bug：原来的写法是「留最后 len(answer_ids) 个
        # token」，而 chat template 会在答案后面追加 `<|im_end|>\n`，
        # 于是窗口整体后移 3 个 token——
        #
        #     期望训练  {"action": "left_click", "x": 475, "y": 251}
        #     实际训练  ": "left_click", "x": 475, "y": 251}<|im_end|>\n
        #               ↑ 开头的 {"action 被当成上下文遮掉了
        #
        # 旧数据没暴露它：答案都是 23 个 token 上下，丢掉开头 3 个还剩
        # 动作名和坐标，大体能学会。**答案越短损失比例越大**——
        # `{"done": true}` 只有 5 个 token，丢 3 个就是六成信号没了，
        # 模型从来没被训练过要输出 `{"done"` 这个开头。
        #
        # 这解释了放宽后 `done` 8 条全错、误报率 0.0%。
        #
        # 现在的做法：找到最后一个 `<|im_start|>assistant\n`，把它及之前
        # 全部遮掉。这样标签正好覆盖「答案 + 结束符」——结束符要训练，
        # 否则模型不知道何时停。
        head = processor.tokenizer("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
        for index in range(labels.shape[0]):
            ids = inputs["input_ids"][index].tolist()
            start = _last_subsequence_end(ids, head)
            if start is None:
                # 找不到就退回全遮，**宁可这条样本不产生梯度，也不要用
                # 一个偏移的窗口去训练**——后者不会报错，只会学错东西。
                labels[index, :] = -100
                logger.warning("样本 %d 找不到 assistant 起始标记，已整条屏蔽", index)
            else:
                labels[index, :start] = -100
        inputs["labels"] = labels
        return inputs

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    resume_dir = resolve_resume(args.resume) if args.resume else None
    if resume_dir is not None:
        run_dir = resume_dir
        print(f"  **续跑** {run_dir}")
        print("     超参必须与上次一致（batch / 梯度累积 / epoch / max_pixels），")
        print("     不然优化器状态对不上，续出来的模型没有意义。")
    else:
        run_dir = OUT_DIR / ("smoke" if args.smoke else time.strftime("%Y%m%d-%H%M%S"))

    # 总优化步数 = 样本数 / (batch × 梯度累积) × epoch。batch 固定为 1。
    if args.smoke:
        total_steps = args.max_steps
    else:
        per_epoch = max(1, len(train_rows) // args.grad_accum)
        total_steps = max(1, int(per_epoch * args.epochs))
    warmup_steps = max(1, int(total_steps * 0.03))
    # 全程存 4-5 次。太密会拖慢训练也吃磁盘，太疏等于没有。
    # `--save-every` 可覆盖：一边用电脑一边训练时，OOM 风险高，
    # 值得用更密的存点换更小的损失。
    save_every = args.save_every or max(5, total_steps // 4)
    # **冒烟模式显式给了 --save-every 就必须真的存。**
    # 原来冒烟恒定 save_steps=10**9，永不存点；而崩溃只发生在存点那一刻，
    # 于是冒烟测试对这类故障完全无效 —— 两次死机都是长跑到一半才暴露。
    do_save = bool(args.save_every) or not args.smoke
    save_steps = save_every if do_save else 10**9
    how = f"每 {save_every} 步存一次" if do_save else "不存权重"
    print(f"  总步数 {total_steps}    预热 {warmup_steps} 步    {how}")

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.smoke else -1,
        learning_rate=args.lr,
        # **不用 paged_adamw_8bit。** 它把优化器状态分页到 CUDA 统一内存
        # (UVM)，而 Windows WDDM 不支持 UVM 超额分配。存检查点时要把分页
        # 出去的状态拉回显存，这一步会让显示驱动超过 TDR 看门狗的 2 秒，
        # 触发 nvlddmkm Event 14 整机死机（本机实测两次，都死在存点那刻）。
        # adamw_8bit 走普通显存，最坏是 OOM —— 那是个干净的报错，不是死机。
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        # **用 warmup_steps 而不是 warmup_ratio。** transformers 5.x 移除了
        # 后者（实测 5.15.1 直接 TypeError），而 warmup_steps 在 4.x / 5.x
        # 都在。显式算步数还有个好处：日志里能直接看到预热几步，
        # 不用再拿比例去乘一个心算的总步数。
        warmup_steps=warmup_steps,
        logging_steps=1 if args.smoke else 10,
        # **按总步数算间隔，不是写死 200。**
        #
        # 写死 200 时踩过一次：本项目总步数只有 136（543 条 ÷ 梯度累积 8
        # × 2 epoch），**一次都不会触发**。跑到第 130 步 OOM，三小时白费。
        # 而这台机器的峰值显存实测 6984MB / 8151MiB，中途开个浏览器就可能
        # 爆——checkpoint 不是可选项。
        save_steps=save_steps,
        save_total_limit=3,
        # **只存模型，不存 optimizer.pt。** 写优化器状态是显存尖峰的来源，
        # 也是上面那条死机链的触发点。代价：不能中途续训，断了要重跑。
        save_only_model=True,
        gradient_checkpointing=True,
        bf16=stats.precision == "bf16",
        fp16=stats.precision == "fp16",
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(train_rows),
        data_collator=collate,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    train_started = time.perf_counter()
    try:
        result = trainer.train(resume_from_checkpoint=bool(resume_dir))
        stats.steps = int(result.global_step)
        stats.epochs = round(float(result.metrics.get("epoch", 0.0)), 3)
    except Exception as exc:  # noqa: BLE001 —— 失败也要留下实测记录
        stats.error = f"{type(exc).__name__}: {exc}"[:400]
        raise
    finally:
        stats.train_seconds = round(time.perf_counter() - train_started, 1)
        if torch.cuda.is_available():
            stats.peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024**2), 1)
        stats.losses = [
            {"step": entry.get("step"), "loss": entry.get("loss")}
            for entry in trainer.state.log_history
            if "loss" in entry
        ]

    if not args.smoke:
        model.save_pretrained(str(run_dir / "adapter"))
        processor.save_pretrained(str(run_dir / "adapter"))
        print(f"  adapter 已保存到 {run_dir / 'adapter'}")
    else:
        print("  冒烟模式：**不保存权重**，只验证链路。")

    (run_dir / "train-stats.json").write_text(
        json.dumps(asdict(stats), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return stats


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Qwen2.5-VL-3B QLoRA 微调（M3 任务 2）")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--data", default=str(DATA_DIR))
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="每几步存一次 checkpoint。默认总步数的 1/4；边用电脑边训练时可调小",
    )
    parser.add_argument(
        "--resume",
        default="",
        help="从断点续跑。给运行目录（finetune/outputs/<时间戳>）或 auto 自动找最近的一次",
    )
    parser.add_argument("--smoke", action="store_true", help="100 步冒烟，不保存权重")
    parser.add_argument("--max-steps", type=int, default=100, help="冒烟模式的步数")
    args = parser.parse_args()

    print("=" * 70)
    print("QLoRA 微调" + ("（冒烟）" if args.smoke else ""))
    print("=" * 70)
    print(f"  模型  {args.model}")
    print(f"  LoRA  r={LORA_R} alpha={LORA_ALPHA} dropout={LORA_DROPOUT}")
    print(f"  目标层 {', '.join(TARGET_MODULES)}")
    print()

    stats = train(args)

    print()
    print("=" * 70)
    print("训练完成")
    print("=" * 70)
    print(f"  步数        {stats.steps}    epoch {stats.epochs}")
    print(f"  加载耗时    {stats.load_seconds}s")
    print(f"  训练耗时    {stats.train_seconds}s")
    print(f"  峰值显存    **{stats.peak_vram_mb} MB**  ← 报告里用这个实测值")
    print(f"  可训练参数  {stats.trainable_params:,}（{stats.trainable_ratio:.4%}）")
    if stats.losses:
        first, last = stats.losses[0], stats.losses[-1]
        print(f"  loss        {first['loss']:.4f} → {last['loss']:.4f}")
    if args.smoke:
        print("\n  **前置件② 完成。** 链路可跑通，把上面的峰值显存记进 M3 计划，")
        print("  D1 下午的本地部署直接沿用这个 max_pixels。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
