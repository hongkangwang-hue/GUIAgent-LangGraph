# 多模态大模型 LoRA 微调与效果对比 —— 第 5 周交付

> 交付物：**微调后的模型权重 + 微调效果对比分析报告**
> 生成日期：2026-09-17　　源仓库分支：`main`

---

## 零、三十秒看懂这个包

| 要什么 | 在哪 |
|---|---|
| **交付物之一：模型权重** | `模型权重/20260827-174327/adapter/`（LoRA，148.7MB） |
| **交付物之二：对比分析报告** | `微调效果对比分析报告.md` |
| 验证报告里的数字 | `python scripts/recompute_report.py`（只用标准库，几秒钟） |
| 验证代码 | `pip install -r requirements.txt` 然后 `python -m pytest` |

**报告的结论要两句一起读：** 微调在离线验证集上效果显著（格式合规 24% → 100%，
动作类型 15.5% → 59.7%）；但在真实客机上唯一一轮环境干净的端到端测试是
**0/25**。报告 §5.3、§5.4 分析了原因，§三 偏差 1 说明了为什么这里要先讲这件事。

---

## 一、大纲四项任务逐条对照

| # | 大纲任务 | 实现位置 | 状态 |
|---|---|---|---|
| 1 | 基于预处理后的公开 GUI 数据集，构建微调训练集与验证集 | `src/finetune/dataset.py`、`src/data/split.py` | ✅ 1783 / 375 条 |
| 2 | 使用 PEFT 库实现对开源多模态模型的 LoRA 微调 | `src/finetune/train_lora.py` | ✅ 权重已交付 |
| 3 | 对比微调前后模型在 GUI 任务理解与动作生成上的效果 | `src/eval/action.py` | ✅ 见报告 §5.1、§5.3、§5.4 |
| 4 | 优化提示词工程，提升模型的任务执行准确率 | `src/prompts/`、`scripts/ablate_prompts.py` | ✅ 见报告 §5.2、附录 A.2 |

### 任务 1 —— 训练集与验证集

`finetune/dataset.py` 把 ScreenAgent 的桌面轨迹转成"截图 + 子任务指令 → 一个动作"
的样本，按 `data/split.py` 冻结的划分写成两个 JSONL。

- **按会话切，不按样本切**：同一会话的相邻截图几乎相同，按样本随机切会泄漏。
  实测两边会话交集为 0。
- **划分落成文件并有指纹**（`c1fd6437d3df7178`），训练前核对，不符直接报错。
- 八种动作：`left_click / double_click / mouse_move / type / key / scroll / wait / done`。
- 坐标归一化到 0~1000，存 `point_norm`，与图片像素尺寸无关。

生成好的 `train.jsonl` / `val.jsonl` 就在本包 `finetune/data/` 下，
MD5 与训练记录一致（`6045cd46…` / `7713ab81…`）。

### 任务 2 —— LoRA 微调

`finetune/train_lora.py` 基于 transformers + PEFT + bitsandbytes：
基座 4-bit NF4 加载，LoRA r=16 / alpha=32 挂在注意力与前馈共 7 类层上。
可训练参数 1.79%，峰值显存 7.4GB，RTX 4090 上 93 分钟。

### 任务 3 —— 微调前后对比

`eval/action.py` 在 375 条验证集上逐条跑模型，记下格式是否合规、动作类型对不对、
坐标差多少，结果逐条写进 JSONL。**格式合规与动作类型分开记**——微调涨的那部分里，
"学会了格式"和"学会了点哪"只有分开才能分辨。

### 任务 4 —— 提示词工程

`prompts/` 下有 8 个 YAML 模板，`scripts/ablate_prompts.py` 做三档阶梯消融
（零样本 → +few-shot → +思维链），`tests/test_prompt_ablation.py` 钉住
"相邻两档只差一个变量"。报告 §5.2 比较了训练提示词与 `executor_v0`。

---

## 二、怎么运行

### 1. 复算报告里的全部数字（任何机器，不装任何包）

```bash
cd <本目录>
python scripts/recompute_report.py
```

只用 Python 标准库。输出按报告节号排列：数据集统计、主结果、提示词对照、
分辨率配对实验、端到端、附录 A 的旧版验证集结论。

### 2. 跑单元测试（不需要 GPU、不需要权重）

```bash
pip install -r requirements.txt          # 6 个包，约 30 秒
python -m pytest                         # 95 passed, 1 skipped
python -m pytest --cov --cov-report=term-missing
```

`pyproject.toml` 里配了 `pythonpath`，**不需要安装本包**，直接跑。

> 覆盖率**不要写成 `--cov=finetune`**：包根目录下有个存放训练数据的 `finetune/data/`，
> coverage 会把 `finetune` 当成那个目录，`finetune` 包就从报告里整个消失，且不报警告。
> 直接用 `--cov`（读 `pyproject.toml` 里写好的 `src/finetune`、`src/eval`）即可。

实测结果（2026-09-17，Windows 11，Windows PowerShell（ANSI 代码页 936），
Python 3.10.20，**从 zip 解压到新目录、全新空虚拟环境**）：

```
95 passed, 1 skipped

Name                         Stmts   Miss  Cover
src/eval/action.py             412    261    37%
src/finetune/dataset.py        157     88    44%
src/finetune/train_lora.py     220    169    23%
TOTAL                          789    518    34%
```

**覆盖率 34%，明显低于前三周（76%~88%），原因是本周代码的主体必须有 GPU 才能跑。**
未覆盖的部分逐个查过，全部属于以下几类：

| 未覆盖 | 为什么单元测试跑不到 |
|---|---|
| `train_lora.train()`、`pick_precision()` | 加载 4-bit 模型、跑训练循环，需要 CUDA 与基座权重 |
| `eval.action.evaluate()`、`_build_predictor()` | 加载模型逐条推理，需要 GPU 或在线 API |
| `dataset.convert()` | 从原始 ScreenAgent 数据集装载，需要先 clone 数据集 |
| 三个文件的 `main()`、`print_summary()` | 命令行入口与终端输出 |

**不依赖 GPU 的纯计算部分都在测试范围内**：坐标归一化、样本构建、训练标签的掩码位置、
提示词与训练数据的契约、评测统计 `summarize()`（报告表 5、表 6 的主结果就是它算的）、
两条评测路径的分母一致性、消融三档只差一个变量。
这些部分的正确性才决定报告里的数字对不对；GPU 路径的正确性由那两份 375 条的
实际评测结果证明。

**那 1 条 skip 是预期的**：`test_resolution_ablation.py` 里有一条要读验证集的真实
图片，而数据集图片不随包分发（见 §三 偏差 5）。源仓库里图片齐全时是 96 passed。

**依赖清单的边界是实测的**：在空环境里逐个卸掉一个包再跑测试，每个包卸掉的后果
写在 `requirements.txt` 每行右边。其中 **pillow 最危险**——卸掉它不会报错，
而是整个分辨率消融测试文件被静默跳过 22 条，只看 "passed" 会以为一切正常。

### 3. 使用权重（需要 CUDA GPU）

```bash
pip install -r requirements.txt -r requirements-train.txt
```

基座模型**不在包内**，需从 HuggingFace 下载 `Qwen/Qwen2.5-VL-3B-Instruct`。
加载方式是先载基座、再挂 adapter：

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

base = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", ...)
model = PeftModel.from_pretrained(base, "模型权重/20260827-174327/adapter")
```

> `adapter_config.json` 里的 `base_model_name_or_path` 是训练服务器上的路径
> `/root/autodl-tmp/qwen3b`，**加载时不会用到它**（上面是显式传入基座），
> 但如果用 `AutoPeftModel` 之类自动找基座的接口会失败。为了保持权重文件与训练产出
> 逐字节一致，没有改这个字段。

先校验权重完整性：

```bash
cd 模型权重/20260827-174327/adapter
md5sum -c ../MD5SUMS.txt
```

### 4. 重跑评测 / 重新训练（需要 GPU + 数据集图片）

先取回数据集图片（与源仓库 `scripts/prepare_datasets.py` 用的是同一个命令；
**这一步没有在交付包里重新实测**）：

```bash
git clone --depth 1 https://github.com/niuzaisheng/ScreenAgent.git data/raw/screenagent_repo
```

训练集与验证集用到的 1787 张图全部在该仓库的 `data/ScreenAgent/train/` 下，不需要解压。

```bash
# 评测（结果写入 docs/m3-action/，会覆盖同名文件，建议换 --tag）
python -m eval.action --local Qwen/Qwen2.5-VL-3B-Instruct \
    --adapter 模型权重/20260827-174327/adapter --tag after-rerun
python -m eval.action --report

# 训练
python -m finetune.train_lora --epochs 2 --lr 1e-4
```

---

## 三、已知偏差（必须随交付说明）

### 1. 本周成绩的边界：离线显著提升，端到端 0/25

大纲第 5 周要求"对比微调前后模型在 GUI 任务理解与动作生成上的效果"，这件事在
**离线验证集**上做完了，结果是显著提升。

但这些提升**没有传到真实桌面**。客机固定 1024×768 后跑的唯一一轮干净端到端测试
是 0/25，此前 1920×1080 下测出的 5/25 **不作为成绩引用**——分辨率与训练不同构，
且 5 次成功全部来自一个被查出很可能是误点了桌面聚焦图标的任务。

之所以把这条放在偏差第一条：只交"微调前后对比表"是完全符合大纲字面要求的，
但那样会让读者以为本地微调模型已经能用了。**它还不能。**

### 2. 离线评测是 bf16，实际部署是 4-bit

报告 §5.1~§5.3 的离线评测按 bf16 加载（复现命令未加 `--load-4bit`，代码默认 bf16）；
部署到客机的 `serve_local_model.py` 默认 4-bit NF4。旧版验证集上量化让微调模型的动作类型
准确率掉了 9.2pp，**新版验证集上 4-bit 的数字没有测过**。
报告 §7「有效性威胁」第一条已写明，表 5 的数字应视为部署配置下的上界。
**这一条是整理本交付包时新发现的，源仓库报告里没有。**

### 3. 权重许可：仅限研究用途

基座 Qwen2.5-VL-3B-Instruct 的许可证是 **`qwen-research`（仅限研究，再分发受限）**——
注意比它大的 7B 反而是 Apache-2.0。本包交付的 LoRA 权重是它的衍生物，因此：

- **只交付 adapter，不含基座权重**；
- 本权重**仅可用于研究目的，不可商用**；
- 再分发前请核对 Qwen 官方许可证全文。

`adapter/` 目录下还有 `tokenizer.json` 等 4 个文件，是训练时随 adapter 一起保存的
基座分词器与处理器配置，同受上述许可约束。

`adapter/README.md` 是 PEFT 自动生成的空白模型卡（全是 `[More Information Needed]`），
**原样保留未修改**，以保证目录内所有文件与训练产出逐字节一致。本 README 与报告替代它的作用。

### 4. src/ 下有七个包不属于本周交付，pythonpath 比 W4 多一项

本周交付的是 `finetune` 与 `eval` 两个包。另外七个包是为了让本交付物
**能独立 import 与跑测试**才一并提供的——范围由实测确定：在源仓库里跑这 4 个测试文件，
记录实际被导入的模块：

| 包 | 属于 | 为什么必须带上 |
|---|---|---|
| `data` | 第 3 周 | `finetune/dataset.py` 依赖 `data.schema` 与 `data.split` |
| `agent`、`llm` | 第 3 周 | `eval/action.py` 读提示词模板、解析模型输出 |
| `grounding` | 第 3 / 5 周 | 测试直接测 `grounding.local_vlm` 的坐标解析，与训练坐标空间必须一致 |
| `core` | 第 4 周 | `agent/session.py` 导入 `core.loop` |
| `control`、`perception` | 第 2 周 | 动作定义与坐标类型 |

`src/eval/` 下**没有**源仓库的 `eval/grounding.py`（ScreenSpot 零样本评测，不属于本周）。

`pyproject.toml` 的 `pythonpath` 是 `["src", "."]`，比 W4 多一个 `"."`：
两个消融测试会 `import scripts.ablate_prompts`，需要包根目录在路径上。

### 5. 数据集图片不随包分发

`train.jsonl` / `val.jsonl` 里每条样本的 `image` 字段指向
`data/raw/screenagent_repo/...` 下的图片。图片来自 ScreenAgent 数据集（Apache-2.0），
共 1787 张，**不打进交付包**，需要时按 §二.4 clone。因此：

- 复算报告、跑单元测试**都不需要图片**；
- 有 1 条测试会因图片不在而跳过（§二.2）；
- 重跑评测与训练需要先 clone。

### 6. 报告附录 A 用的是旧版验证集，其中一份原始文件需要修正分母

附录 A 的三项结论来自数据集重建之前的旧版验证集（142 条、只有点击类），
与正文**口径不同、不可并列**，原始文件单独放在 `docs/m3-action/旧口径/`，
这样 `python -m eval.action --report`（只扫描 `docs/m3-action/` 顶层）不会把它们
混进正文那两份结果里一起打印。

其中 `旧口径/api-8b.jsonl` 是在一个评测缺陷修复**之前**跑的：22 条"模型输出解析
不出来"被记成了"调用失败"并剔出分母。**对它直接跑 `summarize()` 会得到 35.8%，
而报告里的可比数字是 30.3%。** `scripts/recompute_report.py` 显式把这 22 条放回
分母，并同时打印修正前后两个值。**原始文件未改动。**

### 7. 训练日志没能取回

训练服务器在下载完权重后失联，`train.log` 与 `train-stats.json` 未能取回。
`模型权重/20260827-174327/训练记录.md` 是当时从服务器读出的数值的本地存档，
**是本包中唯一不能从原始日志重算的一组数**（训练时长、峰值显存、loss、
告警数）。权重文件本身已逐文件 MD5 校验。

`训练记录.md` 末尾的"未完成"清单是 **2026-08-27 训练当天的快照**，其中评测与端到端
两项后来都已完成（结果即本报告）。为保持原样，未修改。

### 8. 报告数字与源仓库报告有六处不同

- **五处相差 0.1**：源仓库报告有五个差值是用已经四舍五入过的百分数再相减得出的；
- **一处出入较大**：源报告写"70% 的训练目标是 `type`/`key`/`done`"，按训练集原始文件
  算是 **62.1%**（连 `wait`、`scroll` 一起算也只有 67.1%），源报告没有留下 70% 的计算依据。

本报告一律按原始计数重算，六处逐条列在报告附录 C，没有一处改变结论。
**源仓库报告未修改**，是否回改由项目负责人决定。

### 9. 合规偏差（与大纲不一致，如实记录）

端到端测试所用的客机**登录着开发者本人的微软账号，OneDrive 在同步个人文件**。
这与大纲「合规与落地说明」第 5 条"不包含任何个人数据或敏感信息"冲突。
该偏差已记录在案，项目决定不清理客机，因此本交付**不声称符合该条**。

本包内**不含任何客机截图**——端到端存档只有结构化的 JSON 摘要。打包前已对全部
原始数据文件扫描 API key、密码、邮箱、`.env`、登录等模式，无命中。

### 10. 交付副本与源码的全部差异

| 类型 | 文件 |
|---|---|
| 源仓库没有的 | `README.md`、`微调效果对比分析报告.md`、`pyproject.toml`、`requirements.txt`、`requirements-train.txt`、`scripts/recompute_report.py`、`模型权重/20260827-174327/MD5SUMS.txt` |
| 同名但改了 | `scripts/ablate_prompts.py`、`scripts/ablate_resolution.py`（各 +3 行，把 `src/` 加进 `sys.path`，与 W4 同样写法） |
| 改了存放位置 | `模型权重/20260827-174327/`（源仓库 `finetune/outputs/20260827-174327/`，被 `.gitignore` 排除）；`docs/m3-action/` 下分出 `对照实验/`、`交叉验证/`、`旧口径/` 三个子目录（源仓库全部平铺在一层） |
| 改写的 | `微调效果对比分析报告.md` 是为交付重写的汇报版，源仓库对应 `docs/m3-微调效果对比分析报告.md`（2002 行工程日志，含新旧两套口径） |

`src/`、`tests/`、`finetune/data/`、`docs/` 下的其余文件与源仓库逐字一致，
`模型权重/` 下的 adapter 文件逐文件 MD5 一致。

---

## 四、目录结构

```
W5-LoRA微调模型与效果对比/
├── README.md                     本文件
├── 微调效果对比分析报告.md        **交付物之二**
├── pyproject.toml                pytest / ruff / 覆盖率配置
├── requirements.txt              跑测试的依赖（边界经实测确认）
├── requirements-train.txt        训练与推理的依赖（需 CUDA）
│
├── 模型权重/20260827-174327/      **交付物之一**
│   ├── adapter/                  LoRA 权重（7 个文件，148.7MB 为主体）
│   ├── MD5SUMS.txt               逐文件校验值
│   └── 训练记录.md                训练配置、开销、loss 轨迹（服务器失联前的存档）
│
├── src/
│   ├── finetune/             ← 本周交付
│   │   ├── dataset.py            样本构建：截图 + 子任务 → 动作
│   │   └── train_lora.py         QLoRA 训练
│   ├── eval/                 ← 本周交付
│   │   └── action.py             动作生成评测，逐条记录，可断点续跑
│   ├── prompts/                  8 个提示词模板（executor_v0~v4、planner_v1/v2 等）
│   └── agent/ control/ core/ data/ grounding/ llm/ perception/   非本周交付，见 §三 偏差 4
│
├── scripts/
│   ├── recompute_report.py       **复算报告全部数字**（交付包专用）
│   ├── ablate_prompts.py         提示词三档消融
│   ├── ablate_resolution.py      分辨率消融
│   └── make_upscaled_val.py      生成放大到 0.622 缩放的验证集
│
├── tests/                        4 个测试文件（95 passed, 1 skipped）
│   ├── test_m3_modules.py        坐标归一化、样本构建、**标签掩码位置**
│   ├── test_eval_denominator.py  两条评测路径的分母必须一致
│   ├── test_prompt_ablation.py   消融三档只差一个变量、照抄检测
│   └── test_resolution_ablation.py  缩图不动真值坐标
│
├── finetune/data/
│   ├── train.jsonl  val.jsonl    训练集 1783 / 验证集 375
│   ├── val-4x3-coords.jsonl      验证集中 99 条坐标类样本（逐字子集，复算脚本会校验）
│   ├── val-upscaled-coords.jsonl 同上 99 条，图片放大到 1664×1248
│   └── meta.json                 划分指纹与动作分布
│
└── docs/
    ├── m3-prereq/
    │   ├── split-screenagent-desktop.json          冻结划分（当前版）
    │   └── split-screenagent-desktop-grounding-旧.json  重建前的旧划分（算新旧会话交叉用）
    ├── m3-action/
    │   ├── before-fixed.jsonl  after-fixed.jsonl   **报告 §5.1 主结果**（各 375 条）
    │   ├── eval_before.log  eval_after.log         两轮评测的终端输出
    │   ├── 对照实验/     提示词对照（§5.2）、分辨率配对（§5.3）
    │   ├── 交叉验证/     另一台机器上重跑的 18 条
    │   └── 旧口径/       附录 A 的 8 份旧版验证集结果
    └── m2-runs/          端到端两轮存档（§5.4）
```

代码规模：`src/` 下共 14,639 行 —— **本周交付的 `finetune` 与 `eval` 两个包 1,880 行**，
§三 偏差 4 里那七个依赖包 12,759 行。另有测试 1,195 行、脚本 813 行。
