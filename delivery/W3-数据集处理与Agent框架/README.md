# 公开 GUI 数据集处理与基础 Agent 框架 —— 第 3 周交付

> 交付物：**数据集预处理脚本 + 基础 Agent 框架代码**
> 生成日期：2026-09-03　　源仓库分支：`main`

---

## 一、大纲四项任务逐条对照

| # | 大纲任务 | 实现位置 | 状态 |
|---|---|---|---|
| 1 | 下载并预处理 ScreenAgent、WebArena、Mind2Web 等公开 GUI 任务数据集 | `scripts/prepare_datasets.py`、`src/data/**` | ✅ 完成（见 §三 偏差 2：换了数据集组合） |
| 2 | 基于 LangChain / LlamaIndex 搭建基础多模态 Agent 框架 | `src/agent/**`、`src/llm/openai_compat.py` | ✅ 完成（见 §三 偏差 1：只用 LangChain 的模型层） |
| 3 | 实现简单的任务拆解与规划能力 | `src/agent/planner.py`、`src/prompts/planner_v*.yaml` | ✅ 完成 |
| 4 | 开发大模型调用接口，支持开源多模态模型的本地部署与 API 调用 | `src/llm/**`、`scripts/serve_local_model.py` | ✅ 完成 |

### 任务 1 —— 数据集下载与预处理

`scripts/prepare_datasets.py` 是一条命令走完的流水线：

```
下载 → 解压 → 装载（三个 loader）→ 清洗（5 条规则）→ 统计 → 出图 → 冻结划分
```

**三个数据集的获取方式各不相同**，这一点容易踩坑：

| 数据集 | 来源 | 备注 |
|---|---|---|
| ScreenSpot | HF `rootsautomation/ScreenSpot` | parquet，截图内嵌 |
| ScreenSpot-v2 | HF `OS-Copilot/ScreenSpot-v2` | JSON + 1.3GB 图片 zip |
| ScreenAgent | **GitHub** `niuzaisheng/ScreenAgent` | **HF 上只有权重没有数据**，按名字在 HF 搜数据集是搜不到的 |

`src/data/schema.py` 定义 `UnifiedSample` —— 三个来源的字段名、坐标约定、
动作词表各不相同，全部归一到这一个结构，下游（统计、划分、微调数据构建）
只认它。**这是整条数据链的单一真相来源。**

清洗（`src/data/clean.py`）用 5 条规则，**剔除而不修补**：

| 规则 | 理由 |
|---|---|
| `bbox_out_of_bounds` | 元素框超出截图。剔除而不裁剪——剔除量异常时正好暴露格式解析错误 |
| `bbox_too_small` | 面积小于 100px²，既难点中也多半是标注噪声 |
| `bbox_degenerate` | 宽或高为 0，通常是 xywh / xyxy 混淆的产物 |
| `point_out_of_bounds` | 落点超出截图 |
| `no_text_no_type` | 既无指令文本也无类型信息，构不成监督信号 |

实测产出（`docs/数据集统计分析报告.md`）：

| 数据集 | 样本 | 不重复截图 | 带边界框 | 带坐标点 |
|---|---:|---:|---:|---:|
| ScreenSpot | 1,272 | 610 | 1,272 | 1,272 |
| ScreenSpot-v2 | 1,271 | 729 | 1,271 | 1,271 |
| ScreenAgent | 4,012 | 2,634 | 218 | 934 |
| **合计** | **6,555** | **3,973** | **2,761** | **3,477** |

清洗保留率 100.0%（6556 → 6555，剔除 1 条 `bbox_out_of_bounds`）。

`src/data/split.py` 做的是**会话级冻结划分**，不是随机划分。理由：同一个
ScreenAgent 会话里的连续步骤共享截图，按行随机切会让训练集和验证集出现
同一张图，**指标会虚高**。划分结果连同指纹写进文件，之后不再重算。

### 任务 2 —— 基础多模态 Agent 框架

框架分三层，每层单独可测：

```
agent/session.py     会话编排：规划 → 逐子任务执行 → 汇总
  ├─ agent/planner.py    任务拆解（任务 3）
  ├─ agent/context.py    上下文窗口与历史裁剪
  └─ agent/prompts.py    提示词模板（YAML 外置，src/prompts/）
llm/                 模型调用抽象（任务 4）
```

`agent/prompts.py` 把提示词外置成 YAML 而不是写死在代码里。这件事的收益在
第 5 周兑现：`src/prompts/` 下 8 个模板（`executor_v0~v4`、`planner_v1/v2`、
`executor_modeb_v1`）能在不改一行代码的前提下做提示词消融。

**动作空间是提示词与数据之间的契约**——`agent/prompts.py` 里的动作清单
不是手写的，是从 `control/actions.py` 的 `ACTION_SPECS` 现场生成的。抄一份
过去的话，动作空间一改，提示词就会静默失配。

### 任务 3 —— 任务拆解与规划

`agent/planner.py` 的 `Planner.plan()` 接「用户指令 + 可选截图」，出
`Plan`（一组 `SubTask`）。三个设计点：

- **上限 `MAX_SUBTASKS`**，防止模型把一句话拆成二十步后自己走丢。
- **`granularity_report()` / `is_fine_grained()`** —— 拆解粒度是可测量的，
  不是感觉。第 5 周的实测正是靠它发现问题：在线 8B 平均拆 3.6 个子任务，
  本地微调 3B 只拆 2.1 个，`send_message` 上更是拆成 1 个然后在这一个上
  烧掉 12 步。**「任务拆解完全合格」那句话就是被这个指标推翻的。**
- **解析容错**（`_parse` / `_strip_numbering`）—— 模型输出的编号、
  markdown 列表、多余引号都要吃得下，否则一个格式抖动就让整轮失败。

### 任务 4 —— 大模型调用接口（本地部署 + API 调用）

`llm/base.py` 定义 `LLMBackend` 抽象，两类实现：

| 通路 | 实现 | 说明 |
|---|---|---|
| **API 调用** | `llm/openai_compat.py` | 走 `langchain_openai.ChatOpenAI`，三家 OpenAI 兼容端点 |
| **本地部署** | `llm/qwen_vl_local.py` | 进程内加载 Qwen2.5-VL，4-bit NF4 量化 + LoRA 适配器 |
| 本地部署（跨机） | `scripts/serve_local_model.py` | 把本地权重包成 OpenAI 兼容 HTTP 接口 |
| 测试替身 | `llm/fake.py` | `ScriptedBackend`，单元测试不碰真实模型 |

`llm/providers.py` 是平台注册表，四个条目：`dashscope`、`zhipu`、
`selfhost`、`nvidia`。各家的差异（鉴权头、图片大小上限、是否支持
`response_format`）全收在这一处，加平台只需加一条记录。

**关于 `selfhost` 走 HTTP 是否还算「本地部署」**：算。M0《硬件与部署环境》
已裁定——客机没有 GPU 直通，而键鼠执行只能在客机内，两条约束逼出
「宿主机跑权重 / 客机跑执行」这个拓扑，HTTP 只是传输方式，**权重和数据都
不出本机**。这与短租远程服务器不同，那种情况权重不在自己机器上，只算
API 调用。`llm/providers.py` 的 `weights_local` 字段就是记这件事的。

---

## 二、怎么运行

### 跑单元测试（不需要真实桌面、不需要 GPU、不需要 API Key）

```bash
cd <本目录>
pip install -r requirements.txt          # 只跑测试的话装第 1 段即可
python -m pytest                         # 369 passed in 4.22s
python -m pytest --cov=data --cov=llm --cov=agent --cov-report=term-missing
```

`pyproject.toml` 里配了 `pythonpath = ["src"]`，**不需要安装**，直接跑。

实测结果（2026-09-03，Windows 11 / conda `gui-agent` / Python 3.10.20）：

```
369 passed in 4.22s
覆盖率（只统计本周交付的三个包）：2071 语句，缺 253，TOTAL 88%
```

| 包 | 覆盖率 |
|---|---:|
| `agent` | 93% ~ 96%（context 95 / planner 93 / prompts 96 / session 94） |
| `data` | 77% ~ 97%（schema 97 / split 97 / clean 89 / loaders 77~90 / stats 86） |
| `llm` | 59% ~ 98%（base 98 / fake 98 / parsing 94 / providers 86 / openai_compat 84 / **qwen_vl_local 62 / factory 59**） |

两个偏低的都在**需要真实外部资源**的路径上：`qwen_vl_local.py` 未覆盖的
是真正加载权重与推理的段落（需要 GPU），`factory.py` 未覆盖的是按
`--provider` 构造真实后端的分支（需要 API Key）。单元测试用
`llm/fake.py` 的 `ScriptedBackend` 替身，**不碰真实模型也不碰网络**。

### 跑数据集预处理

```bash
python scripts/prepare_datasets.py --download    # 下载 / clone 三个数据集
python scripts/prepare_datasets.py --extract     # 解压两个 zip（1.3GB + 50MB）
python scripts/prepare_datasets.py               # 装载→清洗→统计→出图→冻结划分
python scripts/prepare_datasets.py --no-charts   # 只出数字，不画图
```

`--download` 与 `--extract` 分开，是因为解压 ScreenSpot-v2 图片包要写 1.3GB
到磁盘，这种动作不该混在别的步骤里悄悄发生。

```bash
python scripts/survey_web_datasets.py            # 抽样查看网页数据集（不下图）
```

### 验证模型调用通路

```bash
python scripts/verify_llm.py --provider dashscope     # API 通路
python scripts/verify_llm.py --provider local         # 本地权重，进程内
python scripts/serve_local_model.py --port 8000       # 本地权重，包成 HTTP 服务
```

> **凭据放 `.env`，`.env` 必须在 `.gitignore`。仓库内不得出现任何 API Key。**
> 本交付包内不含 `.env`，也不含任何凭据。

---

## 三、已知偏差（必须随交付说明）

### 1. LangChain 只用了模型层，没有用它的 Agent / Chain 抽象

大纲写的是「基于 LangChain / LlamaIndex 搭建基础多模态 Agent 框架」。
**LangChain 确实在用**——`llm/openai_compat.py` 走 `langchain_openai.ChatOpenAI`
构造请求、用 `langchain_core.messages` 组装多模态消息。

**但任务拆解、上下文管理、执行循环是自研的**，没有用 LangChain 的
`AgentExecutor` / `Chain` / `Tool` 那一套。理由有二：

- 本项目的循环需要在**每一步之间插入截图与帧差判定**，LangChain 的
  Agent 循环把这一段封在内部，插桩要绕。
- 第 6 周要做的 Reflector 级联判定需要拿到每一步的原始输出与执行结果，
  自研循环（`core/loop.py`）的轨迹记录是逐步落盘的，这是第 6 周
  405 步错误分类数据的来源。

**这构成对大纲字面的偏离，故在此声明。** 若评审要求必须用 LangChain 的
Agent 抽象，替换点是 `agent/session.py` 一个文件。

### 2. 数据集组合与大纲点名的不完全一致

大纲点名「ScreenAgent、WebArena、Mind2Web **等**」。实际采用：

| 大纲点名 | 实际 | 原因 |
|---|---|---|
| ScreenAgent | ✅ 采用，4,012 条 | 唯一提供「截图 + 任务目标 → 动作」的，正是动作生成需要的监督信号 |
| Mind2Web | 🟡 **测量了但未混入训练** | 全量 7,775 行分布测量见 `docs/数据集mind2web结果分析.md`。截图高度中位 4,179px、最大 44,771px，`max_pixels` 下采样后元素不可辨；且它的 `SELECT` 动作在本项目动作空间里没有对应物 |
| WebArena | ❌ 未采用 | 它是**交互式环境**不是静态数据集，需要起一整套 Docker 网站副本，八周工期内不可行 |
| （补充）ScreenSpot / ScreenSpot-v2 | ✅ 采用，2,543 条 | 补桌面 grounding 样本。**ScreenSpot 固定为零样本测试集**，不参与训练、验证、提示词选择与超参调优 |

「等」字容得下这个替换，但**替换了什么、为什么**必须写明，故有本条。

### 3. src/ 下有五个包不属于本周交付

本周交付的是 `data` / `llm` / `agent` 三个包。另外五个包是为了让本交付物
**能独立 import 与跑测试**才一并提供的：

| 包 | 属于 | 为什么必须带上 |
|---|---|---|
| `control` | 第 2 周 | `llm/base.py` 与 `agent/prompts.py` 依赖 `control.actions` 的动作定义 |
| `perception` | 第 2 周 | `data/schema.py` 依赖 `perception.types` 的 `BBox` / `Point` |
| `core` | 第 4 / 6 周 | `agent/session.py` 依赖 `core.loop` 与 `core.trajectory` |
| `grounding` | 第 3 / 5 周 | `core.loop` 依赖 `grounding.base`。`native.py` 是模型直接出坐标的通路（模式 B），`local_vlm.py` 是本地 grounding 模型 |
| `finetune` | 第 5 周 | `llm/qwen_vl_local.py` 从 `finetune.train_lora` 读 `SYSTEM_PROMPT` 与 `DEFAULT_MAX_PIXELS`，**而不是抄一份数字过去**——抄了就会在训练侧改动后静默失配 |

`pyproject.toml` 的覆盖率统计**只算 `data` / `llm` / `agent`**，否则会把
本周的覆盖率稀释成一个没有意义的数。

### 4. 交付副本里 `scripts/` 下四个脚本比源码多三行

源仓库把各个包放在根目录，交付包放在 `src/` 下，所以脚本原有的

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

在这里指向的是交付包根目录而不是 `src/`。四个脚本各加了三行：

```python
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "src")
)  # 交付包把各个包放在 src/ 下
```

**这是交付副本与源码之间唯一的差异**，与第 2 周交付包的做法一致
（`delivery/W2-桌面感知与控制模块/scripts/` 下四个脚本同样多这三行）。
`src/`、`tests/`、`docs/` 下所有文件与源仓库逐字一致，已用逐文件二进制
比对核过。

### 5. `data/charts.py` 无单元测试

出图模块，依赖 matplotlib，产物是 `docs/数据集统计分析报告.md` 里引用的
四张图。已在覆盖率统计里 `omit`，不计入本周覆盖率——留着会显示成 0%，
那是误导。

---

## 四、目录结构

```
W3-数据集处理与Agent框架/
├── README.md                     本文件
├── pyproject.toml                pytest / ruff / 覆盖率配置
├── requirements.txt              依赖清单（四段，按用途分）
├── src/
│   ├── data/                     ← 本周交付：数据集处理（6 模块 + 3 个 loader）
│   │   ├── schema.py                 统一样本结构，整条数据链的单一真相来源
│   │   ├── clean.py                  5 条清洗规则
│   │   ├── split.py                  会话级冻结划分（防截图泄漏）
│   │   ├── stats.py / charts.py      统计与出图
│   │   └── loaders/                  三个数据集各自的装载器
│   ├── llm/                      ← 本周交付：模型调用接口（8 模块）
│   │   ├── base.py                   LLMBackend 抽象
│   │   ├── openai_compat.py          API 通路（LangChain 模型层）
│   │   ├── qwen_vl_local.py          本地部署通路（4-bit + LoRA）
│   │   ├── providers.py              四家平台注册表
│   │   └── factory.py / fake.py / parsing.py
│   ├── agent/                    ← 本周交付：Agent 框架（5 模块）
│   │   ├── planner.py                任务拆解与规划
│   │   ├── context.py                上下文窗口
│   │   ├── prompts.py                提示词模板加载
│   │   └── session.py                会话编排
│   ├── prompts/                  8 个 YAML 模板
│   ├── control/ perception/ core/ finetune/ grounding/   非本周交付，见 §三 偏差 3
├── scripts/
│   ├── prepare_datasets.py       **交付物之一：数据集预处理脚本**
│   ├── survey_web_datasets.py    网页数据集抽样查看（不下图）
│   ├── verify_llm.py             模型调用通路自检
│   └── serve_local_model.py      本地权重包成 OpenAI 兼容 HTTP 服务
├── tests/                        8 个测试文件，369 个用例（全部通过）
└── docs/
    ├── 数据集统计分析报告.md          prepare_datasets.py 的产物
    ├── 数据集mind2web结果分析.md      Mind2Web 全量分布测量与不采用的依据
    ├── 网页数据集抽样查看.md          survey_web_datasets.py 的产物
    └── 模型与数据集许可证对照表.md     许可证核对（Qwen2.5-VL-3B 为 Qwen-Research，仅限研究）
```

代码规模：`src/` 下共 12,157 行 —— **本周交付的 `data` / `llm` / `agent`
三个包 6,790 行**，§三 偏差 3 里那五个依赖包 5,367 行。
