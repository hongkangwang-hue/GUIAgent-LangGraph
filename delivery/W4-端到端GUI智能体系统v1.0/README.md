# 端到端 GUI 智能体系统原型 v1.0 —— 第 4 周交付

> 交付物：**端到端 GUI 智能体系统原型 v1.0 + 基础任务测试报告**
> 生成日期：2026-09-03　　源仓库分支：`main`

---

## 一、大纲四项任务逐条对照

| # | 大纲任务 | 实现位置 | 状态 |
|---|---|---|---|
| 1 | 将感知模块、控制模块与 Agent 框架进行无缝集成 | `src/core/loop.py`、`src/agent/session.py` | ✅ 完成 |
| 2 | 实现「用户指令→屏幕感知→任务规划→动作执行→结果反馈」的完整闭环 | 同上 + `src/core/trajectory.py`、`src/core/verify.py` | ✅ 完成 |
| 3 | 开发简单的命令行交互界面 | `src/cli/main.py`、`src/cli/panel.py` | ✅ 完成 |
| 4 | 测试并调试 5 个基础桌面任务 | `tasks/basic_tasks.yaml`、`scripts/run_basic_tasks.py` | ✅ 完成，见 `基础任务测试报告.md` |

### 任务 1、2 —— 完整闭环

大纲要求的五个环节，各自落在哪：

```
用户指令  →  agent/session.py     接自然语言，调规划器
屏幕感知  →  perception/capture   截图 + DPI 感知 + 坐标换算
任务规划  →  agent/planner.py     拆成若干子任务
动作执行  →  control/executor.py  结构化动作 → 真实键鼠
结果反馈  →  core/verify.py       程序化判定 + core/trajectory.py 逐步落盘
```

`core/loop.py`（289 语句）是把这五段串起来的那一层。每一步做四件事：
截图 → 问模型 → 执行动作 → 记轨迹。**每步都落盘**，不是跑完再写——
中途崩溃或人工中断时数据不丢，这是后续三个里程碑所有分析的数据来源。

**「结果反馈」用的是程序化判定，不是让模型自己说完成了。**
`core/verify.py` 提供三类判据：进程是否在运行、窗口标题是否匹配、文件内容
是否符合预期。这个选择有实测依据——`基础任务测试报告.md` §四：

> 模型的自我判断与任务实际状态**无关**，不是偏乐观，是无关。脏环境下它
> 自报完成 20/25 而实测 11/25（高估）；干净环境下自报 16/24 而实测 19/24
> （低估）。两个方向都错过。

配套的还有 **precondition（起点检查）**：每一轮开始前先确认环境处在判据的
**反方向**——例如 `open_browser` 的判据是「`msedge.exe` 在运行」，那么起点
必须先确认它**没在运行**。否则上一轮的残留会让这一轮凭空成功。

### 任务 3 —— 命令行交互界面

`cli/main.py` 基于 typer，四个子命令：

| 命令 | 作用 |
|---|---|
| `run` | 执行一个任务或一份任务清单 |
| `replay` | 逐帧回放一条历史轨迹 |
| `label` | 给失败步骤打错误类型标签 |
| `config` | 查看配置、模板与历史轨迹 |

`cli/panel.py` 用 rich 做实时面板，跑的时候能看到当前子任务、第几步、
模型在想什么、动作是什么、判定结果。

**`run` 默认只演练（dry-run）**：走完整链路——截图、问模型、解析动作、
记轨迹——但**不发出任何键鼠事件**。要真正操作桌面必须显式加 `--execute`，
并且会先弹一次确认。这条默认值是刻意选的：本项目在 M2 期间发生过三次
真实安全事件（Agent 修改并保存了 `.env`、在真实微软账号登录框连按 12 次
回车、进到 Edge 的 Chrome 数据导入对话框且「保存的密码」处于勾选状态）。

### 任务 4 —— 五个基础任务

`tasks/basic_tasks.yaml` 定义了大纲点名的五个任务，每个任务包含
指令、起点检查、程序化判据、重置脚本、步数上限：

| 任务 | 指令 | 判据 |
|---|---|---|
| `open_browser` | 打开 Microsoft Edge 浏览器 | 进程 `msedge.exe` 在运行 |
| `search_content` | 打开 Edge，在地址栏搜索「Python 官方文档」 | 进程 + 窗口标题匹配 `Python.*Edge` |
| `open_file` | 用记事本打开 `C:\agent-test\测试文档.txt` | 进程 `notepad.exe` + 标题含「测试文档」 |
| `send_message` | 在「测试消息」程序里发送「你好世界」 | `messages.log` 文件内容匹配 |
| `close_app` | 关闭计算器窗口 | 进程 `CalculatorApp.exe` 已退出 |

`send_message` 用的是本项目自带的 `tasks/mock_messenger.py`（tkinter 小程序，
消息只写本地文件、不联网）——**不接入任何真实通讯软件**，避免误发消息。

---

## 二、怎么运行

### 跑单元测试（不需要真实桌面、不需要 GPU、不需要 API Key）

```bash
cd <本目录>
pip install -r requirements.txt          # 跑测试装第 1 段即可
python -m pytest                         # 221 passed in 3.08s
python -m pytest --cov=core --cov=cli --cov-report=term-missing
```

`pyproject.toml` 里配了 `pythonpath = ["src"]`，**不需要安装**，直接跑。

实测结果（2026-09-03，Windows 11 / conda `gui-agent` / Python 3.10.20，
**zip 解压到干净临时目录后执行**）。完整记录见 `基础任务测试报告.md` §八：

```
221 passed in 3.08s
覆盖率（只统计本周交付的 core 与 cli 两个包）：1125 语句，缺 263，TOTAL 77%
```

### 跑命令行界面

```bash
# 看有哪些命令
python gui-agent.py --help

# 演练：走完整链路但不碰键鼠。任何机器都能跑
python gui-agent.py run "打开记事本" --provider dashscope

# 真机执行：**只能在隔离虚拟机客机内跑**
python gui-agent.py run "打开记事本" --provider dashscope --execute

# 批量跑五个基础任务
python scripts/run_basic_tasks.py --execute --provider dashscope

# 回放一条轨迹 / 查看配置
python gui-agent.py replay <trajectory-id>
python gui-agent.py config --show
```

> 源仓库里用的是 `python -m cli`。交付包把各个包放在 `src/` 下，`python -m cli`
> 找不到包，因此新增了根目录的 `gui-agent.py` 作为入口，见 §三 偏差 5。

> **`--execute` 会真的操作鼠标键盘。** 本项目全局约束要求一切键鼠执行
> 只发生在隔离虚拟机客机内，宿主机不受控制。急停：鼠标甩到屏幕左上角
> 触发 PyAutoGUI FAILSAFE，或按全局热键 `Ctrl+Alt+Q`。

### 客机内准备五个任务的环境

```bash
python tasks/setup_env.py            # 建 C:\agent-test 与测试文件
python tasks/mock_messenger.py       # 起「测试消息」小程序
python tasks/reset_desktop.py        # 每轮之间重置桌面
```

### 凭据

凭据放 `.env`，`.env` 必须在 `.gitignore`。**仓库内不得出现任何 API Key。**
本交付包内不含 `.env`，也不含任何凭据。

---

## 三、已知偏差（必须随交付说明）

### 1. 「原型 v1.0」的确切含义：不含 Reflector

大纲第 6 周要交付「优化后的系统 v2.0」，因此本周这一版定为 **v1.0**。
两者的差别只有一处：v2.0 打开 `--reflector` 开关，在模型报告完成时用帧差
做一次否决判定；v1.0 没有这一层，模型说完成就是完成。

`src/core/reflector.py` 属于第 6 周，它在 `core` 包里、被 `loop.py` 引用，
所以随包提供，但**默认关闭**，也**不计入本周覆盖率**（它的 17 条测试在
主仓库的 `tests/test_reflector.py`）。

### 2. 基础任务测试报告的 19/24 由两批运行拼接而成

`基础任务测试报告.md` 的主结果不是一次跑出来的：前四个任务取
`20260824-003649-all-exec.json`（R4，00:36），`发送消息` 取
`20260824-100516-send_message-exec.json`（R5，10:05 补跑）。分母 24 而非 25
是因为 R5 第 2 轮执行期间操作者碰了鼠标，该轮在存档里标记 `excluded`
并排除出分子分母——**标记而非删除**，否则没人知道这一批从 5 轮变成了 4 轮。

两批的环境配置一致，但 API 延迟相差 8.97 倍，**耗时一列不可跨批比较**。

本交付包 `docs/m2-runs/` 下放的正是这三个文件（含 R3 脏环境组），报告里的
19/24、16/24、11/25 三个数都可以在包内重算复现。

### 3. 基础任务测试报告的模型口径是在线 8B，不是本地微调模型

`基础任务测试报告.md` 的 19/24 = 79% 是用阿里云百炼的
`qwen3-vl-8b-instruct` 跑出来的。本地微调模型的端到端数据属于第 5、6 周
的内容，不在本报告口径内。

**这份报告最该被记住的不是那个 79%。** 同一套代码、同一个模型，在一天之内
测出过 44%、75%、79% 三个数，差异全部来自评测环境而非模型——报告 §3 逐次
说明了每个数字之间改了什么。这也是本项目后来把「起点检查 + 程序化判据 +
快照名进存档」定为硬性要求的直接原因。

### 4. src/ 下有六个包不属于本周交付

本周交付的是 `core` 与 `cli` 两个包。另外六个包是为了让本交付物
**能独立 import 与跑测试**才一并提供的：

| 包 | 属于 | 为什么必须带上 |
|---|---|---|
| `agent`、`llm` | 第 3 周 | `core/loop.py` 与 `cli/main.py` 依赖规划器、提示词与模型后端 |
| `grounding` | 第 3 / 5 周 | `core/loop.py` 依赖 `grounding.base`；`native.py` 是模式 A 的实现 |
| `control`、`perception` | 第 2 周 | 动作执行与屏幕感知，闭环的两端 |
| `finetune` | 第 5 周 | `llm/qwen_vl_local.py` 从 `finetune.train_lora` 读训练常量，**而不是抄一份数字过去** |

`finetune` 只带了 `__init__.py` 与 `train_lora.py` 两个文件，未带
`dataset.py`——后者依赖整个 `data` 包，而本周的任何代码路径都用不到它。

### 5. 交付副本比源码多三行 ×3，另加一个源仓库没有的入口文件

源仓库把各个包放在根目录，交付包放在 `src/` 下，所以脚本原有的

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

在这里指向的是交付包根目录而不是 `src/`。`scripts/run_basic_tasks.py`、
`scripts/compare_runs.py`、`tasks/wait_for_process.py` 各加了三行：

```python
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "src")
)  # 交付包把各个包放在 src/ 下
```

此外新增了一个**源仓库里没有的文件** `gui-agent.py`（根目录）。原因同上：
`pyproject.toml` 里的 `pythonpath = ["src"]` 只对 pytest 生效，对
`python -m cli` 不生效，所以交付包里 `python -m cli` 会报
`No module named cli`。`gui-agent.py` 把 `src` 加进 `sys.path` 再转交给
`cli.main`，让命令行用法与源仓库一致。文件开头写明了它是交付包专用。

还有一处：`基础任务测试报告.md`（315 行）是**为交付改写的汇报版**，与源仓库
的 `docs/m2-basic-tasks-report.md`（607 行）内容不同。两者的数据完全一样，
差别在写法——源仓库那份是 M2 阶段一路记下来的工程日志，含四轮测量的完整
推导过程与逐条交叉引用；交付这份重新组织成十节，并把单元测试的内容并进
第八节（大纲第 4 周只要求一份「基础任务测试报告」，不要求单元测试报告）。
**报告里的每个数字都没有改动**，都能用 `docs/m2-runs/` 下的原始数据重算。

**以上就是交付副本与源码之间的全部差异**：

| 类型 | 文件 |
|---|---|
| 源仓库没有的 | `README.md`、`pyproject.toml`、`requirements.txt`、`gui-agent.py` |
| 同名但改了 | `scripts/run_basic_tasks.py`、`scripts/compare_runs.py`、`tasks/wait_for_process.py`（各 +3 行 path） |
| 改写的 | `基础任务测试报告.md` |

`src/`、`tests/`、`tasks/`（除 `wait_for_process.py`）、`docs/` 下的全部文件
与源仓库逐字一致，已用逐文件二进制比对核过。

### 6. CLI 两个模块的覆盖率偏低（58% / 55%）

`cli/main.py` 与 `cli/panel.py` 未覆盖的部分是需要真实终端与实时刷新的
交互路径（rich Live 循环、进度面板重绘、执行前的确认提示）。这些路径由
真机运行覆盖，不适合放进单元测试。详见 `基础任务测试报告.md` §八。

---

## 四、目录结构

```
W4-端到端GUI智能体系统v1.0/
├── README.md                     本文件
├── gui-agent.py                  命令行入口（**交付包专用**，见 §三 偏差 5）
├── 基础任务测试报告.md            **交付物之二**：五个任务的实测结果 + 单元测试
├── pyproject.toml                pytest / ruff / 覆盖率配置
├── requirements.txt              依赖清单（三段，第 1 段边界经实测确认）
├── src/
│   ├── core/                 ← 本周交付：执行闭环（6 模块）
│   │   ├── loop.py               Agent Loop，把五个环节串起来
│   │   ├── trajectory.py         逐步落盘的轨迹记录
│   │   ├── verify.py             程序化判定（进程 / 窗口 / 文件）
│   │   ├── retry.py              重试策略（升级动作，不是重复同一个）
│   │   └── reflector.py          第 6 周的 Reflector，默认关闭
│   ├── cli/                  ← 本周交付：命令行界面（4 模块）
│   │   ├── main.py               run / replay / label / config
│   │   └── panel.py              rich 实时面板
│   ├── agent/ llm/ grounding/ control/ perception/ finetune/   非本周交付，见 §三 偏差 4
│   └── prompts/                  8 个 YAML 模板
├── tasks/
│   ├── basic_tasks.yaml          **五个基础任务的定义**（指令 / 起点 / 判据 / 重置）
│   ├── mock_messenger.py         「测试消息」小程序，只写本地文件、不联网
│   ├── setup_env.py              客机环境准备
│   ├── reset_desktop.py          每轮之间重置桌面
│   └── wait_for_process.py       等待进程就绪（UWP 冷启动超过 3 秒）
├── scripts/
│   ├── run_basic_tasks.py        **批量跑五个基础任务**
│   └── compare_runs.py           多次运行的汇总对比
├── tests/                        8 个测试文件，221 个用例（全部通过）
└── docs/                         支撑实测记录（与报告附录「原始数据」一一对应）
    ├── m2-runs/
    │   ├── 20260824-000924-all-exec.json        R3 脏环境全量 25 轮
    │   ├── 20260824-003649-all-exec.json        R4 清场后全量 25 轮
    │   └── 20260824-100516-send_message-exec.json  R5 补跑 5 轮（含 1 轮剔除）
    ├── m2-behavior-stats.json    行为统计（重复动作、步数）
    ├── m2-error-labels.json      失败步骤的错误标签
    ├── m2-stability-report.md    连续运行稳定性
    └── m2-stability-raw.json     稳定性测试的逐点采样
```

代码规模：`src/` 下共 10,589 行 —— **本周交付的 `core` 与 `cli` 两个包
2,713 行**，§三 偏差 3 里那六个依赖包 7,876 行。
