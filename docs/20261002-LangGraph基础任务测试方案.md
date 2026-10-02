# LangGraph 基础任务 5×5 实测方案（2026-10-02）

> 执行状态：客机控制台已报告 5 项各 5/5、合计 25/25；存档
> `docs/m2-runs/20261002-082800-all-exec-online-lg-5x5-v8v5.json`
> 尚未同步到主机核验。首步成功 19/25、人工干预 0/25、安全事件 0。
> 以下保留本批固定条件和复核口径。

## 固定条件

| 项目 | 本批设置 |
|---|---|
| 客机仓库 | `C:\Users\22900\GUIAgent-LangGraph`，运行前核对提交 `63cf0ab` |
| 任务清单 | `tasks/basic_tasks.yaml`，5 个任务各 5 次，共 25 轮 |
| 模型 | 阿里云百炼 `dashscope / qwen3-vl-8b-instruct` |
| 引擎 | `langgraph` |
| 提示词 | `planner_v8` + `executor_v5` |
| 等待、Reflector、动作升级 | 保持脚本默认值；本批不加相关开关 |
| 客机基线 | 记录实际 VMware 快照名；确认记事本不恢复上次会话，测试消息程序可启动，屏幕分辨率和 DPI 不变 |

主机仓库更新代码并推送后，客机 `git pull --ff-only`。开始前在客机运行
`python tasks/setup_env.py --check`；若缺测试目录或文件，运行
`python tasks/setup_env.py` 后重查。不要把客机 `.env` 或截图带出客机。

```powershell
cd C:\Users\22900\GUIAgent-LangGraph
git pull --ff-only origin main
git rev-parse --short HEAD
python tasks/setup_env.py --check
$snapshot = '实际使用的 VMware 快照名'
python scripts/run_basic_tasks.py --execute --repeats 5 --provider dashscope --model qwen3-vl-8b-instruct --engine langgraph --planner-template planner_v8 --executor-template executor_v5 --guest-snapshot $snapshot --tag lg-5x5-v8v5
```

`--execute` 会要求输入 `yes`。运行期间不要碰客机键鼠；若触发急停，保留脚本生成的部分存档，并核对环境后重新开批。每轮的 `reset` 和起点判定由脚本执行；无效轮不计成功率分母，也不能把不足 5 个有效样本的任务写成“5 次稳定”。

## 验收与失败分析

本批以 `docs/m2-runs/<时间戳>-all-exec-online-lg-5x5-v8v5.json` 为原始证据。先核对 `partial=false`、`scope=all`、`executed=true`、`repeats=5`、`engine=langgraph`、在线后端、快照名、模板、屏幕设置、25 条记录及每任务 5 条。随后只用 `precondition_ok=true` 且未 `excluded` 的轮次计算成功率，分别列出程序化判定、自报完成、步数、耗时、安全事件与人工干预。

失败轮按 `trajectory_id` 回查客机 `outputs/trajectories/`：区分起点无效、规划失败、零动作报完成、重复无变化动作、错误窗口/焦点、判定器误判、超时与安全熔断。导出结构化轨迹时使用 `scripts/export_trajectory.py --no-frames --only <轨迹 ID> --out <目录>`；该脚本会扫描文本并拒绝导出疑似敏感内容。截图留在客机。

完成 LangGraph 批次后，再把客机恢复到**同一基线快照**，先保存本批存档，再固定相同提交、任务、模型、模板、屏幕和默认开关，仅改 `--engine legacy` 与标记，运行同规模 5×5。比较两个批次的有效轮成功率、平均步数及耗时；旧版或不同快照的单轮结果不进入该 A/B。

此前五项各自的单轮 1/1 仅说明那一轮成功，不作为本批统计值。
