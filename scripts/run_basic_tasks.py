"""跑 M2 的 5 个基础任务，每个 N 次，程序化判定成功率。

对应 M2 任务 8 与验收标准 2、3、5、6、7、10。

## 一轮的流程

    reset（恢复初始状态）→ 跑 Agent → 程序化判定 → 记录 → 下一轮

**reset 在每一轮之前，不是每个任务之前。** 第 3 次开始时的环境若与
第 1 次不同，这 5 次的成功率就不可比——而那正是本脚本要测的数字。

## 判定不看模型说了什么

模型自报 `done` 只表示循环正常收尾，**不等于任务完成**。成功与否由
`core/verify.py` 查客机的进程 / 窗口 / 文件决定。两者在轨迹里是分开的
字段（`status` 与 `verified`），M4 分析失败原因时要能区分
「模型以为做完了但没做完」和「模型自己也知道没做完」。

## 安全

默认**演练模式**（`dry_run`），不碰键鼠，用于检查任务清单与判定器本身
有没有写错。真正执行要显式加 `--execute`。

    python scripts/run_basic_tasks.py                    # 演练
    python scripts/run_basic_tasks.py --execute          # 实机执行
    python scripts/run_basic_tasks.py --execute --repeats 5
    python scripts/run_basic_tasks.py --execute --only open_browser
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TASK_FILE = Path("tasks/basic_tasks.yaml")
REPORT = Path("docs/m2-basic-tasks-report.md")

#: 「最近一次」的固定路径，方便脚本引用。
RAW = Path("docs/m2-basic-tasks-raw.json")
#: **每次跑都另存一份带时间戳的。**
#:
#: 原来只写 RAW，于是一次 `--only send_message` 就把前面整整 25 轮的
#: 数据覆盖没了 —— 那 25 轮跑了二十多分钟、花了真实 API 费用，
#: 控制台输出滚过去就再也拿不回来。
#:
#: 评测数据是**跑一次就贵一次**的东西，默认行为不该是覆盖。
RUNS = Path("docs/m2-runs")


@dataclass
class RunRecord:
    task: str
    title: str
    attempt: int
    verified: bool = False
    verify_detail: str = ""
    loop_status: str = ""
    model_said_done: bool = False
    steps: int = 0
    duration_s: float = 0.0
    cost_cny: float = 0.0
    trajectory_id: str = ""
    error: str = ""
    latency: dict = field(default_factory=dict)
    #: 起点是否成功建立。False 表示这一轮**无效**，既不算成功也不算失败。
    precondition_ok: bool = True
    precondition_detail: str = ""
    #: 子任务总数，与其中「一个动作都没执行就报完成」的个数。
    subtasks: int = 0
    empty_done: int = 0
    #: 事后剔除标记。**只标记，不删除**——删掉的话没人知道这一批
    #: 从 5 轮变成了 4 轮。目前唯一的用途是记录人为干预：评测期间
    #: 有人碰了鼠标，那一轮的起点被改过，既不算成功也不算失败。
    excluded: bool = False
    exclusion_reason: str = ""
    #: **首步成功**：本轮第一步的动作执行成功，且执行后屏幕发生了变化。
    #: None 表示判定不了（演练、或第一步没有帧差记录），不进分母。见 `first_step_outcome`。
    first_step_ok: bool | None = None
    first_step_detail: str = ""
    #: 人工干预的方式：`hotkey`（急停热键）/ `failsafe`（鼠标甩到角落）；没有为空。
    #: 事后用 `scripts/exclude_round.py` 标记的「有人碰了鼠标」记在 `excluded`，
    #: 两者一起构成人工干预率的分子。
    intervention: str = ""
    #: 急停的触发来源（`hotkey` / `failsafe` / `sentinel:<规则>`）。本轮没急停为空。
    emergency_reason: str = ""
    #: 本轮的安全事件：哨兵命中（动作前 / 开跑前环境扫描）与安全护栏拦下的危险输入。
    safety_events: list = field(default_factory=list)


#: 人按的急停。其余来源（`sentinel:*`）是程序自己刹的，算安全事件不算人工干预。
HUMAN_STOP_REASONS = ("hotkey", "failsafe")

#: `SafetyGuard` 拦截里**算安全事件**的规则。`out_of_bounds`（坐标越界）与
#: `stub_action`（未实现的动作）是模型输出错误，计入会把安全事件数灌水。
GUARD_SAFETY_RULES = ("dangerous_text", "dangerous_keys")


def first_step_outcome(steps: list) -> tuple[bool | None, str]:
    """本轮第一步是否有效：动作执行成功，且执行后屏幕发生了变化。

    **为什么看屏幕变化，不只看执行状态。** `execution_status == "ok"` 只说明键鼠事件
    发出去了：2026-08-25 的实测里「单击桌面图标」连续六次 ok，屏幕纹丝不动。
    **为什么只看第一步。** 第一步从同一个干净起点出发，各轮之间可比；它失败时，
    后面的步骤都建立在错误的状态上。
    """
    if not steps:
        return False, "本轮没有产生任何步骤"
    first = steps[0]
    status = getattr(first, "execution_status", "") or ""
    if status == "no_action":
        return False, "第一步就报告完成，没有执行任何动作"
    if status != "ok":
        error_type = getattr(first, "error_type", "") or ""
        return False, f"第一步未执行成功：{status or '无状态'} {error_type}".strip()
    change = (getattr(first, "meta", None) or {}).get("change")
    if not change:
        return None, "第一步执行了，但没有帧差记录，无法判定"
    if change.get("changed"):
        return True, f"第一步执行成功且屏幕发生变化（ratio={change.get('ratio')}）"
    return False, f"第一步执行了但屏幕没有变化（ratio={change.get('ratio')}）"


def round_safety_events(results: list) -> list[dict]:
    """从本轮的动作执行结果里挑出安全事件：哨兵命中与危险输入拦截。"""
    events = []
    for result in results:
        verdict = getattr(result, "verdict", None)
        if verdict is None or verdict.allowed:
            continue
        if verdict.rule.startswith("sentinel:") or verdict.rule in GUARD_SAFETY_RULES:
            events.append(
                {
                    "rule": verdict.rule,
                    "reason": verdict.reason,
                    "evidence": verdict.evidence,
                    "source": "action",
                }
            )
    return events


def safety_metrics(records: list) -> dict:
    """首步成功率、人工干预率、安全事件数。进存档，也打印在汇总里。

    - **首步成功率** = 首步成功轮数 / 能判定首步的有效轮数（起点建立、未剔除）
    - **人工干预率** = （热键急停 + FAILSAFE + 事后标记有人碰过的轮数）/ Agent 实际开跑的轮数
    - **安全事件数** = 哨兵命中（含开跑前环境扫描）+ 危险输入拦截，按规则分类

    `records` 可以是 `RunRecord`，也可以是从存档读回的 dict——`scripts/exclude_round.py`
    事后剔除一轮时要据此刷新存档里的指标，否则人工干预率就停在剔除之前的值。
    """

    def get(r, key, default=None):
        return r.get(key, default) if isinstance(r, dict) else getattr(r, key, default)

    started = [r for r in records if get(r, "precondition_ok", True)]
    valid = [r for r in started if not get(r, "excluded", False)]
    known = [r for r in valid if get(r, "first_step_ok") is not None]
    first_ok = sum(1 for r in known if get(r, "first_step_ok"))

    by_type: dict[str, int] = {}
    intervened = 0
    for r in started:
        kinds = [
            k for k in (get(r, "intervention", ""), "excluded" if get(r, "excluded") else "") if k
        ]
        if kinds:
            intervened += 1
        for kind in kinds:
            by_type[kind] = by_type.get(kind, 0) + 1

    events = [e for r in records for e in (get(r, "safety_events") or [])]
    by_rule: dict[str, int] = {}
    for event in events:
        by_rule[event["rule"]] = by_rule.get(event["rule"], 0) + 1

    return {
        "first_step_success": {
            "ok": first_ok,
            "known": len(known),
            "rate": round(first_ok / len(known), 4) if known else None,
        },
        "human_intervention": {
            "rounds": intervened,
            "started": len(started),
            "rate": round(intervened / len(started), 4) if started else None,
            "by_type": by_type,
        },
        "safety_events": {
            "total": len(events),
            "rounds_with_event": sum(1 for r in records if get(r, "safety_events")),
            "by_rule": by_rule,
        },
    }


def _console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def run_reset(commands: list[str], dry_run: bool) -> None:
    """执行 reset 命令。失败不中断——`taskkill` 在进程本就不存在时会返回
    非零，那是正常情况，不是错误。"""
    for command in commands or []:
        if dry_run:
            print(f"      [演练] reset: {command}")
            continue
        subprocess.run(
            command,
            shell=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    if not dry_run and commands:
        time.sleep(1.5)  # 给进程退出与窗口关闭留时间


def _slug(text: str) -> str:
    """把 `--tag` 变成能安全进文件名的东西。

    只留字母数字、连字符、下划线。文件名里混进空格或冒号，后面用通配符
    找文件、或者把路径贴进命令行时都会出岔子。
    """
    cleaned = "".join(c if c.isalnum() or c in "-_" else "-" for c in text.strip())
    return cleaned.strip("-")[:40]


def latency_breakdown(records: list) -> dict:
    """把每步的四段延迟汇总。M2 验收标准 3 要求按 API / grounding /
    执行 / 截图 分解。"""
    total = {"api_ms": 0.0, "grounding_ms": 0.0, "execute_ms": 0.0, "screenshot_ms": 0.0}
    for record in records:
        for key in total:
            total[key] += float((record.latency or {}).get(key, 0.0))
    return {k: round(v, 1) for k, v in total.items()}


def build_parser() -> argparse.ArgumentParser:
    """抽成函数是为了让测试能检查参数表。

    与 `eval/action.py`、`serve_local_model.py` 同一条护栏：
    `main()` 读了一个从没注册过的参数,要到跑起来才炸。
    """
    # 引擎名单从 core.loop 读，不在这里抄一份——抄了就会在加引擎时漏改
    from core.loop import ENGINES

    parser = argparse.ArgumentParser(description="M2 基础任务批量执行与成功率统计")
    parser.add_argument("--execute", action="store_true", help="真正操作键鼠（默认演练）")
    parser.add_argument("--repeats", type=int, default=5, help="每个任务跑几次")
    parser.add_argument("--only", default="", help="只跑某个任务（按 name）")
    parser.add_argument("--tasks", default=str(TASK_FILE))
    parser.add_argument("--max-steps", type=int, default=0, help="覆盖清单里的 max_steps")
    parser.add_argument(
        "--tag",
        default="",
        help="给这次跑起个名字，进存档文件名与 JSON。对比实验必填，例如 base / lora",
    )
    parser.add_argument(
        "--verify-delay",
        type=float,
        default=0.0,
        help="程序化判定之前额外等待的秒数，叠加在固定的 1.5 秒之上。"
        "**判决性实验专用**：Reflector 打开后每个子任务多出的重复 done "
        "只贡献耗时不贡献动作，用这个参数在不开 Reflector 时补上等量延时，"
        "就能分辨那 20 个百分点是纠错还是计时假象。见 docs/m4-错误分类体系.md 7.5",
    )
    parser.add_argument(
        "--guest-snapshot",
        default="",
        help="本轮从哪个客机快照起跑，原样记进存档。"
        "**VMware 的快照名从客机内部读不到，只能由操作者传入。** "
        "不传会打一条醒目警告——本项目已经因为没记快照名，"
        "两次拿到无法验证环境一致性的对比数据（17%% 那轮、12%%→40%% 那对）",
    )
    parser.add_argument(
        "--reflector",
        action="store_true",
        help="模型报 done 时先做级联判定，上一步没让屏幕变就否决并要求重试。"
        "M4 任务 1 / 大纲 W6。**默认关闭**——打开会改变行为，"
        "而 M2/M3 的实测都是关着跑的。打开会自动启用帧差。"
        "实测依据见 docs/m4-错误分类体系.md：94.4%% 的子任务在 0~1 个动作后就报完成",
    )
    parser.add_argument(
        "--engine",
        choices=ENGINES,
        default="legacy",
        help="执行引擎。legacy 是自写循环 core/loop.py；langgraph 是同一套单步逻辑"
        "改由 LangGraph 状态图编排（core/graph_loop.py）。**默认 legacy**——"
        "M2~M4 的全部实测都跑在它上面。两者并存期间用同一快照、同一模型、同一模板"
        "各跑一轮做端到端对照，结果在噪声范围内一致后删掉 legacy",
    )
    parser.add_argument(
        "--escalate-on-no-change",
        action="store_true",
        help="动作执行后屏幕没变化时自动升级策略(单击→双击),并把「上一次点了没反应」"
        "回传给模型。大纲 W6 任务 2。**默认关闭**——打开会改变行为,"
        "而 M2/M3 的实测都是关着跑的",
    )
    parser.add_argument(
        "--planner-template",
        default="",
        help="规划器提示词模板，留空用 SessionConfig 的默认值。"
        "**与 --executor-template executor_v3 配套用 planner_v2**："
        "v1 的两份示例串成了一条固定回路(规划器产出「点击任务栏的开始按钮」→"
        "执行器吐出示例常数 (470,750) → 桌面壁纸)",
    )
    parser.add_argument(
        "--allowed-actions",
        default="",
        help="逗号分隔，限定执行器与规划器只用这些动作。"
        "**2026-08-26 起微调版 3B 不要再收窄**——训练集已含 8 种动作 + done："
        "它的训练集里 type / key 各 0 条,而提示词照样说「你可以 type」,"
        "结果五个任务里四个连掷骰子的机会都没有。留空表示不限制",
    )
    parser.add_argument(
        "--executor-template",
        default="",
        help="执行器提示词模板，留空用 SessionConfig 的默认值。**微调模型建议用 executor_v0（零样本）**："
        "2026-08-25 查出离线组 64%% 的动作是在逐字背诵 executor_v1 few-shot "
        "示例里的坐标 (470, 750)，而微调模型格式合规已达 97.9%%，不需要示例",
    )
    # 在线/离线由这组参数切换：
    #   在线      --provider dashscope
    #   离线-基座 --provider local
    #   离线-微调 --provider local --adapter finetune/outputs/<run>/adapter
    # **同一个脚本、同一套任务、同一套提示词**，否则跑出来的差值里
    # 混进了脚本差异，就不是模型的差值了。
    from llm.factory import add_backend_args

    add_backend_args(parser)
    return parser


def main() -> int:
    _console()

    from llm.factory import build_backend, describe, is_offline

    args = build_parser().parse_args()
    if args.execute and not args.guest_snapshot:
        # **本项目已经因为没记快照名吃过两次亏**（17% 那轮、12%→40% 那对），
        # 所以漏传要显眼，但不阻断 —— 数据本身仍然有价值。
        print(
            "\n  [!] 未传 --guest-snapshot。存档里不会有快照名，"
            "这批数据将无法与其他轮次验证环境一致性。"
            "\n      见 docs/m4-错误分类体系.md 7.5。\n"
        )

    import yaml

    from agent.session import Session, SessionConfig
    from control.executor import ActionExecutor
    from core.loop import LoopConfig
    from core.verify import SuccessCheck
    from grounding.native import NativeGrounding
    from perception.capture import ScreenCapturer
    from perception.coordinate import CoordinateScaler

    spec = yaml.safe_load(Path(args.tasks).read_text(encoding="utf-8"))
    tasks = spec["tasks"]
    if args.only:
        tasks = [t for t in tasks if t["name"] == args.only]
        if not tasks:
            raise SystemExit(f"任务清单里没有 {args.only!r}")

    # **后端跨轮复用，不是每轮新建。**
    #
    # 原来每轮 `OpenAICompatBackend(config)` 一个，从不 close()。每个后端
    # 持有一个 ChatOpenAI，它持有 openai SDK 的 httpx 连接池 —— 稳定性测试
    # 实测 22 分钟 16 轮之后句柄从 369 涨到 1521，单调不降。
    #
    # 成本要按轮统计，用 `reset_cost()` 而不是新建实例：前者是为这件事
    # 准备的，后者顺带泄漏了一个连接池。本地后端更不能每轮新建——
    # 每次都要重新把 2GB 权重搬进显存。
    backend = build_backend(args)
    offline = is_offline(backend)

    print("=" * 74)
    print("M2 基础任务批量执行")
    print("=" * 74)
    print(
        f"  任务数    {len(tasks)}    每个跑 {args.repeats} 次    共 {len(tasks) * args.repeats} 轮"
    )
    print(f"  后端      {describe(backend)}")
    print(f"  数据边界  {'截图不出本机' if offline else '截图上传到平台服务器'}")
    print(f"  模式      {'**实机执行**（会真的操作键鼠）' if args.execute else '演练（不碰键鼠）'}")
    if args.execute:
        print("\n  急停：Ctrl+Alt+Q，或把鼠标甩到屏幕角落触发 FAILSAFE")
        print("  执行期间请勿操作鼠标键盘。")
        if input("\n  确认开始？(yes/N) ").strip().lower() not in ("yes", "y"):
            print("  已取消")
            return 1
    print()

    capturer = ScreenCapturer()
    probe = capturer.capture(fresh=True)
    scaler = CoordinateScaler(probe.region)
    # **留空时解析成 SessionConfig 的默认值，并写回 args。**
    # 不写字面量默认值：那样 SessionConfig 改了默认、这里不跟着改，
    # 存档记下的就不是实际用的那份模板——而这正是 §10.8 那个混淆的形态。
    args.executor_template = args.executor_template or SessionConfig.executor_template
    args.planner_template = args.planner_template or SessionConfig.planner_template
    allowed = tuple(x.strip() for x in args.allowed_actions.split(",") if x.strip())

    space = SessionConfig().coordinate_space
    scaler.register("planner", *space)
    executor = ActionExecutor(scaler, space_name="planner", dry_run=not args.execute)
    # **必须 start()，急停热键是在这里挂上的。**
    #
    # 此前本脚本只构造、从不 start()：M2 到 M4 的全部批量实测里，
    # Ctrl+Alt+Q 按了也没有任何效果——三次安全事件恰好都发生在这些批量跑里。
    # FAILSAFE 没受影响，只因为 PyAutoGUI 自己的默认值就是开着的。
    # 2026-09-17 整理安全事件时实测确认（构造后 is_armed=False，start() 后为 True）。
    executor.start()
    if args.execute and not executor.emergency_stop.is_armed:
        print("\n  [!] 急停热键没有挂上（多半是缺 pynput）。此刻只剩 FAILSAFE 一道人工刹车，")
        print("      自动哨兵仍然有效。建议装好 pynput 再跑。\n")

    # **存档路径开跑前就定好，每轮写一次。**
    #
    # 原来只在最后 `render()` 里写一次，于是中断 = 全丢：2026-08-25 一次
    # 重启把跑了一半的离线-微调组整个抹掉了，前面两小时白费。而且它还
    # 逼着人「要么全跑完，要么什么都别看」——想先跑十分钟探探路都不行。
    #
    # 每轮一次 JSON 写盘，几百 KB，相对于一轮几十秒到十分钟的耗时可以忽略。
    RUNS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = f"-{_slug(args.tag)}" if args.tag else ""
    archive = RUNS / (
        f"{stamp}-{args.only or 'all'}-{'exec' if args.execute else 'dry'}"
        f"-{'offline' if offline else 'online'}{suffix}.json"
    )
    label = describe(backend)

    #: 急停之后整批停下的原因。非空时后面的轮次一律不跑。
    abort_reason = ""
    safety_setup = {
        "hotkey_armed": executor.emergency_stop.is_armed,
        "hotkey": executor.emergency_stop.hotkey,
        "sentinel": executor.sentinel is not None,
    }

    def save(partial: bool = True) -> None:
        archive.write_text(
            archive_payload(
                records,
                args,
                label,
                offline,
                partial=partial,
                aborted=abort_reason,
                safety_setup=safety_setup,
            ),
            encoding="utf-8",
        )

    print(f"  存档     {archive}（每轮实时写入，中断也不丢）")

    records: list[RunRecord] = []
    for task in tasks:
        if abort_reason:
            break
        check = SuccessCheck.from_spec(task.get("success_check"))
        pre = SuccessCheck.from_spec(task.get("precondition")) if task.get("precondition") else None
        max_steps = args.max_steps or task.get("max_steps", 12)
        print(f"── {task['title']}（{task['name']}）  每子任务上限 {max_steps} 步")

        for attempt in range(1, args.repeats + 1):
            print(f"   第 {attempt}/{args.repeats} 次")
            run_reset(task.get("reset"), dry_run=not args.execute)

            record = RunRecord(task=task["name"], title=task["title"], attempt=attempt)

            # **开跑前扫一遍环境。** 哨兵拦得住「对着登录框按回车」，拦不住「点岔的第一下」
            # ——那一下发出时前台还是 Edge。所以 reset 之后、Agent 动手之前，屏幕上若还留着
            # 登录框 / Copilot / .env，这一轮不跑、整批停下：环境没恢复干净，跑了也不可比，
            # 而且正是 M2 那三次事件的起点形态。
            if args.execute and executor.sentinel is not None:
                executor.sentinel.reset()
                hazards = executor.sentinel.scan_environment()
                if hazards:
                    record.precondition_ok = False
                    record.precondition_detail = "环境隐患：" + "；".join(
                        f"[{h.rule}] {h.evidence}" for h in hazards
                    )
                    record.safety_events = [h.as_dict() for h in hazards]
                    abort_reason = f"{task['name']} 第 {attempt} 次开跑前发现环境隐患，整批停止：{record.precondition_detail}"
                    print(f"      [停止] {abort_reason[:160]}")
                    records.append(record)
                    save()
                    break

            # **起点检查在 reset 之后、Agent 之前。**
            # reset 不一定成功，而「关闭应用」的判据是「进程应已退出」——
            # 计算器没启动时它会直接打勾，Agent 什么都没做也算成功。
            # 起点没建立的轮次记 invalid，不进成功率的分子也不进分母：
            # 那是环境的失败，混进 Agent 的成功率里两个数字都不可信。
            if pre is not None and args.execute:
                record.precondition_ok, record.precondition_detail = pre.run()
                if not record.precondition_ok:
                    print(f"      [无效轮] 起点未建立：{record.precondition_detail[:120]}")
                    print("        本轮不计入成功率。检查 reset 命令与测试环境。")
                    records.append(record)
                    save()
                    continue

            backend.reset_cost()
            session = Session(
                backend,
                NativeGrounding(*space),
                executor,
                capturer,
                config=SessionConfig(
                    loop=LoopConfig(
                        max_iterations=max_steps,
                        escalate_on_no_change=args.escalate_on_no_change,
                        reflector=args.reflector,
                        engine=args.engine,
                    ),
                    executor_template=args.executor_template,
                    planner_template=args.planner_template,
                    allowed_actions=allowed,
                ),
            )

            started = time.perf_counter()
            history_start = len(executor.history)
            all_steps: list = []
            try:
                result = session.run(task["instruction"])
                record.loop_status = result.status
                record.model_said_done = bool(result.succeeded)
                record.steps = result.total_steps
                record.trajectory_id = result.trajectory_id
                record.cost_cny = round(result.cost.cost_cny, 6)
                # 每个子任务的 LoopResult 里带着该子任务的全部 StepRecord
                all_steps = [
                    step
                    for outcome in result.outcomes
                    for step in getattr(outcome.result, "records", [])
                ]
                record.latency = latency_breakdown(all_steps)

                # **零动作报完成**：子任务以 done 收尾，但一个动作都没执行成功。
                # 实测里这是失败任务的主要形态——模型的 thinking 甚至写着
                # 「截图中未显示任何测试消息程序或发送按钮」，然后输出
                # {"done": true}。Loop 收到 done 就切下一个子任务，于是
                # 6 个子任务「全部成功」、整条轨迹 status=success，而任务
                # 实际什么都没做。
                #
                # M2 是基线，不在这里加闸拦它——那是 M3 该做的改进，
                # 提前塞进基线里 M3 就没有可比的对照。但**必须量出来**，
                # 否则 M4 的错误分类只能靠人翻轨迹。
                record.subtasks = len(result.outcomes)
                record.empty_done = sum(
                    1
                    for outcome in result.outcomes
                    if getattr(outcome.result, "status", "") == "done"
                    and not any(
                        getattr(step, "execution_status", "") == "ok"
                        for step in getattr(outcome.result, "records", [])
                    )
                )
            except Exception as exc:  # noqa: BLE001 —— 单轮崩溃不该中断整批
                record.error = f"{type(exc).__name__}: {exc}"[:200]
                print(f"      [异常] {record.error}")

            record.duration_s = round(time.perf_counter() - started, 1)

            # **判定必须在 reset 之前、执行之后。** 顺序错了就什么都查不到。
            if args.execute:
                # 等界面稳定，避免拿到中间态。
                #
                # **`--verify-delay` 是给判决性实验用的**，见
                # `docs/m4-错误分类体系.md` §7.5：Reflector 打开后成功率
                # 20% → 40%，但纠错率实测为 0%，多出来的 43 步全是重复的
                # `done`，只贡献了耗时（close_app +7.8s / send_message +15.1s）。
                # 用这个参数在**不开 Reflector** 的情况下补上等量延时，
                # 就能判断那 20 个百分点到底是纠错还是计时假象。
                time.sleep(1.5 + args.verify_delay)
                record.verified, record.verify_detail = check.run()
            else:
                # 演练也把判据跑一遍，但**不计入成功**。
                # 判据全是只读的（查进程、查窗口标题、读文件），跑一遍没有副作用；
                # 而 YAML 里判据写错（类型名拼错、参数对不上、路径写错）如果要等到
                # 实机 25 轮跑完才发现，那 25 轮就白跑了。这里让它当场暴露。
                _, detail = check.run()
                record.verified = False
                record.verify_detail = f"[演练不计分] 判据当前状态：{detail}"

            mark = "✓" if record.verified else "✗"
            print(
                f"      {mark} 判定{'通过' if record.verified else '未通过'}"
                f"  步数 {record.steps}  用时 {record.duration_s}s"
                f"  模型自报完成={record.model_said_done}"
            )
            if not record.verified:
                print(f"        {record.verify_detail[:150]}")

            # --- 安全与干预指标 ---
            if args.execute:
                record.first_step_ok, record.first_step_detail = first_step_outcome(all_steps)
            record.safety_events.extend(round_safety_events(executor.history[history_start:]))

            # **任何急停都停整批，不自动复位。**
            # 此前急停状态会一直保持置位，后面每一轮的第一个动作都被拒绝，
            # 却照常记成「失败轮」进了成功率的分母——人的刹车被算成了模型的失败。
            if executor.emergency_stop.is_triggered:
                reason = executor.emergency_stop.trigger_reason or "unknown"
                record.emergency_reason = reason
                if reason in HUMAN_STOP_REASONS:
                    record.intervention = reason
                abort_reason = (
                    f"{task['name']} 第 {attempt} 次触发急停（{reason}），整批停止，等人工确认"
                )
                print(f"      [急停] {abort_reason}")
                for event in record.safety_events:
                    print(f"        [{event['rule']}] {event['evidence']}")

            records.append(record)
            save()
            if abort_reason:
                break
        print()

    save(partial=bool(abort_reason))
    executor.stop()
    backend.close()
    render(
        records,
        args,
        backend_label=label,
        offline=offline,
        archive=archive,
        aborted=abort_reason,
        safety_setup=safety_setup,
    )
    return 3 if abort_reason else 0


def _screen_info() -> dict:
    """当前屏幕分辨率与 DPI 缩放。取不到就留空,不要猜。

    分辨率与缩放**必须一起记**:1920x1080 @100% 和 @150% 下,同一个按钮
    在截图里的像素尺寸差 1.5 倍。只记一个说不清模型看到的元素有多大。
    """
    try:
        from perception.capture import ScreenCapturer

        with ScreenCapturer() as cap:
            shot = cap.capture()
        info = {"resolution": f"{shot.width}x{shot.height}"}
    except Exception:  # noqa: BLE001 —— 取不到屏幕信息不该让整轮跑不起来
        info = {}
    try:
        from perception.dpi import describe as dpi_describe

        info["dpi"] = dpi_describe()
    except Exception:  # noqa: BLE001
        pass
    return info


def archive_payload(
    records: list[RunRecord],
    args,
    backend_label: str,
    offline: bool,
    partial: bool,
    aborted: str = "",
    safety_setup: dict | None = None,
) -> str:
    """存档的 JSON 文本。增量写与最终写共用同一份构造。

    ``partial=True`` 表示这是跑到一半的快照 —— **必须标出来**，否则一份
    因中断而只有 12 轮的存档，看起来和一次「只跑 12 轮」的正常实验一模一样。
    """
    return json.dumps(
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "executed": bool(args.execute),
            "repeats": args.repeats,
            "scope": args.only or "all",
            # **后端必须进存档。** 在线与离线两份存档除了数字之外长得一模一样，
            # 不记下来，隔几天就分不清哪份是哪份——而这两份正是对比报告的全部依据。
            "backend": backend_label,
            "offline": offline,
            "tag": args.tag,
            # **提示词模板必须进存档。** 2026-08-25 查出离线组 64% 的动作是在
            # 逐字背诵 `executor_v1` few-shot 示例里的坐标 (470, 750)。当时
            # 存档里没记用了哪份模板,这个混淆是靠翻客机轨迹才发现的。
            "executor_template": args.executor_template,
            "planner_template": args.planner_template,
            "escalate_on_no_change": args.escalate_on_no_change,
            "reflector": args.reflector,
            # **执行引擎必须进存档。** legacy 与 langgraph 并存期间要做端到端对照，
            # 两份存档除了这一项之外长得一模一样，不记下来就分不清哪份是哪个引擎跑的。
            "engine": args.engine,
            "verify_delay": args.verify_delay,
            # 动作集也进存档——同 executor_template,它能左右结论。
            "allowed_actions": [x.strip() for x in args.allowed_actions.split(",") if x.strip()],
            # **屏幕设置也必须进存档。** 同一天查出报告写错了客机分辨率
            # (把宿主机的 2560x1600 当成了客机的),而 §9 已证明分辨率值
            # 2 倍坐标误差——一个能左右结论的变量,存档里却查不到。
            "screen": _screen_info(),
            # **快照名决定这批数据能和谁比。** 读不到就只能靠人传，
            # 漏传时下面会打警告，但不阻断——数据本身仍然有价值。
            "guest_snapshot": args.guest_snapshot or None,
            # 跑完了没有。中断的存档不能当完整数据用。
            "partial": partial,
            # 急停后整批停下的原因。与 partial 分开：partial 可能只是跑到一半的快照，
            # aborted 非空才说明是**因为安全原因**停的。
            "aborted": aborted or None,
            # 这一批的刹车是否真的挂上了。**此前热键在批量跑里从未挂载，而存档里查不到这件事。**
            "safety_setup": safety_setup,
            "safety_metrics": safety_metrics(records),
            "records": [asdict(r) for r in records],
        },
        ensure_ascii=False,
        indent=2,
    )


def render_safety(
    records: list[RunRecord], aborted: str = "", safety_setup: dict | None = None
) -> None:
    """汇总里的安全一节。**排在成本之前**：先说这批跑得安不安全，再说花了多少。"""
    metrics = safety_metrics(records)
    first = metrics["first_step_success"]
    human = metrics["human_intervention"]
    events = metrics["safety_events"]

    print("")
    print("  **安全与干预**")
    if safety_setup is not None:
        print(
            f"    刹车   热键 {'已挂载' if safety_setup.get('hotkey_armed') else '未挂载 ← 需排查'}"
            f"   自动哨兵 {'开' if safety_setup.get('sentinel') else '关（演练模式）'}"
        )
    if first["known"]:
        print(f"    首步成功率   {first['ok']}/{first['known']} = {first['rate']:.0%}")
    else:
        print("    首步成功率   无法判定（演练模式，或第一步没有帧差记录）")
    if human["started"]:
        detail = "、".join(f"{k} {v}" for k, v in human["by_type"].items()) or "无"
        print(
            f"    人工干预率   {human['rounds']}/{human['started']} = {human['rate']:.0%}（{detail}）"
        )
    print(f"    安全事件数   {events['total']}", end="")
    if events["by_rule"]:
        print("（" + "、".join(f"{k} {v}" for k, v in events["by_rule"].items()) + "）")
    else:
        print("")
    if aborted:
        print(f"\n  **本批因急停提前结束：{aborted}**")
        print("  急停不会自动复位。确认桌面状态、处理完隐患后再重新开跑。")


def render(
    records: list[RunRecord],
    args,
    backend_label: str = "",
    offline: bool = False,
    archive: Path | None = None,
    aborted: str = "",
    safety_setup: dict | None = None,
) -> None:
    from collections import defaultdict

    by_task: dict[str, list[RunRecord]] = defaultdict(list)
    for r in records:
        by_task[r.task].append(r)

    print("=" * 74)
    print("结果汇总")
    print("=" * 74)
    print(f"{'任务':<14}{'成功率':>10}{'占比':>8}{'平均步数':>10}{'平均耗时':>10}{'模型自报':>10}")
    print("-" * 74)

    # **无效轮与剔除轮都不进分母。** 起点没建立、或人为干预过的轮次
    # 既不是成功也不是失败，把它当失败会低估能力，当成功会高估，
    # 只有排除掉这个数字才有意义。
    for group in by_task.values():
        valid = [r for r in group if r.precondition_ok and not r.excluded]
        if not valid:
            print(f"{group[0].title:<14}{'全部无效':>10}{'':>8}{'':>10}{'':>10}{'':>10}")
            continue
        ok = sum(r.verified for r in valid)
        said = sum(r.model_said_done for r in valid)
        steps = sum(r.steps for r in valid) / len(valid)
        secs = sum(r.duration_s for r in valid) / len(valid)
        print(
            f"{group[0].title:<14}"
            f"{f'{ok}/{len(valid)}':>10}"
            f"{ok / len(valid):>8.0%}"
            f"{steps:>10.1f}"
            f"{secs:>9.1f}s"
            f"{f'{said}/{len(valid)}':>10}"
        )

    valid_all = [r for r in records if r.precondition_ok and not r.excluded]
    invalid = [r for r in records if not r.precondition_ok]
    dropped = [r for r in records if r.precondition_ok and r.excluded]
    total_ok = sum(r.verified for r in valid_all)
    total_said = sum(r.model_said_done for r in valid_all)
    print("-" * 74)
    if valid_all:
        print(
            f"{'合计':<14}"
            f"{f'{total_ok}/{len(valid_all)}':>10}"
            f"{total_ok / len(valid_all):>8.0%}"
            f"{'':>10}{'':>10}"
            f"{f'{total_said}/{len(valid_all)}':>10}"
        )
    else:
        print("  没有一轮的起点建立成功，全部无效。")

    if invalid:
        print("")
        print(f"  **无效轮 {len(invalid)}/{len(records)}（起点未建立，已排除出成功率）**")
        for r in invalid[:5]:
            print(f"    {r.title} 第{r.attempt}次：{r.precondition_detail[:90]}")

    # 模型自报完成 vs 程序化判定 —— 两者的差就是「模型以为做完了但没做完」
    gap = [r for r in valid_all if r.model_said_done and not r.verified]
    if gap:
        print(f"\n  **模型自报完成但判定未通过：{len(gap)} 次**")
        print("  这个差值是 M4 错误分类的重点素材——模型的自我判断不可信到什么程度，")
        print("  只有程序化判定能量出来。")
        for r in gap[:5]:
            print(f"    {r.title} 第{r.attempt}次：{r.verify_detail[:90]}")

    subtasks = sum(r.subtasks for r in valid_all)
    empty = sum(r.empty_done for r in valid_all)
    if subtasks:
        print("")
        print(f"  **零动作报完成的子任务：{empty}/{subtasks}（{empty / subtasks:.0%}）**")
        print("  子任务以 done 收尾，却一个动作都没执行成功。这是失败任务的主要形态，")
        print("  也是 M3 要改的第一个东西——朴素 Loop 无条件相信模型的 done。")

    if dropped:
        print("")
        print(f"  **事后剔除 {len(dropped)} 轮（已排除出成功率）**")
        for r in dropped:
            print(f"    {r.title} 第{r.attempt}次：{r.exclusion_reason}")

    render_safety(records, aborted, safety_setup)

    cost = sum(r.cost_cny for r in records)
    if offline:
        # 离线版的 API 费用**真的是 0**，不是「未知」。但这不等于没有成本，
        # 成本转移到了硬件与耗时上——所以这里只说清是哪一种 0，不硬折算成钱。
        print(f"\n  API 费用 0 元（本地推理，{len(records)} 轮）")
        print("  离线版的开销体现在延迟与显存上，见上面的平均耗时。")
    else:
        print(f"\n  累计成本 {cost:.4f} 元（{len(records)} 轮）")
        if records:
            print(f"  单任务平均 {cost / len(records):.4f} 元")

    payload = archive_payload(
        records,
        args,
        backend_label,
        offline,
        partial=bool(aborted),
        aborted=aborted,
        safety_setup=safety_setup,
    )
    # **只有「在线 + 全量 + 实机」这一种跑法才配写 M2 的交付物。**
    #
    # `RAW` 是 M2 验收报告引用的那份数据，说的是在线版 19/24。任何别的跑法
    # 顺手把它写掉，报告里的数字和它引用的文件就对不上——**这不是假想**：
    #
    #   - M1 的 `verify_llm.py` 把一份两平台的报告覆盖成了 1/1
    #   - 本脚本自己也犯过：`--only send_message --repeats 5` 跑完之后，
    #     `RAW` 里只剩 4 条 send_message 记录，而报告仍写着 19/24。
    #     两个数字都对，只是不再来自同一个文件。
    #
    # 存档目录里每一次跑都有独立文件，那才是所有跑法的落点。
    skip_raw = offline or bool(args.only) or not args.execute
    if skip_raw:
        why = "离线运行" if offline else ("只跑了单个任务" if args.only else "演练模式")
        print(f"  （{why}，未覆盖 {RAW}——那是在线全量实机跑的 M2 交付数据）")
    else:
        RAW.parent.mkdir(parents=True, exist_ok=True)
        RAW.write_text(payload, encoding="utf-8")

    # 存档已由主循环每轮写过了，这里不重复写——路径也由主循环给，
    # 不能再算一次：两处各自 `datetime.now()` 会得到不同的时间戳，
    # 于是同一次运行留下两个文件。
    if not skip_raw:
        print(f"  原始数据 {RAW}")
    print(f"  存档     {archive}")
    if not args.execute:
        print("\n  [演练模式] 未真正执行，成功率无意义。加 --execute 实机跑。")


if __name__ == "__main__":
    raise SystemExit(main())
