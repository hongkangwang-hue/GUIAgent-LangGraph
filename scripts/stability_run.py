"""连续跑 N 分钟，看系统会不会崩、会不会泄漏 —— M2 验收标准 10。

## 它和 run_basic_tasks 的区别

`run_basic_tasks.py` 问的是「做得对不对」，这个脚本问的是
**「一直跑会不会坏」**。两者关心的东西不重叠：

- 成功率在这里**不是指标**。任务失败无所谓，进程崩了才是问题。
- 记的是**内存、句柄、线程、异常**，以及它们随时间的走向。

## 为什么要单独测

M2 的每一批评测都在 20 分钟以内，而且中间人一直看着。M5 的 60 轮正式
评测是无人值守的长跑，那时候才发现 dxcam 的帧缓冲没释放、UIA 的 COM
对象越积越多、或者某个异常会让整批停在第 17 轮 —— 代价是重跑几小时。

**长跑暴露的问题和短跑不是同一类。** 短跑测正确性，长跑测资源与容错。

## 泄漏怎么判

不看绝对值，看**趋势**：把整段时间的 RSS 采样做一次最小二乘拟合，
斜率（MB/分钟）显著为正才算可疑。绝对值没有意义 —— PaddleOCR 一加载
就是几百 MB，那不是泄漏。

## 单轮失败不中断

长跑脚本自己因为一个异常停下来，等于没测。所有异常记录下来继续跑，
最后按类型汇总。**「跑完 30 分钟且异常可归类」比「零异常」更重要** ——
零异常也可能只是没跑够久。

## 用法

    python scripts/stability_run.py                      # 演练，不碰键鼠
    python scripts/stability_run.py --execute            # 实机跑 30 分钟
    python scripts/stability_run.py --execute --minutes 45
    python scripts/stability_run.py --execute --only close_app   # 只用最短的任务

**实机跑期间手离开键鼠。** 急停 Ctrl+Alt+Q。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TASK_FILE = Path("tasks/basic_tasks.yaml")
REPORT = Path("docs/m2-stability-report.md")
RAW = Path("docs/m2-stability-raw.json")

#: 采样间隔。30 分钟约 60 个点，够拟合趋势又不至于把日志撑爆。
SAMPLE_EVERY_S = 30.0


@dataclass
class Sample:
    """一次资源采样。"""

    elapsed_s: float
    rss_mb: float
    threads: int
    handles: int
    rounds_done: int


@dataclass
class RoundRecord:
    index: int
    task: str
    elapsed_s: float
    duration_s: float
    steps: int
    status: str = ""
    error: str = ""


@dataclass
class Report:
    started_at: str = ""
    minutes: float = 0.0
    executed: bool = False
    model: str = ""
    engine: str = ""
    planner_template: str = ""
    executor_template: str = ""
    aborted_reason: str = ""
    rounds: list = field(default_factory=list)
    samples: list = field(default_factory=list)
    errors: dict = field(default_factory=dict)
    rss_slope_mb_per_min: float = 0.0
    handle_slope_per_min: float = 0.0
    thread_slope_per_min: float = 0.0
    #: 后半段（稳态）斜率。**判泄漏只看这一组。**
    rss_slope_steady: float = 0.0
    handle_slope_steady: float = 0.0
    crashed: bool = False


def _console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _sample(process, started: float, rounds: int) -> Sample:
    """采一次资源。取不到的字段填 0，不让采样本身成为崩溃源。"""
    rss = threads = handles = 0
    try:
        rss = process.memory_info().rss / (1024 * 1024)
        threads = process.num_threads()
        handles = getattr(process, "num_handles", lambda: 0)()
    except Exception:  # noqa: BLE001
        pass
    return Sample(
        elapsed_s=round(time.perf_counter() - started, 1),
        rss_mb=round(rss, 1),
        threads=threads,
        handles=handles,
        rounds_done=rounds,
    )


def _slope(samples: list[Sample], field_name: str = "rss_mb") -> float:
    """某个资源指标对时间的最小二乘斜率，单位 <该指标单位>/分钟。

    **看趋势不看绝对值。** PaddleOCR 一加载就是几百 MB，那不是泄漏；
    每分钟稳定涨才是。

    句柄同样要算。第一次实测里 RSS 斜率 +1.69MB/分钟已经报了可疑，
    而句柄从 369 涨到 1521（+52/分钟、4.1 倍）**更刺眼却没被评估** ——
    脚本只盯着内存，漏掉了更强的那个信号。
    """
    points = [
        (s.elapsed_s / 60.0, float(getattr(s, field_name)))
        for s in samples
        if getattr(s, field_name) > 0
    ]
    n = len(points)
    if n < 3:
        return 0.0
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    if denominator == 0:
        return 0.0
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in points)
    return round(numerator / denominator, 3)


def steady_slope(samples: list[Sample], field_name: str) -> tuple[float, float]:
    """返回 (整段斜率, 后半段斜率)。**判泄漏要看后半段。**

    第一版只拟合整段，在这份实测数据上直接给出了错误结论：

        22.2 分钟：RSS 72→189MB，句柄 369→1521
        30.6 分钟：RSS 72→189MB，句柄 371→1520

    跑久了 8 分钟，两个终值几乎一模一样 —— 真泄漏的话跑更久必然更高。
    实际是启动期爬升后收敛到稳态（加载 OCR/UIA/dxcam、线程池与连接池
    预热），后半段已经平了。

    而整段拟合把启动期的陡坡摊到全程，报出「持续上涨」。更糟的是句柄
    那一项算出 +9.9/分钟，阈值 10.0，**差 0.1 就翻转结论** ——
    一个挪一点阈值就变号的指标不能用来下判断。

    所以分开看：整段斜率描述「这一跑总共涨了多少」，后半段斜率回答
    「稳态之后还在不在涨」。**只有后者能判泄漏。**
    """
    overall = _slope(samples, field_name)
    half = samples[len(samples) // 2 :]
    return overall, _slope(half, field_name)


def reanalyze(path: Path) -> int:
    """用已有的采样数据重算结论。

    **判定口径改了不该重跑 30 分钟。** 采样是昂贵的（真实 API 费用 +
    半小时机器时间），而从采样到结论那一段是纯计算 —— 两者分开，
    改口径就只是重算。

    第一版没有这个入口，于是「整段斜率报错了结论」这件事一被发现，
    唯一的选择就是再跑一次。
    """
    if not path.exists():
        print(f"找不到 {path}")
        return 1
    payload = json.loads(path.read_text(encoding="utf-8"))
    report = Report(**{k: v for k, v in payload.items() if k in Report.__dataclass_fields__})
    samples = [Sample(**s) for s in report.samples]
    if len(samples) < 3:
        print("采样点太少，算不出趋势。")
        return 1
    report.rss_slope_mb_per_min, report.rss_slope_steady = steady_slope(samples, "rss_mb")
    report.handle_slope_per_min, report.handle_slope_steady = steady_slope(samples, "handles")
    report.thread_slope_per_min = _slope(samples, "threads")
    print(f"（重算 {path}，未跑新测试）")
    render(report, samples, output=path)
    return 0


def main() -> int:  # noqa: PLR0915 —— 长跑脚本，线性叙事比拆函数好读
    _console()
    parser = argparse.ArgumentParser(description="连续跑 N 分钟的稳定性测试（M2 验收 10）")
    parser.add_argument("--execute", action="store_true", help="真正操作键鼠（默认演练）")
    parser.add_argument("--minutes", type=float, default=30.0, help="跑多少分钟")
    parser.add_argument("--only", default="", help="只用某个任务，可减少环境扰动")
    parser.add_argument("--tasks", default=str(TASK_FILE))
    parser.add_argument("--provider", default="dashscope")
    parser.add_argument("--model", default="", help="模型 ID；留空则从客机配置读取")
    parser.add_argument("--engine", choices=("legacy", "langgraph"), default="legacy")
    parser.add_argument("--planner-template", default="")
    parser.add_argument("--executor-template", default="")
    parser.add_argument("--output", default="", help="本次原始记录的独立 JSON 路径")
    parser.add_argument("--max-steps", type=int, default=6, help="每子任务步数上限，长跑取小值")
    parser.add_argument(
        "--analyze",
        default="",
        help="不跑测试，只用已有的 raw json 重算结论。改了判定口径时用，省一次 30 分钟。",
    )
    args = parser.parse_args()

    if args.analyze:
        return reanalyze(Path(args.analyze))

    import psutil
    import yaml

    from agent.session import Session, SessionConfig
    from control.executor import ActionExecutor
    from core.loop import LoopConfig
    from core.verify import SuccessCheck
    from grounding.native import NativeGrounding
    from llm.openai_compat import OpenAICompatBackend
    from llm.providers import load_dotenv_if_present, resolve
    from perception.capture import ScreenCapturer
    from perception.coordinate import CoordinateScaler
    from scripts.run_basic_tasks import run_reset

    spec = yaml.safe_load(Path(args.tasks).read_text(encoding="utf-8"))
    tasks = spec["tasks"]
    if args.only:
        tasks = [t for t in tasks if t["name"] == args.only]
        if not tasks:
            raise SystemExit(f"任务清单里没有 {args.only!r}")

    load_dotenv_if_present()
    config = resolve(args.provider, model=args.model or None)

    print("=" * 74)
    print("稳定性测试")
    print("=" * 74)
    print(f"  时长      {args.minutes} 分钟")
    print(f"  任务      {'、'.join(t['name'] for t in tasks)}（轮流跑）")
    print(f"  模型      {config.model}")
    print(f"  引擎      {args.engine}")
    print(
        f"  提示词    {args.planner_template or SessionConfig.planner_template}"
        f" + {args.executor_template or SessionConfig.executor_template}"
    )
    print(f"  模式      {'**实机执行**' if args.execute else '演练（不碰键鼠）'}")
    print("\n  **成功率在这里不是指标。** 关心的是崩没崩、有没有泄漏。")
    if args.execute:
        print("  执行期间请勿操作鼠标键盘。急停 Ctrl+Alt+Q。")
        if input("\n  确认开始？(yes/N) ").strip().lower() not in ("yes", "y"):
            print("  已取消")
            return 1
    print()

    report = Report(
        started_at=datetime.now().isoformat(timespec="seconds"),
        minutes=args.minutes,
        executed=bool(args.execute),
        model=config.model,
        engine=args.engine,
        planner_template=args.planner_template or SessionConfig.planner_template,
        executor_template=args.executor_template or SessionConfig.executor_template,
    )

    process = psutil.Process()
    capturer = ScreenCapturer()
    probe = capturer.capture(fresh=True)
    scaler = CoordinateScaler(probe.region)
    space = SessionConfig().coordinate_space
    scaler.register("planner", *space)
    executor = ActionExecutor(scaler, space_name="planner", dry_run=not args.execute)
    executor.start()
    if args.execute and not executor.emergency_stop.is_armed:
        print("急停热键未挂载，停止一小时实机测试。")
        executor.stop()
        capturer.close()
        return 2

    # 后端跨轮复用。每轮新建会泄漏 httpx 连接池 —— 这个泄漏正是本脚本
    # 第一次跑就测出来的（22 分钟句柄 369 → 1521）。
    backend = OpenAICompatBackend(config)

    started = time.perf_counter()
    deadline = started + args.minutes * 60
    next_sample = started
    errors: Counter = Counter()
    index = 0

    report.samples.append(asdict(_sample(process, started, 0)))

    try:
        while time.perf_counter() < deadline:
            task = tasks[index % len(tasks)]
            index += 1
            elapsed = time.perf_counter() - started
            record = RoundRecord(
                index=index,
                task=task["name"],
                elapsed_s=round(elapsed, 1),
                duration_s=0.0,
                steps=0,
            )

            if executor.emergency_stop.is_triggered:
                report.aborted_reason = executor.emergency_stop.trigger_reason or "emergency_stop"
                break
            reset_errors = run_reset(task.get("reset"), dry_run=not args.execute)
            if reset_errors:
                report.aborted_reason = "reset: " + "；".join(reset_errors)
                break
            if executor.emergency_stop.is_triggered:
                report.aborted_reason = executor.emergency_stop.trigger_reason or "emergency_stop"
                break
            if args.execute:
                if executor.sentinel is not None:
                    executor.sentinel.reset()
                    hazards = executor.sentinel.scan_environment()
                    if hazards:
                        report.aborted_reason = "environment: " + "；".join(
                            f"[{h.rule}] {h.evidence}" for h in hazards
                        )
                        break
                if task.get("precondition"):
                    ready, detail = SuccessCheck.from_spec(task["precondition"]).run()
                    if not ready:
                        report.aborted_reason = "precondition: " + detail
                        break

            round_started = time.perf_counter()
            try:
                backend.reset_cost()
                session = Session(
                    backend,
                    NativeGrounding(*space),
                    executor,
                    capturer,
                    config=SessionConfig(
                        loop=LoopConfig(
                            max_iterations=args.max_steps,
                            engine=args.engine,
                            max_unchanged_click_repeats=4,
                        ),
                        planner_template=args.planner_template or SessionConfig.planner_template,
                        executor_template=args.executor_template or SessionConfig.executor_template,
                    ),
                )
                result = session.run(task["instruction"])
                record.status = result.status
                record.steps = result.total_steps
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 —— 长跑不能被单轮异常终止
                # 记录类型与首行即可。完整栈进 stderr，不塞进报告
                record.error = f"{type(exc).__name__}: {exc}"[:200]
                record.status = "exception"
                errors[type(exc).__name__] += 1
                traceback.print_exc(file=sys.stderr)

            record.duration_s = round(time.perf_counter() - round_started, 1)
            report.rounds.append(asdict(record))
            if executor.emergency_stop.is_triggered:
                report.aborted_reason = executor.emergency_stop.trigger_reason or "emergency_stop"
                print(f"  急停触发：{report.aborted_reason}；长跑停止")
                break

            mark = "!" if record.error else "."
            print(
                f"  [{record.elapsed_s / 60:5.1f}min] #{index:<3} {task['name']:<16}"
                f" {record.status:<18} {record.steps:>2}步 {record.duration_s:>6.1f}s {mark}"
            )

            now = time.perf_counter()
            if now >= next_sample:
                sample = _sample(process, started, index)
                report.samples.append(asdict(sample))
                next_sample = now + SAMPLE_EVERY_S
                print(
                    f"           采样 RSS {sample.rss_mb:.0f}MB"
                    f"  线程 {sample.threads}  句柄 {sample.handles}"
                )
    except KeyboardInterrupt:
        print("\n  被中断（Ctrl+C）。已跑的数据仍会写出。")
    except BaseException:  # noqa: BLE001 —— 连这里都挂了就是真的崩了
        report.crashed = True
        traceback.print_exc(file=sys.stderr)

    executor.stop()
    backend.close()
    capturer.close()
    report.samples.append(asdict(_sample(process, started, index)))
    report.errors = dict(errors)
    samples = [Sample(**s) for s in report.samples]
    report.rss_slope_mb_per_min, report.rss_slope_steady = steady_slope(samples, "rss_mb")
    report.handle_slope_per_min, report.handle_slope_steady = steady_slope(samples, "handles")
    report.thread_slope_per_min = _slope(samples, "threads")

    render(report, samples, output=Path(args.output) if args.output else RAW)
    return 3 if report.aborted_reason else (1 if report.crashed else 0)


def _verdict(steady: float, threshold: float, what: str) -> str:
    """把稳态斜率翻译成一句话。

    **阈值只在稳态斜率上用。** 第一版把阈值用在整段斜率上，句柄那项
    算出 +9.9 而阈值是 10.0 —— 差 0.1 结论就翻转。稳态斜率在没有泄漏时
    应当贴近 0，所以阈值离它很远，不会出现这种擦边情况。
    """
    if steady > threshold:
        return f"**可疑：稳态之后{what}仍在上涨，长跑会出问题。**"
    if steady < -threshold:
        return f"{what}在回落，可能是缓存释放，不必处理。"
    return f"{what}已进入稳态，未见泄漏。"


def render(report: Report, samples: list[Sample], output: Path = RAW) -> None:
    total_min = samples[-1].elapsed_s / 60 if samples else 0.0
    rounds = report.rounds
    failed = [r for r in rounds if r["error"]]
    rss = [s.rss_mb for s in samples if s.rss_mb > 0]

    print()
    print("=" * 74)
    print("稳定性结果")
    print("=" * 74)
    print(f"  实际跑了   {total_min:.1f} 分钟 / {len(rounds)} 轮")
    print(f"  进程崩溃   {'**是**' if report.crashed else '否'}")
    if report.aborted_reason:
        print(f"  提前停止   {report.aborted_reason[:160]}")
    print(f"  异常轮次   {len(failed)}/{len(rounds)}")
    if report.errors:
        for name, count in report.errors.items():
            print(f"     {name}: {count} 次")
    if rss:
        print(f"  RSS        起 {rss[0]:.0f}MB → 止 {rss[-1]:.0f}MB，峰值 {max(rss):.0f}MB")
        print(
            f"  RSS 斜率   整段 {report.rss_slope_mb_per_min:+.2f}"
            f"    **稳态 {report.rss_slope_steady:+.2f} MB/分钟**"
        )
        # **判泄漏只看稳态。** 整段拟合会把启动期的爬升摊到全程，
        # 在一条先涨后平的曲线上报出「持续上涨」
        print("     " + _verdict(report.rss_slope_steady, 0.5, "内存"))
    if samples:
        print(
            f"  线程       {samples[0].threads} → {samples[-1].threads}"
            f"（{report.thread_slope_per_min:+.2f}/分钟）"
        )
        if samples[-1].handles:
            print(
                f"  句柄       {samples[0].handles} → {samples[-1].handles}"
                f"（整段 {report.handle_slope_per_min:+.1f}"
                f"    **稳态 {report.handle_slope_steady:+.1f}/分钟**）"
            )
            print("     " + _verdict(report.handle_slope_steady, 5.0, "句柄"))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = REPORT if output == RAW else output.with_suffix(".md")

    lines = [
        "# M2 稳定性测试报告",
        "",
        f"> 由 `python scripts/stability_run.py` 生成，{report.started_at}。",
        "> 对应 M2 验收标准 10。**本文件是生成物，改它没有意义，改脚本。**",
        "",
        "## 结论",
        "",
        f"- 连续运行 **{total_min:.1f} 分钟**，完成 **{len(rounds)}** 轮",
        f"- 进程崩溃：**{'是' if report.crashed else '否'}**",
        f"- 异常轮次：**{len(failed)}/{len(rounds)}**",
    ]
    if report.aborted_reason:
        lines.append(f"- 提前停止：**{report.aborted_reason[:160]}**")
    if rss:
        lines += [
            f"- RSS：起 {rss[0]:.0f}MB → 止 {rss[-1]:.0f}MB，峰值 {max(rss):.0f}MB",
            f"- RSS 斜率：整段 {report.rss_slope_mb_per_min:+.2f}，"
            f"**稳态 {report.rss_slope_steady:+.2f} MB/分钟**"
            + ("（可疑）" if report.rss_slope_steady > 0.5 else "（已进入稳态）"),
        ]
    if samples and samples[-1].handles:
        lines += [
            f"- 句柄：{samples[0].handles} → {samples[-1].handles}，"
            f"整段 {report.handle_slope_per_min:+.1f}，"
            f"**稳态 {report.handle_slope_steady:+.1f}/分钟**"
            + ("（可疑）" if report.handle_slope_steady > 5.0 else "（已进入稳态）"),
        ]
    lines += [
        "",
        "**成功率在本测试中不是指标。** 任务失败不影响结论，进程崩溃、",
        "资源持续上涨、异常无法归类才是。",
        "",
    ]
    if report.errors:
        lines += ["## 异常分类", "", "| 类型 | 次数 |", "|---|---:|"]
        lines += [f"| `{k}` | {v} |" for k, v in report.errors.items()]
        lines += [""]
    lines += [
        "## 资源采样",
        "",
        "| 时间(min) | RSS(MB) | 线程 | 句柄 | 已完成轮次 |",
        "|---:|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {s.elapsed_s / 60:.1f} | {s.rss_mb:.0f} | {s.threads} | {s.handles} | {s.rounds_done} |"
        for s in samples
    ]
    lines += ["", f"原始数据：`{output}`", ""]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  报告 {report_path}")
    print(f"  原始 {output}")


if __name__ == "__main__":
    raise SystemExit(main())
