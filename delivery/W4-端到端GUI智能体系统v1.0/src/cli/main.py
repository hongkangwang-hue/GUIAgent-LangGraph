"""命令行入口。

M2 任务 7 要求的五个命令：``run`` / ``run --task-file`` / ``replay`` /
``label`` / ``config --show``。

## 默认演练，实机执行要显式开

``run`` 默认 ``--dry-run``：走完整条链路——真截图、真调模型、真做坐标
转换——但**不发键鼠事件**。要真正操作桌面必须显式加 ``--execute``。

这不是保守，是有依据的：本项目的执行隔离是硬性前置条件——Agent 的键鼠
动作只应发生在隔离虚拟机内（M0 全局约束 1）。把实机执行放在默认值里，
意味着一次手滑就会让模型在宿主机的真实桌面上乱点。

急停已实测有效（热键触发到动作被拒绝 247ms，见
`docs/m1-control-verification.md`），但那是刹车，不是许可。

演练模式本身也有用：单步延迟四段分解、成本实测、提示词效果，这些都能在
不碰桌面的前提下测出来。
"""

from __future__ import annotations

import contextlib
import json
import logging
import sys
from pathlib import Path

try:
    import typer
except ImportError:  # pragma: no cover - 依赖缺失
    print("缺少 typer，请先 pip install typer", file=sys.stderr)
    raise

from agent.session import Session, SessionConfig
from cli.panel import LivePanel, PanelState
from core.loop import LoopConfig
from core.trajectory import (
    DEFAULT_ROOT,
    ERROR_LABELS,
    TrajectoryReader,
    describe_labels,
    list_trajectories,
    validate_labels,
)
from llm.providers import (
    PROVIDERS,
    ProviderNotConfigured,
    available_providers,
    load_dotenv_if_present,
    resolve,
)

app = typer.Typer(
    add_completion=False,
    help="基于多模态大模型的桌面 GUI 智能体",
    no_args_is_help=True,
)


def _setup_console() -> None:
    """Windows 控制台默认 cp936（GBK），而模型输出里带 emoji 是常态。

    实测百炼回过 "OK! 😊"，U+1F60A 不在 GBK 码表里，print 直接抛
    UnicodeEncodeError。CLI 的全部价值就是把过程显示给人看，不能因为
    模型多打了个表情就崩。
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            # 重定向到文件、管道等情况下可能不支持，忽略即可
            with contextlib.suppress(Exception):
                stream.reconfigure(encoding="utf-8", errors="replace")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


# ---------------------------------------------------------------------- #
# run
# ---------------------------------------------------------------------- #


@app.command()
def run(
    instruction: str = typer.Argument("", help="要执行的任务，中文自然语言"),
    task_file: Path = typer.Option(None, "--task-file", help="批量执行：YAML/JSON 任务清单"),
    execute: bool = typer.Option(
        False,
        "--execute",
        help="真正操作桌面。不加此项则只演练（走完整链路但不发键鼠事件）",
    ),
    provider: str = typer.Option(None, "--provider", help=f"平台：{'/'.join(sorted(PROVIDERS))}"),
    model: str = typer.Option(None, "--model", help="覆盖平台默认模型"),
    max_iterations: int = typer.Option(
        8, "--max-iterations", help="单个子任务的步数上限。调试期建议 8"
    ),
    cost_limit: float = typer.Option(1.0, "--cost-limit", help="单任务成本上限（元）"),
    executor_prompt: str = typer.Option("executor_v1", "--prompt", help="执行提示词模板版本"),
    planner_prompt: str = typer.Option("planner_v1", "--planner-prompt", help="拆解模板版本"),
    history_k: int = typer.Option(3, "--history-k", help="回传给模型的历史步数"),
    no_panel: bool = typer.Option(False, "--no-panel", help="关掉实时面板，只打日志"),
    output: Path = typer.Option(None, "--output", help="把结果摘要写成 JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """执行一个任务（默认只演练，不碰真实键鼠）。"""
    _setup_console()
    _setup_logging(verbose)
    load_dotenv_if_present()

    tasks = _load_tasks(task_file) if task_file else [{"name": "", "instruction": instruction}]
    if not tasks or not any(t["instruction"].strip() for t in tasks):
        typer.secho("没有任务可执行。给一句指令，或用 --task-file 指定清单。", fg="red")
        raise typer.Exit(1)

    if execute:
        _warn_before_execute()

    try:
        config = resolve(provider, model=model)
    except ProviderNotConfigured as exc:
        typer.secho(str(exc), fg="red")
        raise typer.Exit(1) from exc

    session_config = SessionConfig(
        planner_template=planner_prompt,
        executor_template=executor_prompt,
        loop=LoopConfig(
            max_iterations=max_iterations,
            cost_limit_cny=cost_limit,
            history_k=history_k,
        ),
    )
    session_config.context = type(session_config.context)(k=history_k)

    summaries = []
    for index, task in enumerate(tasks, start=1):
        if len(tasks) > 1:
            typer.secho(
                f"\n[{index}/{len(tasks)}] {task['name'] or task['instruction']}",
                fg="cyan",
                bold=True,
            )
        summaries.append(
            _run_one(task, config, session_config, execute=execute, show_panel=not no_panel)
        )

    _print_summary(summaries)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
        typer.echo(f"结果已写入 {output}")

    if not all(s["status"] == "completed" for s in summaries):
        raise typer.Exit(1)


def _warn_before_execute() -> None:
    """实机执行前的确认。

    这个确认不是形式：真正跑起来之后鼠标会被程序抢走，那时再想起来"急停
    热键是什么"已经晚了。
    """
    typer.secho("\n⚠ 即将真正操作桌面", fg="red", bold=True)
    typer.echo("  · 急停热键：Ctrl+Alt+Q（全局，不依赖鼠标）")
    typer.echo("  · 备用刹车：把鼠标甩到屏幕左上角触发 PyAutoGUI FAILSAFE")
    typer.echo("  · 建议在虚拟机内执行，且客机中不要有真实账号与个人数据")
    if not typer.confirm("确认继续？", default=False):
        raise typer.Abort()


def _run_one(task: dict, provider_config, session_config, execute: bool, show_panel: bool) -> dict:
    from control.executor import ActionExecutor
    from grounding.native import NativeGrounding
    from llm.openai_compat import OpenAICompatBackend
    from perception.capture import ScreenCapturer
    from perception.coordinate import CoordinateScaler

    # 注意这里用的是**坐标系**尺寸而不是图片尺寸：模型在归一化空间里
    # 作答，scaler 与 grounding 都得按那把尺子来。见 SessionConfig 里的实测记录
    width, height = session_config.coordinate_space
    backend = OpenAICompatBackend(config=provider_config, image_size=session_config.image_size)

    with ScreenCapturer() as capturer:
        probe = capturer.capture(fresh=True)
        scaler = CoordinateScaler(probe.region)
        scaler.register(session_config.space_name, width, height)

        executor = ActionExecutor(scaler, space_name=session_config.space_name, dry_run=not execute)
        state = PanelState(
            instruction=task["instruction"],
            dry_run=not execute,
            backend=backend.name,
            model=backend.model,
            priced=provider_config.price is not None,
        )

        with LivePanel(state, enabled=show_panel) as panel:

            def on_step(record) -> None:
                state.record_step(record)
                panel.refresh()

            session = Session(
                backend,
                NativeGrounding(width, height),
                executor,
                capturer,
                config=session_config,
                on_step=on_step,
            )

            executor.start()
            try:
                result = session.run(task["instruction"])
            finally:
                executor.stop()

            state.trajectory_id = result.trajectory_id
            if result.plan:
                state.subtasks = result.plan.goals()
            state.status = result.status
            panel.refresh()

    return {"name": task.get("name", ""), **result.as_dict()}


def _load_tasks(path: Path) -> list[dict]:
    """读任务清单。支持 YAML 与 JSON，形如::

    tasks:
      - name: open_notepad
        instruction: 打开记事本
    """
    if not path.exists():
        typer.secho(f"任务文件不存在：{path}", fg="red")
        raise typer.Exit(1)

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)

    items = data.get("tasks") if isinstance(data, dict) else data
    if not isinstance(items, list):
        typer.secho(f"{path} 里没有任务列表", fg="red")
        raise typer.Exit(1)

    tasks = []
    for item in items:
        if isinstance(item, str):
            tasks.append({"name": "", "instruction": item})
        elif isinstance(item, dict):
            tasks.append(
                {
                    "name": str(item.get("name", "")),
                    "instruction": str(item.get("instruction") or item.get("task") or ""),
                }
            )
    return tasks


def _print_summary(summaries: list[dict]) -> None:
    typer.echo("")
    for item in summaries:
        mark = "完成" if item["status"] == "completed" else "未完成"
        name = item.get("name") or item["instruction"]
        typer.echo(
            f"  [{mark}] {name}  —  {item['succeeded_subtasks']}/{item['subtasks']} 子任务，"
            f"{item['total_steps']} 步，{item['duration_s']:.1f}s"
        )
        if item.get("reason"):
            typer.echo(f"          {item['reason']}")
        typer.echo(f"          轨迹 {item['trajectory_id']}")

    cost = sum((item.get("cost") or {}).get("cost_cny", 0.0) for item in summaries)
    tokens = sum((item.get("cost") or {}).get("total_tokens", 0) for item in summaries)
    priced = all((item.get("cost") or {}).get("priced", True) for item in summaries)
    typer.echo(
        f"\n合计 {tokens} tokens，" + (f"{cost:.4f} 元" if priced else "成本未知（未配单价）")
    )


# ---------------------------------------------------------------------- #
# replay
# ---------------------------------------------------------------------- #


@app.command()
def replay(
    trajectory: str = typer.Argument(None, help="轨迹 ID 或目录。省略则用最近一条"),
    root: Path = typer.Option(DEFAULT_ROOT, "--root", help="轨迹根目录"),
    step: int = typer.Option(None, "--step", help="只看某一步"),
    show_thinking: bool = typer.Option(True, "--thinking/--no-thinking"),
    show_raw: bool = typer.Option(False, "--raw", help="显示模型原始输出"),
) -> None:
    """逐帧回放一条历史轨迹。"""
    _setup_console()
    reader = _open_trajectory(trajectory, root)
    meta = reader.meta

    typer.secho(f"\n轨迹 {meta.trajectory_id}", bold=True)
    typer.echo(f"  指令   {meta.instruction}")
    typer.echo(f"  后端   {meta.backend} / {meta.model}   模式 {meta.mode}")
    typer.echo(f"  结果   {meta.status}   {meta.total_steps} 步   {meta.duration_s:.1f}s")
    typer.echo(f"  成本   {meta.total_cost_cny:.4f} 元 / {meta.total_tokens} tokens")
    if meta.subtasks:
        typer.echo("  拆解：")
        for index, goal in enumerate(meta.subtasks, start=1):
            typer.echo(f"    {index}. {goal}")
    if meta.error:
        typer.secho(f"  终止原因：{meta.error}", fg="yellow")

    typer.echo("")
    for record in reader.iter_steps():
        if step is not None and record.step != step:
            continue
        color = "green" if record.succeeded else ("red" if record.failed else "cyan")
        typer.secho(f"  {record.summary()}", fg=color)

        if show_thinking and record.model_thinking:
            typer.echo(f"      思考：{record.model_thinking}")
        if record.action_real_coords.get("x") is not None:
            typer.echo(
                f"      坐标：模型 {record.action_model_coords.get('x')},"
                f"{record.action_model_coords.get('y')}"
                f"  →  屏幕 {record.action_real_coords.get('x')},"
                f"{record.action_real_coords.get('y')}"
            )
        if record.grounding:
            typer.echo(
                f"      定位：{record.grounding.get('source')}"
                + (f"  {record.grounding.get('error')}" if record.grounding.get("error") else "")
            )
        latency = record.latency or {}
        if latency:
            typer.echo(
                f"      延迟：api {latency.get('api_ms', 0):.0f}ms / "
                f"grounding {latency.get('grounding_ms', 0):.1f}ms / "
                f"执行 {latency.get('execute_ms', 0):.0f}ms / "
                f"截图 {latency.get('screenshot_ms', 0):.0f}ms"
            )
        if record.screenshot_before:
            typer.echo(f"      截图：{reader.frame(record.screenshot_before)}")
        if show_raw and record.raw_output:
            typer.echo(f"      原始：{record.raw_output[:300]}")
        if record.labels:
            typer.secho(
                f"      标注：{', '.join(record.labels)}  {record.label_note}", fg="magenta"
            )
        typer.echo("")


# ---------------------------------------------------------------------- #
# label
# ---------------------------------------------------------------------- #


@app.command()
def label(
    trajectory: str = typer.Argument(None, help="轨迹 ID 或目录。省略则用最近一条"),
    root: Path = typer.Option(DEFAULT_ROOT, "--root"),
    step: int = typer.Option(None, "--step", help="非交互：指定步号"),
    labels: str = typer.Option(None, "--labels", help="非交互：逗号分隔的标签"),
    note: str = typer.Option("", "--note", help="非交互：补充说明"),
    show_labels: bool = typer.Option(False, "--list", help="只列出可用标签"),
) -> None:
    """给失败步骤打错误类型标签。

    M2 要求"每一条失败轨迹都要当场打标，不允许攒到 M4 集中人工复盘"——
    隔几天再看，当时屏幕上是什么样早就忘了。
    """
    _setup_console()

    if show_labels:
        typer.echo("可用标签：\n" + describe_labels())
        return

    reader = _open_trajectory(trajectory, root)

    if step is not None:
        if not labels:
            typer.secho("--step 需要配合 --labels 使用", fg="red")
            raise typer.Exit(1)
        _apply_labels(reader, step, [x.strip() for x in labels.split(",") if x.strip()], note)
        return

    # 挑的不只是「执行失败」的步 —— 实测 43 条轨迹里一个 failed 都没有，
    # 真正要标的是**执行成功但没用**的步（原地重复、零动作报完成）。
    # 详见 TrajectoryReader.steps_to_label 的说明。
    pending = reader.steps_to_label()
    if not pending:
        typer.secho("这条轨迹没有需要打标的步骤。", fg="green")
        return

    typer.echo("")
    typer.echo(f"{len(pending)} 个步骤待打标。可用标签：")
    typer.echo(describe_labels())
    typer.echo("")
    for record, why in pending:
        typer.secho(f"  {record.summary()}", fg="red")
        typer.secho(f"      挑中原因：{why}", fg="yellow")
        if record.model_thinking:
            typer.echo(f"      思考：{record.model_thinking}")
        if record.raw_output:
            typer.echo(f"      原始：{record.raw_output[:200]}")
        if record.screenshot_before:
            typer.echo(f"      截图：{reader.frame(record.screenshot_before)}")
        if record.labels:
            typer.echo(f"      已标：{', '.join(record.labels)}")

        answer = typer.prompt("      标签（逗号分隔，回车跳过）", default="", show_default=False)
        if not answer.strip():
            continue
        chosen = [x.strip() for x in answer.split(",") if x.strip()]
        remark = typer.prompt("      补充说明（可空）", default="", show_default=False)
        _apply_labels(reader, record.step, chosen, remark)


def _apply_labels(reader: TrajectoryReader, step: int, chosen: list[str], note: str) -> None:
    known, unknown = validate_labels(chosen)
    if unknown:
        # 不拒绝：现场打标时卡住人不合适。但要提醒——自由文本会让 M4
        # 统计前还要人工归并同义写法
        typer.secho(
            f"      未知标签 {unknown}（仍会记录）。建议用固定词表：{', '.join(ERROR_LABELS)}",
            fg="yellow",
        )
    if reader.write_labels(step, chosen, note):
        typer.secho(f"      已记录 #{step}：{', '.join(chosen)}", fg="green")
    else:
        typer.secho(f"      轨迹里没有第 {step} 步", fg="red")


# ---------------------------------------------------------------------- #
# config
# ---------------------------------------------------------------------- #


@app.command()
def config(
    show: bool = typer.Option(False, "--show", help="显示当前配置"),
    prompts: bool = typer.Option(False, "--prompts", help="列出可用提示词模板"),
    trajectories: bool = typer.Option(False, "--trajectories", help="列出历史轨迹"),
    root: Path = typer.Option(DEFAULT_ROOT, "--root"),
) -> None:
    """查看配置、模板与历史轨迹。"""
    _setup_console()
    load_dotenv_if_present()

    if not (show or prompts or trajectories):
        show = True

    if show:
        typer.secho("\n模型平台", bold=True)
        for name, configured in available_providers():
            provider = PROVIDERS[name]
            if configured:
                detail = resolve(name, require_key=False).masked()
                typer.secho(
                    f"  ✓ {provider.label:12} {detail['model']:28} {detail['api_key']}"
                    + ("  已配单价" if detail["priced"] else "  未配单价（成本熔断关闭）"),
                    fg="green",
                )
            else:
                typer.echo(f"  · {provider.label:12} 未配置 {provider.api_key_env}")
            if provider.notes:
                typer.echo(f"      {provider.notes}")

        typer.secho("\n动作空间", bold=True)
        from control.actions import CORE_ACTIONS, STUB_ACTIONS

        typer.echo(
            f"  已实现 {len(CORE_ACTIONS)} 个：{', '.join(sorted(a.value for a in CORE_ACTIONS))}"
        )
        typer.echo(
            f"  接口桩 {len(STUB_ACTIONS)} 个：{', '.join(sorted(a.value for a in STUB_ACTIONS))}"
        )

    if prompts:
        from agent.prompts import list_templates, load_template

        typer.secho("\n提示词模板", bold=True)
        for name in list_templates():
            template = load_template(name)
            typer.echo(
                f"  {name:16} {template.version:4} few-shot {len(template.few_shot)} 组"
                f"  特性 {', '.join(template.features) or '—'}"
            )

    if trajectories:
        typer.secho("\n历史轨迹", bold=True)
        found = list_trajectories(root)
        if not found:
            typer.echo(f"  {root} 下还没有轨迹")
        for path in found[:20]:
            meta = TrajectoryReader(path).meta
            typer.echo(
                f"  {meta.trajectory_id}  {meta.status:14} {meta.total_steps:2d} 步  "
                f"{meta.total_cost_cny:.4f} 元  {meta.instruction[:30]}"
            )


# ---------------------------------------------------------------------- #


def _open_trajectory(trajectory: str | None, root: Path) -> TrajectoryReader:
    """按 ID、目录或"最近一条"打开轨迹。

    支持省略参数是有意的：调试时最常见的动作就是"刚跑完那条给我看看"，
    每次去翻 ID 是纯粹的摩擦。
    """
    if trajectory:
        candidate = Path(trajectory)
        if not candidate.is_dir():
            candidate = Path(root) / trajectory
        if not candidate.is_dir():
            typer.secho(f"找不到轨迹：{trajectory}", fg="red")
            raise typer.Exit(1)
        return TrajectoryReader(candidate)

    found = list_trajectories(root)
    if not found:
        typer.secho(f"{root} 下还没有任何轨迹", fg="red")
        raise typer.Exit(1)
    return TrajectoryReader(found[0])


def main() -> None:
    app()


if __name__ == "__main__":
    main()
