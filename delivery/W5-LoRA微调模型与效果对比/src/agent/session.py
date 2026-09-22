"""Session —— 承接一条用户指令，从头跑到尾。

## 它填的是哪个空

在此之前各层都能单独工作，但没人负责把它们接起来：

- `Planner` 会拆任务，但不知道拆完交给谁
- `AgentLoop` 会跑子任务，但不知道子任务从哪来
- 提示词模板躺在 YAML 里，**没有任何地方把它塞进后端**——这一条最要命：
  不注入的话模型根本不知道动作空间长什么样，跑起来只会胡乱输出
- `ContextWindow` 算好了每帧该用什么分辨率，但决策传不到编码那一步

Session 就是这些接线的所在地。M2 任务 7 的原话是"打通完整链路：指令 →
Planner 拆解 → 逐子任务进入 Agent Loop → 落盘 → 返回结果"，这个类是那句
话的实现。

## 为什么在 agent/ 而不是 core/

依赖方向。Session 依赖 `core.loop`，反过来不行——Loop 是被编排的一层。
把 Session 放 core 里会让 core 反向依赖 agent，从此两层纠缠在一起，M3
想单独替换编排逻辑就得连 Loop 一起动。

## 一个任务一条轨迹

`TrajectoryWriter` 由 Session 持有，跨所有子任务。步号在整条轨迹里连续，
子任务号单独记——回放时才不会出现两个 #1。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from agent.context import ContextPolicy, ContextWindow, Conversation
from agent.planner import Plan, PlanError, Planner
from agent.prompts import PromptTemplate, load_template
from core.loop import STOP_DONE, AgentLoop, LoopConfig, LoopResult
from core.trajectory import DEFAULT_ROOT, TrajectoryWriter
from llm.base import CostInfo, LLMBackend

logger = logging.getLogger(__name__)


@dataclass
class SessionConfig:
    """一次会话的全部可调项。

    分成三组：拆解、执行、上下文。**每一项都会进轨迹的 meta**，因为
    M3 的消融要按配置分组统计，配置没落盘就没法分组。
    """

    #: 提示词模板名。M3 消融换版本就是改这两个
    planner_template: str = "planner_v1"
    executor_template: str = "executor_v1"

    #: **这个模型真的会哪些动作。** 空表示不限制（全部 CORE_ACTIONS）。
    #:
    #: 2026-08-25 查出的结构性错位：微调模型的训练集里 `type` / `key`
    #: 各 0 条（ScreenAgent 的键盘动作因不带坐标被整批过滤），而提示词
    #: 照样告诉它「你可以 type」。五个基础任务里四个需要打字，模型
    #: **连掷骰子的机会都没有**——25 轮跑完只看到一片 0/5，系统里没有
    #: 任何一处发现原因。
    #:
    #: 填上之后，执行器提示词与**规划器提示词**同时收窄。收窄规划器
    #: 是关键：否则它照样拆出「输入 Microsoft Edge」这种执行器做不到的
    #: 步骤，然后整轮卡在那里烧完 max_steps。
    #:
    #: **2026-08-26 之后不要再按旧指引收窄到四个鼠标动作。**
    #:
    #: 那条指引写于训练集只有 3 种动作的时候，当时收窄是对的——模型确实
    #: 不会 `type`。放宽后的训练集有 8 种动作 + `done`（`type` 216 条、
    #: `done` 897 条），照旧指引填的话等于把模型刚学会的能力关掉，
    #: **而且不会报错**：日志里看到的只是它又一次没打字。
    #:
    #: 留空即用完整动作空间。确实要收窄时（比如复现旧实验），显式写全。
    allowed_actions: tuple[str, ...] = ()

    #: 拆解时给不给模型看当前屏幕
    plan_with_screenshot: bool = True

    space_name: str = "planner"

    #: **送给模型的图片尺寸**。只影响上传字节数与视觉 token 数，
    #: 不影响坐标解释
    image_size: tuple[int, int] = (1024, 768)

    #: **模型作答所用的坐标系**。与 image_size 是两回事，这一点是实测出来的。
    #:
    #: 实验：同一张 2560×1600 桌面截图问"点任务栏开始按钮"，分别告诉模型
    #: 画布是 1024×768 和 1024×640，各 4 次——
    #:
    #:     告诉它 768 高 → y = 968, 968, 970, 963
    #:     告诉它 640 高 → y = 973, 967, 967, 970
    #:
    #: **y 完全不随声明的高度变化**，恒在 968 上下。968/1000 = 96.8%，
    #: 正是任务栏在 1600 高屏幕上的位置。结论：qwen3-vl 输出的是归一化到
    #: [0,1000) 的坐标，各轴独立，与我们送多大的图无关。
    #:
    #: 把两者当成同一个东西，后果是每次点击都系统性偏移——而且偏移量随
    #: 目标在屏幕上的位置变化，看起来极像"模型定位不准"。M1 的
    #: `CoordinateScaler` 文档里其实预留过这一条（"Qwen2.5-VL 归一化输出"）。
    #:
    #: 输出图像像素坐标的模型（部分 Qwen2.5-VL 版本）应改成与 image_size
    #: 一致。M3 接新后端时这是**第一件要实测确认的事**。
    coordinate_space: tuple[int, int] = (1000, 1000)

    loop: LoopConfig = field(default_factory=LoopConfig)
    context: ContextPolicy = field(default_factory=ContextPolicy)

    #: 子任务之间要不要清空历史。
    #:
    #: 默认清空：上一个子任务的操作对下一个的决策价值很低（"点了开始菜单"
    #: 对"在记事本里打字"没帮助），却要一直付 token。M2"一次只带一个子目标
    #: 进 Loop"说的就是这种隔离。
    clear_history_between_subtasks: bool = True

    def as_dict(self) -> dict:
        return {
            "planner_template": self.planner_template,
            "executor_template": self.executor_template,
            "plan_with_screenshot": self.plan_with_screenshot,
            "space_name": self.space_name,
            "image_size": list(self.image_size),
            # **进轨迹存档。** 动作集是能左右结论的变量,不记下来事后
            # 分不清「它没变」和「它变了但没人看见」。
            "allowed_actions": list(self.allowed_actions),
            # 重试升级的开关在 `LoopConfig` 上，**这里只是把它抄进存档**。
            # 两处各存一份的话，一处改了另一处不跟着改，存档记的就不是
            # 实际生效的那个值——这是本项目栽过两次的形态（提示词模板、
            # 屏幕分辨率）。
            "escalate_on_no_change": self.loop.escalate_on_no_change,
            "reflector": self.loop.reflector,
            "reflector_max_rejects": self.loop.reflector_max_rejects,
            "coordinate_space": list(self.coordinate_space),
            "max_iterations": self.loop.max_iterations,
            "cost_limit_cny": self.loop.cost_limit_cny,
            "settle_seconds": self.loop.settle_seconds,
            "dry_run": None,  # 由 Session 填，执行器才知道
            "context": self.context.as_dict(),
        }


@dataclass
class SubtaskOutcome:
    """一个子任务的执行结果。"""

    id: int
    goal: str
    expected: str
    result: LoopResult

    @property
    def succeeded(self) -> bool:
        return self.result.succeeded

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "goal": self.goal,
            "expected": self.expected,
            **self.result.as_dict(),
        }


@dataclass
class SessionResult:
    """一条指令跑完的完整结果。"""

    instruction: str = ""
    trajectory_id: str = ""
    trajectory_dir: str = ""
    plan: Plan | None = None
    outcomes: list[SubtaskOutcome] = field(default_factory=list)
    #: completed / plan_failed / subtask_failed / aborted
    status: str = "completed"
    reason: str = ""
    cost: CostInfo | None = None
    duration_s: float = 0.0

    @property
    def succeeded(self) -> bool:
        """**所有子任务都正常收尾**才算成功。

        注意这不等于任务真的做成了——M2 要求"成功判定必须程序化：进程
        检查、窗口标题检查、文件内容检查，不用人工目测"。模型说自己做完
        和它真做完是两回事，后者由任务定义里的校验器判定。这个属性只回答
        "有没有中途被刹车拦下"。
        """
        return self.status == "completed" and all(o.succeeded for o in self.outcomes)

    @property
    def total_steps(self) -> int:
        return sum(o.result.steps for o in self.outcomes)

    def as_dict(self) -> dict:
        return {
            "instruction": self.instruction,
            "trajectory_id": self.trajectory_id,
            "status": self.status,
            "reason": self.reason,
            "subtasks": len(self.outcomes),
            "succeeded_subtasks": sum(1 for o in self.outcomes if o.succeeded),
            "total_steps": self.total_steps,
            "duration_s": round(self.duration_s, 2),
            "cost": self.cost.as_dict() if self.cost else None,
            "outcomes": [o.as_dict() for o in self.outcomes],
        }


# ---------------------------------------------------------------------- #


class Session:
    """一条指令的完整生命周期。

    典型用法::

        session = Session(backend, grounding, executor, capturer)
        result = session.run("打开记事本")
    """

    def __init__(
        self,
        backend: LLMBackend,
        grounding,
        executor,
        capturer,
        config: SessionConfig | None = None,
        trajectory_root: Path | str = DEFAULT_ROOT,
        on_step=None,
        on_subtask=None,
    ) -> None:
        self.backend = backend
        self.grounding = grounding
        self.executor = executor
        self.capturer = capturer
        self.config = config or SessionConfig()
        self.trajectory_root = trajectory_root
        #: 每**步**执行完的回调（收到 StepRecord），CLI 的实时面板用它
        self.on_step = on_step
        #: 每个**子任务**结束的回调（收到 SubtaskOutcome）
        self.on_subtask = on_subtask

        self.executor_template: PromptTemplate = load_template(self.config.executor_template)
        self.planner_template: PromptTemplate = load_template(self.config.planner_template)
        self.conversation = Conversation(window=ContextWindow(self.config.context))

        self._check_coordinate_space()

    # ------------------------------------------------------------------ #

    def _check_coordinate_space(self) -> None:
        """确认送模型的图与坐标系同尺寸。

        这是**最容易错、错了最难查**的一处：模型在它看到的那张图的像素
        坐标系里作答，尺寸对不上时每次点击都会偏，而且偏得很有规律——
        看起来像"模型定位不准"，实际是坐标系错配。开工时喊一声，比事后
        对着一堆偏移量猜要便宜得多。
        """
        scaler = getattr(self.executor, "scaler", None)
        if scaler is None:
            return
        try:
            space = scaler.get(self.config.space_name)
        except KeyError:
            logger.error(
                "坐标系 %r 未在 CoordinateScaler 里注册，执行时会直接失败",
                self.config.space_name,
            )
            return

        if (space.width, space.height) != tuple(self.config.coordinate_space):
            logger.error(
                "模型作答坐标系是 %s，而 CoordinateScaler 里注册的 %r 是 %d×%d —— "
                "两者必须一致，否则每次点击都会系统性偏移",
                self.config.coordinate_space,
                self.config.space_name,
                space.width,
                space.height,
            )

    def _prepare_backend(self) -> None:
        """把执行阶段的提示词注入后端。

        **这一步不做，整个系统就是哑的**：模型不知道有哪些动作、不知道
        坐标系多大、不知道该输出什么格式，只能靠猜。之前各层单测都过而
        端到端跑不通，缺的就是这里。
        """
        # 提示词里说的尺寸必须是**坐标系**的尺寸，不是图片的尺寸——
        # 模型要按哪把尺子作答，就告诉它那把尺子
        width, height = self.config.coordinate_space
        self.backend.system_prompt = self.executor_template.render_system(
            width=width, height=height, allowed_actions=self.config.allowed_actions or None
        )
        self.backend.few_shot = self.executor_template.few_shot_pairs(
            self.config.allowed_actions or None
        )
        self.backend.user_template = self.executor_template.user_template
        # 用户模板里的 {width}/{height} 也必须是**坐标系**尺寸。
        # 只把它传给系统提示、却让用户模板去读 image_size，结果是同一次
        # 请求里两句话互相矛盾（系统说 [0,1000)，用户说 1024×768）。
        # 见 `LLMBackend.coordinate_space`。
        self.backend.coordinate_space = self.config.coordinate_space
        if hasattr(self.backend, "image_size"):
            self.backend.image_size = self.config.image_size

    # ------------------------------------------------------------------ #

    def run(self, instruction: str) -> SessionResult:
        """跑完一条指令。"""
        started = time.perf_counter()
        result = SessionResult(instruction=instruction)

        writer = TrajectoryWriter(instruction, root=self.trajectory_root)
        result.trajectory_id = writer.trajectory_id
        result.trajectory_dir = str(writer.root)
        writer.meta.backend = self.backend.name
        writer.meta.model = self.backend.model
        writer.meta.grounding_backend = getattr(self.grounding, "name", "")
        writer.meta.mode = "A"
        writer.meta.environment = self._describe_environment()
        writer.meta.meta = {**self.config.as_dict(), "dry_run": self._dry_run()}
        writer.write_meta()

        try:
            return self._run_inner(instruction, writer, result, started)
        except Exception as exc:  # noqa: BLE001 - 任何意外都要把轨迹收尾
            result.status = "aborted"
            result.reason = f"{type(exc).__name__}: {exc}"
            result.duration_s = time.perf_counter() - started
            writer.finish(status="aborted", error=result.reason)
            logger.exception("会话异常终止")
            raise
        finally:
            result.cost = self.backend.get_cost()

    def _describe_environment(self) -> dict:
        """记下这条轨迹跑在什么环境里。

        分辨率、DPI、截图引擎三者任一变化都会让成功率与延迟失去可比性。
        采集失败不能影响任务——所以整段包在 try 里，拿不到就留空。
        """
        env: dict = {}
        try:
            from perception.dpi import describe as dpi_describe

            env["dpi"] = dpi_describe()
        except Exception as exc:  # noqa: BLE001
            env["dpi_error"] = str(exc)[:120]

        capturer = getattr(self, "capturer", None)
        engine = getattr(capturer, "engine_name", None) or getattr(capturer, "engine", None)
        if engine is not None:
            env["capture_engine"] = str(engine)

        scaler = getattr(self.executor, "scaler", None)
        if scaler is not None and hasattr(scaler, "region"):
            region = scaler.region
            env["resolution"] = f"{region.width}x{region.height}"
        return env

    def _run_inner(
        self,
        instruction: str,
        writer: TrajectoryWriter,
        result: SessionResult,
        started: float,
    ) -> SessionResult:
        # --- 1. 拆解 ---
        planner = Planner(
            self.backend,
            template=self.planner_template,
            with_screenshot=self.config.plan_with_screenshot,
            allowed_actions=self.config.allowed_actions,
        )
        screenshot = self.capturer.capture(fresh=True) if self.config.plan_with_screenshot else None

        try:
            plan = planner.plan(instruction, screenshot)
        except PlanError as exc:
            result.status = "plan_failed"
            result.reason = str(exc)
            result.duration_s = time.perf_counter() - started
            writer.meta.subtasks = []
            writer.finish(status="failed", error=f"拆解失败：{exc}")
            logger.error("拆解失败：%s", exc)
            return result

        result.plan = plan
        writer.meta.subtasks = plan.goals()
        writer.meta.meta["plan"] = plan.as_dict()
        writer.meta.meta["granularity_warnings"] = plan.granularity_report()
        writer.write_meta()

        # --- 2. 逐子任务执行 ---
        # 拆解阶段动过后端的提示词配置，执行前必须重新注入执行模板
        self._prepare_backend()

        loop = AgentLoop(
            llm=self.backend,
            grounding=self.grounding,
            executor=self.executor,
            capturer=self.capturer,
            writer=writer,
            config=self.config.loop,
            history_selector=self.conversation.window.select_steps,
            on_step=self.on_step,
        )
        loop.history = self.conversation.steps  # 共用同一份，便于按子任务清空

        for subtask in plan.subtasks:
            if self.config.clear_history_between_subtasks:
                self.conversation.clear()

            outcome = SubtaskOutcome(
                id=subtask.id,
                goal=subtask.goal,
                expected=subtask.expected,
                result=loop.run_subtask(subtask.goal, subtask_id=subtask.id),
            )
            result.outcomes.append(outcome)
            if self.on_subtask:
                self.on_subtask(outcome)

            if not outcome.succeeded:
                # GUI 操作有强顺序依赖，前一步没成往下走全是无效点击
                result.status = "subtask_failed"
                result.reason = (
                    f"子任务 #{subtask.id}（{subtask.goal}）以 "
                    f"{outcome.result.status} 结束：{outcome.result.reason}"
                )
                break

        # --- 3. 收尾 ---
        result.duration_s = time.perf_counter() - started
        cost = self.backend.get_cost()
        writer.meta.total_cost_cny = cost.cost_cny
        writer.meta.total_tokens = cost.total_tokens
        writer.finish(
            status="success" if result.status == "completed" else "failed",
            error=result.reason,
        )
        logger.info(
            "会话结束：%s，%d/%d 子任务完成，%d 步，%.4f 元",
            result.status,
            sum(1 for o in result.outcomes if o.succeeded),
            len(plan.subtasks),
            result.total_steps,
            cost.cost_cny,
        )
        return result

    # ------------------------------------------------------------------ #

    def _dry_run(self) -> bool:
        return bool(getattr(self.executor, "dry_run", False))

    def __repr__(self) -> str:
        return (
            f"<Session backend={self.backend.name!r} "
            f"executor={self.config.executor_template!r} dry_run={self._dry_run()}>"
        )


__all__ = [
    "STOP_DONE",
    "Session",
    "SessionConfig",
    "SessionResult",
    "SubtaskOutcome",
]
