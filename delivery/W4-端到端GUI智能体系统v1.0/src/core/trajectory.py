"""轨迹日志 —— 每一步执行的结构化记录。

## 这是后面三个里程碑唯一的数据源

M2 文档写得很直接："轨迹日志是 M3、M4、M5 全部分析的唯一数据源。"具体说：

- **M3** 的六档横评要按后端分组统计成功率、步数、延迟、成本
- **M4** 的错误分类与恢复策略要从失败轨迹里归纳错误类型分布
- **M5** 的消融实验要对比模式 A/B 的定位精度

这三件事都是**事后分析**，跑的时候记漏了就再也补不回来了——重跑一次
不是同一条轨迹，模型的输出本来就有随机性。因此这里的原则是**记全**：
拿不准要不要记的字段，记。存储很便宜，重跑很贵。

## 与 HistoryStep 的分工

`llm.base.HistoryStep` 是**给模型看的**，要精简，每个 token 都要付费。
这里的 `StepRecord` 是**给人和分析脚本看的**，要完整。两者刻意不复用。

## 落盘形态：JSONL + 截图帧目录

一行一步，追加写。选 JSONL 而不是数据库的理由见 M2「明确不做」第 11 条，
但还有一条更实际的：**Agent 跑崩了，已经写进去的行还在**。数据库要等
事务提交，进程被急停热键杀掉时未提交的就没了——而急停恰恰在最需要看
日志的时候触发。

每步落盘后立即 flush，代价是一点点 IO，换来的是崩溃现场可复盘。
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: 轨迹根目录。每条轨迹一个子目录，内含 steps.jsonl 与 frames/
DEFAULT_ROOT = Path("outputs/trajectories")


# ---------------------------------------------------------------------- #
# 延迟拆分
# ---------------------------------------------------------------------- #


@dataclass
class LatencyBreakdown:
    """单步延迟的四段拆分。

    M2 验收标准第 3 条要求"单步平均延迟已记录，并按 API / grounding /
    执行 / 截图 四段分解"。**必须分开记**：只记总延迟的话，看到"单步 6 秒"
    完全不知道该优化什么——是模型慢、定位慢、还是动作后那个固定等待太长。
    M4 的性能优化要靠这四个数决定优先级。
    """

    api_ms: float = 0.0
    grounding_ms: float = 0.0
    execute_ms: float = 0.0
    screenshot_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.api_ms + self.grounding_ms + self.execute_ms + self.screenshot_ms

    def as_dict(self) -> dict:
        return {
            "api_ms": round(self.api_ms, 2),
            "grounding_ms": round(self.grounding_ms, 2),
            "execute_ms": round(self.execute_ms, 2),
            "screenshot_ms": round(self.screenshot_ms, 2),
            "total_ms": round(self.total_ms, 2),
        }


# ---------------------------------------------------------------------- #
# 一步的记录
# ---------------------------------------------------------------------- #


@dataclass
class StepRecord:
    """一步执行的完整记录，对应 JSONL 的一行。

    字段清单直接来自 M2 任务拆解第 6 条，另加三项：``grounding``（完整
    结果而不只是来源）、``labels``（`label` 命令写回的错误标注）、
    ``retry_count``（输出解析重试次数，M3 比较各模型格式稳定性时要用）。
    """

    trajectory_id: str = ""
    step: int = 0
    #: 所属子任务的序号。Planner 拆出来的第几个子任务
    subtask_id: int = 0
    subtask: str = ""
    timestamp: str = ""

    #: 截图路径，相对轨迹目录。**存路径不存 base64**——一条 25 步的轨迹
    #: 内联截图能让 JSONL 涨到几十 MB，用文本编辑器都打不开
    screenshot_before: str = ""
    screenshot_after: str = ""

    model_thinking: str = ""
    #: 模型原始输出。解析失败时这是唯一线索
    raw_output: str = ""
    #: 模型给出的动作意图（grounding 之前的原始形态）
    action_intent: dict = field(default_factory=dict)
    #: grounding 的完整结果：来源、置信度、是否纠偏
    grounding: dict = field(default_factory=dict)

    #: 最终执行的动作。**同时记模型坐标与真实坐标**——验收标准第 7 条要求
    #: "模型输出坐标与实际点击坐标的映射关系在日志中可追溯，无坐标错位"，
    #: 只记一套就查不了错位
    action_model_coords: dict = field(default_factory=dict)
    action_real_coords: dict = field(default_factory=dict)

    #: ok / failed / rejected(安全拦截) / no_action / error
    execution_status: str = ""
    error: str = ""
    error_type: str = ""
    #: 输出解析失败后的重试次数
    retry_count: int = 0

    latency: dict = field(default_factory=dict)
    tokens: dict = field(default_factory=dict)
    cost_cny: float = 0.0
    request_id: str = ""
    #: 后端标识，M3 横评按它分组
    backend: str = ""

    #: `label` 命令写回的错误标签。M2 要求失败轨迹**当场打标**，
    #: 不允许攒到 M4 集中人工复盘
    labels: list[str] = field(default_factory=list)
    label_note: str = ""

    meta: dict = field(default_factory=dict)

    #: 真正算失败的状态。**no_action 不在其中**——模型报告子任务完成的
    #: 那一步没有动作可执行，它既不是"成功执行了动作"，也不是失败。
    #:
    #: 把它算成失败有实际后果：`failed_steps` 会把每条轨迹的收尾步都拉
    #: 进待打标清单，逼人给一次正常完成打错误标签；按步统计成功率时，
    #: 每条轨迹还会凭空多一次失败。
    FAILURE_STATUSES = ("failed", "rejected", "error")

    @property
    def succeeded(self) -> bool:
        """这一步成功执行了动作。"""
        return self.execution_status == "ok"

    @property
    def failed(self) -> bool:
        """这一步真的出错了。与 `succeeded` **不是互补关系**。"""
        return self.execution_status in self.FAILURE_STATUSES

    @property
    def is_terminal(self) -> bool:
        """模型在这一步报告子任务完成。"""
        return self.execution_status == "no_action"

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> StepRecord:
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in data.items() if k in known}
        extra = {k: v for k, v in data.items() if k not in known}
        record = cls(**kwargs)
        if extra:
            # 旧版本轨迹里的字段不认识也别丢——分析脚本可能还要用
            record.meta.setdefault("_unknown_fields", {}).update(extra)
        return record

    def summary(self) -> str:
        """一行人类可读摘要，`replay` 命令用。

        **只用 ASCII 记号**。Windows 控制台默认 cp936（GBK），项目里满屏
        中文都能打，但 U+2713 这类符号不在 GBK 码表里，``print`` 会直接抛
        ``UnicodeEncodeError``——摘要是出问题时用来看的东西，它本身不能
        成为新的崩溃源。
        """
        action = self.action_model_coords.get("action", "-")
        if self.is_terminal:
            mark, action = "DONE", action if action != "-" else "（模型报告完成）"
        else:
            mark = "OK  " if self.succeeded else "FAIL"
        line = f"{mark} #{self.step} (sub{self.subtask_id}) {action}"
        if self.error:
            line += f" - {self.error}"
        if self.labels:
            line += f" [{', '.join(self.labels)}]"
        return line


# ---------------------------------------------------------------------- #
# 一条轨迹的元信息
# ---------------------------------------------------------------------- #


@dataclass
class TrajectoryMeta:
    """轨迹级别的信息，落在 meta.json。

    和步骤分开存：步骤是追加写的流，元信息在开始与结束时各写一次。混在
    一个文件里会让"追加"变成"改写"，崩溃时更容易丢数据。
    """

    trajectory_id: str = ""
    instruction: str = ""
    subtasks: list[str] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""
    #: running / success / failed / aborted / cost_limit / max_iterations
    status: str = "running"
    backend: str = ""
    model: str = ""
    grounding_backend: str = ""
    mode: str = "A"
    total_steps: int = 0
    total_cost_cny: float = 0.0
    total_tokens: int = 0
    duration_s: float = 0.0
    #: 程序化成功判定的结果。M2 要求"成功判定必须程序化，不用人工目测"
    verified: bool | None = None
    verify_note: str = ""
    screen_region: list = field(default_factory=list)
    model_space: dict = field(default_factory=dict)
    #: **这条轨迹跑在什么环境里。**
    #:
    #: 分辨率、DPI 缩放、截图引擎——三者任何一个变了，成功率与延迟都不再
    #: 可比。而 M5 的结题报告要引用 M2 的成本与成功率作参照，跨了三周半，
    #: 中间还会恢复无数次快照。
    #:
    #: 「两次跑在同一环境」如果只靠人记得核对，那它是个**假设**；记进每条
    #: 轨迹，它才是**事实**，事后还能查。
    environment: dict = field(default_factory=dict)
    error: str = ""
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> TrajectoryMeta:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------- #
# 写入
# ---------------------------------------------------------------------- #


def new_trajectory_id(prefix: str = "traj") -> str:
    """时间戳 + 短随机串。

    时间戳在前是为了目录按时间自然排序；带随机串是因为同一秒可能起两条
    （批量任务），纯时间戳会撞。
    """
    return f"{prefix}-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


class TrajectoryWriter:
    """把一条轨迹写到磁盘。

    目录结构::

        outputs/trajectories/<trajectory_id>/
            meta.json          轨迹级信息
            steps.jsonl        一行一步
            frames/            截图帧

    用作上下文管理器时，退出会自动补写 ``finished_at`` 与状态——**包括
    异常退出**。轨迹半截断掉却没记状态，是复盘时最讨厌的情况。
    """

    def __init__(
        self,
        instruction: str = "",
        trajectory_id: str | None = None,
        root: Path | str = DEFAULT_ROOT,
    ) -> None:
        self.trajectory_id = trajectory_id or new_trajectory_id()
        self.root = Path(root) / self.trajectory_id
        self.frames_dir = self.root / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        self.meta = TrajectoryMeta(
            trajectory_id=self.trajectory_id,
            instruction=instruction,
            started_at=datetime.now().isoformat(timespec="seconds"),
        )
        self._steps_path = self.root / "steps.jsonl"
        self._meta_path = self.root / "meta.json"
        self._start = time.perf_counter()
        self._step_count = 0
        self.write_meta()

    # ------------------------------------------------------------------ #

    @property
    def steps_path(self) -> Path:
        return self._steps_path

    @property
    def step_count(self) -> int:
        return self._step_count

    def next_step_index(self) -> int:
        return self._step_count + 1

    # ------------------------------------------------------------------ #

    def append(self, record: StepRecord) -> StepRecord:
        """追加一步并**立即 flush**。

        flush 的代价是每步一次磁盘同步；换来的是急停热键把进程杀掉时，
        已执行的步骤仍在文件里。急停恰恰在最需要看日志的时候触发。
        """
        record.trajectory_id = self.trajectory_id
        if not record.step:
            record.step = self.next_step_index()
        if not record.timestamp:
            record.timestamp = datetime.now().isoformat(timespec="milliseconds")

        with self._steps_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(record.as_dict(), ensure_ascii=False, default=_json_default) + "\n"
            )
            handle.flush()

        self._step_count = max(self._step_count, record.step)
        self.meta.total_steps = self._step_count
        self.meta.total_cost_cny += record.cost_cny
        self.meta.total_tokens += int(record.tokens.get("total_tokens", 0) or 0)
        return record

    def frame_path(self, step: int, phase: str) -> Path:
        """截图帧的落盘路径。``phase`` 取 before / after。"""
        return self.frames_dir / f"step{step:03d}-{phase}.png"

    def relative(self, path: Path | str) -> str:
        """帧路径转成相对轨迹目录的形式，写进 JSONL。

        存相对路径而不是绝对路径：轨迹目录整个拷到另一台机器上分析时，
        绝对路径全部失效。M3 的分析大概率不在跑任务的那台机器上做。
        """
        try:
            return str(Path(path).relative_to(self.root)).replace("\\", "/")
        except ValueError:
            return str(path)

    def write_meta(self) -> None:
        self._meta_path.write_text(
            json.dumps(self.meta.as_dict(), ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )

    def finish(self, status: str = "success", error: str = "") -> TrajectoryMeta:
        self.meta.status = status
        self.meta.error = error
        self.meta.finished_at = datetime.now().isoformat(timespec="seconds")
        self.meta.duration_s = round(time.perf_counter() - self._start, 2)
        self.write_meta()
        logger.info(
            "轨迹 %s 结束：%s，%d 步，%.4f 元",
            self.trajectory_id,
            status,
            self.meta.total_steps,
            self.meta.total_cost_cny,
        )
        return self.meta

    # ------------------------------------------------------------------ #

    def __enter__(self) -> TrajectoryWriter:
        return self

    def __exit__(self, exc_type, exc, _tb) -> None:
        if exc_type is not None:
            # 异常退出也要把状态落下去，否则轨迹停在 running 上，事后
            # 分不清是"跑崩了"还是"还在跑"
            self.finish(status="aborted", error=f"{exc_type.__name__}: {exc}")
        elif self.meta.status == "running":
            self.finish()


# ---------------------------------------------------------------------- #
# 读取与打标
# ---------------------------------------------------------------------- #


class TrajectoryReader:
    """读一条已落盘的轨迹，供 `replay` 与 `label` 使用。"""

    def __init__(self, path: Path | str) -> None:
        self.root = Path(path)
        if not self.root.is_dir():
            raise FileNotFoundError(f"轨迹目录不存在：{self.root}")
        self._steps_path = self.root / "steps.jsonl"
        self._meta_path = self.root / "meta.json"

    @property
    def meta(self) -> TrajectoryMeta:
        if not self._meta_path.exists():
            return TrajectoryMeta(trajectory_id=self.root.name)
        return TrajectoryMeta.from_dict(json.loads(self._meta_path.read_text(encoding="utf-8")))

    def steps(self) -> list[StepRecord]:
        return list(self.iter_steps())

    def iter_steps(self) -> Iterator[StepRecord]:
        """逐行读。

        单行解析失败时跳过并告警，**不让整条轨迹读不出来**——进程被杀在
        写一半的时候，最后一行天然可能是残缺的，那不该毁掉前面所有步骤。
        """
        if not self._steps_path.exists():
            return
        with self._steps_path.open(encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield StepRecord.from_dict(json.loads(line))
                except (json.JSONDecodeError, TypeError) as exc:
                    logger.warning(
                        "轨迹 %s 第 %d 行无法解析，已跳过：%s", self.root.name, lineno, exc
                    )

    def frame(self, relative_path: str) -> Path:
        return self.root / relative_path

    def failed_steps(self) -> list[StepRecord]:
        """动作执行本身失败的步骤。

        用 `StepRecord.failed` 而不是 ``not succeeded``：后者会把模型报告
        完成的收尾步也算进来，于是每条轨迹都至少有一条"待打标"，而它其实
        什么问题都没有。

        **但这个集合不足以支撑打标**，见 `steps_to_label()`。
        """
        return [s for s in self.iter_steps() if s.failed]

    def steps_to_label(self) -> list[tuple[StepRecord, str]]:
        """值得人工打标的步骤，附上「为什么挑中它」。

        ## 为什么不能只挑执行失败的步

        M2 的 43 条真实轨迹里，`execution_status` 只出现过 `ok`（282 步）
        与 `no_action`（162 步）——**一个 `failed` 都没有**。于是
        `failed_steps()` 在每条轨迹上都返回空，`label` 一律回答
        「没有失败步骤，不需要打标」，M2 验收标准 6 无从执行。

        当初那个筛选条件的前提就错了：真正要标的不是「动作没执行成功」，
        而是**「动作执行成功但没用」**。实测里的三种形态：

        1. **连续重复同一动作**（占总步数 33%，最长连发 12 次）。
           每一次点击都返回 ok，可屏幕纹丝不动。
        2. **零动作报完成**：子任务以 `done` 收尾，而这个子任务里
           一个动作都没成功执行过。
        3. 动作执行失败（原有的那一类，实测未出现，但保留）。

        ## 每段重复只挑一步

        连发 12 次就offer 12 步会把打标变成体力活。每段连续重复只offer
        **第二步**（即第一次重复），说明里带上这一段连发了几次——
        人看一眼就知道发生了什么，标一次即可。
        """
        steps = list(self.iter_steps())
        picked: dict[int, str] = {}

        # --- 1. 执行失败 ---
        for record in steps:
            if record.failed:
                picked[record.step] = f"动作执行失败：{record.error or record.execution_status}"

        # --- 2. 连续重复同一动作，每段只挑第一次重复 ---
        run_start = 0
        for index in range(1, len(steps)):
            previous, current = steps[index - 1], steps[index]
            same = (
                current.subtask_id == previous.subtask_id
                and current.action_intent
                and current.action_intent == previous.action_intent
                and not current.action_intent.get("done")
            )
            if not same:
                run_start = index
                continue
            if index - 1 == run_start:  # 这一段的第一次重复
                length = 2
                probe = index + 1
                while probe < len(steps) and steps[probe].action_intent == current.action_intent:
                    length += 1
                    probe += 1
                picked.setdefault(current.step, f"原地重复：同一动作连发 {length} 次")

        # --- 3. 零动作报完成 ---
        by_subtask: dict[int, list[StepRecord]] = {}
        for record in steps:
            by_subtask.setdefault(record.subtask_id, []).append(record)
        for group in by_subtask.values():
            last = group[-1]
            if last.is_terminal and not any(r.succeeded for r in group):
                picked.setdefault(
                    last.step,
                    f"零动作报完成：子任务 #{last.subtask_id} 共 {len(group)} 步，无一步执行成功",
                )

        return [(r, picked[r.step]) for r in steps if r.step in picked]

    # ------------------------------------------------------------------ #

    def write_labels(self, step: int, labels: list[str], note: str = "") -> bool:
        """给某一步写回错误标签。

        整文件重写而不是原地改：JSONL 每行长度不定，原地改会破坏后续行。
        轨迹最多几十步，重写的代价可以忽略。

        先写临时文件再替换，避免打标过程中断掉把原轨迹也毁了。
        """
        records = self.steps()
        hit = False
        for record in records:
            if record.step == step:
                record.labels = list(labels)
                record.label_note = note
                hit = True
        if not hit:
            logger.warning("轨迹 %s 中没有第 %d 步", self.root.name, step)
            return False

        temp = self._steps_path.with_suffix(".jsonl.tmp")
        with temp.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(record.as_dict(), ensure_ascii=False, default=_json_default) + "\n"
                )
        temp.replace(self._steps_path)
        return True


def list_trajectories(root: Path | str = DEFAULT_ROOT) -> list[Path]:
    """按时间倒序列出所有轨迹目录。"""
    root = Path(root)
    if not root.is_dir():
        return []
    dirs = [p for p in root.iterdir() if p.is_dir() and (p / "steps.jsonl").exists()]
    return sorted(dirs, key=lambda p: p.name, reverse=True)


#: `label` 命令的候选错误类型。
#:
#: 固定候选而不是自由文本，是为了让 M4 能直接对标签做频次统计——自由
#: 文本会出现"点错位置""坐标不对""click 偏了"三种写法指同一件事，统计时
#: 还要再人工归并一遍，等于把 M2 省下的功夫在 M4 加倍还回去。
#:
#: 需要补充时在这里加，不要在命令行里现编。
ERROR_LABELS: dict[str, str] = {
    "grounding_off": "定位偏差：动作类型对，但坐标点错了地方",
    "wrong_action": "动作选错：该点击时输入、该滚动时点击一类",
    "premature_done": "过早收工：子任务没完成就报告 done",
    "stuck_loop": "原地打转：反复执行同一个无效动作",
    "parse_error": "输出不可解析：模型没按 schema 返回",
    "missing_coords": "缺坐标：模式 A 下模型没给 x/y",
    "safety_blocked": "被安全白名单拦截",
    "app_state": "界面状态不符预期：弹窗、加载未完成、焦点丢失",
    "planner_bad": "拆解有问题：子任务粒度过粗或顺序错误",
    "timeout": "超时或无响应",
    "other": "其他（在 note 里写清楚）",
}


def describe_labels() -> str:
    """候选标签的可读清单，CLI 提示用。"""
    return "\n".join(f"  {key:16} {desc}" for key, desc in ERROR_LABELS.items())


def validate_labels(labels: list[str]) -> tuple[list[str], list[str]]:
    """分出已知与未知标签。未知的不拒绝，只提醒——现场打标时卡住人不合适。"""
    known = [label for label in labels if label in ERROR_LABELS]
    unknown = [label for label in labels if label not in ERROR_LABELS]
    return known, unknown


def _json_default(value: Any) -> Any:
    """兜底序列化。轨迹落盘不该因为某个字段不可序列化就整步丢掉。"""
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if hasattr(value, "as_tuple"):
        return value.as_tuple()
    return str(value)
