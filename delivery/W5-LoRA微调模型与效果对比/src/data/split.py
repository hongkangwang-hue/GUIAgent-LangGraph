"""三分划定与冻结 —— M3 前置件①。

M2 任务 9 把这件事定在第 4 周，理由写得很直白："划分错了 M3 整周训练结果
全部作废"。所以划分要满足三件事：**可复现**（记种子）、**不泄漏**（验证集
与训练集不重叠）、**可核对**（落成文件，M3 开工第一件事是核对齐全）。

## 按会话切，不按样本切

ScreenAgent 的一个会话（session）是一串连续操作，**同一张截图会被同一会话
里的多条标注引用**。按样本随机切分的话，同一张截图会同时出现在训练集和
验证集里——模型在验证集上的表现会虚高，而且高得看不出来。

因此 `freeze_split()` 的划分单位是会话，样本跟着会话走。这也意味着验证集
的实际条数不会正好等于请求值，只能取最接近的会话组合。**宁可条数不整，
不可泄漏。**

## 三分的角色，不能混用

| 集合 | 来源 | 唯一允许的用途 |
|---|---|---|
| 训练集 | ScreenAgent train 划分中有坐标的样本 | LoRA 微调 |
| 验证集 | 同上，按会话隔离切出 | **唯一允许用于调参与提示词选型的集合** |
| 零样本测试集 | ScreenSpot / ScreenSpot-v2 全量 | 只在最终评测跑一次 |

测试集来自完全不同的数据集，本身就没有泄漏风险——但 ScreenSpot 与
ScreenSpot-v2 之间**互相**高度重叠（实测 586/610 张截图共用），两者不能
当成两个独立评测集来报告平均分。这一条写进 `docs/m3-prereq/`。
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from data.schema import Platform, UnifiedSample, is_trainable

#: 划分种子。**改这个数就等于换了一份训练集**，M3 的所有结果将不可比。
#: 定成常量而不是参数默认值，是为了让它在代码里只出现一次。
SPLIT_SEED = 20260913

#: 大纲给 M3 定的训练集目标区间
TARGET_TRAIN_MIN = 3000
TARGET_TRAIN_MAX = 4000
TARGET_VAL = 500

#: 验证集占池子的比例上限。
#:
#: 大纲的 500 条是按 3000-4000 条训练集配的（约 12%-17%）。实测可用池只有
#: 716 条，照搬 500 会切出 503 验证 / 213 训练——验证集比训练集还大，
#: 完全跑偏。所以 500 只作为**目标值**，实际取 `min(500, 池子 × 本比例)`。
#:
#: 20% 是常规取值。池子小的时候验证集也必然小，这是样本量不足的直接后果，
#: 不该靠挪用训练数据来掩盖。
MAX_VAL_RATIO = 0.2


def training_pool(
    samples: Iterable[UnifiedSample],
    *,
    datasets: Sequence[str] = ("screenagent",),
    splits: Sequence[str] = ("train",),
    platforms: Sequence[Platform] = (Platform.DESKTOP,),
) -> list[UnifiedSample]:
    """挑出可以作为**动作生成**监督信号的样本。

    这是**选取**不是清洗：被排除的样本没有任何问题，只是缺了该动作必需的
    字段，或不属于目标场景。两件事分开的理由见 `data.clean._no_location`。

    ## 2026-08-26：这个函数原名 `grounding_pool`，条件是一行

        and s.resolve_point() is not None

    只有带坐标的样本能通过，于是 ScreenAgent 4012 条里只剩 716 条，
    `type` / `key` / `done` / `wait` / `scroll` 全部落在门外。

    根因不是过滤写错了，是**它写于「grounding 层」时期**——那时任务定义
    就是「元素描述 → 坐标」，"必须有坐标"正是任务本身。2026-08-24 目标
    改为动作生成后，这个条件没跟着改，函数名也没改。

    > 教训：**目标改了要回头查过滤器。** 一个按旧目标写的筛选条件，在新
    > 目标下不会报错，只会安静地把大部分数据挡在门外，而计数看起来很正常。

    准入判据现在是 `data.schema.is_trainable`——按动作类型各查各的必需字段。
    """
    wanted_platforms = {p.value for p in platforms}
    return [
        s
        for s in samples
        if s.source_dataset in datasets
        and s.split in splits
        and s.platform.value in wanted_platforms
        and is_trainable(s)
    ]


@dataclass
class FrozenSplit:
    """一次冻结的划分结果。"""

    seed: int
    #: 分组键的名字，用于说明按什么单位切的
    group_by: str
    train_ids: list[str] = field(default_factory=list)
    val_ids: list[str] = field(default_factory=list)
    train_groups: list[str] = field(default_factory=list)
    val_groups: list[str] = field(default_factory=list)
    #: 池子总量与目标值的差距，供报告直接引用
    pool_size: int = 0
    requested_val: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def train_size(self) -> int:
        return len(self.train_ids)

    @property
    def val_size(self) -> int:
        return len(self.val_ids)

    def fingerprint(self) -> str:
        """划分内容的指纹。

        M3 开工核对时比这个字符串，比逐条 diff 快，也比只看条数可靠——
        条数一样但内容不同的划分是完全可能出现的（换了种子就是）。
        """
        digest = hashlib.sha256()
        for name, ids in (("train", self.train_ids), ("val", self.val_ids)):
            digest.update(name.encode())
            for sample_id in sorted(ids):
                digest.update(sample_id.encode())
        return digest.hexdigest()[:16]

    def leakage(self) -> set[str]:
        """训练集与验证集共享的样本 id。**必须为空。**"""
        return set(self.train_ids) & set(self.val_ids)

    def to_json_dict(self) -> dict:
        return {
            "seed": self.seed,
            "group_by": self.group_by,
            "fingerprint": self.fingerprint(),
            "pool_size": self.pool_size,
            "requested_val": self.requested_val,
            "train_size": self.train_size,
            "val_size": self.val_size,
            "train_groups": sorted(self.train_groups),
            "val_groups": sorted(self.val_groups),
            "train_ids": sorted(self.train_ids),
            "val_ids": sorted(self.val_ids),
            "notes": self.notes,
        }

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_json_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
            newline="\n",
        )
        return path

    @classmethod
    def read(cls, path: str | Path) -> FrozenSplit:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        split = cls(
            seed=payload["seed"],
            group_by=payload["group_by"],
            train_ids=payload["train_ids"],
            val_ids=payload["val_ids"],
            train_groups=payload.get("train_groups", []),
            val_groups=payload.get("val_groups", []),
            pool_size=payload.get("pool_size", 0),
            requested_val=payload.get("requested_val", 0),
            notes=payload.get("notes", []),
        )
        stored = payload.get("fingerprint")
        if stored and stored != split.fingerprint():
            raise ValueError(
                f"划分文件指纹不符：文件记的是 {stored}，按内容重算得到 "
                f"{split.fingerprint()}。文件被改过，不要用它训练。"
            )
        return split


def _group_of(sample: UnifiedSample) -> str:
    """样本的分组键。见模块文档「按会话切」。

    ScreenAgent 有 session_id；其他数据集没有会话概念，退回到截图——
    同一张截图的多条标注仍然不该跨集合，这是同一个道理的弱化版。
    """
    session = sample.meta.get("session_id")
    if session:
        return f"session:{session}"
    return f"image:{sample.screenshot_path}"


def freeze_split(
    pool: Sequence[UnifiedSample],
    *,
    val_size: int = TARGET_VAL,
    seed: int = SPLIT_SEED,
) -> FrozenSplit:
    """按会话切出验证集，其余为训练集。

    做法：把会话打乱，按顺序往验证集里放，直到样本数达到或超过实际目标。
    最后一个会话可能让验证集略微超出目标值——接受这一点，因为拆开会话就是
    泄漏。

    `val_size` 是**目标**不是承诺：实际取值还受 `MAX_VAL_RATIO` 约束，
    见该常量的说明。
    """
    groups: dict[str, list[UnifiedSample]] = {}
    for sample in pool:
        groups.setdefault(_group_of(sample), []).append(sample)

    effective_val = min(val_size, int(len(pool) * MAX_VAL_RATIO))

    names = sorted(groups)  # 先排序，让 shuffle 的结果不依赖输入顺序
    random.Random(seed).shuffle(names)

    val_groups: list[str] = []
    count = 0
    for name in names:
        if count >= effective_val:
            break
        val_groups.append(name)
        count += len(groups[name])
    train_groups = [n for n in names if n not in set(val_groups)]

    split = FrozenSplit(
        seed=seed,
        group_by="session",
        train_ids=[s.sample_id for n in train_groups for s in groups[n]],
        val_ids=[s.sample_id for n in val_groups for s in groups[n]],
        train_groups=train_groups,
        val_groups=val_groups,
        pool_size=len(pool),
        requested_val=val_size,
    )

    if split.leakage():
        raise AssertionError(f"划分泄漏：{len(split.leakage())} 条样本同时出现在两侧")

    if split.train_size < TARGET_TRAIN_MIN:
        split.notes.append(
            f"训练集 {split.train_size} 条，低于大纲目标下限 {TARGET_TRAIN_MIN} 条"
            f"（缺口 {TARGET_TRAIN_MIN - split.train_size} 条）。"
            "M3 的微调目标需据此下调，见 docs/m3-prereq/"
        )
    if effective_val < val_size:
        split.notes.append(
            f"验证集目标由 {val_size} 条下调为 {effective_val} 条：池子只有 "
            f"{len(pool)} 条，照搬 {val_size} 会让验证集占到 {val_size / len(pool):.0%}，"
            f"训练集反而更小。按 MAX_VAL_RATIO={MAX_VAL_RATIO:.0%} 取上限"
        )
    if split.val_size != effective_val:
        split.notes.append(
            f"验证集实得 {split.val_size} 条，目标 {effective_val} 条——"
            "按会话切分不拆会话，条数取最接近值"
        )
    return split


__all__ = [
    "SPLIT_SEED",
    "TARGET_TRAIN_MAX",
    "TARGET_TRAIN_MIN",
    "TARGET_VAL",
    "FrozenSplit",
    "freeze_split",
    "training_pool",
]
