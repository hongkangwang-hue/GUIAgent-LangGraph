"""装载器基类。"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from data.schema import UnifiedSample

logger = logging.getLogger(__name__)


@dataclass
class LoaderResult:
    """一次装载的结果。

    `samples` 为空不代表失败——也可能是数据集没下载。两者的区别写在
    `available` 与 `reason` 里，报告要据此区分"这个数据集没有可用样本"
    和"这个数据集我们还没准备好"。
    """

    dataset: str
    available: bool
    samples: list[UnifiedSample] = field(default_factory=list)
    reason: str = ""
    #: 装载过程中跳过的条目及原因（缺图、字段缺失等）。清洗之前就丢掉的那些。
    skipped: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.samples)


class DatasetLoader(ABC):
    """一个数据集的装载器。

    子类只需实现 `available()` 与 `load()`；`run()` 负责统一异常处理，
    保证单个数据集出问题不会带倒整条流水线。
    """

    #: 注册名，也是 `UnifiedSample.source_dataset` 的取值
    name: str = ""

    def __init__(self, root: str | Path = "data/raw", **kwargs) -> None:
        self.root = Path(root)
        self.options = kwargs

    @property
    def dataset_dir(self) -> Path:
        return self.root / self.name

    @abstractmethod
    def available(self) -> tuple[bool, str]:
        """数据是否就绪。返回 (就绪, 未就绪时的原因说明)。

        原因要能直接告诉人下一步做什么，而不是只说"文件不存在"。
        """

    @abstractmethod
    def load(self) -> Iterator[UnifiedSample]:
        """逐条产出统一样本。用生成器：ScreenSpot 的图是内嵌在 parquet 里的，
        全量读进内存是 1GB 量级。"""

    def run(self) -> LoaderResult:
        ok, reason = self.available()
        if not ok:
            logger.warning("数据集 %s 未就绪：%s", self.name, reason)
            return LoaderResult(dataset=self.name, available=False, reason=reason)
        self._skipped: dict[str, int] = {}
        samples = list(self.load())
        return LoaderResult(
            dataset=self.name,
            available=True,
            samples=samples,
            skipped=dict(getattr(self, "_skipped", {})),
        )

    def _skip(self, reason: str) -> None:
        """记一次跳过。计数而不是逐条打日志——ScreenSpot 一个越界字段能刷几百行。"""
        store = getattr(self, "_skipped", None)
        if store is None:
            store = self._skipped = {}
        store[reason] = store.get(reason, 0) + 1


def image_size(path: str | Path) -> tuple[int, int] | None:
    """读图片尺寸。

    用 PIL 而不是 OpenCV：`Image.open` 只解析文件头就能给出尺寸，不解码
    像素。全量解码 2000+ 张 1024×768 与 1300 张桌面截图纯属浪费，而且
    `cv2.imread` 在中文路径上会静默返回 None（本项目在 M1 已经踩过一次）。
    """
    try:
        from PIL import Image

        with Image.open(path) as img:
            return img.size
    except Exception as exc:  # noqa: BLE001 —— 坏图不该带倒整个装载
        logger.debug("读取图片尺寸失败 %s: %s", path, exc)
        return None
