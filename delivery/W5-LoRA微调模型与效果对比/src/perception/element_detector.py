"""双通道元素识别：UIA + OCR 并行 → 融合去重。

## 两个通道的坐标系不一样，这是最容易埋雷的地方

- **UIA 给的是屏幕绝对坐标**（虚拟桌面原点起算）
- **OCR 给的是图像相对坐标**（截图区域左上角为 0,0）

截全屏时两者恰好重合，于是这个 bug 在开发期不会暴露；一旦改成截某个
区域或副显示器，OCR 的框就会整体偏移。本模块在融合前**统一把 OCR 结果
平移到屏幕绝对坐标**，并且只在这一处做，避免各处重复换算。

## 并行方式：OCR 进工作线程，UIA 留在调用线程

不是随便选的。`uiautomation` 依赖 COM，而 COM 有线程套间（apartment）
的概念——在没有 `CoInitialize` 的工作线程里调 UIA 会以各种诡异的方式
失败。OCR 是纯计算，进哪个线程都无所谓。因此把 OCR 甩出去、UIA 留在
原地，既拿到了并行收益（OCR 是慢的那个，几百毫秒），又完全绕开了 COM
线程模型的坑。
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from perception.capture import Screenshot
from perception.ocr_engine import OCREngine
from perception.types import BBox, ElementSource, UIElement, dedupe_by_iou
from perception.uia_tree import UIATree

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """一次识别的产出与过程数据。

    过程数据不是可选的调试信息：M1 验收标准要求"按来源分项统计召回率"，
    M4 要按失败案例调参数，两者都依赖这里的分项计数。
    """

    elements: list[UIElement]
    region: BBox
    uia_count: int = 0
    ocr_count: int = 0
    uia_ms: float = 0.0
    ocr_ms: float = 0.0
    total_ms: float = 0.0
    uia_stats: dict = field(default_factory=dict)

    @property
    def fused_count(self) -> int:
        return len(self.elements)

    @property
    def dropped_by_dedupe(self) -> int:
        return self.uia_count + self.ocr_count - self.fused_count

    def by_source(self, source: ElementSource) -> list[UIElement]:
        return [e for e in self.elements if e.source is source]

    def summary(self) -> dict:
        """可直接进表格 / 日志的摘要。"""
        return {
            "region": self.region.as_tuple(),
            "uia_raw": self.uia_count,
            "ocr_raw": self.ocr_count,
            "fused": self.fused_count,
            "dropped_by_dedupe": self.dropped_by_dedupe,
            "uia_ms": round(self.uia_ms, 1),
            "ocr_ms": round(self.ocr_ms, 1),
            "total_ms": round(self.total_ms, 1),
            "uia": self.uia_stats,
        }


class ElementDetector:
    """把屏幕像素转成结构化的 UI 元素列表。

    典型用法::

        detector = ElementDetector(ocr_engine=PaddleOCREngine())
        shot = capturer.capture()
        result = detector.detect(shot)
        for element in result.elements:
            print(element.index, element.label(), element.click_point)
    """

    def __init__(
        self,
        ocr_engine: OCREngine | None = None,
        uia_tree: UIATree | None = None,
        iou_threshold: float = 0.5,
        parallel: bool = True,
    ) -> None:
        self.ocr_engine = ocr_engine
        self.uia_tree = uia_tree if uia_tree is not None else UIATree()
        #: 融合去重阈值。M4 会按实测失败案例调这个值——阈值太高会留下
        #: 重复框，太低会把相邻的两个真控件误并成一个
        self.iou_threshold = iou_threshold
        self.parallel = parallel

    # ------------------------------------------------------------------ #

    def detect(
        self,
        screenshot: Screenshot,
        use_uia: bool = True,
        use_ocr: bool = True,
        foreground_only: bool = True,
    ) -> DetectionResult:
        """识别截图中的 UI 元素。

        返回的元素坐标一律是**屏幕绝对坐标**，可以直接交给
        `ActionExecutor` 点击，不需要再做任何换算。
        """
        start = time.perf_counter()
        region = screenshot.region

        ocr_future = None
        executor = None
        if use_ocr and self.ocr_engine is not None and self.parallel:
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr")
            ocr_future = executor.submit(self._run_ocr, screenshot)

        uia_elements: list[UIElement] = []
        uia_ms = 0.0
        uia_stats: dict = {}
        if use_uia and self.uia_tree.is_available():
            uia_start = time.perf_counter()
            try:
                uia_elements = (
                    self.uia_tree.capture_foreground(clip=region)
                    if foreground_only
                    else self.uia_tree.capture_desktop(clip=region)
                )
            except Exception as exc:  # noqa: BLE001 —— UIA 挂掉不该让整次感知失败
                logger.warning("UIA 通道失败，本次仅用 OCR：%s", exc)
            uia_ms = (time.perf_counter() - uia_start) * 1000.0
            uia_stats = self.uia_tree.stats.as_dict()

        ocr_elements: list[UIElement] = []
        ocr_ms = 0.0
        if ocr_future is not None:
            ocr_elements, ocr_ms = ocr_future.result()
        elif use_ocr and self.ocr_engine is not None:
            ocr_elements, ocr_ms = self._run_ocr(screenshot)
        if executor is not None:
            executor.shutdown(wait=False)

        fused = dedupe_by_iou(uia_elements + ocr_elements, iou_threshold=self.iou_threshold)

        result = DetectionResult(
            elements=fused,
            region=region,
            uia_count=len(uia_elements),
            ocr_count=len(ocr_elements),
            uia_ms=uia_ms,
            ocr_ms=ocr_ms,
            total_ms=(time.perf_counter() - start) * 1000.0,
            uia_stats=uia_stats,
        )
        logger.info("元素识别 %s", result.summary())
        return result

    # ------------------------------------------------------------------ #

    def _run_ocr(self, screenshot: Screenshot) -> tuple[list[UIElement], float]:
        """跑 OCR 并把结果平移到屏幕绝对坐标。"""
        start = time.perf_counter()
        try:
            raw = self.ocr_engine.recognize(screenshot.image)
        except Exception as exc:  # noqa: BLE001
            logger.warning("OCR 通道失败，本次仅用 UIA：%s", exc)
            return [], (time.perf_counter() - start) * 1000.0

        offset_x, offset_y = screenshot.region.left, screenshot.region.top
        elements = [
            UIElement(
                bbox=BBox(
                    item.bbox.left + offset_x,
                    item.bbox.top + offset_y,
                    item.bbox.right + offset_x,
                    item.bbox.bottom + offset_y,
                ),
                source=ElementSource.OCR,
                text=item.text,
                confidence=item.confidence,
                meta={"polygon": item.polygon},
            )
            for item in raw
        ]
        return elements, (time.perf_counter() - start) * 1000.0
