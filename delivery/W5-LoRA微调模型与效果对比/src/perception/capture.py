"""截图模块。

## 三个引擎，按实测速度排序

M1 的验收标准是单帧低于 15ms。在本机（2560×1600）上实测：

| 引擎 | 机制 | 全屏 p50 | 能否达标 |
|---|---|---|---|
| **dxcam** | DXGI 桌面复制，GPU 侧 | **~7ms** | ✅ |
| mss | GDI BitBlt，CPU 侧 | ~37ms | ❌ |
| PyAutoGUI | PIL ImageGrab | 更慢 | ❌ |

**mss 在 2560×1600 上达不到 15ms**，这是实测结论而非猜测：`grab()` 本身
就要 41ms，与后续转换无关。分辨率越高差距越大——1920×1080 下 mss 约
24ms，仍然超标。因此 Windows 上优先用 dxcam，mss 降为 fallback。

## dxcam 必须用连续捕获模式

DXGI 桌面复制的 `grab()` 在**画面无变化时直接返回 None**（静止屏幕下
实测 60 次有 53 次返回 None）。一次性 grab 因此不可靠：Agent 需要的是
"当前屏幕长什么样"，而不是"屏幕刚才变了吗"。

正确用法是 `start()` 起后台捕获线程 + `get_latest_frame()` 取最新帧，
后者必定返回有效帧。代价是常驻一个线程，收益是 7ms 且语义正确。

需要注意：这样测出的 7ms 是**取帧耗时**（一次显存到内存的拷贝），不是
从画面变化到拿到帧的端到端延迟。后者还要加上显示器刷新周期（60Hz 下
约 16ms）。M4 做自适应等待时用的是端到端延迟，不能直接套这个数。

## 图像格式

统一为 **BGR 三通道 ndarray**（OpenCV 约定），因为 M1 的预处理链和 M4 的
帧差检测都在 OpenCV 里做。需要 PIL 时用 `Screenshot.to_pil()`。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from perception.dpi import enable_dpi_awareness
from perception.types import BBox

logger = logging.getLogger(__name__)


@dataclass
class Screenshot:
    """一帧截图及其元信息。

    `region` 记录这帧对应的**真实屏幕区域**（含多显示器偏移），
    `CoordinateScaler` 依赖它把模型坐标还原回屏幕坐标。
    """

    image: np.ndarray  # (H, W, 3) BGR
    region: BBox
    engine: str
    captured_at: float = field(default_factory=time.time)
    latency_ms: float = 0.0

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]

    def to_pil(self):
        """转为 PIL Image（RGB）。绘制中文标签时需要。"""
        from PIL import Image

        return Image.fromarray(self.image[:, :, ::-1])

    def save(self, path: str) -> str:
        """存盘。返回实际写入的路径。

        **不用裸 `cv2.imwrite`**：它在中文路径下会静默失败——不抛异常、
        不报错，只是不写文件。Agent Loop 每步要存前后两帧，踩上这个就是
        整条轨迹一张图都没有，而且要等到回放时才发现。

        绕法与 `perception.visualizer.save_annotated` 一致：先 imencode
        到内存再用 Python 的文件对象写出去。
        """
        import os

        import cv2

        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        ok, buffer = cv2.imencode(os.path.splitext(path)[1] or ".png", self.image)
        if not ok:
            raise RuntimeError(f"编码截图失败：{path}")
        with open(path, "wb") as handle:
            handle.write(buffer.tobytes())
        return path

    def resize_to(self, width: int, height: int) -> np.ndarray:
        """缩放到模型画布尺寸。

        缩小用 INTER_AREA——它对每个目标像素做区域平均，比默认的双线性
        更好地保留文字笔画，而 GUI 截图几乎全是细小文字。
        """
        import cv2

        interp = (
            cv2.INTER_AREA if (width < self.width or height < self.height) else cv2.INTER_LINEAR
        )
        return cv2.resize(self.image, (width, height), interpolation=interp)


class CaptureError(RuntimeError):
    """所有截图引擎都不可用。"""


class _DXCamEngine:
    """DXGI 桌面复制。Windows 上的首选，见模块文档。

    ## 用一次性 grab + 自己缓存，而不是后台捕获线程

    本机实测三种策略（2560×1600，165Hz 屏）：

    | 策略 | p50 | p95 |
    |---|---|---|
    | `start(target_fps=60)` + `get_latest_frame()` | 16.63ms | 17.84ms |
    | `start(target_fps=144)` + `get_latest_frame()` | 7.13ms | 13.56ms |
    | **一次性 `grab()` + 缓存** | **0.14ms** | **4.14ms** |

    `get_latest_frame()` 是**帧同步**的——它阻塞等下一帧，因此耗时被显示
    器刷新周期钉住（60Hz 下 16.7ms）。这解释了第一行为什么正好是一个帧周期。

    而 DXGI 在 `grab()` 返回 `None` 时的语义是"**画面自上次抓取以来没有
    变化**"。既然没变化，上一帧就是当前画面，直接复用即可——不需要等，
    也不会拿到过期数据。这个语义让缓存策略既最快又正确。

    唯一要小心的是**刚发出动作之后**：界面可能正在重绘、DXGI 还没投递新帧，
    此时复用缓存会拿到动作前的画面。因此提供 `fresh=True` 强制等新帧
    （实测 p50 5.96ms），供动作后的观察使用。
    """

    name = "dxcam"

    #: 等新帧的上限。超时说明画面确实没变，返回缓存
    FRESH_TIMEOUT_S = 0.5

    def __init__(self) -> None:
        import dxcam

        # dxcam 对同一 device/output 是单例：重复 create 会返回已有实例。
        # 一个进程本来就只该有一个捕获器，因此不做规避。
        self._camera = dxcam.create(output_color="BGR")
        if self._camera is None:
            raise RuntimeError("dxcam.create() 返回 None，DXGI 桌面复制不可用")

        self._cache: np.ndarray | None = None
        self._reused = 0
        self._fetched = 0

        # 首帧必须拿到，否则后续没有可复用的缓存
        deadline = time.perf_counter() + 2.0
        while self._cache is None and time.perf_counter() < deadline:
            self._cache = self._camera.grab()
        if self._cache is None:
            raise RuntimeError("dxcam 两秒内取不到首帧，DXGI 可能被独占或不可用")

        self._height, self._width = self._cache.shape[:2]

    def monitors(self) -> list[dict]:
        full = {"left": 0, "top": 0, "width": self._width, "height": self._height}
        return [full, full]

    def grab(self, region: dict, fresh: bool = False) -> np.ndarray:
        if fresh:
            deadline = time.perf_counter() + self.FRESH_TIMEOUT_S
            frame = None
            while frame is None and time.perf_counter() < deadline:
                frame = self._camera.grab()
            if frame is not None:
                self._cache = frame
                self._fetched += 1
            else:
                # 超时不是错误：说明这段时间画面确实没变
                self._reused += 1
                logger.debug("等新帧超时 %.0fms，画面未变化，复用缓存", self.FRESH_TIMEOUT_S * 1000)
        else:
            frame = self._camera.grab()
            if frame is None:
                self._reused += 1
            else:
                self._cache = frame
                self._fetched += 1

        return self._crop(self._cache, region)

    def _crop(self, frame: np.ndarray, region: dict) -> np.ndarray:
        left, top = region["left"], region["top"]
        right, bottom = left + region["width"], top + region["height"]
        if (left, top, right, bottom) == (0, 0, self._width, self._height):
            return frame
        # 裁切产生非连续视图，OpenCV 的多数算子要求连续内存
        return np.ascontiguousarray(frame[top:bottom, left:right])

    def stats(self) -> dict:
        total = self._fetched + self._reused
        return {
            "fetched": self._fetched,
            "reused_cache": self._reused,
            "reuse_rate": round(self._reused / total, 3) if total else 0.0,
        }

    def close(self) -> None:
        self._cache = None


class _MSSEngine:
    """GDI BitBlt。跨平台可用，但在高分辨率下达不到 15ms 预算。"""

    name = "mss"

    def __init__(self) -> None:
        import mss  # 延迟导入：未安装时只在实例化处报错，不影响模块导入

        self._sct = mss.mss()

    def monitors(self) -> list[dict]:
        return list(self._sct.monitors)

    def grab(self, region: dict, fresh: bool = False) -> np.ndarray:
        # GDI 每次都是实抓，没有"复用上一帧"的概念，fresh 无意义
        import cv2

        raw = self._sct.grab(region)
        # mss 的 ScreenShot 支持缓冲区协议，np.asarray 是零拷贝视图；
        # 而 np.frombuffer(raw.bgra, ...) 会先触发 .bgra 属性构造一份
        # bytes（本机实测多花 7ms）。
        #
        # BGRA → BGR 用 cv2.cvtColor 而不是 arr[:, :, :3] + ascontiguousarray：
        # 后者在 2560×1600 上实测 24ms，前者 7ms —— numpy 的跨步切片拷贝
        # 打不过 OpenCV 的 SIMD 实现。
        return cv2.cvtColor(np.asarray(raw), cv2.COLOR_BGRA2BGR)

    def close(self) -> None:
        self._sct.close()


class _PyAutoGUIEngine:
    """fallback。比 mss 慢得多，只在 mss 不可用时启用。"""

    name = "pyautogui"

    def __init__(self) -> None:
        import pyautogui

        self._pyautogui = pyautogui

    def monitors(self) -> list[dict]:
        w, h = self._pyautogui.size()
        full = {"left": 0, "top": 0, "width": int(w), "height": int(h)}
        return [full, full]

    def grab(self, region: dict, fresh: bool = False) -> np.ndarray:
        shot = self._pyautogui.screenshot(
            region=(region["left"], region["top"], region["width"], region["height"])
        )
        return np.ascontiguousarray(np.array(shot)[:, :, ::-1])  # RGB → BGR

    def close(self) -> None:
        pass


class ScreenCapturer:
    """统一截图入口。

    实例化时会自动声明 DPI 感知——**这必须发生在任何截图之前**，否则在
    非 100% 缩放的机器上截图尺寸与坐标查询结果会不一致（见 `perception.dpi`）。
    """

    #: 按实测速度排序。dxcam 只在 Windows 且有 DXGI 时可用，失败自动降级
    ENGINE_ORDER = (_DXCamEngine, _MSSEngine, _PyAutoGUIEngine)

    def __init__(self, prefer: str | None = None) -> None:
        enable_dpi_awareness()

        order = list(self.ENGINE_ORDER)
        if prefer is not None:
            # 指定引擎时把它提到最前，但保留其余作为降级路径 —— 强制单一
            # 引擎会让"某台机器上 dxcam 不可用"直接变成整个系统起不来
            order.sort(key=lambda cls: cls.name != prefer)

        self._engine: Any = None
        errors: list[str] = []
        for engine_cls in order:
            try:
                self._engine = engine_cls()
                logger.info("截图引擎：%s", engine_cls.name)
                break
            except Exception as exc:  # noqa: BLE001 —— 任何导入/初始化失败都应降级
                errors.append(f"{engine_cls.name}: {exc}")
                logger.warning("截图引擎 %s 不可用：%s", engine_cls.name, exc)

        if self._engine is None:
            raise CaptureError("没有可用的截图引擎。\n" + "\n".join(errors))

    @property
    def engine_name(self) -> str:
        return self._engine.name

    # ------------------------------------------------------------------ #
    # 显示器
    # ------------------------------------------------------------------ #

    def list_monitors(self) -> list[BBox]:
        """全部显示器区域。索引 0 是所有显示器合成的虚拟桌面，1 起为各显示器。"""
        return [
            BBox.from_xywh(m["left"], m["top"], m["width"], m["height"])
            for m in self._engine.monitors()
        ]

    def monitor_region(self, index: int = 1) -> BBox:
        """指定显示器的区域。默认 1 = 主显示器。"""
        monitors = self.list_monitors()
        if not 0 <= index < len(monitors):
            raise IndexError(f"显示器索引 {index} 越界，共 {len(monitors)} 项（0 为虚拟桌面）")
        return monitors[index]

    # ------------------------------------------------------------------ #
    # 截图
    # ------------------------------------------------------------------ #

    def capture(self, monitor: int = 1, fresh: bool = False) -> Screenshot:
        """截取整个显示器。"""
        return self.capture_region(self.monitor_region(monitor), fresh=fresh)

    def capture_full(self, fresh: bool = False) -> Screenshot:
        """截取全部显示器合成的虚拟桌面。"""
        return self.capture_region(self.monitor_region(0), fresh=fresh)

    def capture_region(self, region: BBox, fresh: bool = False) -> Screenshot:
        """截取指定的真实屏幕区域。

        `fresh=True` 时强制等待一帧新画面。**发出动作之后的观察必须用它**
        ——界面可能正在重绘，此时直接取会拿到动作前的画面。代价是多花
        一个帧周期（本机实测 p50 5.96ms），只有 dxcam 引擎有这个区分。
        """
        if region.width <= 0 or region.height <= 0:
            raise ValueError(f"截图区域非法：{region!r}")

        start = time.perf_counter()
        image = self._engine.grab(
            {
                "left": region.left,
                "top": region.top,
                "width": region.width,
                "height": region.height,
            },
            fresh=fresh,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        return Screenshot(
            image=image,
            region=region,
            engine=self._engine.name,
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------ #
    # 基准测试
    # ------------------------------------------------------------------ #

    def benchmark(self, n: int = 100, monitor: int = 1, fresh: bool = False) -> dict:
        """连续截图 n 次，返回延迟统计（毫秒）。

        验收标准是单帧低于 15ms。**看 p95 而不是均值**——偶发的慢帧会让
        Agent 的动作时序错乱，而均值会把它们平均掉。

        `fresh=False`（默认）测的是"取当前画面"的耗时，这是 Agent 大多数
        时候做的事；`fresh=True` 测的是"等一帧新画面"，下限是显示器的刷新
        周期，**不可能低于它**——这是物理限制，不是实现问题。
        """
        region = self.monitor_region(monitor)
        self.capture_region(region, fresh=fresh)  # 预热，排除首帧的初始化开销

        samples = [self.capture_region(region, fresh=fresh).latency_ms for _ in range(n)]
        samples.sort()
        result = {
            "engine": self._engine.name,
            "n": n,
            "mode": "fresh" if fresh else "latest",
            "resolution": f"{region.width}×{region.height}",
            "mean_ms": round(sum(samples) / len(samples), 3),
            "p50_ms": round(samples[len(samples) // 2], 3),
            "p95_ms": round(samples[int(len(samples) * 0.95)], 3),
            "min_ms": round(samples[0], 3),
            "max_ms": round(samples[-1], 3),
        }
        if hasattr(self._engine, "stats"):
            result.update(self._engine.stats())
        return result

    # ------------------------------------------------------------------ #

    def close(self) -> None:
        if self._engine is not None:
            self._engine.close()

    def __enter__(self) -> ScreenCapturer:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
