"""OCR 预处理链。

链路：灰度化 → CLAHE 对比度增强 → 2× 超分 → 自适应二值化。

## 每一步都可以单独关掉，而且默认不全开

这不是偷懒。**现代 OCR 模型（PaddleOCR / EasyOCR 的检测识别网络）是在自然
图像上训练的，喂给它二值化后的图往往比原图更差**——笔画被阈值切断、抗
锯齿信息丢失，模型反而认不出来。经典的"二值化提升 OCR"经验来自 Tesseract
时代的传统算法，对深度学习 OCR 不一定成立。

因此默认配置只做温和的增强（灰度 + CLAHE），二值化与超分默认关闭。
真正该开哪几步，由 M1 的双引擎对照实验和 M4 的准确率优化用**实测失败
案例**来决定——M4 明确要求"改动必须有失败案例支撑，禁止凭直觉调参数"。

## 2× 超分为什么从默认里去掉了（M1 实测）

同一张 2560×1600 桌面截图，PaddleOCR CPU，`min_confidence=0.5`：

| 配置 | 耗时 | 框数 | 平均置信度 |
|---|---|---|---|
| `UPSCALE2X`（灰度+CLAHE+2× 超分） | 93.32s | 65 | — |
| 灰度+CLAHE，不超分 | 43.51s | 64 | 0.9804 |
| `PASSTHROUGH`（空跑） | **42.08s** | 64 | **0.9873** |

2× 超分用 **+51 秒换 +1 个框**，CLAHE 还把平均置信度压低了 0.7 个点。

原因是尺度错配：2560×1600 放大成 5120×3200 后超过 PaddleOCR 的
``max_side_limit=4000``，它自己又缩回来——放大再缩小，白付一次三次样条
插值的钱，信息一点没多。"放大能救小字"这条经验对**局部裁剪**成立，对
整屏截图不成立。M4 若要对小控件做局部 OCR，应在裁剪后单独开超分，而不
是把整屏图放大。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PreprocessConfig:
    """预处理链配置。

    默认值是保守的：只做不会丢信息的增强。
    """

    #: 转灰度。彩色转灰度对文字识别几乎无损，且能省 2/3 的计算
    grayscale: bool = True
    #: CLAHE 限制对比度自适应直方图均衡。对深色主题、低对比度界面有效
    clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8
    #: 放大倍数。**默认 1.0（不放大）**，理由见模块文档的实测表。
    #: 小字号确实是 GUI 截图 OCR 的主要失败源，但那条经验适用于**局部裁剪**；
    #: 对整屏截图，放大会先撞上 OCR 自己的 max_side_limit 再被缩回去。
    upscale: float = 1.0
    #: 自适应二值化。**默认关闭**，理由见模块文档
    binarize: bool = False
    binarize_block_size: int = 11
    binarize_c: int = 2
    #: 轻度去噪。截图几乎无噪声，默认关闭
    denoise: bool = False

    def enabled_steps(self) -> list[str]:
        """本次实际会执行的步骤，写进日志便于复盘。"""
        steps = []
        if self.grayscale:
            steps.append("grayscale")
        if self.clahe:
            steps.append(f"clahe(clip={self.clahe_clip_limit})")
        if self.upscale != 1.0:
            steps.append(f"upscale({self.upscale}x)")
        if self.binarize:
            steps.append("binarize")
        if self.denoise:
            steps.append("denoise")
        return steps


#: 什么都不做。双引擎对照实验的基线组——**必须有这一组**，否则无法证明
#: 预处理到底是帮忙还是帮倒忙。
PASSTHROUGH = PreprocessConfig(grayscale=False, clahe=False, upscale=1.0)

#: 默认：温和增强，**不放大**
DEFAULT = PreprocessConfig()

#: 灰度 + CLAHE + 2× 超分。M1 之前的默认值，现降级为对照实验的一个臂——
#: 实测它在整屏截图上是净亏损（见模块文档），保留是为了让这个结论可复现。
UPSCALE2X = PreprocessConfig(upscale=2.0)

#: 激进：全开。留给 M4 在有失败案例支撑时试
AGGRESSIVE = PreprocessConfig(binarize=True, denoise=True, upscale=2.0)


def preprocess(image: np.ndarray, config: PreprocessConfig = DEFAULT) -> np.ndarray:
    """按配置执行预处理链。

    输入 BGR 或灰度 ndarray，输出处理后的 ndarray。**不修改输入。**

    注意返回的图像尺寸可能因超分而变化——调用方若要把 OCR 结果映射回
    原图坐标，必须除以 ``config.upscale``。`OCREngine` 已经代为处理，
    直接用 `perception.ocr_engine` 的话不需要自己换算。
    """
    import cv2

    result = image.copy()

    if config.grayscale and result.ndim == 3:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    if config.clahe:
        clahe = cv2.createCLAHE(
            clipLimit=config.clahe_clip_limit,
            tileGridSize=(config.clahe_grid_size, config.clahe_grid_size),
        )
        if result.ndim == 2:
            result = clahe.apply(result)
        else:
            # 彩色图在 LAB 空间只对亮度通道做均衡，避免色偏
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if config.upscale != 1.0:
        new_size = (
            int(result.shape[1] * config.upscale),
            int(result.shape[0] * config.upscale),
        )
        # 放大用 CUBIC：比 LINEAR 保边更好，比 LANCZOS 快
        result = cv2.resize(result, new_size, interpolation=cv2.INTER_CUBIC)

    if config.denoise:
        if result.ndim == 2:
            result = cv2.fastNlMeansDenoising(result, h=7)
        else:
            result = cv2.fastNlMeansDenoisingColored(result, h=7)

    if config.binarize:
        if result.ndim == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        result = cv2.adaptiveThreshold(
            result,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            config.binarize_block_size,
            config.binarize_c,
        )

    return result
