"""调试可视化：把识别结果画回截图。

M1 阶段还没有大模型参与，**感知效果好不好只能靠人眼判断**。这个模块是
M1 唯一的评估手段，也是交付物《感知效果可视化图集》（不少于 10 张标注
截图）的产出工具。

## 中文标签必须用 PIL 画

OpenCV 的 `cv2.putText` 只认 ASCII，中文会画成一串问号。Windows 上的
UI 文字大量是中文，用 OpenCV 画等于什么都看不见。因此绘制走 PIL，
只在最后转回 ndarray。
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np

from perception.types import ElementSource, UIElement

logger = logging.getLogger(__name__)

#: 按来源着色（RGB）。UIA 绿、OCR 蓝，与 M1 设计思路一致
SOURCE_COLORS: dict[ElementSource, tuple[int, int, int]] = {
    ElementSource.UIA: (46, 204, 113),
    ElementSource.OCR: (52, 152, 219),
}
FALLBACK_COLOR = (149, 165, 166)

#: 中文字体候选，按优先级。找不到就退回 PIL 默认字体（中文会变方块，
#: 但至少边界框还在，不至于整张图作废）
FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",      # 微软雅黑
    r"C:\Windows\Fonts\msyhl.ttc",
    r"C:\Windows\Fonts\simhei.ttf",    # 黑体
    r"C:\Windows\Fonts\simsun.ttc",    # 宋体
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/System/Library/Fonts/PingFang.ttc",
)

_font_cache: dict[int, object] = {}


def _load_font(size: int):
    """加载中文字体，结果缓存——每次绘制都重新加载字体会明显拖慢批量出图。"""
    if size in _font_cache:
        return _font_cache[size]

    from PIL import ImageFont

    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                _font_cache[size] = font
                return font
            except OSError as exc:
                logger.debug("字体 %s 加载失败：%s", path, exc)

    logger.warning(
        "未找到中文字体（已尝试 %d 个候选），标签中的中文将显示为方块。"
        "平台：%s", len(FONT_CANDIDATES), sys.platform,
    )
    font = ImageFont.load_default()
    _font_cache[size] = font
    return font


def draw_elements(
    image: np.ndarray,
    elements: list[UIElement],
    origin: tuple[int, int] = (0, 0),
    font_size: int = 14,
    box_width: int = 2,
    show_labels: bool = True,
    show_click_points: bool = True,
) -> np.ndarray:
    """在截图上画出识别结果，返回新的 BGR ndarray。

    `origin` 是这张图对应的**屏幕区域左上角**。`UIElement` 的坐标是屏幕
    绝对坐标，画到图上需要减掉这个偏移——截全屏时是 (0,0)，截区域或副
    显示器时不是。传错了整张图的框会集体偏移。
    """
    from PIL import Image, ImageDraw

    pil_image = Image.fromarray(image[:, :, ::-1] if image.ndim == 3 else image).convert("RGB")
    draw = ImageDraw.Draw(pil_image)
    font = _load_font(font_size)
    off_x, off_y = origin

    for element in elements:
        color = SOURCE_COLORS.get(element.source, FALLBACK_COLOR)
        box = element.bbox
        left, top = box.left - off_x, box.top - off_y
        right, bottom = box.right - off_x, box.bottom - off_y

        draw.rectangle([left, top, right, bottom], outline=color, width=box_width)

        if show_click_points:
            center = element.click_point
            cx, cy = center.x - off_x, center.y - off_y
            draw.line([cx - 4, cy, cx + 4, cy], fill=color, width=1)
            draw.line([cx, cy - 4, cx, cy + 4], fill=color, width=1)

        if show_labels:
            _draw_label(draw, element.label(), left, top, color, font, pil_image.height)

    return np.array(pil_image)[:, :, ::-1]


def _draw_label(draw, text: str, left: int, top: int, color, font, image_height: int) -> None:
    """在框的左上角外侧画带底色的标签。

    贴着框顶部画，顶部空间不够时改画在框内——否则最上排的元素标签会
    被裁掉，而最上排往往是菜单栏这类关键控件。
    """
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:  # 老版本 PIL 没有 textbbox
        text_w, text_h = draw.textsize(text, font=font)

    pad = 2
    label_top = top - text_h - pad * 2
    if label_top < 0:
        label_top = min(top, image_height - text_h - pad * 2)

    draw.rectangle(
        [left, label_top, left + text_w + pad * 2, label_top + text_h + pad * 2],
        fill=color,
    )
    draw.text((left + pad, label_top + pad), text, fill=(255, 255, 255), font=font)


def draw_legend(image: np.ndarray, stats: dict | None = None) -> np.ndarray:
    """左上角画来源图例与统计摘要。

    图集里每张图都该带上这个——离开上下文看图时，"绿框是 UIA、蓝框是
    OCR、这次融合后剩多少个"这些信息不该靠记忆。
    """
    from PIL import Image, ImageDraw

    pil_image = Image.fromarray(image[:, :, ::-1]).convert("RGB")
    draw = ImageDraw.Draw(pil_image)
    font = _load_font(14)

    lines = [("UIA（无障碍树）", SOURCE_COLORS[ElementSource.UIA]),
             ("OCR（文字识别）", SOURCE_COLORS[ElementSource.OCR])]
    if stats:
        summary = f"UIA {stats.get('uia_raw', 0)} + OCR {stats.get('ocr_raw', 0)} → 融合 {stats.get('fused', 0)}"
        lines.append((summary, (255, 255, 255)))
        lines.append((f"耗时 {stats.get('total_ms', 0)}ms", (255, 255, 255)))

    box_h = 22 * len(lines) + 12
    draw.rectangle([8, 8, 320, 8 + box_h], fill=(0, 0, 0))
    for i, (text, color) in enumerate(lines):
        y = 14 + i * 22
        draw.rectangle([16, y + 3, 30, y + 15], fill=color)
        draw.text((38, y), text, fill=(255, 255, 255), font=font)

    return np.array(pil_image)[:, :, ::-1]


def save_annotated(
    image: np.ndarray,
    elements: list[UIElement],
    path: str,
    origin: tuple[int, int] = (0, 0),
    stats: dict | None = None,
) -> str:
    """画好并保存。返回实际写入的路径。"""
    import cv2

    annotated = draw_elements(image, elements, origin=origin)
    annotated = draw_legend(annotated, stats)

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    # 中文路径下 cv2.imwrite 会静默失败，走 imencode + 文件写入绕开
    ok, buffer = cv2.imencode(os.path.splitext(path)[1] or ".png", annotated)
    if not ok:
        raise RuntimeError(f"编码图片失败：{path}")
    with open(path, "wb") as handle:
        handle.write(buffer.tobytes())
    logger.info("已保存标注图：%s（%d 个元素）", path, len(elements))
    return path
