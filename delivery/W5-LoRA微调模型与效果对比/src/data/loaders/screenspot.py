"""ScreenSpot 与 ScreenSpot-v2 —— M3 的零样本定位评测集。

## 两个版本的 bbox 格式不一样，这是本模块最要紧的一件事

|  | 来源 | bbox 含义 | 基准 |
|---|---|---|---|
| ScreenSpot   | HF `rootsautomation/ScreenSpot`，parquet + 内嵌 PNG | `[x1, y1, x2, y2]` | **归一化 [0, 1]** |
| ScreenSpot-v2| HF `OS-Copilot/ScreenSpot-v2`，JSON + 图片 zip | `[x, y, w, h]` | **绝对像素** |

同一张截图（如 `pc_ede36f9b-….png`）在两个版本里都出现，标注也高度重合，
很容易想当然地认为格式相同。**假设错了不会报错**，只会让所有框静默错位——
v1 的 0.94 被当成绝对像素就是屏幕最左边，v2 的 910 被当成归一化就直接越界。
两条路径因此分成两个类，各自把格式写死在代码里，而不是靠嗅探。

## 动作类型标成 LOCATE 而不是 CLICK

ScreenSpot 的 `data_type` 是 text / icon，说的是"这是个什么元素"，不是
"要对它做什么"。标成 CLICK 等于凭空替数据集断言这些元素都该被点击，而
"动作类型分布"这张图会因此凭空多出 1272 条 click。原始值保留在
`meta["element_kind"]`。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

from data.loaders.base import DatasetLoader, image_size
from data.schema import ActionType, Platform, UnifiedSample
from perception.types import BBox

logger = logging.getLogger(__name__)

#: 文件名前缀 → 平台。ScreenSpot 两个版本都用这套前缀。
_PREFIX_PLATFORM = {
    "pc": Platform.DESKTOP,
    "mobile": Platform.MOBILE,
    "web": Platform.WEB,
}


def _platform_of(file_name: str) -> Platform:
    return _PREFIX_PLATFORM.get(file_name.split("_", 1)[0], Platform.UNKNOWN)


def _clamped_bbox(
    left: float, top: float, right: float, bottom: float, width: int, height: int
) -> BBox:
    """四舍五入取整并保证 right > left、bottom > top。

    不在这里裁剪越界——越界样本要留给 `data.clean` 统计和剔除。这里只保证
    构造出的 BBox 不会因为 right < left 而抛异常（`perception.types.BBox`
    的不变式），否则越界样本会在装载阶段崩掉，根本走不到清洗那一步。
    """
    x1, y1 = int(round(left)), int(round(top))
    x2, y2 = int(round(right)), int(round(bottom))
    return BBox(x1, y1, max(x2, x1), max(y2, y1))


class ScreenSpotLoader(DatasetLoader):
    """ScreenSpot v1：parquet，bbox 归一化 xyxy，截图以字节内嵌。

    截图会被抽出来落到 `data/raw/screenspot/images/`——统一 schema 的
    `screenshot_path` 必须是一个真能打开的路径，M3 的评测脚本要按它读图。
    610 张不重复截图，抽取一次约十几秒。
    """

    name = "screenspot"

    def available(self) -> tuple[bool, str]:
        shards = sorted(self.dataset_dir.glob("data/*.parquet"))
        if not shards:
            return False, (
                f"未找到 parquet：{self.dataset_dir / 'data'}。"
                "运行 `python scripts/prepare_datasets.py --download` 下载"
            )
        return True, ""

    def load(self) -> Iterator[UnifiedSample]:
        import pandas as pd

        image_dir = self.dataset_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        shards = sorted(self.dataset_dir.glob("data/*.parquet"))
        #: file_name → (width, height)。同一张图被多条标注引用（1272 条标注
        #: 只对应 610 张图），缓存避免反复解码文件头。
        sizes: dict[str, tuple[int, int]] = {}

        for shard in shards:
            frame = pd.read_parquet(shard)
            for index, row in frame.iterrows():
                file_name = str(row["file_name"])
                target = image_dir / file_name

                if file_name not in sizes:
                    if not target.exists():
                        payload = row["image"]
                        raw = payload.get("bytes") if isinstance(payload, dict) else None
                        if not raw:
                            self._skip("截图字节缺失")
                            continue
                        target.write_bytes(raw)
                    size = image_size(target)
                    if size is None:
                        self._skip("截图无法解析")
                        continue
                    sizes[file_name] = size
                width, height = sizes[file_name]

                box = list(row["bbox"])
                if len(box) != 4:
                    self._skip("bbox 字段长度异常")
                    continue
                # 归一化 xyxy → 绝对像素
                bbox = _clamped_bbox(
                    box[0] * width,
                    box[1] * height,
                    box[2] * width,
                    box[3] * height,
                    width,
                    height,
                )

                yield UnifiedSample(
                    sample_id=f"screenspot-{shard.stem}-{index}",
                    screenshot_path=str(target),
                    resolution=(width, height),
                    instruction=str(row["instruction"]),
                    action_type=ActionType.LOCATE,
                    platform=_platform_of(file_name),
                    source_dataset=self.name,
                    bbox=bbox,
                    app=str(row["data_source"]),
                    split="test",
                    meta={
                        "element_kind": str(row["data_type"]),
                        "file_name": file_name,
                        "bbox_normalized": [float(v) for v in box],
                    },
                )


class ScreenSpotV2Loader(DatasetLoader):
    """ScreenSpot-v2：三个 JSON（desktop / mobile / web），bbox 绝对 xywh。

    图片来自 `screenspotv2_image.zip`（1.3GB）。zip 没解压时本装载器报
    未就绪而不是报错——1.3GB 的解压是个需要人知情的动作。
    """

    name = "screenspot_v2"

    _SPLITS = {
        "screenspot_desktop_v2.json": Platform.DESKTOP,
        "screenspot_mobile_v2.json": Platform.MOBILE,
        "screenspot_web_v2.json": Platform.WEB,
    }

    @property
    def image_dir(self) -> Path:
        return self.dataset_dir / "screenspotv2_image"

    def available(self) -> tuple[bool, str]:
        missing = [n for n in self._SPLITS if not (self.dataset_dir / n).exists()]
        if missing:
            return False, f"缺少标注文件 {missing}，运行 prepare_datasets.py --download"
        if not self.image_dir.is_dir():
            return False, (
                f"图片未解压：{self.dataset_dir / 'screenspotv2_image.zip'} "
                f"→ {self.image_dir}（1.3GB，用 --extract 解压）"
            )
        return True, ""

    def load(self) -> Iterator[UnifiedSample]:
        sizes: dict[str, tuple[int, int]] = {}

        for file_name, platform in self._SPLITS.items():
            rows = json.loads((self.dataset_dir / file_name).read_text(encoding="utf-8"))
            for index, row in enumerate(rows):
                image_name = str(row["img_filename"])
                path = self.image_dir / image_name

                if image_name not in sizes:
                    size = image_size(path)
                    if size is None:
                        self._skip("截图缺失或无法解析")
                        continue
                    sizes[image_name] = size
                width, height = sizes[image_name]

                box = list(row["bbox"])
                if len(box) != 4:
                    self._skip("bbox 字段长度异常")
                    continue
                # 绝对 xywh → xyxy
                x, y, w, h = (float(v) for v in box)
                bbox = _clamped_bbox(x, y, x + w, y + h, width, height)

                yield UnifiedSample(
                    sample_id=f"screenspot_v2-{platform.value}-{index}",
                    screenshot_path=str(path),
                    resolution=(width, height),
                    instruction=str(row["instruction"]),
                    action_type=ActionType.LOCATE,
                    platform=platform,
                    source_dataset=self.name,
                    bbox=bbox,
                    app=str(row.get("data_source", "")),
                    split="test",
                    meta={
                        "element_kind": str(row.get("data_type", "")),
                        "file_name": image_name,
                        "bbox_xywh": [x, y, w, h],
                    },
                )
