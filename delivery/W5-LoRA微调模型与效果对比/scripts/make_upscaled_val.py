"""把验证集等比放大，用于验证 x 坐标塌陷是否由**缩放强度**造成。

## 要验的假设

宽高比与提示词两个候选已被排除（见 `docs/m3-letterbox-实验结果.md`）。
重新算 `max_pixels = 896² = 802,816` 的触发情况：

    ScreenAgent 训练 / 离线   786,432 px   缩放系数 1.000   一个像素都不缩
    letterbox 实验          1,048,320    0.875           轻微，无塌陷
    客机 1920×1080          2,073,600    0.622           激进，未测

**训练数据恰好卡在 max_pixels 以下。** 而客机要缩到 0.622，
比 letterbox 那轮激进 2.4 倍——这条线还没测到头。

## 做法

等比放大 1.625 倍：1024×768 → **1664×1248**

    1664 × 1248 = 2,076,672 px   缩放系数 0.6218   ≈ 客机的 0.622

**宽高比保持 4:3 不变**，与已排除的宽高比变量彻底分开。

## 为什么这次不用变换坐标

等比放大不改变归一化坐标：原图 x_norm 对应像素 x_norm/1000*1024，
放大后是 x_norm/1000*1664，归一回去仍是 x_norm。

> letterbox 那轮因为补黑边改变了内容占比，必须变换四处真值字段
> （`point_norm` / `point_px` / `answer` / `params.point`），
> 而第一版漏了 `point_norm`——`eval/action.py` 读的正是它。
> **等比放大天然没有这个风险**，只改图片路径与 `resolution`。

## 用法

    python scripts/make_upscaled_val.py
    python -m eval.action --val finetune/data/val-upscaled-coords.jsonl \
        --local Qwen/Qwen2.5-VL-3B-Instruct \
        --adapter finetune/outputs/20260827-174327/adapter \
        --tag after-upscaled --no-resume
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image

SRC = Path("finetune/data/val-4x3-coords.jsonl")
DST = Path("finetune/data/val-upscaled-coords.jsonl")
IMG_DIR = Path("data/upscaled-0622")
FACTOR = 1.625
MAX_PIXELS = 896 * 896


def main() -> int:
    if not SRC.exists():
        print(f"缺 {SRC}。它由 letterbox 那一轮生成，重跑 scripts/make_letterbox_val.py 即可。")
        return 1

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in SRC.open(encoding="utf-8") if line.strip()]
    print(f"读入 {len(rows)} 条")

    cache: dict[str, tuple[str, int, int]] = {}
    out = []
    for row in rows:
        src = row["image"]
        if src not in cache:
            img = Image.open(src).convert("RGB")
            w, h = img.size
            nw, nh = round(w * FACTOR), round(h * FACTOR)
            big = img.resize((nw, nh), Image.LANCZOS)
            name = f"{abs(hash(src)) % (10**12):012d}.png"
            path = IMG_DIR / name
            big.save(path)
            cache[src] = (str(path).replace("\\", "/"), nw, nh)
        path, nw, nh = cache[src]

        # **坐标一个字都不用改。** 等比放大不改变归一化坐标。
        new_row = dict(row)
        new_row["image"] = path
        new_row["resolution"] = [nw, nh]
        out.append(new_row)

    with DST.open("w", encoding="utf-8", newline="\n") as fh:
        for row in out:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    _, nw, nh = next(iter(cache.values()))
    px = nw * nh
    scale = 1.0 if px <= MAX_PIXELS else 1 / math.sqrt(px / MAX_PIXELS)
    print(f"去重图片 {len(cache)} 张 → {IMG_DIR}")
    print(f"  {nw}x{nh} = {px:,} px   缩放系数 {scale:.4f}   （客机 0.622）")
    print(f"写出 {len(out)} 条 → {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
