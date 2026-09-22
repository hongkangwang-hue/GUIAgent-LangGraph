"""数据集装载器 —— 把各家原始格式翻译成 `UnifiedSample`。

## 装载器为什么要能"不可用"

三个数据集的获取方式完全不同：ScreenSpot 走 HuggingFace，ScreenAgent 只
在作者的 GitHub 仓库里（HF 上只有权重，没有数据集），ScreenSpot-v2 的图片
是一个 1.3GB 的 zip 要另行解压。任何一个没准备好都不该让整条流水线崩掉——
统计报告里写"ScreenAgent 未就绪"远比一个 FileNotFoundError 有用。

所以每个装载器都有 `available()`，`build_all()` 只装载就绪的那些，并把
缺席者记进结果。

## 装载器不做清洗

装载器只负责"翻译"，越界框、超小元素一律照原样放行，交给 `data.clean`。
这样清洗规则的剔除量才是可统计、可写进报告的数字；如果装载器顺手扔掉了
一些，报告里的"剔除 N 条"就不完整了。
"""

from __future__ import annotations

from data.loaders.base import DatasetLoader, LoaderResult
from data.loaders.screenagent import ScreenAgentLoader
from data.loaders.screenspot import ScreenSpotLoader, ScreenSpotV2Loader

#: 深度处理的数据集。大纲指定 ScreenSpot / ScreenSpot-v2 作为 M3 定位评测集，
#: ScreenAgent 作为 grounding 训练样本来源。
#: Mind2Web / WebArena 只抽样查看，见 `data.survey`，不进这个注册表。
LOADERS: dict[str, type[DatasetLoader]] = {
    "screenspot": ScreenSpotLoader,
    "screenspot_v2": ScreenSpotV2Loader,
    "screenagent": ScreenAgentLoader,
}


def build_all(root: str = "data/raw", **kwargs) -> list[LoaderResult]:
    """装载所有就绪的数据集，缺席的也返回一条记录说明原因。"""
    results = []
    for cls in LOADERS.values():
        loader = cls(root=root, **kwargs)
        results.append(loader.run())
    return results


__all__ = [
    "LOADERS",
    "DatasetLoader",
    "LoaderResult",
    "ScreenAgentLoader",
    "ScreenSpotLoader",
    "ScreenSpotV2Loader",
    "build_all",
]
