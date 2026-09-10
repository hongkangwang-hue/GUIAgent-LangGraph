"""屏幕变化检测 —— 判断一次动作到底有没有产生效果。

## 为什么需要它

2026-08-25 的实测（`open_browser`，微调 3B）：

    子任务「点击任务栏上的Microsoft Edge图标」
       left_click (15, 190) ×6      execution_status 全是 ok

**六次点击机械上都"成功"了，屏幕纹丝不动。** 系统里没有任何一处发现
这件事——`execution_status` 只说"键鼠事件发出去了"，不说"发出去之后
有没有用"。

同一批数据里的对照很干净：

    轮次里出现过 double_click    2/2 成功
    全程只有 left_click          0/8 成功        p = 0.0222

单击桌面图标只是选中它，不会打开。**而"选中"和"打开"在
`execution_status` 里长得一模一样。**

## 判据：变化像素的占比，不是"完全一样"

「完全没变」这个判据太严：任务栏的时钟每分钟跳一次，鼠标指针本身也占
几十个像素。反过来，「变了就算有效」又太松：单击桌面图标会给它加个
高亮框，那也是变化，但任务没有推进。

所以量的是**变化像素占全屏的比例**，并给出三档参考量级（本机 1920×1080
实测）：

    指针移动 / 时钟跳字        < 0.1%
    图标高亮 / 按钮悬停        0.2% ~ 0.5%
    菜单展开 / 窗口打开        > 5%

`DEFAULT_THRESHOLD` 取 **1%**：落在"高亮"和"开窗"中间，两边都留了 2 倍
余量。**这个数是拍出来的，不是测出来的**——它需要一批标注过的
（动作，前后截图）样本才能定准，而本项目没有。所以它是可配置的，
且 `ChangeReport` 把 `ratio` 原样带出来，让调用方自己判断。

## 不做的事

**不做结构化对比**（控件树 diff、SSIM、感知哈希）。那些更准，但：

- UIA 穿不透虚拟机边界（M0 全局约束），控件树这条路在本项目里不通
- SSIM 对整体亮度变化敏感，而 GUI 截图经常整块变暗（模态框的遮罩）
- 这里要的只是一个粗判：**"什么都没发生"还是"发生了点什么"**

像素占比够用，而且它的失败形态是可预测的（全屏动画会误判为"有变化"），
比一个说不清什么时候会错的相似度分数好交代。
"""

from __future__ import annotations

from dataclasses import dataclass

#: 变化像素占比超过它就算"屏幕动了"。见模块文档——这个数是拍的，可配置。
DEFAULT_THRESHOLD = 0.01

#: 单个像素灰度差超过它才算"这个像素变了"。压掉 JPEG 噪点与抗锯齿抖动。
PIXEL_TOLERANCE = 24


@dataclass(frozen=True)
class ChangeReport:
    """一次前后对比的结果。

    `ratio` 一定要带出来。只返回一个布尔值的话，阈值定得对不对就再也
    看不出来了——而这个阈值是拍的。
    """

    changed: bool
    ratio: float
    threshold: float
    #: 前后截图尺寸不一致时为真。此时 `changed` 恒为真（分辨率变了当然算变了），
    #: 但要标出来：它多半意味着环境出了状况，不是动作起了效果。
    size_mismatch: bool = False

    def as_dict(self) -> dict:
        return {
            "changed": self.changed,
            "ratio": round(self.ratio, 5),
            "threshold": self.threshold,
            "size_mismatch": self.size_mismatch,
        }


def compare(before, after, threshold: float = DEFAULT_THRESHOLD) -> ChangeReport:
    """对比两张截图（`numpy` 数组或有 `.image` 的 `Screenshot`）。

    任一为 None 时返回 `changed=False`, `ratio=0.0`——**拍不到就当没变化**，
    而不是当成变化了。理由：这个信号是用来触发"换个策略"的，
    宁可漏报（继续原策略）也不要误报（明明没动却以为动了，从而不再升级策略）。
    """
    import numpy as np

    left = _as_array(before)
    right = _as_array(after)
    if left is None or right is None:
        return ChangeReport(changed=False, ratio=0.0, threshold=threshold)

    if left.shape[:2] != right.shape[:2]:
        return ChangeReport(changed=True, ratio=1.0, threshold=threshold, size_mismatch=True)

    diff = np.abs(_gray(left).astype(np.int16) - _gray(right).astype(np.int16))
    ratio = float((diff > PIXEL_TOLERANCE).mean())
    return ChangeReport(changed=ratio > threshold, ratio=ratio, threshold=threshold)


def _as_array(source):
    if source is None:
        return None
    image = getattr(source, "image", source)
    return image if getattr(image, "size", 0) else None


def _gray(image):
    """转灰度。已经是单通道就原样返回。

    不用 `cv2.cvtColor`：这个模块要能在没装 OpenCV 的环境里被导入
    （评测机、CI）。通道平均与加权亮度对"有没有变"这个粗判没有区别。
    """
    if image.ndim == 2:
        return image
    return image.mean(axis=2)
