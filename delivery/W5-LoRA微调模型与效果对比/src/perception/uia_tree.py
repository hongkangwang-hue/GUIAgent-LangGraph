"""Windows UI Automation 元素树抓取。

UIA 给出的是控件的**真实边界与语义类型**，比 OCR 的文字外接框可靠得多，
因此在双通道融合中优先级更高。它也是 M4 Reflector 程序化判据的基础——
"某个按钮是否出现"这种判断，查元素树比让模型看截图既快又准。

## 必须有时间预算

UIA 遍历会**很慢**：一棵完整的桌面元素树可能有上万个节点，某些应用
（Electron、老 WinForms）的跨进程 COM 调用单次就要几十毫秒。不加限制地
深度遍历，一次感知能耗掉十几秒，Agent 的动作时序会完全乱掉。

因此本模块有三重刹车：**深度上限、元素数上限、时间预算**。任何一个触顶
就停止并在结果里标记 `truncated`——宁可给出不完整但及时的结果，也不要
让上层无限等待。三个参数的具体取值留给 M4 用实测失败案例来调，M1 只给
一组保守的默认值。
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass

from perception.types import BBox, ElementSource, UIElement

logger = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"

#: 这些控件类型即使没有 Name 也值得保留——它们是可交互的
INTERACTIVE_TYPES = frozenset(
    {
        "ButtonControl",
        "CheckBoxControl",
        "ComboBoxControl",
        "EditControl",
        "HyperlinkControl",
        "ListItemControl",
        "MenuItemControl",
        "RadioButtonControl",
        "SliderControl",
        "SplitButtonControl",
        "TabItemControl",
        "TreeItemControl",
    }
)

#: 这些是纯容器，本身不可点击，抓来只会污染结果
CONTAINER_TYPES = frozenset(
    {
        "PaneControl",
        "GroupControl",
        "CustomControl",
        "ToolBarControl",
        "TitleBarControl",
    }
)


@dataclass
class UIATraversalStats:
    """一次遍历的统计。慢的时候靠它定位是深度、数量还是耗时触顶。"""

    visited: int = 0
    kept: int = 0
    max_depth_reached: int = 0
    elapsed_ms: float = 0.0
    truncated_by: str = ""  # "" / "depth" / "count" / "time"

    def as_dict(self) -> dict:
        return {
            "visited": self.visited,
            "kept": self.kept,
            "max_depth_reached": self.max_depth_reached,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "truncated_by": self.truncated_by or "none",
        }


class UIATree:
    """UIA 元素树抓取器。

    典型用法::

        tree = UIATree()
        elements = tree.capture_foreground()   # 只抓当前前台窗口，快
        elements = tree.capture_desktop()      # 抓整个桌面，慢
    """

    def __init__(
        self,
        max_depth: int = 12,
        max_elements: int = 400,
        time_budget_ms: float = 1500.0,
        min_area: int = 64,
        include_containers: bool = False,
    ) -> None:
        self.max_depth = max_depth
        self.max_elements = max_elements
        self.time_budget_ms = time_budget_ms
        #: 过滤掉 8×8 以下的元素——点不中，也不会是有意义的控件
        self.min_area = min_area
        self.include_containers = include_containers
        self._stats = UIATraversalStats()

    @property
    def stats(self) -> UIATraversalStats:
        """最近一次遍历的统计。"""
        return self._stats

    # ------------------------------------------------------------------ #

    @staticmethod
    def is_available() -> bool:
        if not IS_WINDOWS:
            return False
        try:
            import uiautomation  # noqa: F401

            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------ #

    def capture_foreground(self, clip: BBox | None = None) -> list[UIElement]:
        """抓取当前前台窗口的元素树。

        **默认用这个而不是 `capture_desktop`。** Agent 操作的总是前台窗口，
        抓整个桌面既慢又会混进后台窗口的控件——那些控件不可见却有合法
        坐标，点下去会点到前台窗口的其他位置。
        """
        import uiautomation as auto

        window = auto.GetForegroundControl()
        if window is None:
            logger.warning("取不到前台窗口，退回抓取桌面")
            return self.capture_desktop(clip)
        return self._walk(window, clip)

    def capture_desktop(self, clip: BBox | None = None) -> list[UIElement]:
        """抓取整个桌面的元素树。慢，仅在确实需要跨窗口时使用。"""
        import uiautomation as auto

        return self._walk(auto.GetRootControl(), clip)

    def capture_window(self, title_substring: str, clip: BBox | None = None) -> list[UIElement]:
        """按标题子串抓取指定窗口。找不到时返回空列表。"""
        import uiautomation as auto

        window = auto.WindowControl(searchDepth=2, SubName=title_substring)
        if not window.Exists(maxSearchSeconds=1):
            logger.warning("未找到标题含 %r 的窗口", title_substring)
            self._stats = UIATraversalStats()
            return []
        return self._walk(window, clip)

    # ------------------------------------------------------------------ #

    def _walk(self, root, clip: BBox | None) -> list[UIElement]:
        """广度优先遍历。

        **用 BFS 而不是 DFS 是有意的**：触顶截断时，BFS 保留的是靠近顶层的
        控件（菜单栏、主要按钮），DFS 保留的是某一条分支钻到底的深层节点。
        前者对 Agent 有用得多。
        """
        start = time.perf_counter()
        stats = UIATraversalStats()
        elements: list[UIElement] = []

        queue: list[tuple[object, int]] = [(root, 0)]
        while queue:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if elapsed_ms > self.time_budget_ms:
                stats.truncated_by = "time"
                logger.warning("UIA 遍历超时预算 %.0fms，已截断", self.time_budget_ms)
                break
            if len(elements) >= self.max_elements:
                stats.truncated_by = "count"
                break

            node, depth = queue.pop(0)
            stats.visited += 1
            stats.max_depth_reached = max(stats.max_depth_reached, depth)

            element = self._to_element(node, clip)
            if element is not None:
                elements.append(element)

            if depth >= self.max_depth:
                stats.truncated_by = stats.truncated_by or "depth"
                continue

            try:
                for child in node.GetChildren():
                    queue.append((child, depth + 1))
            except Exception as exc:  # noqa: BLE001 —— 跨进程 COM 调用什么都可能抛
                logger.debug("读取子控件失败（depth=%d）：%s", depth, exc)

        stats.kept = len(elements)
        stats.elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._stats = stats
        logger.debug("UIA 遍历 %s", stats.as_dict())
        return elements

    def _to_element(self, node, clip: BBox | None) -> UIElement | None:
        """把一个 UIA 控件转成 `UIElement`，不合格的返回 None。"""
        try:
            if node.IsOffscreen:
                return None

            rect = node.BoundingRectangle
            # 无效矩形：不可见控件、尚未布局的控件都会给出零矩形
            if rect is None or rect.width() <= 0 or rect.height() <= 0:
                return None

            bbox = BBox(int(rect.left), int(rect.top), int(rect.right), int(rect.bottom))
            if bbox.area < self.min_area:
                return None
            if clip is not None and bbox.intersection_area(clip) == 0:
                return None

            name = (node.Name or "").strip()
            control_type = node.ControlTypeName or ""

            if not self.include_containers and control_type in CONTAINER_TYPES:
                return None
            # 无名且非交互类型的控件对 Agent 没有价值
            if not name and control_type not in INTERACTIVE_TYPES:
                return None

            return UIElement(
                bbox=bbox,
                source=ElementSource.UIA,
                text=name,
                control_type=control_type,
                confidence=1.0,
                meta={
                    "automation_id": getattr(node, "AutomationId", "") or "",
                    "class_name": getattr(node, "ClassName", "") or "",
                    "is_enabled": bool(getattr(node, "IsEnabled", True)),
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("转换 UIA 控件失败：%s", exc)
            return None


# ---------------------------------------------------------------------- #
# M4 Reflector 会用到的程序化判据
# ---------------------------------------------------------------------- #


def find_by_name(elements: list[UIElement], name: str, exact: bool = False) -> UIElement | None:
    """按控件名查找。M4 Reflector 的"某控件是否出现"判据。"""
    for element in elements:
        if (element.text == name) if exact else (name in element.text):
            return element
    return None


def get_foreground_window_title() -> str:
    """当前前台窗口标题。M4 的窗口标题判据用它，比截图判断快几个数量级。"""
    if not IS_WINDOWS:
        return ""
    try:
        import uiautomation as auto

        window = auto.GetForegroundControl()
        return (window.Name or "") if window else ""
    except Exception as exc:  # noqa: BLE001
        logger.debug("取前台窗口标题失败：%s", exc)
        return ""
