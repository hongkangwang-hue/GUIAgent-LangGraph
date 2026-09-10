"""存档必须记下**能左右结论的每一个变量**。

## 为什么单开一个文件守这个

2026-08-25 一天之内栽了两次，形态完全相同：

1. **提示词模板没进存档。** 离线组 64% 的动作是在逐字背诵 `executor_v1`
   few-shot 示例里的坐标 `(470, 750)`——而存档里查不到用了哪份模板，
   是靠翻客机轨迹才发现的。
2. **屏幕设置没进存档。** 报告把宿主机的 2560×1600 当成了客机的分辨率，
   一直写在 §9 里；客机实际是 1920×1080。而 §9 自己刚证明分辨率值
   2 倍坐标误差。

两次都不是「记错了」，是**根本没记**。而没记下来的变量，事后无法区分
「它没变」和「它变了但没人看见」——**两个存档摆在一起，看不出任何异常。**
"""

from __future__ import annotations

import json

import pytest

from scripts.run_basic_tasks import _screen_info, build_parser


def make_args(**overrides):
    """**从真实 parser 构造，不手搓桩。**

    手搓的 `Args` 类会随 parser 长出新参数而漏字段——加
    `--allowed-actions` 那次就漏了，三条测试一起变红，而它们本该只关心
    自己那一项。从 parser 出发，新参数自动带上默认值。
    """
    args = build_parser().parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TestArchiveRecordsProvenance:
    def _payload(self) -> dict:
        from scripts.run_basic_tasks import archive_payload

        args = make_args(execute=True, repeats=5, tag="lora", executor_template="executor_v0")
        return json.loads(archive_payload([], args, "离线 / 本地", offline=True, partial=False))

    def test_记下用了哪份提示词模板(self):
        assert self._payload()["executor_template"] == "executor_v0"

    def test_记下屏幕设置(self):
        screen = self._payload()["screen"]
        assert "resolution" in screen or screen == {}, "取不到就该是空，不能猜一个"

    def test_已有的溯源字段没被弄丢(self):
        """`backend` / `offline` / `tag` / `partial` 是既有的护栏，一起钉住。"""
        payload = self._payload()
        for field in ("backend", "offline", "tag", "partial", "repeats", "scope"):
            assert field in payload


class TestScreenInfo:
    def test_分辨率与缩放一起取(self):
        """**1920×1080 @100% 和 @150% 下按钮差 1.5 倍。**

        只记一个说不清模型看到的元素有多大。
        """
        info = _screen_info()
        if not info:
            pytest.skip("这台机器取不到屏幕信息")
        assert "resolution" in info
        assert "dpi" in info and "scale_factor" in info["dpi"]

    def test_分辨率格式是宽x高(self):
        info = _screen_info()
        if not info.get("resolution"):
            pytest.skip("取不到分辨率")
        width, _, height = info["resolution"].partition("x")
        assert width.isdigit() and height.isdigit()

    def test_取不到时返回空而不是抛(self, monkeypatch):
        """**采集失败不该让整轮跑不起来。** 存档少一项，比 25 轮跑不成好。"""
        import scripts.run_basic_tasks as mod

        monkeypatch.setitem(
            __import__("sys").modules, "perception.capture", None
        )  # 触发 import 失败
        assert isinstance(mod._screen_info(), dict)


class TestTemplateFlag:
    def test_默认留空而不是写死字面量(self):
        """写死 `"executor_v1"` 的话，`SessionConfig` 改了默认这里不跟着改，

        **存档记下的就不是实际用的那份**——正是它要防的那个混淆的形态。
        """
        assert build_parser().parse_args([]).executor_template == ""

    def test_留空时解析成_SessionConfig_的默认值(self):
        import inspect

        import scripts.run_basic_tasks as mod

        src = inspect.getsource(mod.main) if hasattr(mod, "main") else inspect.getsource(mod)
        assert "args.executor_template or SessionConfig.executor_template" in src

    def test_可以显式指定(self):
        args = build_parser().parse_args(["--executor-template", "executor_v0"])
        assert args.executor_template == "executor_v0"

    def test_三档模板都能指定(self):
        from agent.prompts import list_templates

        available = list_templates()
        for name in ("executor_v0", "executor_v1", "executor_v2"):
            assert name in available
            assert (
                build_parser().parse_args(["--executor-template", name]).executor_template == name
            )


class TestAllowedActionsProvenance:
    """`--allowed-actions` 与 `executor_template`、`screen` 同一条护栏。"""

    def test_动作集进存档(self):
        import json

        from scripts.run_basic_tasks import archive_payload

        args = make_args(allowed_actions="left_click, mouse_move")
        payload = json.loads(archive_payload([], args, "离线", offline=True, partial=False))
        assert payload["allowed_actions"] == ["left_click", "mouse_move"]

    def test_留空时存档记的是空列表而不是缺字段(self):
        """**「没限制」要能看出来，不能靠字段缺失来表达。**"""
        import json

        from scripts.run_basic_tasks import archive_payload

        payload = json.loads(archive_payload([], make_args(), "在线", offline=False, partial=False))
        assert payload["allowed_actions"] == []


class TestPlannerTemplateProvenance:
    def test_规划器模板也进存档(self):
        """**两份模板都能左右结论。** 2026-08-25 的事故正是两者的示例串成回路。"""
        import json

        from scripts.run_basic_tasks import archive_payload

        args = make_args(planner_template="planner_v2", executor_template="executor_v3")
        payload = json.loads(archive_payload([], args, "离线", offline=True, partial=False))
        assert payload["planner_template"] == "planner_v2"
        assert payload["executor_template"] == "executor_v3"

    def test_默认留空并在_main_里解析(self):
        import inspect

        import scripts.run_basic_tasks as mod
        from scripts.run_basic_tasks import build_parser

        assert build_parser().parse_args([]).planner_template == ""
        assert "args.planner_template or SessionConfig.planner_template" in inspect.getsource(mod)
