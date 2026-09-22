"""分辨率消融的性质测试。

## 这里守的是什么

这个消融能在验证集上做，靠的是**真值坐标与图片像素尺寸无关**——
`point_norm` 存的是归一化 1000 空间，模型也被要求在 1000 空间作答。
缩图不动真值。

**若哪天真值改成像素坐标，这套消融会静默地全错**：图缩了、真值没缩，
每一档的坐标误差都会按缩放倍数虚高，而表格照样打印得整整齐齐。
所以这里把「缩图不动真值」钉成断言。
"""

from __future__ import annotations

import pytest

PIL = pytest.importorskip("PIL.Image")


def _img(width: int, height: int):
    from PIL import Image

    return Image.new("RGB", (width, height), "white")


class TestDownscale:
    def test_按倍数缩(self):
        from eval.action import downscale

        assert downscale(_img(1024, 768), 0.5).size == (512, 384)

    def test_倍数为一时原样返回(self):
        from eval.action import downscale

        image = _img(1024, 768)
        assert downscale(image, 1.0) is image

    def test_只降不升(self):
        """**放大不产生细节**，只会把插值出来的模糊像素喂给模型。

        测出来的就成了「模型怕不怕插值」，不是「模型吃不吃分辨率」。
        """
        from eval.action import downscale

        image = _img(1024, 768)
        assert downscale(image, 2.0) is image
        assert downscale(image, 1.5) is image

    def test_极小倍数不产生零尺寸(self):
        from eval.action import downscale

        assert downscale(_img(1024, 768), 0.0001).size == (1, 1)

    def test_不改变宽高比(self):
        from eval.action import downscale

        out = downscale(_img(1024, 768), 0.375)
        assert abs(out.size[0] / out.size[1] - 1024 / 768) < 0.01


class TestTruthIsResolutionIndependent:
    """**这一组是整个消融成立的前提。**"""

    def _rows(self):
        import json
        from pathlib import Path

        path = Path("finetune/data/val.jsonl")
        if not path.exists():
            pytest.skip("验证集不在，跳过")
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_真值存的是归一化坐标而不是像素(self):
        """`point_norm` 与 `point_px` 必须是两个不同的字段，且评测用前者。

        它们相等就说明分辨率恰好是 1000×1000，那是巧合不是设计。

        **不能只看第一行。** 训练集放宽后第一行可能是 `key` / `type` / `done`
        这类本来就没有坐标的动作——2026-08-26 这条测试就是这么红的。
        """
        rows = [
            r for r in self._rows() if r["action"] in ("left_click", "double_click", "mouse_move")
        ]
        assert rows, "验证集里一条带坐标的动作都没有，那才是真出问题了"
        for row in rows[:20]:
            assert "point_norm" in row and "point_px" in row
            assert row["point_norm"] != row["point_px"]

    def test_不涉及坐标的动作不该有坐标字段(self):
        """**留一个空坐标比没有坐标更糟。**

        `llm.parsing` 里就有同样的处理：抠不出成对坐标时把半截的也清掉，
        否则 `needs_grounding` 会判成「不缺坐标」，然后在 `Action.validate`
        那里才炸，错误信息离根因太远。
        """
        for row in self._rows():
            if row["action"] in ("type", "key", "wait", "done", "scroll"):
                assert "point_norm" not in row and "point_px" not in row

    def test_坐标空间与图片尺寸是两回事(self):
        from agent.session import SessionConfig

        config = SessionConfig()
        assert config.coordinate_space == (1000, 1000)
        assert config.image_size != config.coordinate_space


class TestResolutionRunner:
    def test_原生分辨率那一档必须在(self):
        """没有 ×1.0 基线，其余三档的「掉了多少」没有参照。"""
        from scripts.ablate_resolution import LADDER

        assert LADDER[0][1] == 1.0

    def test_阶梯是单调下降的(self):
        from scripts.ablate_resolution import LADDER

        scales = [s for _, s in LADDER]
        assert scales == sorted(scales, reverse=True)

    def test_标签不重名(self):
        from scripts.ablate_resolution import LADDER, tag_of

        tags = [tag_of("", s) for _, s in LADDER]
        assert len(set(tags)) == len(tags)

    def test_prefix_区分基座与微调两轮(self):
        from scripts.ablate_resolution import tag_of

        assert tag_of("", 0.5) == "res50"
        assert tag_of("lora", 0.5) == "lora-res50"

    def test_与提示词消融的结果文件不撞名(self):
        """两个消融写同一个目录，撞名会互相覆盖且不报错。"""
        from scripts.ablate_prompts import LADDER as PROMPT_LADDER
        from scripts.ablate_prompts import tag_of as prompt_tag
        from scripts.ablate_resolution import LADDER as RES_LADDER
        from scripts.ablate_resolution import tag_of as res_tag

        a = {prompt_tag("lora", t) for _, t in PROMPT_LADDER}
        b = {res_tag("lora", s) for _, s in RES_LADDER}
        assert not a & b

    def test_缺档时报告不崩也不用零填(self, capsys, monkeypatch, tmp_path):
        from scripts import ablate_resolution

        monkeypatch.setattr("eval.action.RESULT_DIR", tmp_path)
        ablate_resolution.report("不存在的前缀")
        out = capsys.readouterr().out
        assert "缺这几档" in out and "—" in out

    def test_报告必须写明它证不了什么(self, capsys, monkeypatch, tmp_path):
        """筛选实验被读成证明，比不做更糟。

        它量的是**低于** 1024 的衰减，客机的问题是**高于** 1024 的细节
        被丢掉——方向相反。这句免责必须打在结果旁边，不能只写在文档里。
        """
        from scripts import ablate_resolution

        monkeypatch.setattr("eval.action.RESULT_DIR", tmp_path)
        ablate_resolution.report("")
        out = capsys.readouterr().out
        assert "证不了" in out

    def test_ablate_resolution_的参数表也不漏接(self):
        import ast
        import inspect

        from scripts import ablate_resolution

        tree = ast.parse(inspect.getsource(ablate_resolution))
        used = {
            n.attr
            for n in ast.walk(tree)
            if isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name)
            and n.value.id == "args"
        }
        parsed = vars(ablate_resolution.build_parser().parse_args([]))
        assert used and not used - parsed.keys()

    def test_不给模型时明确报错(self):
        """跑四档要二十几分钟，参数缺了要立刻说。"""
        import sys

        from scripts import ablate_resolution

        saved = sys.argv
        try:
            sys.argv = ["ablate_resolution.py"]
            with pytest.raises(SystemExit):
                ablate_resolution.main()
        finally:
            sys.argv = saved


class TestEvaluateWiring:
    def test_evaluate_接受_image_scale(self):
        import inspect

        from eval.action import evaluate

        assert "image_scale" in inspect.signature(evaluate).parameters

    def test_api_路径也吃缩放(self):
        """**两条路都要接。** 只接本地那条的话，`--provider` 下四档会跑出

        四组一模一样的数——而那看起来正是一个干净的「分辨率无影响」结论。
        """
        import cv2
        import numpy as np

        import eval.action

        path = "finetune/data/val.jsonl"
        import json
        from pathlib import Path

        if not Path(path).exists():
            pytest.skip("验证集不在，跳过")
        row = json.loads(Path(path).read_text(encoding="utf-8").splitlines()[0])
        if not Path(row["image"]).exists():
            pytest.skip("验证集图片不在，跳过")
        full = eval.action._screenshot_of(row)
        half = eval.action._screenshot_of(row, 0.5)
        assert half.image.shape[1] == round(full.image.shape[1] * 0.5)
        assert isinstance(half.image, np.ndarray) and cv2 is not None

    def test_api_predict_把缩放传下去了(self):
        import inspect

        import eval.action

        src = inspect.getsource(eval.action._build_predictor)
        assert "_screenshot_of(row, image_scale)" in src, "API 路径漏传 image_scale"

    def test_预测路径确实用了缩放(self):
        """护栏：`--image-scale` 接上了但预测里没用，四档会跑出一模一样的数。

        **这种失败最危险**——四条曲线完全重合，看起来像一个干净的
        「分辨率无影响」结论，而实际上什么都没测。
        """
        import inspect

        import eval.action

        src = inspect.getsource(eval.action._build_predictor)
        assert "downscale(" in src

    def test_cli_有_image_scale_且默认为原样(self):
        from eval.action import build_parser

        args = build_parser().parse_args([])
        assert args.image_scale == 1.0

    def test_main_读到的每个属性都存在(self):
        """`serve_local_model` 上栽过的那个坑，现在 `eval.action` 也守上了。

        **用 AST 而不是正则。** 正则扫整个模块源码会连 docstring 里的
        `args.xxx` 一起匹配——写这条测试时就当场被自己的说明文字骗了一次，
        报了个不存在的 `max_new_tokens`。AST 只看真实的属性访问。
        """
        import ast
        import inspect

        import eval.action

        tree = ast.parse(inspect.getsource(eval.action))
        used = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "args"
        }
        parsed = vars(eval.action.build_parser().parse_args([]))
        assert used, "一个 args.xxx 都没找到，说明这条测试自己坏了"
        assert not used - parsed.keys()
