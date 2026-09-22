"""**同一种失败，在两条路上必须记成同一件事。**

## 这里守的是什么

`eval/action.py` 有两条预测路径（本地 transformers / API）。它们跑的是
同一批样本、同一套提示词、同一套解析，结果要并排放进一张表里比较。

那么「模型输出解析不出来」这件事，两条路必须**记成同一种东西**。

2026-08-25 的 8B 评测上没做到：

    本地：`extract()` 静默返回 None  → 留在分母里，记 format_ok=False
    API ：`predict_action` 抛异常     → 记 error，被 summarize() 剔出分母

于是 8B 那 22 条（15.5%）格式失败**从分母里消失了**：

    格式合规率   分母 120 → 45.8%      分母 142 → 38.7%
    联合命中@50  分母 120 → 35.8%      分母 142 → 30.3%

而它正要跟本地那组 142/142、0 error 的数字比。**分母不一致的两列数
并排放着，看不出任何异常。**
"""

from __future__ import annotations

import pytest


class TestParseFailureStaysInDenominator:
    def test_解析失败带着原文抛(self):
        """没有原文就分不清是模型的问题还是解析层的问题。

        `llm/base.py` 的注释一直写着这条，但代码没兑现——`LLMBackendError`
        当时不接受 `raw`，22 条失败样本的 `raw` 全是空字符串。
        """
        from llm.base import LLMBackendError

        exc = LLMBackendError("x", kind="parse_error", raw="模型说的话")
        assert exc.raw == "模型说的话"

    def test_默认没有原文时是空串而不是报错(self):
        from llm.base import LLMBackendError

        assert LLMBackendError("x").raw == ""

    def test_predict_action_把原文塞进异常(self):
        """端到端确认：解析失败时拿得到模型原话。

        用一个只实现 `complete()` 的最小后端——`ScriptedBackend` 直接返回
        `ActionIntent`，压根不走解析那一步，测不到这里。
        """
        from llm.base import LLMBackend, LLMBackendError, RawResponse

        class 只会胡说的后端(LLMBackend):
            name = "babbler"

            def complete(self, prompt, screenshot=None, history=None):
                return RawResponse(text="这不是 JSON，随便说点什么")

        with pytest.raises(LLMBackendError) as caught:
            只会胡说的后端(model="m").predict_action("打开浏览器", None)

        assert caught.value.kind == "parse_error"
        assert caught.value.raw == "这不是 JSON，随便说点什么"

    def test_api_路径只吞解析失败不吞网络失败(self):
        """**网络超时、限流、余额不足确实不是模型的失败**，剔出分母才对。

        全吞掉的话，一次限流会被记成「模型格式不合规」，
        把后端的锅算到模型头上。
        """
        import inspect

        import eval.action

        src = inspect.getsource(eval.action._build_predictor)
        assert 'exc.kind != "parse_error"' in src
        assert "raise" in src

    def test_api_路径返回原文而不是抛出去(self):
        import inspect

        import eval.action

        src = inspect.getsource(eval.action._build_predictor)
        assert "return exc.raw" in src


class TestSummarizeDenominator:
    """`summarize()` 剔除 error 是对的——前提是 error 里没有模型的失败。"""

    def _write(self, tmp_path, rows):
        import json

        path = tmp_path / "t.jsonl"
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
        )
        return path

    BASE = {
        "sample_id": "x",
        "instruction": "",
        "truth_action": "left_click",
        "truth_xy": [0, 0],
        "pred_action": "left_click",
        "pred_xy": [0, 0],
        "action_ok": True,
        "format_ok": True,
        "distance": 0.0,
        "error": "",
        "latency_ms": 1.0,
        "raw": "{}",
        "parroted": False,
    }

    def test_不合规的样本留在分母里(self, tmp_path):
        from eval.action import summarize

        rows = [dict(self.BASE, sample_id=str(i)) for i in range(8)]
        for row in rows[:4]:
            row.update(format_ok=False, action_ok=False, distance=-1.0)
        stats = summarize(self._write(tmp_path, rows))
        assert stats["total"] == 8
        assert stats["format_acc"] == 0.5

    def test_error_样本被剔出分母(self, tmp_path):
        """这是既有行为，钉住它——**它只在 error 里没有模型失败时才正确**。"""
        from eval.action import summarize

        rows = [dict(self.BASE, sample_id=str(i)) for i in range(8)]
        for row in rows[:4]:
            row.update(error="Timeout", format_ok=False, action_ok=False, distance=-1.0)
        stats = summarize(self._write(tmp_path, rows))
        assert stats["total"] == 8
        assert stats["errors"] == 4
        assert stats["format_acc"] == 1.0, "剔掉 4 条 error 后，剩下 4 条全合规"

    def test_两种口径的差在报告里必须看得见(self, tmp_path):
        """`summarize()` 同时给 `total` 和 `errors`，读的人才算得出真分母。

        只给一个比率、不给 error 数，分母不一致就彻底看不出来了。
        """
        from eval.action import summarize

        rows = [dict(self.BASE, sample_id=str(i)) for i in range(8)]
        rows[0]["error"] = "boom"
        stats = summarize(self._write(tmp_path, rows))
        assert stats["total"] == 8 and stats["errors"] == 1
