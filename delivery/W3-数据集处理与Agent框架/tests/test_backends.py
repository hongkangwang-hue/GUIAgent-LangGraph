"""双层后端抽象的单元测试。

重点不在"调用能不能成功"，而在**模型不听话时系统怎么反应**。开源规划
模型的输出稳定性明显低于前沿模型，M2 设计思路里专门写了两条针对开源
模型的必要设计——这里测的是同一件事的另一面：模型给了不合规的输出，
系统是老实记录下来，还是崩掉、或者更糟，悄悄当成功处理。
"""

from __future__ import annotations

import pytest

from control.actions import ActionType, ActionValidationError
from grounding.native import NativeGrounding
from llm.base import (
    ActionIntent,
    CostInfo,
    HistoryStep,
    LLMBackendError,
    PriceSheet,
    TokenUsage,
)
from llm.fake import ScriptedBackend
from perception.types import Point

# ===================================================================== #
# 计费
# ===================================================================== #


def test_price_sheet_basic() -> None:
    price = PriceSheet("qwen-vl", input_per_1k=0.002, output_per_1k=0.006)
    assert price.cost_of(1000, 500) == pytest.approx(0.002 + 0.003)


def test_cached_tokens_billed_at_cache_rate() -> None:
    """缓存命中的输入按缓存单价计，不能按原价。"""
    price = PriceSheet("m", input_per_1k=0.010, output_per_1k=0.0, cached_input_per_1k=0.001)
    # 1000 输入里 800 命中缓存：200 按 0.010，800 按 0.001
    assert price.cost_of(1000, 0, cached_tokens=800) == pytest.approx(0.002 + 0.0008)


def test_cache_rate_falls_back_to_input_rate() -> None:
    """平台不支持缓存时按原价计，不能白送。"""
    price = PriceSheet("m", input_per_1k=0.010, output_per_1k=0.0)
    assert price.cost_of(1000, 0, cached_tokens=800) == pytest.approx(0.010)


def test_cached_tokens_exceeding_prompt_does_not_go_negative() -> None:
    """平台报的命中数偶尔大于总输入，不能算出负费用。"""
    price = PriceSheet("m", input_per_1k=0.010, output_per_1k=0.0, cached_input_per_1k=0.0)
    assert price.cost_of(100, 0, cached_tokens=500) == pytest.approx(0.0)


def test_unknown_price_marks_cost_unreliable() -> None:
    """没配单价就别猜。假成本比"未知"更有害——M0 的估算要靠这些数校准。"""
    backend = ScriptedBackend([{"action": "wait", "duration": 1}], price=None)
    backend.predict_action("x", None)
    cost = backend.get_cost()
    assert cost.priced is False
    assert cost.cost_cny == 0.0
    assert cost.total_tokens > 0  # token 照记，只是折不成钱


def test_cost_accumulates_across_calls() -> None:
    price = PriceSheet("m", input_per_1k=0.001, output_per_1k=0.002)
    backend = ScriptedBackend(
        [{"action": "wait", "duration": 1}] * 3,
        price=price,
        usage_per_call=TokenUsage(prompt_tokens=1000, completion_tokens=1000),
    )
    for _ in range(3):
        backend.predict_action("x", None)
    cost = backend.get_cost()
    assert cost.requests == 3
    assert cost.total_tokens == 6000
    assert cost.cost_cny == pytest.approx(3 * 0.003)
    assert cost.priced is True


def test_reset_cost() -> None:
    backend = ScriptedBackend([{"action": "wait", "duration": 1}])
    backend.predict_action("x", None)
    backend.reset_cost()
    assert backend.get_cost().requests == 0


def test_cache_hit_rate() -> None:
    info = CostInfo(prompt_tokens=1000, cached_tokens=250)
    assert info.cache_hit_rate == pytest.approx(0.25)
    assert CostInfo().cache_hit_rate == 0.0


# ===================================================================== #
# ActionIntent
# ===================================================================== #


def test_intent_with_coordinates_needs_no_grounding() -> None:
    intent = ActionIntent(action_type="left_click", params={"x": 10, "y": 20})
    assert intent.needs_grounding is False


def test_intent_missing_coordinates_needs_grounding() -> None:
    """模式 A 的模型漏给坐标，判据要能识别出来。"""
    intent = ActionIntent(action_type="left_click", target_description="地址栏")
    assert intent.needs_grounding is True


def test_coordinate_free_action_never_needs_grounding() -> None:
    """打字、等待这类动作没有坐标，不该被送去定位。"""
    assert ActionIntent(action_type="type", params={"text": "你好"}).needs_grounding is False
    assert ActionIntent(action_type="wait", params={"duration": 1}).needs_grounding is False


def test_done_intent_needs_no_grounding() -> None:
    assert ActionIntent(done=True).needs_grounding is False


def test_unknown_action_type_deferred_to_to_action() -> None:
    """未知动作类型不在 needs_grounding 里炸，留给 to_action 报清楚的错。"""
    intent = ActionIntent(action_type="teleport", params={})
    assert intent.needs_grounding is False
    with pytest.raises(ActionValidationError):
        intent.to_action()


def test_with_point_returns_new_instance() -> None:
    """必须返回新实例：原始意图要原样进轨迹日志。

    模式 B 下"模型说了什么"与"grounding 定位到哪"是两条独立证据，就地
    覆盖会把前一条抹掉，M3 分析 grounding 误差时就无从对比。
    """
    original = ActionIntent(action_type="left_click", target_description="地址栏")
    filled = original.with_point(100, 200)

    assert filled is not original
    assert filled.params == {"x": 100, "y": 200}
    assert original.params == {}  # 原件未被修改
    assert filled.target_description == "地址栏"


def test_to_action_carries_thinking_as_reasoning() -> None:
    intent = ActionIntent(action_type="left_click", params={"x": 1, "y": 2}, thinking="点地址栏")
    action = intent.to_action()
    assert action.type is ActionType.LEFT_CLICK
    assert action.reasoning == "点地址栏"


def test_to_action_rejects_invalid_params() -> None:
    with pytest.raises(ActionValidationError):
        ActionIntent(action_type="left_click", params={"x": 1}).to_action()


def test_history_step_summary_marks_failure() -> None:
    from control.actions import Action

    step = HistoryStep(action=Action(type="left_click", x=1, y=2), success=False, error="坐标越界")
    assert "坐标越界" in step.summary()
    assert HistoryStep().summary() == "（无动作）"


# ===================================================================== #
# ScriptedBackend
# ===================================================================== #


def test_scripted_backend_follows_script() -> None:
    backend = ScriptedBackend(
        [
            {"action": "left_click", "x": 1, "y": 2},
            {"action": "type", "text": "你好"},
        ]
    )
    assert backend.predict_action("a", None).action_type == "left_click"
    assert backend.predict_action("a", None).params["text"] == "你好"


def test_scripted_backend_records_calls() -> None:
    backend = ScriptedBackend([{"action": "wait", "duration": 1}])
    backend.predict_action("打开浏览器", None, history=[HistoryStep(), HistoryStep()])
    assert backend.calls[0]["instruction"] == "打开浏览器"
    assert backend.calls[0]["history_len"] == 2


def test_scripted_backend_done_when_exhausted() -> None:
    backend = ScriptedBackend([])
    assert backend.predict_action("a", None).done is True


def test_scripted_backend_repeat_mode() -> None:
    """repeat 用来构造"模型原地打转"，测 Loop 的迭代上限。"""
    backend = ScriptedBackend([{"action": "left_click", "x": 1, "y": 1}], on_exhausted="repeat")
    for _ in range(5):
        assert backend.predict_action("a", None).action_type == "left_click"


def test_scripted_backend_raise_mode() -> None:
    backend = ScriptedBackend([], on_exhausted="raise")
    with pytest.raises(LLMBackendError) as info:
        backend.predict_action("a", None)
    assert info.value.kind == "script_exhausted"


def test_scripted_backend_can_raise_from_script() -> None:
    """脚本里塞异常，用来测 Loop 的错误处理分支。"""
    boom = LLMBackendError("限流", retryable=True, kind="rate_limit")
    backend = ScriptedBackend([boom])
    with pytest.raises(LLMBackendError) as info:
        backend.predict_action("a", None)
    assert info.value.retryable is True


def test_scripted_backend_rejects_bad_on_exhausted() -> None:
    with pytest.raises(ValueError):
        ScriptedBackend([], on_exhausted="explode")


def test_repeated_intent_does_not_share_params() -> None:
    """repeat 模式下同一条脚本返回多次，params 不能被上游改花。"""
    backend = ScriptedBackend(
        [ActionIntent(action_type="left_click", params={"x": 1, "y": 1})], on_exhausted="repeat"
    )
    first = backend.predict_action("a", None)
    first.params["x"] = 999
    assert backend.predict_action("a", None).params["x"] == 1


# ===================================================================== #
# NativeGrounding（模式 A）
# ===================================================================== #


def test_native_grounding_passes_through() -> None:
    intent = ActionIntent(action_type="left_click", params={"x": 100, "y": 200})
    result = NativeGrounding(1024, 768).locate(None, "", intent)
    assert result.found
    assert result.point == Point(100, 200)
    assert result.source == "native"


def test_native_grounding_rejects_missing_coordinates() -> None:
    """模型该给坐标却没给。这是开源模型的常见失误，要当数据记，不能崩。"""
    intent = ActionIntent(action_type="left_click", target_description="地址栏")
    result = NativeGrounding(1024, 768).locate(None, "地址栏", intent)
    assert not result.found
    assert result.meta["reason"] == "missing_coordinates"


def test_native_grounding_rejects_out_of_space() -> None:
    """越界要在这里拦住并标明根因。

    放它流到执行器，报出来的是"坐标越界被安全白名单拒绝"，看不出根因
    是模型输出不合规。M3 统计模式 A 的失败原因时，这两者必须分得开。
    """
    intent = ActionIntent(action_type="left_click", params={"x": 1500, "y": 200})
    result = NativeGrounding(1024, 768).locate(None, "", intent)
    assert not result.found
    assert result.meta["reason"] == "out_of_space"
    assert result.meta["raw_point"] == (1500, 200)


@pytest.mark.parametrize("point", [(-1, 10), (10, -1), (1024, 10), (10, 768)])
def test_native_grounding_boundary(point) -> None:
    """右下边界是开区间：1024×768 的坐标系里 x=1024 已经越界。"""
    intent = ActionIntent(action_type="left_click", params={"x": point[0], "y": point[1]})
    assert not NativeGrounding(1024, 768).locate(None, "", intent).found


def test_native_grounding_accepts_last_valid_pixel() -> None:
    intent = ActionIntent(action_type="left_click", params={"x": 1023, "y": 767})
    assert NativeGrounding(1024, 768).locate(None, "", intent).found


def test_native_grounding_rejects_non_integer() -> None:
    intent = ActionIntent(action_type="left_click", params={"x": "左边", "y": 10})
    result = NativeGrounding(1024, 768).locate(None, "", intent)
    assert not result.found
    assert result.meta["reason"] == "non_integer_coordinates"


def test_native_grounding_without_intent() -> None:
    result = NativeGrounding(1024, 768).locate(None, "地址栏", None)
    assert not result.found
    assert "ActionIntent" in result.error


def test_grounding_result_records_latency() -> None:
    intent = ActionIntent(action_type="left_click", params={"x": 1, "y": 1})
    assert NativeGrounding(1024, 768).locate(None, "", intent).latency_ms >= 0


def test_grounding_result_corrected_flag() -> None:
    from grounding.base import GroundingResult

    plain = GroundingResult(point=Point(10, 10), source="native")
    assert plain.corrected is False
    assert "original_point" not in plain.as_dict()

    fixed = GroundingResult(
        point=Point(12, 10), source="uia_fallback", original_point=Point(10, 10)
    )
    assert fixed.corrected is True
    assert fixed.as_dict()["original_point"] == (10, 10)
