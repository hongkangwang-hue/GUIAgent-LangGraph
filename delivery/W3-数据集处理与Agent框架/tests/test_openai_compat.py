"""OpenAI 兼容后端与平台注册表的单元测试。

**不发真实请求。** 用假 client 把响应钉死，测的是这一层自己的逻辑：
消息怎么拼、图怎么压、用量怎么读、错误怎么归类。这些和模型好不好无关，
但每一条错了都会让 M3 的横评数据失真。

真实连通性由 `scripts/verify_llm.py` 单独验证，那个要花钱，不进 CI。
"""

from __future__ import annotations

import base64

import numpy as np
import pytest

from llm.base import HistoryStep, LLMBackendError
from llm.openai_compat import (
    DEFAULT_IMAGE_SIZE,
    MIN_JPEG_QUALITY,
    OpenAICompatBackend,
    _extract_usage,
    _request_id,
    _response_text,
    classify_error,
    encode_screenshot,
)
from llm.providers import (
    PROVIDERS,
    ProviderNotConfigured,
    available_providers,
    load_dotenv_if_present,
    resolve,
)
from perception.capture import Screenshot
from perception.types import BBox

SCREEN = BBox(0, 0, 2560, 1600)


def _shot(width=2560, height=1600, noisy=True) -> Screenshot:
    """构造一张截图。

    默认填随机噪声：纯色图 JPEG 能压到几百字节，测不出体积上限的逻辑。
    """
    if noisy:
        rng = np.random.default_rng(0)
        image = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    else:
        image = np.zeros((height, width, 3), dtype=np.uint8)
    return Screenshot(image=image, region=BBox(0, 0, width, height), engine="fake")


class FakeResponse:
    def __init__(self, content, usage_metadata=None, response_metadata=None) -> None:
        self.content = content
        self.usage_metadata = usage_metadata
        self.response_metadata = response_metadata or {}


class FakeClient:
    def __init__(self, response, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.received: list = []

    def invoke(self, messages):
        self.received.append(messages)
        if self.error:
            raise self.error
        return self.response


@pytest.fixture
def backend(monkeypatch):
    """一个不需要真实 key 的后端。"""
    for provider in PROVIDERS.values():
        monkeypatch.setenv(provider.api_key_env, "test-key-0123456789")
    instance = OpenAICompatBackend(config=resolve("dashscope"))
    return instance


def wire(backend, content, usage=None, metadata=None, error=None) -> FakeClient:
    client = FakeClient(FakeResponse(content, usage, metadata), error)
    backend._client = client
    return client


# ===================================================================== #
# 平台注册表
# ===================================================================== #


def test_providers_registered() -> None:
    """三家云平台 + 一条自建服务。

    `selfhost` 是离线版的接入点：权重跑在宿主机 GPU 上，客机经 HTTP 调用。
    它与三家云平台共用同一个 `OpenAICompatBackend`——这正是当初把后端写成
    通用兼容层换来的好处，加一条记录即可，后端类一行不改。
    """
    assert set(PROVIDERS) == {"dashscope", "zhipu", "nvidia", "selfhost"}


def test_only_selfhost_has_local_weights() -> None:
    """`weights_local` 是给自建服务开的口子，不能顺手把「不猜单价」废掉。

    商业平台未配单价时必须继续是「未知」（priced=False），否则成本恒为 0
    却看起来像真的——这正是 llm/providers.py 开头要避免的那种数字。
    """
    assert {k for k, p in PROVIDERS.items() if p.weights_local} == {"selfhost"}


def test_resolve_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("ZHIPU_API_KEY", "zk-123456789")
    monkeypatch.setenv("ZHIPU_MODEL", "glm-4.6v-flash")
    config = resolve("zhipu")
    assert config.model == "glm-4.6v-flash"
    assert config.base_url.startswith("https://open.bigmodel.cn")


def test_explicit_model_overrides_env(monkeypatch) -> None:
    """横评要在同一平台上换模型跑，显式传入必须优先。"""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-1")
    monkeypatch.setenv("PLANNER_MODEL", "qwen3-vl-8b-instruct")
    assert resolve("dashscope", model="qwen3-vl-30b").model == "qwen3-vl-30b"


def test_missing_key_is_a_clear_error(monkeypatch) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(ProviderNotConfigured) as info:
        resolve("nvidia")
    assert "NVIDIA_API_KEY" in str(info.value)


def test_unknown_provider(monkeypatch) -> None:
    with pytest.raises(ProviderNotConfigured):
        resolve("openai")


def test_require_key_can_be_waived(monkeypatch) -> None:
    """`config --show` 要能在没配 key 时列出平台。"""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    assert resolve("nvidia", require_key=False).api_key == ""


def test_price_is_none_unless_configured(monkeypatch) -> None:
    """不猜单价。假成本比"未知"更有害。"""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-1")
    monkeypatch.delenv("DASHSCOPE_PRICE_IN_PER_1K", raising=False)
    monkeypatch.delenv("DASHSCOPE_PRICE_OUT_PER_1K", raising=False)
    assert resolve("dashscope").price is None


def test_price_from_env(monkeypatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-1")
    monkeypatch.setenv("DASHSCOPE_PRICE_IN_PER_1K", "0.002")
    monkeypatch.setenv("DASHSCOPE_PRICE_OUT_PER_1K", "0.006")
    price = resolve("dashscope").price
    assert price is not None
    assert price.cost_of(1000, 1000) == pytest.approx(0.008)


def test_blank_price_is_treated_as_unset(monkeypatch) -> None:
    """.env 里留空行是常态，空字符串必须等同于没填。

    os.getenv 对 ``ZHIPU_PRICE_IN_PER_1K=`` 返回 "" 而不是 None，按"已
    配置"处理会造出一张 0 元单价表并标成可信——成本恒为 0 却看起来像
    真的，正是"不猜单价"这条规矩要避免的东西。
    """
    monkeypatch.setenv("ZHIPU_API_KEY", "zk-1")
    monkeypatch.setenv("ZHIPU_PRICE_IN_PER_1K", "")
    monkeypatch.setenv("ZHIPU_PRICE_OUT_PER_1K", "   ")
    assert resolve("zhipu").price is None


def test_partial_price_still_counts_as_configured(monkeypatch) -> None:
    """只填了输入单价也算配置了——缺的那半按 0 算，总比整张表作废好。"""
    monkeypatch.setenv("ZHIPU_API_KEY", "zk-1")
    monkeypatch.setenv("ZHIPU_PRICE_IN_PER_1K", "0.001")
    monkeypatch.delenv("ZHIPU_PRICE_OUT_PER_1K", raising=False)
    price = resolve("zhipu").price
    assert price is not None
    assert price.cost_of(1000, 1000) == pytest.approx(0.001)


def test_bad_price_is_treated_as_unset(monkeypatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-1")
    monkeypatch.setenv("DASHSCOPE_PRICE_IN_PER_1K", "很便宜")
    assert resolve("dashscope").price is None


def test_api_key_is_masked(monkeypatch) -> None:
    """key 只留头尾。

    轨迹日志、控制台输出、贴给别人看的报错都可能带出这个 dict，
    一次疏忽就是一把废掉的凭据。
    """
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-abcdefghijklmnop")
    masked = resolve("dashscope").masked()
    assert "sk-abcdefghijklmnop" not in str(masked)
    assert masked["api_key"] == "sk-a…mnop"


def test_short_key_fully_masked(monkeypatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "abc")
    assert resolve("dashscope").masked()["api_key"] == "***"


def test_available_providers(monkeypatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-1")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    status = dict(available_providers())
    assert status["dashscope"] is True
    assert status["nvidia"] is False


def test_dotenv_does_not_override_existing(tmp_path, monkeypatch) -> None:
    """命令行里显式设的值应当优先于文件。"""
    env_file = tmp_path / ".env"
    env_file.write_text("DASHSCOPE_API_KEY=from-file\nZHIPU_API_KEY=zk-file\n", encoding="utf-8")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "from-shell")
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)

    assert load_dotenv_if_present(str(env_file)) is True
    import os

    assert os.environ["DASHSCOPE_API_KEY"] == "from-shell"
    assert os.environ["ZHIPU_API_KEY"] == "zk-file"


def test_dotenv_missing_file() -> None:
    assert load_dotenv_if_present("查无此文件.env") is False


# ===================================================================== #
# 图片编码
# ===================================================================== #


def test_screenshot_is_resized_to_coordinate_space() -> None:
    """送给模型的图必须和坐标系同尺寸。

    模型在它看到的那张图的像素坐标系里作答。送 2560×1600 却按 1024×768
    反算真实坐标，每次点击都会偏，而且偏得很有规律——看起来像"模型定位
    不准"，实际是坐标系错配。
    """
    _url, meta = encode_screenshot(_shot(), size=(1024, 768))
    assert (meta["sent_width"], meta["sent_height"]) == (1024, 768)


def test_encoded_as_jpeg_data_url() -> None:
    url, _ = encode_screenshot(_shot(), size=(320, 240))
    assert url.startswith("data:image/jpeg;base64,")
    base64.b64decode(url.split(",", 1)[1])  # 解不开就是坏的


def test_jpeg_beats_png_on_size() -> None:
    """选 JPEG 的理由之一：上传字节直接进 token 账单。"""
    import cv2

    shot = _shot(1024, 768)
    _url, meta = encode_screenshot(shot, size=(1024, 768))
    ok, png = cv2.imencode(".png", shot.image)
    assert ok
    assert meta["bytes"] < len(png.tobytes())


def test_quality_drops_to_meet_byte_limit() -> None:
    """NVIDIA 有内联图片硬上限，超了整条请求被拒。"""
    _url, meta = encode_screenshot(_shot(), size=(1024, 768), max_bytes=20_000)
    assert meta["bytes"] <= 20_000 or meta["downscale"] < 1.0
    assert meta["jpeg_quality"] < 85


def test_resolution_is_the_last_resort() -> None:
    """降分辨率会破坏坐标系，只有降质量到下限还超时才用。"""
    _url, meta = encode_screenshot(_shot(), size=(1024, 768), max_bytes=2_000)
    assert meta["jpeg_quality"] == MIN_JPEG_QUALITY
    assert meta["downscale"] < 1.0


def test_no_downscale_when_within_limit() -> None:
    _url, meta = encode_screenshot(_shot(), size=(1024, 768), max_bytes=5_000_000)
    assert meta["downscale"] == 1.0
    assert meta["jpeg_quality"] == 85


def test_nvidia_limit_is_applied(monkeypatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "nv-1")
    config = resolve("nvidia")
    assert config.max_image_bytes > 0
    _url, meta = encode_screenshot(_shot(), size=(1024, 768), max_bytes=config.max_image_bytes)
    assert meta["bytes"] <= config.max_image_bytes


def test_dashscope_has_no_byte_limit(monkeypatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-1")
    assert resolve("dashscope").max_image_bytes == 0


# ===================================================================== #
# 错误归类
# ===================================================================== #


@pytest.mark.parametrize(
    "message",
    ["Request timed out", "429 Too Many Requests", "503 Service Unavailable", "Connection reset"],
)
def test_transient_errors_are_retryable(message) -> None:
    kind, retryable = classify_error(RuntimeError(message))
    assert retryable is True
    assert kind == "transient"


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Insufficient balance", "quota"),
        ("401 Unauthorized", "auth"),
        ("Invalid API key provided", "auth"),
        ("model not found", "bad_request"),
    ],
)
def test_fatal_errors_are_not_retried(message, expected) -> None:
    kind, retryable = classify_error(RuntimeError(message))
    assert retryable is False
    assert kind == expected


def test_quota_beats_auth_on_shared_status_code() -> None:
    """额度耗尽与鉴权失败的 HTTP 码会撞，必须先判额度。

    百炼实测返回 403 + AllocationQuota.FreeTierOnly。若先判 auth，"403"
    一命中就报成"API key 无效"，让人去查错方向——真正要做的是充值或关掉
    "仅用免费额度"开关。两者都不可重试，所以不影响控制流，但会让 M3 的
    失败原因统计归错类。
    """
    real_message = (
        "Error code: 403 - Free quota exhausted. To continue accessing the model "
        "on a paid basis, please add funds or disable the use-free-tier-only mode "
        "in the management console. code=AllocationQuota.FreeTierOnly"
    )
    kind, retryable = classify_error(RuntimeError(real_message))
    assert kind == "quota"
    assert retryable is False


def test_fatal_wins_over_transient() -> None:
    """限流响应里也可能带 connection 字样。

    顺序反了会把余额不足当成网络抖动，一直重试到把重试次数耗光。
    """
    kind, retryable = classify_error(RuntimeError("connection closed: insufficient balance"))
    assert kind == "quota"
    assert retryable is False


def test_unknown_error_is_not_retried() -> None:
    """认不出来就别重试——盲目重试只是把一次失败变成两次。"""
    assert classify_error(RuntimeError("某种没见过的错")) == ("unknown", False)


# ===================================================================== #
# 响应拆解
# ===================================================================== #


def test_text_from_plain_content() -> None:
    assert _response_text(FakeResponse("hello")) == "hello"


def test_text_from_content_blocks() -> None:
    """带 reasoning 的模型会返回内容块列表。"""
    blocks = [{"type": "text", "text": "第一段"}, {"type": "text", "text": "第二段"}]
    assert _response_text(FakeResponse(blocks)) == "第一段\n第二段"


def test_usage_from_usage_metadata() -> None:
    usage = _extract_usage(
        FakeResponse("x", usage_metadata={"input_tokens": 1200, "output_tokens": 45})
    )
    assert usage.prompt_tokens == 1200
    assert usage.total_tokens == 1245


def test_usage_falls_back_to_response_metadata() -> None:
    """各平台经兼容层透传的字段名不完全一致，要兜一道。"""
    usage = _extract_usage(
        FakeResponse(
            "x", response_metadata={"token_usage": {"prompt_tokens": 10, "completion_tokens": 2}}
        )
    )
    assert usage.prompt_tokens == 10


def test_cached_tokens_are_read() -> None:
    usage = _extract_usage(
        FakeResponse(
            "x",
            usage_metadata={
                "input_tokens": 1000,
                "output_tokens": 10,
                "input_token_details": {"cache_read": 800},
            },
        )
    )
    assert usage.cached_tokens == 800


def test_missing_usage_records_zero_not_an_estimate() -> None:
    """读不到就记 0。估出来的 token 数会污染成本实测数据。"""
    usage = _extract_usage(FakeResponse("x"))
    assert usage.total_tokens == 0


def test_request_id_extracted() -> None:
    assert _request_id(FakeResponse("x", response_metadata={"id": "req-42"})) == "req-42"


def test_request_id_absent() -> None:
    assert _request_id(FakeResponse("x")) == ""


# ===================================================================== #
# predict_action
# ===================================================================== #


def test_predict_action_happy_path(backend) -> None:
    wire(
        backend,
        '{"action":"left_click","x":100,"y":200,"thinking":"点地址栏"}',
        usage={"input_tokens": 1500, "output_tokens": 30},
    )
    intent = backend.predict_action("打开浏览器", _shot())
    assert intent.action_type == "left_click"
    assert intent.params == {"x": 100, "y": 200}
    assert intent.usage.prompt_tokens == 1500
    assert intent.latency_ms > 0


def test_predict_action_keeps_raw_text(backend) -> None:
    """原文必须留着：解析失败时它是唯一线索，也是复盘的依据。"""
    raw = '```json\n{"action":"wait","duration":1}\n```'
    wire(backend, raw)
    assert backend.predict_action("x", _shot()).raw_text == raw


def test_predict_action_tolerates_messy_output(backend) -> None:
    """脏输出不该被判失败——那等于用解析层的脆弱把模型能力测低。"""
    wire(backend, '好的。\n{"action"："click"，"point":[512,384]}\n以上。')
    intent = backend.predict_action("x", _shot())
    assert intent.action_type == "left_click"
    assert intent.params == {"x": 512, "y": 384}


def test_unparseable_output_is_retryable(backend) -> None:
    """解析失败判为可重试：模型下一次很可能就给对了。"""
    wire(backend, "我觉得应该点那个按钮。")
    with pytest.raises(LLMBackendError) as info:
        backend.predict_action("x", _shot())
    assert info.value.kind == "parse_error"
    assert info.value.retryable is True


def test_api_error_is_classified(backend) -> None:
    wire(backend, "", error=RuntimeError("429 rate limit exceeded"))
    with pytest.raises(LLMBackendError) as info:
        backend.predict_action("x", _shot())
    assert info.value.retryable is True


def test_usage_is_recorded_even_when_parsing_fails(backend) -> None:
    """调用已经发生、钱已经花了，用量就必须记上。

    只在成功路径上计费会让实测成本偏低，而成本实测正是 M2 的交付物。
    """
    wire(backend, "这不是 JSON", usage={"input_tokens": 900, "output_tokens": 12})
    with pytest.raises(LLMBackendError):
        backend.predict_action("x", _shot())
    assert backend.get_cost().prompt_tokens == 900


def test_done_response(backend) -> None:
    wire(backend, '{"done": true, "thinking": "已完成"}')
    assert backend.predict_action("x", _shot()).done is True


# ===================================================================== #
# 消息构造
# ===================================================================== #


def test_messages_include_system_and_image(backend) -> None:
    backend.system_prompt = "你是一个桌面助手。"
    messages = backend.build_messages("点开始菜单", _shot(), [])
    assert messages[0].content == "你是一个桌面助手。"

    blocks = messages[-1].content
    assert blocks[0]["type"] == "text"
    assert blocks[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_few_shot_sits_before_history(backend) -> None:
    """few-shot 属于规则说明。混进历史里模型会以为那是本次真发生过的步骤。"""
    backend.system_prompt = "系统"
    backend.few_shot = [{"input": "示例输入", "output": '{"action":"left_click","x":1,"y":1}'}]
    history = [HistoryStep(thinking="历史一", screenshot=None)]

    messages = backend.build_messages("目标", _shot(), history)
    contents = [m.content if isinstance(m.content, str) else "<image>" for m in messages]
    assert contents.index("示例输入") < contents.index("（无动作）")


def test_only_recent_history_carries_images(backend) -> None:
    """每张图都要按 token 计费，而三步之前的界面对当前决策几乎没价值。"""
    backend.history_images = 1
    history = [HistoryStep(screenshot=_shot(320, 240)) for _ in range(3)]

    messages = backend.build_messages("目标", _shot(), history)
    image_messages = [m for m in messages if isinstance(m.content, list)]
    assert len(image_messages) == 2  # 历史里 1 张 + 当前 1 张


def test_history_without_images_is_still_summarized(backend) -> None:
    """文字摘要保留"做过什么、成没成"，几乎不花钱。"""
    from control.actions import Action

    history = [HistoryStep(action=Action(type="left_click", x=1, y=2), success=False, error="越界")]
    messages = backend.build_messages("目标", _shot(), history)
    assert any("越界" in str(m.content) for m in messages)


def test_prompt_states_the_coordinate_space(backend) -> None:
    """告诉模型按哪把尺子作答——**是坐标系尺寸，不是图片尺寸**。

    这条原来断言的是 image_size，把一个缺陷钉住了：系统提示用坐标系
    尺寸渲染（「x ∈ [0, 1000)」），用户消息却用图片尺寸（「坐标按
    1024×768 的范围给出」），同一次请求里两句话互相矛盾。矛盾的后果不是
    报错，是模型按哪把尺子作答不确定——看起来像「模型定位不稳」。

    见 `LLMBackend.coordinate_space`。
    """
    messages = backend.build_messages("目标", _shot(), [])
    text = messages[-1].content[0]["text"]
    space = backend.coordinate_space
    assert f"{space[0]}×{space[1]}" in text
    # 图片尺寸恰好等于坐标系时这条断言会失去意义，先钉住两者确实不同
    assert tuple(space) != tuple(DEFAULT_IMAGE_SIZE)
    assert f"{DEFAULT_IMAGE_SIZE[0]}×{DEFAULT_IMAGE_SIZE[1]}" not in text


def test_image_meta_is_exposed(backend) -> None:
    """图被压到多小、有没有被迫降分辨率，要能进轨迹日志。"""
    backend.build_messages("目标", _shot(), [])
    assert backend.last_image_meta["bytes"] > 0
    assert backend.last_image_meta["jpeg_quality"] == 85


def test_backend_name_follows_provider(monkeypatch) -> None:
    """轨迹里的 backend 字段要能区分三家，M3 横评按它分组。"""
    monkeypatch.setenv("ZHIPU_API_KEY", "zk-1")
    assert OpenAICompatBackend(config=resolve("zhipu")).name == "zhipu"
