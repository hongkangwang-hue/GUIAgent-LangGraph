"""离线后端与在线后端的**同构性**测试。

## 这里守的是什么

老师补充的要求是「做一个在线版和一个离线版」，而两个版本要能对比，
前提是它们只在「请求发到哪」这一点上不同。一旦消息顺序、提示词渲染、
坐标空间任何一处漂移，跑出来的差值里就混进了实现差异——**而这种漂移
不会抛异常，只会让报告里的结论悄悄变成错的**。

所以这些用例几乎都是「两边必须相等」的形式，不是在测某个功能好不好用。

全部不加载模型权重：构造与消息构造都不碰 GPU，只有 `complete()` 才会。
"""

from __future__ import annotations

import numpy as np
import pytest

from agent.session import SessionConfig
from control.actions import Action
from finetune.dataset import COORD_SPACE
from llm.base import HistoryStep, TokenUsage
from llm.factory import build_backend, describe, is_offline
from llm.qwen_vl_local import QwenVLLocalBackend
from perception.capture import Screenshot
from perception.types import BBox


def _shot(width: int = 2560, height: int = 1600) -> Screenshot:
    return Screenshot(
        image=np.zeros((height, width, 3), dtype=np.uint8),
        region=BBox(0, 0, width, height),
        engine="fake",
    )


class TestCoordSpaceChain:
    def test_训练坐标空间与运行坐标空间一致(self):
        """训练输出 [0,1000)，Agent 也按 [0,1000) 换算，两者必须相等。

        不等的话，微调后的模型接进来每一次点击都系统性偏移，而且偏移量
        随目标位置变化——看起来极像「模型定位不准」，实际是尺子不同。
        `CoordinateScaler` 一行都不用改，靠的就是这条。
        """
        assert SessionConfig().coordinate_space == COORD_SPACE


class TestPromptRendering:
    """系统提示与用户消息不能互相矛盾。

    这是一个实测出来的缺陷：`_prepare_backend` 用坐标系尺寸渲染系统提示，
    却把 `image_size` 交给用户模板，于是 M2 的每一次请求都同时说了
    「x ∈ [0, 1000)」和「坐标按 1024×768 的范围给出」。
    """

    def test_用户模板按坐标系渲染而不是图片尺寸(self):
        backend = QwenVLLocalBackend()
        backend.coordinate_space = (1000, 1000)
        backend.image_size = (1024, 768)
        backend.user_template = "坐标按 {width}×{height} 的范围给出"
        assert backend._user_prompt("x") == "坐标按 1000×1000 的范围给出"

    def test_图片尺寸仍然可取(self):
        """需要描述图片本身时用 image_width/image_height，两者不再混用。"""
        backend = QwenVLLocalBackend()
        backend.image_size = (1024, 768)
        backend.user_template = "这张图是 {image_width}×{image_height}"
        assert backend._user_prompt("x") == "这张图是 1024×768"

    def test_模板里的_json_花括号不被当占位符(self):
        """模板里有 done 那个 JSON 示例，用 str.format 会直接炸。"""
        backend = QwenVLLocalBackend()
        backend.user_template = '目标：{instruction}。完成则输出 {"done": true}'
        assert backend._user_prompt("打开记事本") == '目标：打开记事本。完成则输出 {"done": true}'

    def test_两个后端的渲染结果逐字相同(self):
        """在线与离线必须给模型看一模一样的文字。"""
        from llm.openai_compat import OpenAICompatBackend
        from llm.providers import Provider, ProviderConfig

        config = ProviderConfig(
            provider=Provider(
                key="t",
                label="t",
                api_key_env="X",
                base_url_env="Y",
                model_env="Z",
                default_base_url="http://x",
                default_model="m",
            ),
            api_key="k",
            base_url="http://x",
            model="m",
        )
        online = OpenAICompatBackend(config)
        offline = QwenVLLocalBackend()
        template = "目标：{instruction}，坐标按 {width}×{height}，图 {image_width}×{image_height}"
        for backend in (online, offline):
            backend.user_template = template
            backend.coordinate_space = (1000, 1000)
            backend.image_size = (1024, 768)
        assert online._user_prompt("点这里") == offline._user_prompt("点这里")


class TestBuildMessages:
    def test_消息顺序与在线后端一致(self):
        """系统 → few-shot → 历史 → 当前。顺序不同，上下文结构就不可比。"""
        backend = QwenVLLocalBackend(
            system_prompt="SYS", few_shot=[{"input": "IN", "output": "OUT"}]
        )
        history = [HistoryStep(action=Action(type="left_click", x=1, y=2), screenshot=_shot())]
        messages, _ = backend.build_messages("点击地址栏", _shot(), history)
        assert [m["role"] for m in messages] == [
            "system",
            "user",  # few-shot 输入
            "assistant",  # few-shot 输出
            "assistant",  # 历史动作摘要
            "user",  # 历史截图
            "user",  # 当前截图与目标
        ]

    def test_图片数量与占位符数量相等(self):
        """HF 的 processor 按占位符顺序去 images 列表里对应。

        **差一个，模型看到的就是上一帧。** 这种错不会报错，只会让每一步
        都基于过期的画面决策，症状是「模型总慢半拍」。
        """
        backend = QwenVLLocalBackend()
        history = [
            HistoryStep(action=Action(type="left_click", x=1, y=2), screenshot=_shot()),
            HistoryStep(action=Action(type="left_click", x=3, y=4), screenshot=_shot()),
        ]
        messages, images = backend.build_messages("目标", _shot(), history)
        placeholders = sum(1 for m in messages for part in m["content"] if part["type"] == "image")
        assert placeholders == len(images)

    def test_只有最近一步带图(self):
        """默认 history_images=1。

        旧帧的边际价值衰减很快，而每张图都要吃视觉 token —— 3B 模型在
        8GB 卡上很快就顶到上限。策略与 API 后端一致，两边上下文长度才可比。
        """
        backend = QwenVLLocalBackend()
        history = [
            HistoryStep(action=Action(type="left_click", x=i, y=i), screenshot=_shot())
            for i in range(4)
        ]
        _, images = backend.build_messages("目标", _shot(), history)
        assert len(images) == 2  # 最近一步的历史图 + 当前图

    def test_历史帧按_image_scale_降采样(self):
        backend = QwenVLLocalBackend()
        backend.image_size = (1024, 768)
        history = [
            HistoryStep(
                action=Action(type="left_click", x=1, y=2),
                screenshot=_shot(),
                with_image=True,
                image_scale=0.5,
            )
        ]
        _, images = backend.build_messages("目标", _shot(), history)
        assert (images[0].width, images[0].height) == (512, 384)
        assert (images[1].width, images[1].height) == (1024, 768)

    def test_没有截图时走纯文本(self):
        """Planner 可以配成不看屏幕拆解，那时不该硬塞一张图。"""
        backend = QwenVLLocalBackend()
        messages, images = backend.build_messages("拆解任务", None, [])
        assert images == []
        assert all(part["type"] == "text" for m in messages for part in m["content"])


class TestPinPrompt:
    def test_默认接受外部注入的提示词(self):
        """默认不钉死：离线与在线跑同一套提示词，差值才是模型的差值。"""
        backend = QwenVLLocalBackend(adapter="fake/adapter")
        backend.system_prompt = "EXTERNAL"
        assert backend.system_prompt == "EXTERNAL"

    def test_钉死时用训练时的提示词(self):
        """跑「分布内」对照组时才打开。"""
        from finetune.train_lora import SYSTEM_PROMPT

        backend = QwenVLLocalBackend(pin_prompt=True)
        assert backend.system_prompt == SYSTEM_PROMPT
        assert backend.user_template == "{instruction}"

    def test_挂_adapter_会告警动作空间被收窄(self, caplog):
        """adapter 只见过四个点击类动作，不认识 type / wait / done。

        **这条不能静默。** 端到端成功率如果因此上不去，读报告的人要能在
        日志里直接看到原因，而不是去猜模型为什么从不报完成。
        """
        with caplog.at_level("WARNING"):
            QwenVLLocalBackend(adapter="fake/adapter")
        assert any("分布外" in record.getMessage() for record in caplog.records)


class TestCostAccounting:
    def test_离线的零成本是可信的零而不是未知(self):
        """基类在 `price is None` 时把 `priced` 置 False，意思是「不可信」。

        离线版的 API 费用**确实**是 0 元，是能写进报告的事实。给一张零
        单价表，读报告的人才分得清「0」和「不知道」。
        """
        backend = QwenVLLocalBackend()
        cost = backend.record_usage(TokenUsage(prompt_tokens=1200, completion_tokens=30))
        assert cost == 0.0
        assert backend.get_cost().priced is True
        assert backend.get_cost().total_tokens == 1230


class TestClose:
    def test_未加载时_close_不报错(self):
        """构造了但没调用过，close 应当是空操作。"""
        QwenVLLocalBackend().close()

    def test_可以当上下文管理器用(self):
        with QwenVLLocalBackend() as backend:
            assert backend.model


class TestFactory:
    def test_按_provider_选后端(self):
        backend = build_backend(provider="local")
        assert isinstance(backend, QwenVLLocalBackend)
        assert is_offline(backend)

    def test_离线标记用类型判断而不是名字(self):
        """名字可以被 --model 改花，类型不会。"""
        backend = build_backend(provider="local", model="随便什么名字")
        assert is_offline(backend)
        assert "离线" in describe(backend)

    def test_adapter_只对本地后端生效(self):
        backend = build_backend(
            provider="local", adapter="finetune/outputs/x/adapter", pin_prompt=True
        )
        assert backend.adapter.endswith("adapter")
        assert "adapter" in describe(backend)


@pytest.mark.parametrize("field", ["max_pixels", "min_pixels"])
def test_视觉_token_上下限与训练时一致(field):
    """训练时 processor 配的是 min 256*28*28 / max 896×896。

    推理时不配，一张 2560×1600 的截图会产生 2000+ 个视觉 token：既撑爆
    显存，也让模型见到的分辨率与训练时不是一回事。
    """
    from finetune.train_lora import DEFAULT_MAX_PIXELS

    backend = QwenVLLocalBackend()
    expected = {"max_pixels": DEFAULT_MAX_PIXELS, "min_pixels": 256 * 28 * 28}[field]
    assert getattr(backend, field) == expected
