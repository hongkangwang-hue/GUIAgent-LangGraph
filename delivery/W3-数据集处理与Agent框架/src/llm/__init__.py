"""LLM 后端抽象层。

三家平台（阿里云百炼 / 智谱 / NVIDIA NIM）都提供 OpenAI 兼容端点，因此
只有 `OpenAICompatBackend` 一个实现，平台差异全部降级成 `providers` 里
的数据。M3 横评时切换平台不改代码，改 `LLM_PROVIDER` 即可。

``langchain_openai`` 与 ``langchain_core`` 都在函数内部导入，因此
``import llm`` 本身很轻——只想用 `ScriptedBackend` 跑单元测试的场景不该
为几秒钟的框架导入买单。
"""

from llm.base import (
    ActionIntent,
    CostInfo,
    HistoryStep,
    LLMBackend,
    LLMBackendError,
    PriceSheet,
    TokenUsage,
)
from llm.fake import ScriptedBackend
from llm.openai_compat import OpenAICompatBackend
from llm.parsing import OutputParseError, parse_action_payload
from llm.providers import (
    PROVIDERS,
    ProviderConfig,
    ProviderNotConfigured,
    available_providers,
    load_dotenv_if_present,
    resolve,
)

__all__ = [
    "PROVIDERS",
    "ActionIntent",
    "CostInfo",
    "HistoryStep",
    "LLMBackend",
    "LLMBackendError",
    "OpenAICompatBackend",
    "OutputParseError",
    "PriceSheet",
    "ProviderConfig",
    "ProviderNotConfigured",
    "ScriptedBackend",
    "TokenUsage",
    "available_providers",
    "load_dotenv_if_present",
    "parse_action_payload",
    "resolve",
]
