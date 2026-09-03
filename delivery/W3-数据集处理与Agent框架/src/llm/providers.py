"""模型平台注册表 —— 三家 OpenAI 兼容端点的差异都收在这里。

## 为什么不是 QwenVLAPIBackend

M2 文档原计划写一个 `QwenVLAPIBackend`，M3 再补 GLM-4V 与 Llama 的实现。
但实际可用的三家平台**都提供 OpenAI 兼容端点**：

| 平台 | 模型 | 端点 |
|---|---|---|
| 阿里云百炼 | qwen3-vl-8b-instruct | dashscope.aliyuncs.com/compatible-mode/v1 |
| 智谱开放平台 | glm-4.6v-flash | open.bigmodel.cn/api/paas/v4 |
| NVIDIA NIM | meta/llama-3.2-11b-vision-instruct | integrate.api.nvidia.com/v1 |

既然协议是同一个，写三个类就是把同一份逻辑抄三遍——而 M3 的横评恰恰
要求三者跑在**完全相同的代码路径**上，否则跑出来的差异分不清是模型带来
的还是实现带来的。所以只有一个 `OpenAICompatBackend`，平台差异全部降级
成这张表里的数据。

M2 验收标准第 8 条"新增一个模型后端不需要改动 Agent 层与 Loop 层"，在
这个结构下变成了更强的一条：**新增平台连后端类都不用改，加一条记录即可。**

## 单价为什么默认是 None

项目规矩：不猜单价。假成本比"未知"更有害——M0 的成本估算要靠 M2 的实测
数据校准，掺了猜测就白测了。

代价是成本熔断在没配单价时不生效（`AgentLoop._over_budget` 会跳过）。
这是有意的取舍，且已在 `CostInfo.priced` 上标明。要启用熔断，就去平台
的计费页面查到真实单价填进 .env，那才是能进报告的数字。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from llm.base import PriceSheet

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Provider:
    """一个模型平台的接入参数与已知限制。"""

    key: str
    label: str
    #: 环境变量名
    api_key_env: str
    base_url_env: str
    model_env: str
    default_base_url: str
    default_model: str

    #: 内联图片的字节上限。超过要降质量或降分辨率。
    #: 0 表示没有已知限制
    max_image_bytes: int = 0

    #: 该平台已知的坑，进日志与报告
    notes: str = ""

    #: **权重跑在自己的机器上。** 一个事实，两个后果。
    #:
    #: 后果一：**数据不出本机**。这是「离线版」的定义，`llm.factory.is_offline`
    #: 靠它判断。判据必须绑在这个字段上而不是后端类型上——同一个
    #: `OpenAICompatBackend`，连百炼是在线、连宿主机的 selfhost 是离线，
    #: 光看类型分不出来。分不出来的后果很具体：存档文件名标错、
    #: 「数据边界」那一栏印反、离线跑覆盖掉 M2 的在线交付数据。
    #:
    #: 后果二：**API 费用确实是 0**，而不是「单价未知」。这个区分不是
    #: 文字游戏。本模块开头写着「不猜单价」，所以未配单价时 `CostInfo.priced`
    #: 置 False，意思是「这个数字不可信」。而自建服务的 0 元是能写进报告的
    #: 事实。两者都显示 0.0000，只有 `priced` 分得清哪个是「确认的 0」。
    #:
    #: 离线版不是没有成本，成本转移到了硬件与耗时上——那部分由延迟数据
    #: 体现，不该硬折算成钱塞进单价表。口径与 `llm.qwen_vl_local.LOCAL_PRICE` 一致。
    #:
    #: ⚠ 只有**自己机器上**的服务才置 True。短租的远程服务器权重在别人
    #: 机器上，截图发出去就出了本机，那只是 API 调用。
    weights_local: bool = False


#: 单个 base64 内联图片的保守上限。
#:
#: NVIDIA NIM 的文档写明：内联图片超过 180KB 就必须改走 NVCF 资产上传
#: 接口，那是另一套协议，不在 OpenAI 兼容路径上。桌面截图很容易超——
#: 2560×1600 的 PNG 动辄几 MB。因此这里按 JPEG 编码并自适应降质量。
#:
#: 取 150KB 而不是 180KB：base64 编码会让体积涨约 33%，留出余量。
NVIDIA_INLINE_LIMIT = 150 * 1024


PROVIDERS: dict[str, Provider] = {
    "dashscope": Provider(
        key="dashscope",
        label="阿里云百炼",
        api_key_env="DASHSCOPE_API_KEY",
        base_url_env="DASHSCOPE_BASE_URL",
        model_env="PLANNER_MODEL",
        default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_model="qwen3-vl-8b-instruct",
        notes="OpenAI 兼容模式。Qwen-VL 系列原生支持 grounding，是模式 A 的主力",
    ),
    "zhipu": Provider(
        key="zhipu",
        label="智谱开放平台",
        api_key_env="ZHIPU_API_KEY",
        base_url_env="ZHIPU_BASE_URL",
        model_env="ZHIPU_MODEL",
        default_base_url="https://open.bigmodel.cn/api/paas/v4",
        default_model="glm-4.6v-flash",
        notes="flash 档为免费或低价模型，适合做横评的成本对照组",
    ),
    "selfhost": Provider(
        key="selfhost",
        label="自建本地服务",
        # 本服务不校验鉴权，但仍走 api_key 这条路：`resolve()` 的
        # require_key 校验对所有平台一视同仁，为它开特例反而多一条分支
        api_key_env="SELFHOST_API_KEY",
        base_url_env="SELFHOST_BASE_URL",
        model_env="SELFHOST_MODEL",
        default_base_url="http://127.0.0.1:8000/v1",
        default_model="Qwen/Qwen2.5-VL-3B-Instruct",
        notes=(
            "宿主机上的 scripts/serve_local_model.py（或 vLLM）。"
            "**权重在本机 GPU 上跑，属于大纲 W3 的「本地部署」**——"
            "M0《硬件与部署环境》已裁定：客机没有 GPU 直通、而键鼠执行只能在客机内，"
            "两条约束逼出这个拓扑，HTTP 只是传输方式，数据不出本机。"
            "注意与短租远程服务器区分：那种情况权重不在自己机器上，只算 API 调用"
        ),
        weights_local=True,
    ),
    "nvidia": Provider(
        key="nvidia",
        label="NVIDIA NIM",
        api_key_env="NVIDIA_API_KEY",
        base_url_env="NVIDIA_BASE_URL",
        model_env="NVIDIA_MODEL",
        default_base_url="https://integrate.api.nvidia.com/v1",
        default_model="meta/llama-3.2-11b-vision-instruct",
        max_image_bytes=NVIDIA_INLINE_LIMIT,
        notes=(
            "内联图片有硬上限，超过要走 NVCF 资产上传（另一套协议）。"
            "桌面截图必须压到限制内，否则整条请求被拒"
        ),
    ),
}

#: 没指定平台时用哪个
DEFAULT_PROVIDER = "dashscope"


class ProviderNotConfigured(RuntimeError):
    """平台缺少必要配置（通常是 API key）。"""


@dataclass
class ProviderConfig:
    """解析完环境变量之后的实际接入参数。"""

    provider: Provider
    api_key: str
    base_url: str
    model: str
    price: PriceSheet | None = None

    @property
    def name(self) -> str:
        return self.provider.key

    @property
    def max_image_bytes(self) -> int:
        return self.provider.max_image_bytes

    def masked(self) -> dict:
        """可以安全打进日志与 `config --show` 的形式。

        **key 只留头尾**。轨迹日志、控制台输出、贴给别人看的报错，都可能
        把这个 dict 带出去；一次疏忽就是一把废掉的凭据。
        """
        return {
            "provider": self.provider.key,
            "label": self.provider.label,
            "model": self.model,
            "base_url": self.base_url,
            "api_key": _mask(self.api_key),
            "priced": self.price is not None,
        }


def _mask(secret: str) -> str:
    if not secret:
        return "（未配置）"
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}…{secret[-4:]}"


def _price_from_env(provider: Provider) -> PriceSheet | None:
    """从 .env 读单价。没配就返回 None —— 不猜。

    变量名形如 ``DASHSCOPE_PRICE_IN_PER_1K`` / ``DASHSCOPE_PRICE_OUT_PER_1K``，
    单位元/千 token，去平台计费页面抄。
    """
    prefix = provider.api_key_env.rsplit("_API_KEY", 1)[0]
    raw_in = _nonblank(f"{prefix}_PRICE_IN_PER_1K")
    raw_out = _nonblank(f"{prefix}_PRICE_OUT_PER_1K")
    if provider.weights_local and raw_in is None and raw_out is None:
        # 自建服务：0 是查证过的事实，不是「没查」。**这不违反「不猜单价」**
        # ——那条规矩针对的是商业平台的真实单价，而自建服务根本没有账单。
        # 仍然允许用 .env 覆盖：真要把电费折算进去时，填上就是了。
        return PriceSheet(model=provider.default_model, input_per_1k=0.0, output_per_1k=0.0)
    # 两个都没填才算未配置。**空字符串等同于没填**——.env 里留着
    # `ZHIPU_PRICE_IN_PER_1K=` 这样的空行是常态，os.getenv 对它返回 ""
    # 而不是 None，按"已配置"处理会造出一张 0 元单价表并标成可信，
    # 于是成本恒为 0 却看起来像真的。这正是本模块开头要避免的那种数字。
    if raw_in is None and raw_out is None:
        return None
    try:
        return PriceSheet(
            model=provider.default_model,
            input_per_1k=float(raw_in or 0.0),
            output_per_1k=float(raw_out or 0.0),
            cached_input_per_1k=_optional_float(f"{prefix}_PRICE_CACHED_IN_PER_1K"),
        )
    except ValueError:
        logger.warning("%s 的单价配置不是数字，按未配置处理", provider.label)
        return None


def _nonblank(name: str) -> str | None:
    """环境变量的值，空白等同于未设置。"""
    raw = os.getenv(name)
    return raw.strip() if raw and raw.strip() else None


def _optional_float(name: str) -> float | None:
    raw = _nonblank(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def load_dotenv_if_present(path: str = ".env") -> bool:
    """把 .env 读进环境变量。

    自己解析而不是拉 python-dotenv：需求只有"KEY=VALUE 一行一条"，
    为此多一个依赖不划算。**已存在的环境变量不覆盖**——命令行里显式
    设的值应当优先于文件。

    ## 编码要容错

    Windows PowerShell 5.1 的 `Add-Content` 默认写 ANSI（简中环境即 GBK），
    往 .env 里追加一行带中文的内容就会混进非 UTF-8 字节。此时严格按 UTF-8
    读会抛 `UnicodeDecodeError`，堆栈停在 codecs 内部——**看不出是哪个
    文件、哪一行、什么字符**，而真正的问题只是某一行编码不对。

    所以按 UTF-8 → GBK → 忽略错误 三级降级读，并在降级时明确告知。
    读得进来比读得纯粹重要：一个坏掉的 .env 不该让整个程序起不来。
    """
    if not os.path.exists(path):
        return False

    text = None
    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            text = Path(path).read_text(encoding=encoding)
        except (UnicodeDecodeError, LookupError):
            continue
        if encoding != "utf-8":
            print(
                f"[提示] {path} 不是 UTF-8（按 {encoding} 读出来了）。"
                f"多半是 PowerShell 的 Add-Content 写进了 ANSI 字节，"
                f"建议用记事本另存为 UTF-8。"
            )
        break
    else:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        print(f"[警告] {path} 编码无法识别，已忽略坏字节读入。请检查文件内容。")

    seen: dict[str, int] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if not key:
            continue
        # 同名多行是手工追加时的常见事故，先到的生效，后面的提示出来。
        # 静默取其一会让人对着一个"明明改了却不生效"的文件排查很久。
        if key in seen:
            print(f"[提示] {path} 第 {number} 行的 {key} 重复，沿用第 {seen[key]} 行的值。")
            continue
        seen[key] = number
        if key not in os.environ:
            os.environ[key] = value
    return True


def resolve(
    name: str | None = None,
    model: str | None = None,
    require_key: bool = True,
    base_url: str | None = None,
) -> ProviderConfig:
    """按平台名解析出可用的接入参数。

    ``model`` 显式传入时覆盖环境变量——横评要在同一平台上换模型跑。
    ``base_url`` 同理，自建服务的地址随机器变，写死在 .env 里不如命令行给。
    """
    key = (name or os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    if key not in PROVIDERS:
        raise ProviderNotConfigured(f"未知平台 {key!r}。已注册：{', '.join(sorted(PROVIDERS))}")
    provider = PROVIDERS[key]

    api_key = os.getenv(provider.api_key_env, "").strip()
    if provider.weights_local and not api_key:
        # **自建服务不校验鉴权，就别逼人去 .env 里填一个假值。**
        #
        # 原来要求所有平台一律配 key，于是客机上多一步手工编辑 .env——
        # 而那一步除了「可能改错、可能存成错的编码」之外不产生任何价值。
        # 少一个手工步骤就少一处出错的地方。
        #
        # 仍然给一个非空占位符：langchain 的 ChatOpenAI 拿到空 key 会在
        # 构造时就报错，而那个错会指向「凭据没配」，与真实原因无关。
        api_key = "local"
    elif require_key and not api_key:
        raise ProviderNotConfigured(
            f"{provider.label} 缺少 API key：请在 .env 里设置 {provider.api_key_env}。"
            f"（.env 已在 .gitignore 中，不会被提交）"
        )

    return ProviderConfig(
        provider=provider,
        api_key=api_key,
        base_url=(
            (base_url or "").strip()
            or os.getenv(provider.base_url_env, "").strip()
            or provider.default_base_url
        ),
        model=(model or os.getenv(provider.model_env, "").strip() or provider.default_model),
        price=_price_from_env(provider),
    )


def available_providers() -> list[tuple[str, bool]]:
    """(平台名, key 是否已配置)。`config --show` 与选型脚本用。"""
    return [
        (key, bool(os.getenv(provider.api_key_env, "").strip()))
        for key, provider in sorted(PROVIDERS.items())
    ]
