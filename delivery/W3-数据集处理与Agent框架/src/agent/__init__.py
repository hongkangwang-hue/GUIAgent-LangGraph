"""Agent 编排层 —— 任务拆解、提示词组织、上下文管理。

三个模块各管一件事：

- `prompts`：提示词模板库。YAML 外置、版本化，M3 消融靠它换版本而不改代码
- `planner`：把一句指令拆成一串原子级子任务
- `context`：决定每步给模型看多少历史、看多清楚

**这一层不知道后端是哪家的。** 提示词配置（`system_prompt` / `few_shot` /
`user_template`）声明在 `LLMBackend` 基类上，因此换后端时这里一行不用改
——这是 M2 验收标准 8 的具体含义。
"""

from agent.context import (
    DEFAULT_POLICY,
    FRUGAL,
    RICH,
    ContextFrame,
    ContextPolicy,
    ContextWindow,
    Conversation,
)
from agent.planner import MAX_SUBTASKS, Plan, PlanError, Planner, SubTask
from agent.prompts import (
    PROMPT_DIR,
    FewShotExample,
    PromptError,
    PromptTemplate,
    list_templates,
    load_template,
    render_action_reference,
)
from agent.session import Session, SessionConfig, SessionResult, SubtaskOutcome

__all__ = [
    "DEFAULT_POLICY",
    "FRUGAL",
    "MAX_SUBTASKS",
    "PROMPT_DIR",
    "RICH",
    "ContextFrame",
    "ContextPolicy",
    "ContextWindow",
    "Conversation",
    "FewShotExample",
    "Plan",
    "PlanError",
    "Planner",
    "PromptError",
    "PromptTemplate",
    "Session",
    "SessionConfig",
    "SessionResult",
    "SubTask",
    "SubtaskOutcome",
    "list_templates",
    "load_template",
    "render_action_reference",
]
