"""Agent 编排层的单元测试。

三块各有各的要害：

- **提示词**：动作清单必须来自 `ACTION_SPECS`，不能是 YAML 里手抄的一份。
  抄一份之后动作参数一改就过期，而模型照着过期说明输出会在
  `Action.validate` 处被拒，报错却指向"模型不听话"，极难查。
- **Planner**：拆得够不够细要**程序化可判**，否则 M2 验收标准 4 的
  "可验证"就变成各说各话。
- **上下文**：降采样不等于丢弃。丢弃会让模型忘掉是自己点开了菜单，
  然后反复点同一个地方（`label` 词表里的 stuck_loop）。
"""

from __future__ import annotations

import pytest

from agent.context import (
    DEFAULT_POLICY,
    FRUGAL,
    RICH,
    ContextPolicy,
    ContextWindow,
    Conversation,
)
from agent.planner import MAX_SUBTASKS, Plan, PlanError, Planner, SubTask
from agent.prompts import (
    PromptError,
    PromptTemplate,
    list_templates,
    load_template,
    render_action_reference,
)
from control.actions import ACTION_SPECS, CORE_ACTIONS, Action
from llm.base import HistoryStep
from llm.fake import ScriptedBackend


def _plan_backend(payload: str) -> ScriptedBackend:
    return ScriptedBackend([{"raw_text": payload, "done": True}])


def _steps(count: int, with_screenshot: bool = True) -> list[HistoryStep]:
    return [
        HistoryStep(
            action=Action(type="left_click", x=index, y=index),
            screenshot=object() if with_screenshot else None,
        )
        for index in range(count)
    ]


# ===================================================================== #
# 提示词模板
# ===================================================================== #


def test_templates_exist() -> None:
    assert {"executor_v1", "planner_v1"} <= set(list_templates())


def test_executor_template_loads() -> None:
    template = load_template("executor_v1")
    assert template.system
    assert template.user_template
    assert len(template.few_shot) >= 2  # M2 要求内置 2-3 组


def test_missing_template_lists_alternatives() -> None:
    with pytest.raises(PromptError) as info:
        load_template("查无此模板")
    assert "executor_v1" in str(info.value)


def test_action_reference_comes_from_action_specs() -> None:
    """动作清单必须从 ACTION_SPECS 生成。

    这是 M1 `control/actions.py` 立的规矩——"给模型看的"和"实际校验的"
    必须同源。这条测试锁住的是：新增一个核心动作后，提示词自动跟着变，
    不需要有人记得去改 YAML。
    """
    reference = render_action_reference()
    for action_type in CORE_ACTIONS:
        assert f"`{action_type.value}`" in reference


def test_action_reference_includes_parameter_names() -> None:
    reference = render_action_reference()
    for spec in ACTION_SPECS[next(iter(CORE_ACTIONS))][1]:
        assert spec.name in reference


def test_action_reference_marks_enum_values() -> None:
    """scroll 的方向是枚举，模型需要知道可选值。"""
    assert "up/down/left/right" in render_action_reference()


def test_system_prompt_injects_action_reference() -> None:
    rendered = load_template("executor_v1").render_system(width=1024, height=768)
    assert "{action_reference}" not in rendered
    assert "left_click" in rendered


def test_system_prompt_states_coordinate_range() -> None:
    """提示词里说的必须是**坐标系**范围，不是图片尺寸。

    实测 qwen3-vl 输出的是归一化到 [0,1000) 的坐标，与我们送多大的图无关。
    告诉它"截图是 1024×768"会误导——它按哪把尺子作答，就该告诉它那把尺子。
    """
    rendered = load_template("executor_v1").render_system(width=1000, height=1000)
    assert "[0, 1000)" in rendered
    assert "无论截图实际多大" in rendered


def test_json_braces_in_template_survive_rendering() -> None:
    """提示词里的 JSON 示例不能被格式化吃掉。

    用 str.format 的话 ``{"action": ...}`` 会被当占位符抛 KeyError，
    而 JSON 示例恰恰是本项目提示词里最重要的部分。
    """
    template = PromptTemplate(
        name="t",
        system='输出 {"action": "left_click", "x": 1} 这种格式。画布 {width}。',
    )
    rendered = template.render_system(width=1024)
    assert '{"action": "left_click", "x": 1}' in rendered
    assert "画布 1024" in rendered


def test_few_shot_pairs_shape() -> None:
    pairs = load_template("executor_v1").few_shot_pairs()
    assert all(set(p) == {"input", "output"} for p in pairs)


def test_few_shot_outputs_are_parseable() -> None:
    """few-shot 示范的输出格式必须是解析层真能吃下的。

    示例和解析器对不上是最隐蔽的一类错：模型老老实实照着学，然后每次
    都被判解析失败，而你会以为是模型不行。
    """
    from llm.parsing import parse_action_payload

    for example in load_template("executor_v1").few_shot:
        payload = parse_action_payload(example.output)
        assert payload["action_type"] or payload["done"]


def test_planner_few_shot_outputs_are_parseable() -> None:
    from llm.parsing import extract_json

    for example in load_template("planner_v1").few_shot:
        data = extract_json(example.output)
        assert isinstance(data.get("subtasks"), list)


def test_template_records_version_for_ablation() -> None:
    """M3 消融要能从轨迹里追溯用的是哪个提示词版本。"""
    payload = load_template("executor_v1").as_dict()
    assert payload["name"] == "executor_v1"
    assert payload["few_shot_count"] >= 2


def test_bad_few_shot_is_rejected(tmp_path) -> None:
    (tmp_path / "broken.yaml").write_text(
        "name: broken\nfew_shot:\n  - input: 只有输入\n", encoding="utf-8"
    )
    with pytest.raises(PromptError):
        load_template("broken", directory=tmp_path)


# ===================================================================== #
# Planner
# ===================================================================== #


def test_plan_parses_subtasks() -> None:
    backend = _plan_backend(
        '{"subtasks":[{"id":1,"goal":"点击开始按钮","expected":"菜单展开"},'
        '{"id":2,"goal":"输入记事本","expected":"出现搜索结果"}]}'
    )
    plan = Planner(backend).plan("打开记事本")
    assert plan.goals() == ["点击开始按钮", "输入记事本"]
    assert plan.subtasks[0].expected == "菜单展开"


def test_plan_rejects_empty_instruction() -> None:
    with pytest.raises(PlanError):
        Planner(_plan_backend("{}")).plan("   ")


def test_plan_rejects_unparseable_output() -> None:
    with pytest.raises(PlanError):
        Planner(_plan_backend("我觉得应该先打开浏览器")).plan("打开浏览器")


def test_plan_rejects_empty_subtask_list() -> None:
    with pytest.raises(PlanError):
        Planner(_plan_backend('{"subtasks": []}')).plan("打开浏览器")


@pytest.mark.parametrize("key", ["subtasks", "plan", "steps", "tasks"])
def test_plan_accepts_alternative_container_keys(key) -> None:
    """模型对容器字段名的叫法不统一，容错口径与 llm.parsing 一致。"""
    backend = _plan_backend(f'{{"{key}":[{{"goal":"点击开始按钮"}}]}}')
    assert Planner(backend).plan("x").goals() == ["点击开始按钮"]


def test_plan_accepts_plain_string_items() -> None:
    backend = _plan_backend('{"subtasks": ["点击开始按钮", "输入记事本"]}')
    assert len(Planner(backend).plan("x").subtasks) == 2


def test_plan_strips_numbering() -> None:
    """子任务已经有 id，描述里再带序号会拼出"目标：3. 按回车"这种别扭文本，
    而且模型给的序号常和截断后的实际顺序对不上。"""
    backend = _plan_backend(
        '{"subtasks":[{"goal":"1. 点击开始按钮"},{"goal":"第2步：输入记事本"}]}'
    )
    assert Planner(backend).plan("x").goals() == ["点击开始按钮", "输入记事本"]


def test_plan_renumbers_sequentially() -> None:
    backend = _plan_backend('{"subtasks":[{"id":7,"goal":"甲"},{"id":9,"goal":"乙"}]}')
    assert [s.id for s in Planner(backend).plan("x").subtasks] == [1, 2]


def test_plan_truncates_beyond_limit() -> None:
    """M2「明确不做」第 6 条：不做超过 8 步的复杂任务。

    截断而非报错——前 8 步照样能演示，且截断数进轨迹，事后看得出来。
    """
    items = ",".join(f'{{"goal":"步骤{i}"}}' for i in range(12))
    plan = Planner(_plan_backend(f'{{"subtasks":[{items}]}}')).plan("x")
    assert len(plan.subtasks) == MAX_SUBTASKS
    assert plan.truncated == 4


def test_plan_records_prompt_version() -> None:
    plan = Planner(_plan_backend('{"subtasks":[{"goal":"甲"}]}')).plan("x")
    assert plan.prompt["name"] == "planner_v1"


def test_plan_cost_joins_the_backend_account() -> None:
    """拆解复用同一个后端，成本自动并入总账——单任务成本是 M2 交付物。"""
    backend = _plan_backend('{"subtasks":[{"goal":"甲"}]}')
    Planner(backend).plan("x")
    assert backend.get_cost().requests == 1


def test_planner_restores_backend_prompt_config() -> None:
    """拆解会临时改后端的提示词配置，用完必须还原。

    不还原的话，紧接着的执行阶段会带着 planner 的系统提示去做单步决策，
    模型会继续输出子任务列表而不是动作——而且这个故障看起来像"模型不会
    执行"，根因很难指向这里。
    """
    backend = _plan_backend('{"subtasks":[{"goal":"甲"}]}')
    backend.system_prompt = "执行阶段的系统提示"
    backend.few_shot = [{"input": "a", "output": "b"}]
    original_template = backend.user_template

    Planner(backend).plan("x")

    assert backend.system_prompt == "执行阶段的系统提示"
    assert backend.few_shot == [{"input": "a", "output": "b"}]
    assert backend.user_template == original_template


def test_planner_restores_config_even_on_failure() -> None:
    backend = _plan_backend("不是 JSON")
    backend.system_prompt = "原始提示"
    with pytest.raises(PlanError):
        Planner(backend).plan("x")
    assert backend.system_prompt == "原始提示"


def test_planner_can_skip_screenshot() -> None:
    """纯文本拆解，M3 消融的一个对照臂。"""
    backend = _plan_backend('{"subtasks":[{"goal":"甲"}]}')
    Planner(backend, with_screenshot=False).plan("x", screenshot=object())
    assert backend.calls[0]["screenshot"] is None


# ===================================================================== #
# 粒度检查（M2 验收标准 4）
# ===================================================================== #


def test_fine_grained_plan_has_no_warnings() -> None:
    plan = Plan(
        subtasks=[
            SubTask(1, "点击任务栏的开始按钮", "开始菜单展开"),
            SubTask(2, "按回车键", "记事本窗口出现"),
        ]
    )
    assert plan.granularity_report() == []
    assert plan.is_fine_grained()


@pytest.mark.parametrize("word", ["并且", "然后", "接着", "同时", "以及"])
def test_conjunction_is_flagged(word) -> None:
    """连词是"塞了不止一件事"最可靠的信号，提示词里也明确禁止过。"""
    plan = Plan(subtasks=[SubTask(1, f"打开记事本{word}保存文件", "文件已存")])
    assert any(w["rule"] == "conjunction" for w in plan.granularity_report())


def test_overlong_goal_is_flagged() -> None:
    from agent.planner import MAX_GOAL_LENGTH

    plan = Plan(subtasks=[SubTask(1, "把" * (MAX_GOAL_LENGTH + 5), "某种结果")])
    assert any(w["rule"] == "too_long" for w in plan.granularity_report())


def test_multiple_verbs_flagged() -> None:
    plan = Plan(subtasks=[SubTask(1, "点击文件菜单保存文档", "已保存")])
    assert any(w["rule"] == "multi_verb" for w in plan.granularity_report())


def test_missing_expected_is_flagged() -> None:
    """没有完成判据就没法判断这一步做完没有，而 M2 不做 Reflector。"""
    plan = Plan(subtasks=[SubTask(1, "点击开始按钮", "")])
    assert any(w["rule"] == "no_expected" for w in plan.granularity_report())


def test_granularity_check_does_not_block_execution() -> None:
    """只警告不拦截：启发式必然有误判，提成硬拒会在演示时卡死任务。"""
    backend = _plan_backend('{"subtasks":[{"goal":"打开记事本并且输入文字","expected":"有字"}]}')
    plan = Planner(backend).plan("x")
    assert plan.subtasks  # 照样返回了
    assert not plan.is_fine_grained()  # 但记了警告


# ===================================================================== #
# 上下文窗口
# ===================================================================== #


def test_default_policy_matches_spec() -> None:
    """M2 任务 3 规定历史窗口默认 3。"""
    assert DEFAULT_POLICY.k == 3


def test_window_keeps_only_k_steps() -> None:
    frames = ContextWindow(ContextPolicy(k=2)).select(_steps(5))
    assert len(frames) == 2


def test_window_orders_oldest_first() -> None:
    """消息要按时间顺序拼，倒序会让模型以为时间是反的。"""
    frames = ContextWindow(ContextPolicy(k=3)).select(_steps(3))
    assert [f.age for f in frames] == [2, 1, 0]


def test_recent_frame_is_full_resolution() -> None:
    """上一步执行完的样子对判断"我刚那下点成没点成"最关键，给全清晰度。"""
    frames = ContextWindow(ContextPolicy(k=3, full_res_steps=1, image_steps=3)).select(_steps(3))
    newest = next(f for f in frames if f.age == 0)
    assert newest.with_image and newest.scale == 1.0


def test_older_frames_are_downscaled_not_dropped() -> None:
    """降采样 ≠ 丢弃。

    丢弃会让模型忘掉是自己点开了菜单，然后反复点同一个地方——也就是
    label 词表里的 stuck_loop。留一张模糊的旧帧就能避免。
    """
    policy = ContextPolicy(k=3, full_res_steps=1, image_steps=3, downscale=0.5)
    frames = ContextWindow(policy).select(_steps(3))
    older = [f for f in frames if f.age >= 1]
    assert older
    assert all(f.with_image and f.scale == 0.5 for f in older)


def test_beyond_image_steps_keeps_text_only() -> None:
    """更早的只留文字摘要。它几乎不花钱，却保住了"做过什么、成没成"。"""
    policy = ContextPolicy(k=4, full_res_steps=1, image_steps=2)
    frames = ContextWindow(policy).select(_steps(4))
    assert [f.with_image for f in frames] == [False, False, True, True]


def test_steps_without_screenshot_never_carry_images() -> None:
    frames = ContextWindow().select(_steps(3, with_screenshot=False))
    assert not any(f.with_image for f in frames)


def test_k_zero_disables_history() -> None:
    assert ContextWindow(ContextPolicy(k=0)).select(_steps(3)) == []


def test_empty_history() -> None:
    assert ContextWindow().select([]) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k": -1},
        {"downscale": 0.0},
        {"downscale": 1.5},
        {"full_res_steps": -1},
        {"image_steps": -1},
    ],
)
def test_policy_rejects_nonsense(kwargs) -> None:
    with pytest.raises(ValueError):
        ContextPolicy(**kwargs)


def test_stats_records_what_the_model_saw() -> None:
    """排查"这步为什么决策错了"，第一个要问的就是"当时模型看到了什么"。"""
    policy = ContextPolicy(k=3, full_res_steps=1, image_steps=2)
    stats = ContextWindow(policy).stats(_steps(5))
    assert stats["history_total"] == 5
    assert stats["selected"] == 3
    assert stats["with_image"] == 2
    assert stats["full_res"] == 1
    assert stats["downscaled"] == 1
    assert stats["policy"]["k"] == 3


def test_named_policies_differ_in_cost() -> None:
    """省钱档带的图必须少于富上下文档，否则这两个名字没意义。"""
    history = _steps(6)
    frugal = ContextWindow(FRUGAL).stats(history)["with_image"]
    rich = ContextWindow(RICH).stats(history)["with_image"]
    assert frugal < rich


def test_conversation_clears_between_subtasks() -> None:
    """每个子任务独立带历史：上一个子任务的操作对下一个决策价值很低，
    却要一直付 token。"""
    conversation = Conversation()
    for step in _steps(3):
        conversation.append(step)
    assert conversation.recent()

    conversation.clear()
    assert conversation.recent() == []


def test_conversation_applies_policy() -> None:
    conversation = Conversation(window=ContextWindow(ContextPolicy(k=2)))
    for step in _steps(5):
        conversation.append(step)
    assert len(conversation.recent()) == 2
    assert len(conversation.steps) == 5  # 原始记录不受策略影响
