"""数据集处理模块的单元测试。

这些测试盯的都是**不会报错、只会让数字悄悄错掉**的地方：

- 两个 ScreenSpot 版本的 bbox 格式不同（归一化 xyxy vs 绝对 xywh）。
  搞混了不会抛异常，只会让所有框错位
- ScreenAgent 的 `mouse_position` 拿 `width`/`height` 当 x/y 的键名
- ScreenAgent 有 1264 个 RLHF 负样本混在标注目录里，按 `*.json` 通配
  就会把模型的错误回答当成训练数据
- 划分泄漏：同一张截图同时进训练集和验证集，验证分数会虚高且看不出来
- 划分的可复现性：种子变了等于换了一份训练集，M3 的结果将不可比
"""

from __future__ import annotations

import json

import pytest

from data.clean import DEFAULT_RULES, OPTIONAL_RULES, clean
from data.loaders.screenagent import ScreenAgentLoader
from data.loaders.screenspot import ScreenSpotLoader, ScreenSpotV2Loader
from data.schema import ActionType, Platform, UnifiedSample, read_jsonl, write_jsonl
from data.split import MAX_VAL_RATIO, FrozenSplit, freeze_split, training_pool
from data.stats import collect, collect_by_dataset, overlap
from perception.types import BBox, Point


def _sample(**kwargs) -> UnifiedSample:
    payload = {
        "sample_id": "s1",
        "screenshot_path": "img.png",
        "resolution": (1000, 500),
        "instruction": "点击关闭按钮",
        "action_type": ActionType.LOCATE,
        "platform": Platform.DESKTOP,
        "source_dataset": "screenspot",
        "bbox": BBox(100, 50, 200, 150),
    }
    payload.update(kwargs)
    return UnifiedSample(**payload)


def _png(path, width: int, height: int) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), (10, 10, 10)).save(path)


# ===================================================================== #
# schema
# ===================================================================== #


def test_point_falls_back_to_bbox_center() -> None:
    assert _sample().resolve_point() == BBox(100, 50, 200, 150).center


def test_explicit_point_wins_over_bbox_center() -> None:
    """标注者真点过的位置比几何中心更能代表"可点的地方"——想想一个很长的菜单项。"""
    sample = _sample(point=Point(111, 60))
    assert sample.resolve_point() == Point(111, 60)


def test_normalized_point_uses_resolution_not_image_size() -> None:
    """模型坐标系是 [0,1000) 归一化，这一点在 M1 真机验证里实测确认过。"""
    sample = _sample(point=Point(500, 250), resolution=(1000, 500))
    assert sample.normalized_point(1000) == (500.0, 500.0)


def test_normalized_point_without_location_is_none() -> None:
    assert _sample(bbox=None, point=None).normalized_point() is None


def test_jsonl_round_trip_is_lossless(tmp_path) -> None:
    """轨迹与数据集都靠 JSONL 落盘，往返丢字段是最难发现的一类损坏。"""
    original = _sample(
        point=Point(7, 8), app="windows", split="test", meta={"element_kind": "icon"}
    )
    path = tmp_path / "u.jsonl"
    assert write_jsonl([original], path) == 1
    restored = next(iter(read_jsonl(path)))

    assert restored.to_json_dict() == original.to_json_dict()
    assert restored.bbox == original.bbox
    assert restored.point == original.point
    assert restored.platform is Platform.DESKTOP


def test_to_row_flattens_for_dataframe() -> None:
    row = _sample().to_row()
    assert row["bbox_area"] == 100 * 100
    assert row["point_x"] == 150 and row["point_y"] == 100
    assert row["has_bbox"] is True


def test_to_row_survives_missing_bbox() -> None:
    row = _sample(bbox=None, point=Point(3, 4)).to_row()
    assert row["bbox_area"] is None
    assert (row["point_x"], row["point_y"]) == (3, 4)


# ===================================================================== #
# ScreenSpot：两个版本的 bbox 格式不同
# ===================================================================== #


def _missing(name: str) -> bool:
    """这台机器上有没有装某个包。

    客机只跑 Agent（感知 / 控制 / HTTP 客户端），数据集预处理与出图都在
    宿主机上做，因此 pandas / pyarrow / matplotlib 在客机上本来就没有。
    **让这些用例失败，等于让客机的测试套件永远有 3 个红色**——而一个总是
    有失败的套件，会训练人忽略失败。跳过才是诚实的表示。
    """
    import importlib.util

    return importlib.util.find_spec(name) is None


def test_screenspot_v1_reads_bbox_as_normalized_xyxy(tmp_path) -> None:
    """v1 的 bbox 是 [0,1] 归一化的 xyxy。当成绝对像素解析会把整张图的
    元素全挤到左上角一个点上——而且不报错。"""
    from io import BytesIO

    import pytest
    from PIL import Image

    pd = pytest.importorskip("pandas", reason="数据集预处理只在宿主机上做")
    # pandas 写 parquet 要一个引擎，两个都是可选依赖。缺了会在
    # `to_parquet` 里抛 ImportError——错误指向 pandas 内部，看不出
    # 是「这台机器没装引擎」还是「代码写错了」。
    if all(_missing(name) for name in ("pyarrow", "fastparquet")):
        pytest.skip("缺少 parquet 引擎（pyarrow / fastparquet），数据集预处理只在宿主机上做")

    buffer = BytesIO()
    Image.new("RGB", (800, 600), (0, 0, 0)).save(buffer, format="PNG")

    root = tmp_path / "screenspot" / "data"
    root.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "file_name": "pc_a.png",
                "bbox": [0.25, 0.5, 0.5, 1.0],
                "instruction": "close",
                "data_type": "icon",
                "data_source": "windows",
                "image": {"bytes": buffer.getvalue()},
            }
        ]
    ).to_parquet(root / "test-00000-of-00001.parquet")

    result = ScreenSpotLoader(root=tmp_path).run()
    assert result.available, result.reason
    sample = result.samples[0]

    assert sample.resolution == (800, 600)
    assert sample.bbox == BBox(200, 300, 400, 600)  # 0.25*800, 0.5*600, …
    assert sample.platform is Platform.DESKTOP
    assert sample.action_type is ActionType.LOCATE  # 不臆断为 CLICK
    assert sample.meta["element_kind"] == "icon"


def test_screenspot_v2_reads_bbox_as_absolute_xywh(tmp_path) -> None:
    """v2 的 bbox 是绝对像素的 xywh。当成 xyxy 解析会得到一个宽高等于
    右下角坐标的巨框——同样不报错。"""
    base = tmp_path / "screenspot_v2"
    _png(base / "screenspotv2_image" / "pc_a.png", 800, 600)
    for name in ("desktop", "mobile", "web"):
        rows = (
            [
                {
                    "img_filename": "pc_a.png",
                    "bbox": [910 % 800, 78, 44, 34],
                    "instruction": "close this window",
                    "data_type": "icon",
                    "data_source": "windows",
                }
            ]
            if name == "desktop"
            else []
        )
        (base / f"screenspot_{name}_v2.json").write_text(json.dumps(rows), encoding="utf-8")

    result = ScreenSpotV2Loader(root=tmp_path).run()
    assert result.available, result.reason
    sample = result.samples[0]

    assert sample.bbox == BBox(110, 78, 110 + 44, 78 + 34)
    assert sample.bbox.width == 44 and sample.bbox.height == 34
    assert sample.platform is Platform.DESKTOP


def test_screenspot_v2_reports_unextracted_zip_instead_of_crashing(tmp_path) -> None:
    """1.3GB 的解压是个要人知情的动作，不该表现为一个 FileNotFoundError。"""
    base = tmp_path / "screenspot_v2"
    base.mkdir(parents=True)
    for name in ("desktop", "mobile", "web"):
        (base / f"screenspot_{name}_v2.json").write_text("[]", encoding="utf-8")

    result = ScreenSpotV2Loader(root=tmp_path).run()
    assert not result.available
    assert "解压" in result.reason


def test_missing_dataset_is_reported_not_raised(tmp_path) -> None:
    result = ScreenSpotLoader(root=tmp_path).run()
    assert not result.available
    assert result.samples == []
    assert "prepare_datasets" in result.reason


# ===================================================================== #
# ScreenAgent
# ===================================================================== #


def _screenagent_fixture(tmp_path, *, split: str = "train", suffix: str = "_translate"):
    """按**真实结构**造 fixture：一步只有一种动作类型。

    这一点是踩过坑的。最初的 fixture 把 PlanAction 和 MouseAction 放进同一个
    step 的 actions 列表里，装载器于是"顺理成章"地把 plan 的描述当成了后续
    点击的指令，测试也就跟着绿了——**而真实数据里这两者从不同步出现**
    （实测 716 个带坐标的鼠标动作，同一步内前方有 PlanAction 的是 0 个）。
    自己编的 fixture 印证的是自己的误解，不是数据。
    """
    session = tmp_path / "screenagent_repo/data/ScreenAgent" / split / "sess1"
    base = {
        "session_id": "sess1",
        "task_prompt_en": "Find information about von Neumann",
        "task_prompt_zh": "在网上查冯诺依曼的资料",
        "video_width": 1024,
        "video_height": 768,
    }

    def step(stamp: str, image: str, actions: list[dict], subtask: str = "") -> None:
        _png(session / "images" / image, 1024, 768)
        payload = {**base, "saved_image_name": image, "actions": actions}
        if subtask:
            # 真实标注里子任务写在 send_prompt 的这句话里，**中文那份是原文**
            payload["send_prompt_zh"] = (
                f"你对 Linux 操作系统非常熟悉。现在的子任务是「{subtask}」，"
                "请根据当前屏幕状态给出下一步动作。"
            )
        (session / f"{stamp}{suffix}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    # 规划步：只有 PlanAction，而且写在**另一张截图**上
    step(
        "2023-12-20_19-31-31",
        "plan.jpg",
        [
            {"action_type": "PlanAction", "element": "the address bar"},
        ],
        subtask="打开浏览器",
    )
    # 执行步：鼠标 + 键盘
    step(
        "2023-12-20_19-33-28",
        "act.jpg",
        [
            {
                "action_type": "MouseAction",
                "mouse_action_type": "click",
                "mouse_button": "left",
                "mouse_position": {"width": 543, "height": 133},
            },
            {
                "action_type": "KeyboardAction",
                "keyboard_action_type": "text",
                "keyboard_text": "hello",
            },
        ],
        subtask="在地址栏输入网址",
    )
    # 评估步
    step(
        "2023-12-20_19-35-28",
        "eval.jpg",
        [
            {"action_type": "EvaluateSubTaskAction", "situation": "sub_task_success"},
        ],
        subtask="确认页面已打开",
    )

    other = "test" if split == "train" else "train"
    (tmp_path / "screenagent_repo/data/ScreenAgent" / other).mkdir(parents=True, exist_ok=True)
    return session


def test_screenagent_mouse_position_width_height_are_x_y(tmp_path) -> None:
    """`{"width": 543, "height": 133}` 里 width 是 x、height 是 y。

    这个命名和字面意思毫无关系，照字面理解会让每个坐标都错成"宽高"。
    """
    _screenagent_fixture(tmp_path)
    samples = ScreenAgentLoader(root=tmp_path).run().samples
    click = next(s for s in samples if s.action_type is ActionType.CLICK)
    assert click.point == Point(543, 133)


def test_screenagent_指令是当前子任务而不是总目标(tmp_path) -> None:
    """**一个会话十几步共用一句总目标，那是矛盾监督。**

    实测：716 条样本只对应 169 个不重复指令，「Insert a circle」出现 17 次
    却对应 17 个不同坐标。改取子任务后不重复指令涨到 1068 个。

    仍然**绝不取 `PlanAction.element`**——它写在另一步、配的是另一张截图，
    而且中英混杂（1168 条里 73% 中文，且该字段没有 _en/_zh 变体）。
    """
    _screenagent_fixture(tmp_path)
    samples = ScreenAgentLoader(root=tmp_path).run().samples
    click = next(s for s in samples if s.action_type is ActionType.CLICK)

    assert click.instruction == "在地址栏输入网址"
    assert click.meta["instruction_source"] == "send_prompt_zh"
    assert click.meta["task_prompt"] == "Find information about von Neumann"
    assert "plan_element" not in click.meta


class TestKeysymTranslation:
    """ScreenAgent 走 VNC，键名是 X11 keysym；本项目执行器用 pyautogui。

    **不翻译的后果不止是执行失败。** `control.safety.rule_dangerous_keys`
    靠按 `+` 切分来匹配危险组合键——喂进去未翻译的键名，危险键规则一条也
    匹配不上，那是安全缺口。
    """

    def test_单键翻译(self):
        from data.loaders.screenagent import _keys_of

        assert _keys_of("Return") == "enter"
        assert _keys_of("Escape") == "esc"
        assert _keys_of("BackSpace") == "backspace"

    def test_列表要合并成组合键(self):
        """**323 条 press 里有 85 条是列表。**

        不合并的话 `params["key"]` 会存成 ``"['Control_L', 'a']"`` —— Python
        列表的字符串形式原样进训练目标，教模型输出它自己都解析不了的东西。
        """
        from data.loaders.screenagent import _keys_of

        assert _keys_of(["Control_L", "a"]) == "ctrl+a"
        assert _keys_of(["Control_L", "Shift_L", "`"]) == "ctrl+shift+`"
        assert _keys_of(["Super_L", "r"]) == "win+r"

    def test_已经是加号连接的原样归一(self):
        from data.loaders.screenagent import _keys_of

        assert _keys_of("Ctrl+S") == "ctrl+s"

    def test_空值返回空串而不是_None_字面量(self):
        from data.loaders.screenagent import _keys_of

        assert _keys_of(None) == ""
        assert _keys_of([]) == ""

    def test_翻译后能被危险键规则读懂(self):
        """这条才是真正要守的东西——**格式对不上时安全规则会静默失效**。"""
        from control.actions import Action, ActionType
        from control.safety import rule_dangerous_keys
        from data.loaders.screenagent import _keys_of

        verdict = rule_dangerous_keys(
            Action(type=ActionType.KEY, keys=_keys_of(["Control_L", "a"]))
        )
        assert verdict.evidence in ("", "ctrl+a"), "规则拿到的应当是归一后的组合键"

    def test_表里没有的原样小写透传(self):
        """单字母与数字本来就一致，不该为它们各写一行。"""
        from data.loaders.screenagent import _keys_of

        assert _keys_of("F") == "f"
        assert _keys_of("2") == "2"


def test_screenagent_规划样本的指令是总目标而不是子任务(tmp_path) -> None:
    """**指令取哪一句，取决于这条样本是给谁训的。**

    规划器的活是「总目标 → 子任务列表」。给它子任务当指令等于把答案当题目。
    一刀切要求子任务时，884 条 PlanAction 被挡到只剩 215 条——规划步的
    `send_prompt` 问的是「请给出实施计划」，本来就没有「现在的子任务是」那句。
    """
    _screenagent_fixture(tmp_path)
    samples = ScreenAgentLoader(root=tmp_path).run().samples
    plan = next(s for s in samples if s.meta.get("raw_action_type") == "PlanAction")

    assert plan.instruction == "Find information about von Neumann"
    assert plan.meta["instruction_source"] == "task_prompt"

    click = next(s for s in samples if s.action_type is ActionType.CLICK)
    assert click.instruction != plan.instruction, "执行器与规划器不该拿到同一句指令"


def test_screenagent_取不到子任务时留空而不是退回总目标(tmp_path) -> None:
    """**退回总目标等于把矛盾监督又放回训练集**，而条数看起来还是满的。

    实测 4012 条里有 669 条两个来源都取不到，这些必须被丢弃而不是补一句
    总目标充数。
    """
    session = _screenagent_fixture(tmp_path)
    for path in session.glob("*_translate.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.pop("send_prompt_zh", None)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    # 规划样本不受影响——它的指令本来就是总目标，见上一条测试
    samples = [
        s
        for s in ScreenAgentLoader(root=tmp_path).run().samples
        if s.meta.get("raw_action_type") != "PlanAction"
    ]
    assert samples, "fixture 自己坏了"
    assert all(not s.instruction for s in samples)
    assert all(s.meta["instruction_source"] == "缺失" for s in samples)
    from data.schema import is_trainable

    assert not any(is_trainable(s) for s in samples), "没有指令的样本不该进训练池"


def test_screenagent_子任务优先中文原文(tmp_path) -> None:
    """英文那份是机翻且引号已经坏了：

        Now the subtask is "enter" Von Neumann "as the search keyword".

    按引号切会切出半句话。而且**部署时模型收到的就是中文**——
    `tasks/basic_tasks.yaml` 写的是「打开 Microsoft Edge 浏览器」。
    """
    session = _screenagent_fixture(tmp_path)
    path = next(session.glob("*19-33-28*.json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["send_prompt_en"] = 'Now the subtask is "enter" Von Neumann "as the keyword".'
    payload["current_task"] = "Enter the URL in the address bar"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    samples = ScreenAgentLoader(root=tmp_path).run().samples
    click = next(s for s in samples if s.action_type is ActionType.CLICK)
    assert click.instruction == "在地址栏输入网址"


def test_screenagent_没有中文时退到_current_task(tmp_path) -> None:
    """test 划分的原始文件里根本没有中文变体，639/639 靠这个字段。

    **这构成 train / test 之间的指令语言差异，报告里必须写明。**
    """
    session = _screenagent_fixture(tmp_path)
    path = next(session.glob("*19-33-28*.json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("send_prompt_zh", None)
    payload["current_task"] = "Enter the URL in the address bar"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    samples = ScreenAgentLoader(root=tmp_path).run().samples
    click = next(s for s in samples if s.action_type is ActionType.CLICK)
    assert click.instruction == "Enter the URL in the address bar"
    assert click.meta["instruction_source"] == "current_task"


def test_screenagent_keeps_plan_text_for_mode_b_reference(tmp_path) -> None:
    """element 对 grounding 没用，但是模式 B target_description 的现成语料，
    不能顺手丢掉。"""
    _screenagent_fixture(tmp_path)
    samples = ScreenAgentLoader(root=tmp_path).run().samples
    plans = [s for s in samples if s.meta.get("raw_action_type") == "PlanAction"]

    assert [s.meta["plan_element"] for s in plans] == ["the address bar"]


def test_screenagent_指令里不混进_plan_的英文(tmp_path) -> None:
    """`plan_element` 一旦被当成指令，语言就会被污染——它没有 _en/_zh 变体。

    子任务一律来自中文原文，所以整个训练集的指令语言是一致的。
    """
    _screenagent_fixture(tmp_path)
    samples = ScreenAgentLoader(root=tmp_path, language="zh").run().samples
    assert all(s.instruction for s in samples)
    assert not any("address bar" in s.instruction for s in samples)


def test_screenagent_excludes_rlhf_negatives(tmp_path) -> None:
    """`_neg_plan` / `_neg_eval` 是模型的**错误**回答，不是标注。"""
    session = _screenagent_fixture(tmp_path)
    (session / "2023-12-20_19-31-31_translate_neg_plan.json").write_text(
        json.dumps(
            {
                "video_width": 1024,
                "video_height": 768,
                "saved_image_name": "shot.jpg",
                "actions": [],
            }
        ),
        encoding="utf-8",
    )

    result = ScreenAgentLoader(root=tmp_path).run()
    assert result.skipped.get("RLHF 负样本") == 1
    assert all("_neg_" not in s.sample_id for s in result.samples)


def test_screenagent_accepts_test_naming_convention(tmp_path) -> None:
    """train 是 `<时间戳>_translate.json`，test 是 `<时间戳>.json`。
    只认一种的话，test 划分会静默地一条都读不出来。"""
    _screenagent_fixture(tmp_path, split="test", suffix="")
    samples = ScreenAgentLoader(root=tmp_path).run().samples
    assert samples and all(s.split == "test" for s in samples)


def test_screenagent_keeps_non_spatial_actions(tmp_path) -> None:
    """规划 / 评估动作没有坐标，但它们是数据集结构的一部分，
    丢掉的话"动作类型分布"就只剩鼠标动作了。"""
    _screenagent_fixture(tmp_path)
    samples = ScreenAgentLoader(root=tmp_path).run().samples
    assert {s.action_type for s in samples} == {ActionType.OTHER, ActionType.CLICK, ActionType.TYPE}
    assert sum(1 for s in samples if s.resolve_point() is None) == 3


# ===================================================================== #
# 清洗
# ===================================================================== #


def test_out_of_bounds_bbox_is_rejected() -> None:
    _, report = clean([_sample(bbox=BBox(900, 400, 1200, 600))])
    assert report.rejected == {"bbox_out_of_bounds": 1}


def test_tiny_bbox_is_rejected() -> None:
    _, report = clean([_sample(bbox=BBox(10, 10, 19, 19))])  # 81px² < 100
    assert report.rejected == {"bbox_too_small": 1}


def test_bbox_at_exact_threshold_is_kept() -> None:
    kept, _ = clean([_sample(bbox=BBox(10, 10, 20, 20))])  # 恰好 100px²
    assert len(kept) == 1


def test_out_of_bounds_point_is_rejected() -> None:
    _, report = clean([_sample(bbox=None, point=Point(1000, 10))])  # 宽 1000，开区间
    assert report.rejected == {"point_out_of_bounds": 1}


def test_sample_without_text_or_type_is_rejected() -> None:
    _, report = clean([_sample(instruction="   ", meta={})])
    assert report.rejected == {"no_text_no_type": 1}


def test_sample_without_text_but_with_type_is_kept() -> None:
    kept, _ = clean([_sample(instruction="", meta={"element_kind": "icon"})])
    assert len(kept) == 1


def test_each_sample_counts_toward_one_rule_only() -> None:
    """一条样本同时违反多条规则时只计第一条，否则各项相加会大于剔除总数，
    报告里没法交代。"""
    both = _sample(bbox=BBox(990, 490, 1200, 700), instruction="")
    _, report = clean([both])
    assert sum(report.rejected.values()) == report.rejected_total == 1


def test_missing_location_is_not_a_cleaning_failure_by_default() -> None:
    """没坐标不是"数据坏了"，是"这条不适合做 grounding"。两件事分开。"""
    kept, _ = clean([_sample(bbox=None, point=None)])
    assert len(kept) == 1

    _, report = clean(
        [_sample(bbox=None, point=None)], (*DEFAULT_RULES, OPTIONAL_RULES["no_location"])
    )
    assert report.rejected == {"no_location": 1}


def test_report_attributes_rejections_to_source_dataset() -> None:
    """越界的都来自哪个数据集，是判断格式解析是否写错的最直接线索。"""
    _, report = clean(
        [
            _sample(bbox=BBox(900, 400, 1200, 600), source_dataset="screenspot_v2"),
            _sample(bbox=BBox(900, 400, 1300, 600), source_dataset="screenspot_v2"),
        ]
    )
    assert report.rejected_by_dataset["bbox_out_of_bounds"] == {"screenspot_v2": 2}


# ===================================================================== #
# 统计
# ===================================================================== #


def test_stats_counts_unique_images_not_samples() -> None:
    """ScreenSpot 一张图对应多条标注，样本数不等于图片数。"""
    stats = collect(
        [
            _sample(sample_id="a", screenshot_path="x.png"),
            _sample(sample_id="b", screenshot_path="x.png"),
        ]
    )
    assert stats.total == 2 and stats.image_count == 1


def test_stats_group_by_dataset() -> None:
    grouped = collect_by_dataset(
        [_sample(source_dataset="screenspot"), _sample(source_dataset="screenagent")]
    )
    assert set(grouped) == {"screenspot", "screenagent"}
    assert grouped["screenspot"].total == 1


def test_overlap_compares_file_names_not_paths() -> None:
    """同一张截图在两个数据集里的存放路径必然不同，比路径永远得 0。"""
    left = [_sample(screenshot_path="a/x.png", meta={"file_name": "x.png"})]
    right = [_sample(screenshot_path="b/x.png", meta={"file_name": "x.png"})]
    report = overlap("l", left, "r", right)
    assert report.shared_images == 1
    assert report.left_leaked_ratio == 1.0


def test_tiny_ratio_uses_relative_area() -> None:
    """绝对像素阈值在 1024×768 和 2560×1600 上含义完全不同。"""
    stats = collect(
        [
            _sample(bbox=BBox(0, 0, 10, 10), resolution=(1000, 1000)),  # 0.01%
            _sample(bbox=BBox(0, 0, 500, 500), resolution=(1000, 1000)),  # 25%
        ]
    )
    assert stats.tiny_ratio == 0.5


# ===================================================================== #
# 划分（M3 前置件①）
# ===================================================================== #


def _pool(sessions: int, per_session: int) -> list[UnifiedSample]:
    return [
        _sample(
            sample_id=f"s{s}-{i}",
            source_dataset="screenagent",
            split="train",
            point=Point(1, 1),
            bbox=None,
            screenshot_path=f"sess{s}.jpg",
            meta={
                "session_id": f"sess{s}",
                "raw_action_type": "MouseAction",
                "raw_action_subtype": "click",
            },
        )
        for s in range(sessions)
        for i in range(per_session)
    ]


def _action(kind: str, subtype: str = "", **kwargs) -> UnifiedSample:
    """一条 ScreenAgent 形状的样本。`kind` 是原始 `raw_action_type`。"""
    meta = {"session_id": "s", "raw_action_type": kind, "raw_action_subtype": subtype}
    meta.update(kwargs.pop("meta", {}))
    return _sample(
        source_dataset="screenagent",
        split="train",
        action_type=kwargs.pop("action_type", ActionType.OTHER),
        bbox=None,
        meta=meta,
        **kwargs,
    )


def test_training_pool_只收目标场景的样本() -> None:
    samples = [
        *_pool(1, 2),
        _action("MouseAction", "click", point=None),  # 缺坐标
        _sample(source_dataset="screenagent", split="test", point=Point(1, 1)),  # 非 train
        _sample(source_dataset="screenspot", split="train", point=Point(1, 1)),  # 非目标数据集
    ]
    assert len(training_pool(samples)) == 2


def test_training_pool_收下没有坐标的动作() -> None:
    """**这是这次放宽的全部意义。**

    原过滤只有一行 `resolve_point() is not None`，于是 ScreenAgent 4012 条里
    只剩 716 条——`type` / `key` / `done` / `wait` 全部落在门外，而端到端实测
    查出来的瓶颈恰恰是训练集里 `type` 与 `done` 各为 0。
    """
    samples = [
        _action("KeyboardAction", "text", point=None, params={"text": "冯诺依曼"}),
        _action("KeyboardAction", "press", point=None, params={"key": "Return"}),
        _action("EvaluateSubTaskAction", point=None, params={"situation": "sub_task_success"}),
        _action("WaitAction", point=None, params={"seconds": 0.5}),
        _action("MouseAction", "scroll_down", point=None, params={"direction": "down"}),
        _action("PlanAction", point=None, params={"element": "打开浏览器"}),
    ]
    assert len(training_pool(samples)) == len(samples)


def test_training_pool_丢掉缺参数的样本() -> None:
    """**这是「字段齐不齐」，不是「样本好不好」。**

    一条没带 `keyboard_text` 的 `type` 生成不出训练目标——留着它，模型学到的
    是「该打字的时候输出空字符串」。
    """
    samples = [
        _action("KeyboardAction", "text", point=None, params={}),
        _action("KeyboardAction", "press", point=None, params={}),
        _action("PlanAction", point=None, params={}),
    ]
    assert training_pool(samples) == []


def test_training_pool_仍然丢掉_drag() -> None:
    """ScreenAgent 的拖拽标注**只有起点没有终点**，构不成可执行的动作。

    丢弃而不是硬凑——拿只有起点的拖拽去训练，教出来的是「拖拽 = 点一下」。
    """
    samples = [
        _action("MouseAction", "drag", point=Point(1, 1)),
        _action("MouseAction", "down", point=Point(1, 1)),
        _action("MouseAction", "up", point=Point(1, 1)),
    ]
    assert training_pool(samples) == []


def test_动作判定不能只看_action_type() -> None:
    """`ActionType.OTHER` 底下同时混着三样完全不同的东西。

    实测：`done` 1184 条、`plan` 1168 条、`mouse_move` 83 条，全是 `other`。
    只看 `action_type` 会把它们当成同一类。
    """
    from data.schema import training_action

    assert training_action(_action("EvaluateSubTaskAction")) == "done"
    assert training_action(_action("PlanAction")) == "plan"
    assert training_action(_action("MouseAction", "move")) == "mouse_move"


def test_必需字段只有一份口径() -> None:
    """`data.split` 与 `finetune.dataset` 必须读同一张表。

    两处各写一份的话，一处放宽了另一处没跟着放宽，池子里有的样本会在生成
    训练记录时被静默丢掉，而两边的计数都「看起来对」——**本项目栽过两次**。
    """
    import inspect

    from data.schema import ACTION_REQUIREMENTS

    assert ACTION_REQUIREMENTS["type"] == ("text",)
    assert "ACTION_REQUIREMENTS" in inspect.getsource(
        __import__("finetune.dataset", fromlist=["x"])
    )


def test_split_never_leaks_a_session_across_sides() -> None:
    """同一会话共用截图。按样本随机切会让同一张图同时进两侧，
    验证分数虚高且看不出来。"""
    split = freeze_split(_pool(40, 5), val_size=50)
    assert not split.leakage()
    assert not set(split.train_groups) & set(split.val_groups)


def test_split_is_reproducible_for_the_same_seed() -> None:
    pool = _pool(30, 4)
    assert freeze_split(pool, seed=7).fingerprint() == freeze_split(pool, seed=7).fingerprint()


def test_split_changes_with_the_seed() -> None:
    """换种子等于换一份训练集，M3 的结果将不可比——所以它必须真的有影响。"""
    pool = _pool(30, 4)
    assert freeze_split(pool, seed=7).fingerprint() != freeze_split(pool, seed=8).fingerprint()


def test_split_does_not_depend_on_input_order() -> None:
    pool = _pool(20, 3)
    assert freeze_split(pool).fingerprint() == freeze_split(pool[::-1]).fingerprint()


def test_validation_set_is_capped_at_a_fraction_of_the_pool() -> None:
    """池子只有 100 条时照搬 val_size=500 会切出 500 验证 / 0 训练。

    这是实测踩到的：可用池 716 条，固定 500 得到 503 验证 / 213 训练。
    """
    split = freeze_split(_pool(50, 2), val_size=500)  # 池子 100 条
    assert split.val_size <= 100 * MAX_VAL_RATIO + 2  # 会话不拆，允许略微超出
    assert split.train_size > split.val_size
    assert any("下调" in note for note in split.notes)


def test_split_warns_when_pool_falls_short_of_the_outline_target() -> None:
    split = freeze_split(_pool(10, 5))
    assert any("低于大纲目标下限" in note for note in split.notes)


def test_frozen_split_round_trips_through_file(tmp_path) -> None:
    split = freeze_split(_pool(20, 4))
    path = split.write(tmp_path / "split.json")
    restored = FrozenSplit.read(path)
    assert restored.fingerprint() == split.fingerprint()
    assert restored.train_ids == sorted(split.train_ids)


def test_tampered_split_file_is_refused() -> None:
    """M3 开工要核对划分。文件被改过而指纹还对得上，是最危险的情况。"""
    import tempfile
    from pathlib import Path

    split = freeze_split(_pool(20, 4))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "split.json"
        split.write(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["train_ids"].append("偷偷加进来的样本")
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        with pytest.raises(ValueError, match="指纹不符"):
            FrozenSplit.read(path)


# ===================================================================== #
# 图表
# ===================================================================== #


def test_all_four_charts_render(tmp_path) -> None:
    """四张图必须都画得出来。

    绘图代码最典型的失效不是画得难看，而是**在某个数据形态上直接抛异常**
    （某个平台一条样本都没有、某个数据集缺席、面积列表为空）。这些在跑
    完整流水线前发现不了，而那时已经花了几分钟装载。
    """
    import pytest

    matplotlib = pytest.importorskip("matplotlib", reason="出图只在宿主机上做")
    matplotlib.use("Agg")
    from data.charts import render_all

    samples = [
        _sample(sample_id="a", source_dataset="screenspot", platform=Platform.DESKTOP),
        _sample(
            sample_id="b",
            source_dataset="screenspot_v2",
            platform=Platform.WEB,
            resolution=(1920, 1080),
            bbox=BBox(0, 0, 60, 40),
        ),
        _sample(
            sample_id="c",
            source_dataset="screenagent",
            platform=Platform.MOBILE,
            bbox=BBox(5, 5, 300, 300),
            meta={"raw_action_type": "MouseAction"},
            action_type=ActionType.CLICK,
        ),
    ]
    by_dataset = collect_by_dataset(samples)
    by_platform = {
        p.value: collect([s for s in samples if s.platform is p], name=p.value) for p in Platform
    }

    paths = render_all(by_dataset, by_platform, tmp_path)
    assert len(paths) == 4
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_element_size_chart_refuses_when_nothing_has_a_bbox(tmp_path) -> None:
    """ScreenAgent 训练集只有落点没有框。硬画会得到一张空图，
    比报错更难发现。"""
    import pytest

    matplotlib = pytest.importorskip("matplotlib", reason="出图只在宿主机上做")
    matplotlib.use("Agg")
    from data.charts import apply_style, chart_element_sizes

    apply_style()
    only_points = [_sample(bbox=None, point=Point(1, 1), platform=Platform.DESKTOP)]
    by_platform = {"desktop": collect(only_points, name="desktop")}

    with pytest.raises(ValueError, match="无法绘制"):
        chart_element_sizes(by_platform, tmp_path)
