"""采集任务集不得与评测任务集重叠 —— **这是测试泄漏的唯一防线**。

## 为什么值得单开一个文件

蒸馏的做法是「让教师在客机上做任务，把过程当训练数据」。如果采集用的任务
就是评测用的任务，那么训练之后再去评测，测的是模型有没有把那几轮背下来，
不是它学会了什么。**而这种失误在结果上表现为「效果特别好」**，最容易被当成
成功接受下来。

ScreenSpot 那条纪律是同一个道理：固定为零样本测试集，不参与训练、验证、
提示词选型、超参调优。这里把同一条纪律用在自己造的任务集上。

## 比什么

不只比任务名（改个名字就绕过去了），而是比**模型实际看到的东西**：
指令原文、目标文件路径、搜索词、消息内容。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

COLLECT = Path("tasks/collect_variants.yaml")
EVAL_SETS = (Path("tasks/basic_tasks.yaml"), Path("tasks/desktop_20.yaml"))

#: 两边必然共用的**环境名词**——它们是被操作的程序，不是模型要生成的内容。
#:
#: 「测试消息」是那个自写 tkinter 程序的窗口标题，采集和评测操作的是同一个
#: 程序；换一个程序就不是同一类任务了。模型真正会记住的是**发了什么内容**
#: （你好世界 / 今天天气不错），那一项仍然严格禁止重叠。
#:
#: **这个名单只能加程序名，不能加搜索词、消息内容或文件名。** 往里加一个
#: 内容类的词，这条防线就废了——而废掉之后，结果表现为「效果特别好」。
SHARED_ENV_NOUNS = frozenset({"测试消息"})


def load_tasks(path: Path) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["tasks"]


def quoted_params(instruction: str) -> set[str]:
    """指令里被「」括起来的部分，以及出现的文件路径。

    这些是任务之间真正不同的地方——搜索词、消息内容、目标文件。
    """
    params = set(re.findall(r"「([^」]+)」", instruction))
    params |= {m.lower() for m in re.findall(r"[A-Za-z]:\\[^\s，。]+", instruction)}
    return params - SHARED_ENV_NOUNS


@pytest.fixture(scope="module")
def collect_tasks() -> list[dict]:
    return load_tasks(COLLECT)


@pytest.fixture(scope="module")
def eval_tasks() -> list[dict]:
    tasks = []
    for path in EVAL_SETS:
        tasks.extend(load_tasks(path))
    return tasks


class TestNoOverlapWithEvaluation:
    def test_指令原文不得相同(self, collect_tasks, eval_tasks):
        """一字不差的指令进了训练集，评测就是在考背诵。"""
        evaluated = {t["instruction"].strip() for t in eval_tasks}
        for task in collect_tasks:
            assert task["instruction"].strip() not in evaluated, (
                f"采集任务 {task['name']} 的指令与评测集重复：{task['instruction']}"
            )

    def test_参数不得相同(self, collect_tasks, eval_tasks):
        """搜索词、消息内容、目标文件这些才是模型真正会记住的东西。"""
        evaluated: set[str] = set()
        for task in eval_tasks:
            evaluated |= quoted_params(task["instruction"])
        for task in collect_tasks:
            shared = quoted_params(task["instruction"]) & evaluated
            assert not shared, f"采集任务 {task['name']} 与评测集共用参数：{shared}"

    def test_任务名不得相同(self, collect_tasks, eval_tasks):
        """名字重了，存档和轨迹目录会混在一起，事后分不清哪轮是采集哪轮是评测。"""
        evaluated = {t["name"] for t in eval_tasks}
        for task in collect_tasks:
            assert task["name"] not in evaluated

    def test_采集任务名都带collect前缀(self, collect_tasks):
        """看一眼存档就知道这轮是采集还是评测，不用翻配置。"""
        for task in collect_tasks:
            assert task["name"].startswith("collect_"), task["name"]


class TestCollectTasksAreRunnable:
    """结构上跑得起来。跑不起来的任务在客机上才暴露，一轮就是半天。"""

    def test_必填字段齐全(self, collect_tasks):
        for task in collect_tasks:
            for field in ("name", "instruction", "success_check", "reset", "max_steps"):
                assert field in task, f"{task.get('name')} 缺 {field}"

    def test_判据类型都是已实现的(self, collect_tasks):
        from core.verify import CHECKERS, SuccessCheck

        for task in collect_tasks:
            check = SuccessCheck.from_spec(task["success_check"])
            assert check.checks, f"{task['name']} 的判据是空的——**空判据不能默认算成功**"
            for spec in check.checks:
                assert spec.get("type") in CHECKERS, f"{task['name']} 用了未实现的判据 {spec}"

    def test_每个任务都有反方向的起点检查(self, collect_tasks):
        """判据是「X 应该在」，起点就必须是「X 不在」。

        假成功全都来自这个方向：起点没清干净，判据一上来就满足。
        """
        for task in collect_tasks:
            assert task.get("precondition"), f"{task['name']} 没有 precondition"

    def test_重置都先清场(self, collect_tasks):
        """屏幕就是 Agent 的全部输入，上一轮遗留的窗口是下一轮输入的一部分。"""
        for task in collect_tasks:
            assert any("reset_desktop" in cmd for cmd in task["reset"]), task["name"]

    def test_采集文件来自采集专用的那套(self, collect_tasks):
        """采集任务只能碰 setup_collect.py 建的文件，不碰评测环境的文件。"""
        from tasks.setup_collect import COLLECT_FILES

        allowed = {name.lower() for name in COLLECT_FILES}
        for task in collect_tasks:
            for match in re.findall(r"[A-Za-z]:\\agent-test\\([^\s，。]+)", task["instruction"]):
                assert match.lower() in allowed, (
                    f"{task['name']} 用了非采集专用文件 {match}，可能与评测环境共用"
                )


class TestSetupCollect:
    def test_采集文件名与评测环境不重叠(self):
        """两套文件混用的话，训练时见过的文件评测时又出现，等于泄漏。"""
        from tasks.setup_collect import COLLECT_FILES
        from tasks.setup_env import TEST_FILE_NAME

        names = {n.lower() for n in COLLECT_FILES}
        assert TEST_FILE_NAME.lower() not in names
        for other in ("源文件.txt", "待改名.txt", "待删除.txt", "长文档.txt", "笔记.txt"):
            assert other.lower() not in names


class TestSharedEnvNounsAllowlist:
    """放行名单本身也要被守住 —— 它是上面那条防线唯一的缺口。"""

    def test_名单里只能有程序名(self):
        """往里加一个搜索词或消息内容，泄漏防线就废了。

        判据：名单里的词必须出现在某个评测任务的 `precondition` 窗口标题里
        （说明它是被操作的程序），而不是只出现在指令的内容位置上。
        """
        from core.verify import SuccessCheck

        titles: set[str] = set()
        for path in EVAL_SETS:
            for task in load_tasks(path):
                # precondition 有「列表」和「带 mode 的字典」两种写法，
                # from_spec 是既有的归一入口，不在测试里再写一遍解析
                for spec in SuccessCheck.from_spec(task.get("precondition") or []).checks:
                    if spec.get("type") == "window_title":
                        titles.add(str(spec.get("pattern", "")))
        for noun in SHARED_ENV_NOUNS:
            assert any(noun in title for title in titles), (
                f"{noun!r} 不是任何评测任务的目标程序，不该出现在放行名单里"
            )

    def test_名单短(self):
        """名单越长，这条防线越接近于不存在。"""
        assert len(SHARED_ENV_NOUNS) <= 3
