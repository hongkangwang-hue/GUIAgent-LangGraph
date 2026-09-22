"""长任务测试集 —— M4 交付物 4 / 验收标准 4。

要求是「3 个 8 步以上、同一应用内」的任务。这个文件守的是那三个定语：
数量、长度、同应用。少一个，这份交付物就不满足标准 4，而那种不满足
只有在验收时才会被发现。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

LONG = Path("tasks/long_tasks.yaml")


@pytest.fixture(scope="module")
def tasks() -> list[dict]:
    return yaml.safe_load(LONG.read_text(encoding="utf-8"))["tasks"]


class TestMeetsCriterion4:
    def test_正好三个(self, tasks):
        assert len(tasks) == 3, "标准 4 要求 3 个长任务"

    def test_步数预算都留够(self, tasks):
        """人工要 8~9 步的任务，上限给 8 是不够的：模型比人多走几步是常态，
        上限太紧会让「能力不足」和「预算不足」混在一起。
        """
        for task in tasks:
            assert task["max_steps"] >= 16, f"{task['name']} 的步数上限太紧"

    def test_都在同一应用内(self, tasks):
        """跨应用的失败大多发生在「切过去」那一步，测的是启动不是长程。"""
        for task in tasks:
            assert "notepad" in " ".join(task["reset"]), f"{task['name']} 没有预先打开目标应用"

    def test_起点要求应用已经开着(self, tasks):
        """这批任务测的是「在应用内做长流程」，不是「能不能把应用打开」。"""
        for task in tasks:
            kinds = [c for c in task["precondition"] if c.get("type") == "process"]
            assert any(c.get("should_run") for c in kinds), f"{task['name']} 没要求应用已启动"


class TestChecksAreHardToFake:
    def test_判据查文件内容而不只是窗口(self, tasks):
        """判据越间接，假成功越多。文件内容是任务真正的副作用。"""
        for task in tasks:
            types = {c.get("type") for c in _checks(task["success_check"])}
            assert types & {"file_contains", "file_exists"}, f"{task['name']} 的判据太间接"

    def test_追加任务能识破清空重写(self, tasks):
        """只查「新内容在不在」的话，把文件清空重写也算通过——那不是追加。"""
        task = _by_name(tasks, "long_append_and_save")
        texts = {c.get("text") for c in _checks(task["success_check"])}
        assert "底稿原有内容" in texts

    def test_替换任务要求旧词消失(self, tasks):
        """只查新词出现的话，在末尾补一行「新词」也能通过。"""
        task = _by_name(tasks, "long_replace_word")
        negative = [c for c in _checks(task["success_check"]) if c.get("should_contain") is False]
        assert negative, "没有反向判据，替换没做干净也会算成功"

    def test_写文件任务起点要求产物不存在(self, tasks):
        """上一轮的产物留着，Agent 什么都不做判据也会打勾。"""
        task = _by_name(tasks, "long_write_three_lines")
        negative = [c for c in task["precondition"] if c.get("should_exist") is False]
        assert negative


class TestResetRestoresStartingPoint:
    def test_每轮都重置底稿(self, tasks):
        """追加与替换会改掉文件。不重置的话第二轮的起点和第一轮不同，
        5 轮的数据就不可比。
        """
        for task in tasks:
            assert any("setup_long_tasks" in cmd for cmd in task["reset"]), task["name"]

    def test_重置先清场(self, tasks):
        for task in tasks:
            assert any("reset_desktop" in cmd for cmd in task["reset"]), task["name"]


class TestNoOverlapWithOtherSets:
    def test_不与评测集和采集集重名(self, tasks):
        """重名会让存档和轨迹目录混在一起，事后分不清哪轮是哪个集合。"""
        others: set[str] = set()
        for path in (
            "tasks/basic_tasks.yaml",
            "tasks/desktop_20.yaml",
            "tasks/collect_variants.yaml",
        ):
            spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
            others |= {t["name"] for t in spec["tasks"]}
        for task in tasks:
            assert task["name"] not in others

    def test_用的是长任务专用文件(self, tasks):
        """碰评测集的文件会污染那边的起点。"""
        from tasks.setup_long_tasks import FIXTURES, GENERATED

        allowed = {n.lower() for n in list(FIXTURES) + list(GENERATED)}
        for task in tasks:
            for check in _checks(task["success_check"]):
                path = check.get("path")
                if path:
                    assert Path(path).name.lower() in allowed, f"{task['name']} 碰了别处的文件"


class TestSetupScript:
    def test_底稿含追加判据要查的那句(self):
        from tasks.setup_long_tasks import FIXTURES

        assert "底稿原有内容" in FIXTURES["长任务底稿.txt"]

    def test_替换底稿里旧词出现多次(self):
        """只有一处的话，「全部替换」和「替换一个」测不出区别。"""
        from tasks.setup_long_tasks import FIXTURES

        assert FIXTURES["长任务替换.txt"].count("旧词") >= 2

    def test_产物被列为每轮要删的(self):
        from tasks.setup_long_tasks import GENERATED

        assert "三行.txt" in GENERATED

    def test_重置真的写文件也真的删产物(self, tmp_path, monkeypatch):
        import tasks.setup_long_tasks as mod

        monkeypatch.setattr(mod, "TEST_DIR", tmp_path)
        (tmp_path).mkdir(exist_ok=True)
        (tmp_path / "三行.txt").write_text("上一轮留下的", encoding="utf-8")
        written, removed = mod.reset()
        assert written == len(mod.FIXTURES) and removed == 1
        assert not (tmp_path / "三行.txt").exists()
        assert (tmp_path / "长任务底稿.txt").read_text(encoding="utf-8").startswith("底稿原有内容")


def _checks(spec) -> list[dict]:
    if isinstance(spec, dict) and "checks" in spec:
        return list(spec["checks"])
    if isinstance(spec, dict):
        return [spec]
    return list(spec or [])


def _by_name(tasks: list[dict], name: str) -> dict:
    return next(t for t in tasks if t["name"] == name)
