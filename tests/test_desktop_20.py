"""M5 桌面任务测试集 —— 大纲第 7 周任务 1。

## 这套测试守的是什么

任务清单是**数据**不是代码，写错了不会报错，只会让评测结果静静地失真。
M2 已经栽过三次，形态都一样：**判据一上来就满足，Agent 什么都没做也打勾**。

    起点没清干净       taskkill 失败，浏览器本来就开着
    记事本会话恢复     杀了再开，Windows 把上次的标签页恢复回来
    评测脚本自己的窗口 标题含 "Python"，被「搜索 Python 官方文档」判为成功

前两个是 precondition 漏了反方向，第三个是判定器看见了自己。所以这里
逐条检查每个任务的 precondition 是不是判据的反方向——**这是数据层面的
不变量，只能用测试守**。
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

BASIC = Path("tasks/basic_tasks.yaml")
FULL = Path("tasks/desktop_20.yaml")


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tasks() -> list[dict]:
    return _load(FULL)["tasks"]


class TestOriginalFiveUnchanged:
    """前 5 个任务必须与 `basic_tasks.yaml` 逐字段相同。

    **它们是 M2/M3 全部实测的对象。** 报告 §7 的端到端 17%、§10.5 的拆解
    粒度分析都建立在它们之上——改一个字，那些结论就不再可比，而且从新的
    数字上完全看不出来。
    """

    def test_前五个逐字段相同(self):
        old = _load(BASIC)["tasks"]
        new = _load(FULL)["tasks"][:5]
        assert [t["name"] for t in new] == [t["name"] for t in old]
        for a, b in zip(old, new, strict=True):
            assert a == b, f"{a['name']} 与 basic_tasks.yaml 不一致"

    def test_新增的在后面(self):
        """顺序有意义：前 5 个是历史对照，改动只能追加在后面。"""
        names = [t["name"] for t in _load(FULL)["tasks"]]
        assert names[:5] == [t["name"] for t in _load(BASIC)["tasks"]]


class TestTaskSetShape:
    def test_正好二十个(self):
        """大纲第 7 周任务 1 要求「包含 20 个不同难度桌面任务的测试集」。"""
        assert len(_load(FULL)["tasks"]) == 20

    def test_名字不重复(self):
        """重名的话结果按名字聚合时会悄悄合并，样本量对不上。"""
        names = [t["name"] for t in _load(FULL)["tasks"]]
        assert len(names) == len(set(names))

    def test_三档难度都有且都不止一个(self):
        """**只报总体成功率会被单步任务主导。**

        大纲要的是「不同难度」，报告要按难度分别给成功率——每档只有一个
        任务的话，那一档的数字就是一次伯努利试验，没有意义。
        """
        import collections

        counts = collections.Counter(t["difficulty"] for t in _load(FULL)["tasks"])
        assert set(counts) == {"单步", "多步", "复杂"}
        assert all(n >= 2 for n in counts.values()), counts


class TestEveryTaskIsWellFormed:
    """每个任务都得有指令、判据、reset、起点检查、步数上限。"""

    @pytest.mark.parametrize("field", ["name", "title", "difficulty", "instruction"])
    def test_必填字段(self, tasks, field):
        missing = [t.get("name", "?") for t in tasks if not t.get(field)]
        assert not missing, f"缺 {field}：{missing}"

    def test_都有判据(self, tasks):
        assert not [t["name"] for t in tasks if not t.get("success_check")]

    def test_都有_reset(self, tasks):
        """没有 reset 的话，第 N 轮的起点是前 N-1 轮的残留，5 轮之间不独立。"""
        assert not [t["name"] for t in tasks if not t.get("reset")]

    def test_reset_第一条都是清桌面(self, tasks):
        """`reset_desktop.py` 负责 Esc 关浮层、Win+D 最小化、杀常驻应用。

        **屏幕就是 Agent 的全部输入。** 上一轮遗留的窗口是下一轮输入的
        一部分——M2 实测：`send_message` 因为没关上一轮的记事本，
        模型整条轨迹都在说「记事本已打开」，5 轮全挂。
        """
        bad = [t["name"] for t in tasks if "reset_desktop.py" not in t["reset"][0]]
        assert not bad, bad

    def test_都有起点检查(self, tasks):
        """没有 precondition，环境失败会被记成 Agent 失败。"""
        assert not [t["name"] for t in tasks if not t.get("precondition")]

    def test_都有步数上限(self, tasks):
        """不封顶的话，一次死循环能把整批评测拖到跑不完。"""
        assert not [t["name"] for t in tasks if not t.get("max_steps")]

    def test_setup_env_clean_命令不带不支持的文件名(self, tasks):
        """setup_env.py 的 --clean 不接受位置参数；错误会被 reset 静默忽略。"""
        for task in tasks:
            for command in task["reset"]:
                if "tasks/setup_env.py --clean" in command:
                    assert command == "python tasks/setup_env.py --clean", task["name"]

    def test_资源管理器与显示桌面的起点可恢复(self, tasks):
        by_name = {task["name"]: task for task in tasks}
        assert by_name["open_explorer"]["reset"][0].endswith("--close-explorer")
        desktop = by_name["minimize_all"]
        assert desktop["precondition"][0]["visible_only"] is True
        assert desktop["success_check"][0]["visible_only"] is True

    def test_关闭记事本按窗口而非后台进程判定(self, tasks):
        task = next(item for item in tasks if item["name"] == "close_notepad")
        assert task["success_check"][0]["type"] == "window_title"
        assert task["success_check"][0]["should_match"] is False
        assert task["precondition"][0]["type"] == "window_title"

    def test_记事本文件任务启动前须无旧进程(self, tasks):
        by_name = {task["name"]: task for task in tasks}
        for name in ("write_note", "append_line", "copy_paste_text"):
            check = by_name[name]["precondition"]
            checks = check["checks"] if isinstance(check, dict) else check
            assert {
                "type": "process",
                "name": "notepad.exe",
                "should_run": False,
            } in checks, name

    def test_步数上限随难度递增(self, tasks):
        """复杂任务的预算不该低于单步任务的。"""
        budget = {"单步": [], "多步": [], "复杂": []}
        for t in tasks:
            budget[t["difficulty"]].append(t["max_steps"])
        assert max(budget["单步"]) <= min(budget["多步"])
        assert max(budget["多步"]) <= min(budget["复杂"])


class TestChecksAreSupported:
    """判据里用的类型和参数，`core.verify` 必须真的支持。

    **写错了不会报错**，只会在评测时把那一条判据整个判成失败——
    而失败看起来就像 Agent 没做到。
    """

    def _all_checks(self, spec) -> list[dict]:
        if spec is None:
            return []
        if isinstance(spec, dict) and "checks" in spec:
            return list(spec["checks"])
        if isinstance(spec, dict):
            return [spec]
        return list(spec)

    def test_判据类型都存在(self, tasks):
        from core.verify import CHECKERS

        for t in tasks:
            for spec in self._all_checks(t["success_check"]) + self._all_checks(
                t.get("precondition")
            ):
                assert spec["type"] in CHECKERS, f"{t['name']}: 未知判据 {spec['type']}"

    def test_判据参数都被接受(self, tasks):
        """逐个参数比对函数签名。**`window_title` 就是这么发现缺反方向的。**"""
        import inspect

        from core.verify import CHECKERS

        for t in tasks:
            for spec in self._all_checks(t["success_check"]) + self._all_checks(
                t.get("precondition")
            ):
                fn = CHECKERS[spec["type"]]
                allowed = set(inspect.signature(fn).parameters)
                extra = set(spec) - {"type"} - allowed
                assert not extra, f"{t['name']}: {spec['type']} 不认识 {extra}"

    def test_四种判据都有反方向(self):
        """M2 的假成功**全部**来自起点没清干净。

        起点检查必须能表达「判据的反面」，所以四种判据一个都不能少这个参数。
        """
        import inspect

        from core.verify import CHECKERS

        reverse = {
            "process": "should_run",
            "window_title": "should_match",
            "file_contains": "should_contain",
            "file_exists": "should_exist",
        }
        for name, param in reverse.items():
            assert param in inspect.signature(CHECKERS[name]).parameters, name


class TestPreconditionIsTheReverse:
    """起点检查必须真的是判据的反方向，不能只是「有就行」。"""

    def test_进程类任务的起点是反的(self, tasks):
        """判据说「进程应在运行」，起点就得是「进程不在运行」。

        `open_browser` 曾经没有这一条：taskkill 失败时浏览器本来就开着，
        Agent 一个动作都不做，判据直接打勾。
        """
        for t in tasks:
            checks = t["success_check"]
            checks = checks if isinstance(checks, list) else [checks]
            for c in checks:
                if c.get("type") != "process":
                    continue
                pres = t.get("precondition") or []
                pres = pres if isinstance(pres, list) else [pres]
                pres = pres[0]["checks"] if pres and "checks" in pres[0] else pres
                same = [p for p in pres if p.get("type") == "process" and p["name"] == c["name"]]
                if not same:
                    continue
                assert same[0].get("should_run", True) != c.get("should_run", True), (
                    f"{t['name']}: 起点与判据同方向，reset 失败时会假成功"
                )

    def test_文件类任务的起点是反的(self, tasks):
        for t in tasks:
            checks = t["success_check"]
            checks = checks["checks"] if isinstance(checks, dict) and "checks" in checks else checks
            checks = checks if isinstance(checks, list) else [checks]
            pres = t.get("precondition") or []
            pres = pres if isinstance(pres, list) else [pres]
            pres = pres[0]["checks"] if pres and "checks" in pres[0] else pres
            for c in checks:
                if c.get("type") not in ("file_contains", "file_exists"):
                    continue
                key = "should_contain" if c["type"] == "file_contains" else "should_exist"
                same = [p for p in pres if p.get("type") == c["type"] and p["path"] == c["path"]]
                if not same:
                    continue
                if c["type"] == "file_contains" and same[0].get("text") != c.get("text"):
                    continue
                assert same[0].get(key, True) != c.get(key, True), (
                    f"{t['name']}: {c['path']} 的起点与判据同方向"
                )


class TestFixturesExist:
    """判据引用的固定文件，`setup_env.py` 必须真的会生成。

    **判据里的字符串和 fixture 的内容是两处写死的同一个值。** 一处改了
    另一处没跟着改，任务会以「文件不含 X」的形式失败，而那看起来
    完全像是 Agent 做错了。
    """

    def test_判据引用的文本都在_fixture_里(self):
        import tasks.setup_env as env

        bodies = " ".join(
            [env.TEST_FILE_BODY, env.SOURCE_BODY, env.RENAME_BODY, env.TEMP_BODY, env.LONG_DOC_BODY]
        )
        # 这些是任务**运行时产生**的，不该在 fixture 里
        produced = {"已审阅", "今天完成了三项任务", "末尾已到达", "579", "63", "你好世界"}

        for t in _load(FULL)["tasks"]:
            spec = t["success_check"]
            spec = spec["checks"] if isinstance(spec, dict) and "checks" in spec else spec
            spec = spec if isinstance(spec, list) else [spec]
            for c in spec:
                if c.get("type") != "file_contains":
                    continue
                text = str(c["text"])
                if text in produced:
                    continue
                assert text in bodies, f"{t['name']} 要查的 {text!r} 不在任何 fixture 里"

    def test_生成物清单与任务一致(self):
        """`GENERATED` 决定 `--clean` 删什么。漏一个，那个任务的起点就建不起来。"""
        import tasks.setup_env as env

        # 所有 precondition 里要求「不存在」的路径
        need_clean = set()
        for t in _load(FULL)["tasks"]:
            pres = t.get("precondition") or []
            pres = pres if isinstance(pres, list) else [pres]
            pres = pres[0]["checks"] if pres and "checks" in pres[0] else pres
            for p in pres:
                if p.get("type") == "file_exists" and p.get("should_exist") is False:
                    need_clean.add(Path(p["path"]).name)
        assert need_clean <= set(env.GENERATED), (
            f"这些要在起点前删掉，但不在 GENERATED 里：{need_clean - set(env.GENERATED)}"
        )
