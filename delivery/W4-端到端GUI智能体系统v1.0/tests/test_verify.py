# ===================================================================== #
# file_contains 的反方向 —— 起点检查用
# ===================================================================== #


class TestFileContainsNegation:
    """`should_contain=False` 是给**起点检查**用的。

    「发送消息」的判据是「日志含你好世界」。上一轮的内容若没被 reset 清掉，
    Agent 什么都不做判据也会打勾——**假成功全都来自这个方向**。
    """

    def test_不含时通过(self, tmp_path):
        from core.verify import check_file_contains

        path = tmp_path / "log.txt"
        path.write_text("", encoding="utf-8")
        assert check_file_contains(str(path), "你好世界", should_contain=False).passed

    def test_含了就不通过(self, tmp_path):
        from core.verify import check_file_contains

        path = tmp_path / "log.txt"
        path.write_text("你好世界\n", encoding="utf-8")
        result = check_file_contains(str(path), "你好世界", should_contain=False)
        assert not result.passed
        assert "却含有" in result.detail  # 说法要跟着方向变，别把排查引反

    def test_文件不存在时不含成立(self, tmp_path):
        """一个不存在的文件当然不含任何东西。

        正方向（要求「含」）时文件不存在是失败，反方向时是成立——
        两者不能共用一条分支。
        """
        from core.verify import check_file_contains

        missing = str(tmp_path / "nope.txt")
        assert check_file_contains(missing, "x", should_contain=False).passed
        assert not check_file_contains(missing, "x", should_contain=True).passed

    def test_默认仍是正方向(self):
        """加参数不能改变已有调用的行为——`success_check` 全都靠默认值。"""
        import inspect

        from core.verify import check_file_contains

        assert inspect.signature(check_file_contains).parameters["should_contain"].default is True


class TestTaskPreconditions:
    """任务清单里每个任务的起点检查都必须**反向**覆盖它的判据。

    判据要「X 在运行」，起点就得「X 没在运行」；否则 reset 失败时
    Agent 一个动作不做也能打勾。**只有 close_app 一开始有这条**，
    另外三个是 2026-08-25 补的。
    """

    def _tasks(self):
        import yaml

        return yaml.safe_load(open("tasks/basic_tasks.yaml", encoding="utf-8"))["tasks"]

    def test_每个任务都有起点检查(self):
        missing = [t["name"] for t in self._tasks() if not t.get("precondition")]
        assert not missing, f"这些任务没有起点检查，reset 失败会被算成成功：{missing}"

    def test_起点检查都构造得出来(self):
        """参数名写错在 YAML 里不会报错，跑起来才炸——而那时已经在实测中了。"""
        from core.verify import SuccessCheck

        for task in self._tasks():
            check = SuccessCheck.from_spec(task["precondition"])
            assert check.checks

    def test_进程类判据有反向起点(self):
        """判据说「X 应在运行」，起点必须说「X 不应在运行」。"""
        for task in self._tasks():
            wants_running = {
                c.get("name")
                for c in _flatten(task.get("success_check"))
                if c.get("type") == "process" and c.get("should_run", True)
            }
            guards_absent = {
                c.get("name")
                for c in task.get("precondition") or []
                if c.get("type") == "process" and c.get("should_run", True) is False
            }
            assert wants_running <= guards_absent, (
                f"{task['name']}：判据要求 {wants_running} 在运行，"
                f"但起点没检查它们一开始是关着的（起点检查了 {guards_absent}）"
            )


def _flatten(spec):
    """`success_check` 支持单条 dict / list / 带 mode 的 dict 三种写法。"""
    if not spec:
        return []
    if isinstance(spec, dict):
        return spec.get("checks", []) if "checks" in spec else [spec]
    return list(spec)


# ===================================================================== #
# 判据本身 —— 这个模块决定每个任务成功还是失败
# ===================================================================== #


class TestSuccessCheckRun:
    """`SuccessCheck.run` 是所有成功率数字的最终出处。"""

    def test_没有判据不能算成功(self):
        """**默认值必须是失败。**

        一个忘了写 success_check 的任务如果默认算成功，整批数据会安静地
        变成 100%——而 100% 看起来像好消息，没人会去查。
        """
        from core.verify import SuccessCheck

        passed, detail = SuccessCheck(checks=[]).run()
        assert passed is False
        assert "不能默认算成功" in detail

    def test_未知判据类型算失败而不是跳过(self, tmp_path):
        """打错一个 type，那条判据必须变成失败，不能被静默忽略。

        忽略的话，`{"type": "flie_contains"}`（拼错）会让一条判据凭空消失，
        剩下的判据全过就算成功。
        """
        from core.verify import SuccessCheck

        passed, detail = SuccessCheck(checks=[{"type": "不存在的判据"}]).run()
        assert passed is False
        assert "未知判据类型" in detail

    def test_参数写错算失败并说清楚(self, tmp_path):
        from core.verify import SuccessCheck

        passed, detail = SuccessCheck(checks=[{"type": "file_exists", "拼错的参数": 1}]).run()
        assert passed is False
        assert "参数不对" in detail

    def test_判据执行抛异常算失败(self, monkeypatch):
        """**判定出错就是判定失败**，不能因为异常就放过去。"""
        from core import verify

        def boom(**_kwargs):
            raise RuntimeError("炸了")

        monkeypatch.setitem(verify.CHECKERS, "process", boom)
        passed, detail = verify.SuccessCheck(checks=[{"type": "process", "name": "x"}]).run()
        assert passed is False
        assert "执行出错" in detail

    def test_mode_all_要全过(self, tmp_path):
        from core.verify import SuccessCheck

        good = tmp_path / "a.txt"
        good.write_text("x", encoding="utf-8")
        spec = {
            "mode": "all",
            "checks": [
                {"type": "file_exists", "path": str(good)},
                {"type": "file_exists", "path": str(tmp_path / "missing.txt")},
            ],
        }
        assert SuccessCheck.from_spec(spec).run()[0] is False

    def test_mode_any_过一条就行(self, tmp_path):
        from core.verify import SuccessCheck

        good = tmp_path / "a.txt"
        good.write_text("x", encoding="utf-8")
        spec = {
            "mode": "any",
            "checks": [
                {"type": "file_exists", "path": str(good)},
                {"type": "file_exists", "path": str(tmp_path / "missing.txt")},
            ],
        }
        assert SuccessCheck.from_spec(spec).run()[0] is True

    def test_每条判据的结果都进说明(self, tmp_path):
        """失败时要能一眼看出是哪一条挂了，否则只能去猜。"""
        from core.verify import SuccessCheck

        _, detail = SuccessCheck(
            checks=[
                {"type": "file_exists", "path": str(tmp_path / "a")},
                {"type": "file_exists", "path": str(tmp_path / "b")},
            ]
        ).run()
        assert detail.count("✗") == 2


class TestFromSpec:
    """任务清单里 `success_check` 支持三种写法。"""

    def test_单条_dict(self):
        from core.verify import SuccessCheck

        check = SuccessCheck.from_spec({"type": "process", "name": "x"})
        assert len(check.checks) == 1
        assert check.mode == "all"

    def test_多条_list(self):
        from core.verify import SuccessCheck

        check = SuccessCheck.from_spec(
            [{"type": "process", "name": "a"}, {"type": "process", "name": "b"}]
        )
        assert len(check.checks) == 2

    def test_带_mode(self):
        from core.verify import SuccessCheck

        check = SuccessCheck.from_spec(
            {"mode": "any", "checks": [{"type": "process", "name": "x"}]}
        )
        assert check.mode == "any"

    def test_乱七八糟的输入得到空判据而不是崩(self):
        """空判据在 `run()` 里会算失败——**降级到失败，不是降级到成功**。"""
        from core.verify import SuccessCheck

        assert SuccessCheck.from_spec(None).checks == []
        assert SuccessCheck.from_spec("字符串").checks == []
        assert SuccessCheck.from_spec(None).run()[0] is False


class TestFileExists:
    def test_两个方向(self, tmp_path):
        from core.verify import check_file_exists

        path = tmp_path / "a.txt"
        assert not check_file_exists(str(path)).passed
        assert check_file_exists(str(path), should_exist=False).passed
        path.write_text("x", encoding="utf-8")
        assert check_file_exists(str(path)).passed
        assert not check_file_exists(str(path), should_exist=False).passed


class TestProcess:
    def test_自己这个_python_进程一定在(self):
        """用当前解释器做正样本，不依赖被测机器装了什么。"""
        import os
        import sys

        import psutil

        from core.verify import check_process

        name = psutil.Process(os.getpid()).name()
        assert check_process(name).passed
        assert not check_process(name, should_run=False).passed
        assert sys.executable  # 只是说明这条用例依赖真实进程表

    def test_不存在的进程(self):
        from core.verify import check_process

        assert not check_process("绝对不存在的进程xyz.exe").passed
        assert check_process("绝对不存在的进程xyz.exe", should_run=False).passed

    def test_说明里带上实际找到了几个(self):
        """「找到 9 个 calculatorapp.exe」这种信息是排查 reset 的关键线索。"""
        import os

        import psutil

        from core.verify import check_process

        detail = check_process(psutil.Process(os.getpid()).name()).detail
        assert "找到" in detail
