"""恢复率统计 —— **这个文件守的是口径，不是功能**。

M4 的恢复率数字上一次就是口径错的：把「安全阀兜底」算成了「恢复」，
报成 90%，真实值 0/30，虚高整整 90 个百分点。

所以这里的每条测试都对着一种会让数字虚高的算法：
分母把判定不了的也算进去、A/B 两类合成一个数、把「后来成功了」
当成「被救回来了」。
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_recovery import analyze, classify, export_labels, subtask_outcome


def step(**overrides) -> dict:
    base = {
        "step": 1,
        "subtask_id": 0,
        "subtask": "点击发送按钮",
        "execution_status": "ok",
        "screenshot_before": "frames/step001-before.png",
        "action_intent": {},
        "meta": {},
    }
    base.update(overrides)
    return base


def rejected(changed: bool | None, **overrides) -> dict:
    meta = {"reflector": {"decision": "reject"}}
    if changed is not None:
        meta["change"] = {"changed": changed, "ratio": 0.2 if changed else 0.0}
    return step(meta=meta, **overrides)


def make_traj(root: Path, name: str, steps: list[dict]) -> Path:
    traj = root / name
    traj.mkdir(parents=True, exist_ok=True)
    (traj / "steps.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in steps) + "\n", encoding="utf-8"
    )
    return traj


class TestClassify:
    def test_屏幕变了是A类(self):
        assert classify(step(meta={"change": {"changed": True}})) == "A"

    def test_屏幕没变是B类(self):
        """B 类是「点空了」：执行成功但界面纹丝不动，实测占 38%。"""
        assert classify(step(meta={"change": {"changed": False}})) == "B"

    def test_没有帧差记录时不猜(self):
        """**算进 A 或 B 都会让那一类虚高**，而分开看正是这个脚本的全部意义。"""
        assert classify(step(meta={})) == "unknown"


class TestSubtaskOutcome:
    def test_以done收尾算成功(self):
        steps = [step(step=1), step(step=2, action_intent={"done": True})]
        assert subtask_outcome(steps, 0) == "success"

    def test_跑到最后没done算步数用尽(self):
        assert subtask_outcome([step(step=1), step(step=2)], 0) == "exhausted"

    def test_被安全拦下单独记(self):
        """拦截和「做不到」是两件事，混在一起会让恢复率看起来更差。"""
        steps = [step(step=1, execution_status="blocked")]
        assert subtask_outcome(steps, 0) == "blocked"

    def test_查不到的子任务返回unknown而不是失败(self):
        """默认算失败会悄悄改变分母。"""
        assert subtask_outcome([step(subtask_id=0)], 99) == "unknown"


class TestAnalyze:
    def test_按错误类型分项而不是合成一个数(self, tmp_path):
        """A 类要模型自己改主意，B 类换个动作就可能救回来。

        **合成一个数会把这个区别抹掉**，而它正是决定下一步改什么的依据。
        """
        make_traj(
            tmp_path,
            "t1",
            [
                rejected(changed=False, step=1, subtask_id=0),
                step(step=2, subtask_id=0, action_intent={"done": True}),
                rejected(changed=True, step=3, subtask_id=1),
                step(step=4, subtask_id=1),
            ],
        )
        buckets = analyze([tmp_path / "t1"]).by_class()
        assert buckets["B"]["n"] == 1 and buckets["B"]["比例"] == 1.0
        assert buckets["A"]["n"] == 1 and buckets["A"]["比例"] == 0.0

    def test_判定不了的单独成一类不混进AB(self, tmp_path):
        make_traj(tmp_path, "t1", [rejected(changed=None, step=1)])
        buckets = analyze([tmp_path / "t1"]).by_class()
        assert "unknown" in buckets
        assert "A" not in buckets and "B" not in buckets

    def test_只统计否决不统计接受(self, tmp_path):
        """接受的判定不构成「恢复」的分母，算进去会把比例稀释。"""
        make_traj(
            tmp_path,
            "t1",
            [
                step(step=1, meta={"reflector": {"decision": "accept"}}),
                rejected(changed=False, step=2),
            ],
        )
        result = analyze([tmp_path / "t1"])
        assert len(result.rejections) == 1
        assert result.judgements["accept"] == 1 and result.judgements["reject"] == 1

    def test_没开reflector时不报错只是没数据(self, tmp_path):
        """基线轮次默认不开 Reflector，脚本要能安静地跑完。"""
        make_traj(tmp_path, "t1", [step(step=1), step(step=2)])
        result = analyze([tmp_path / "t1"])
        assert result.rejections == [] and result.steps == 2

    def test_多条轨迹一起统计(self, tmp_path):
        make_traj(tmp_path, "t1", [rejected(changed=False, step=1)])
        make_traj(tmp_path, "t2", [rejected(changed=False, step=1)])
        result = analyze([tmp_path / "t1", tmp_path / "t2"])
        assert result.trajectories == 2 and len(result.rejections) == 2

    def test_残缺的轨迹不会毁掉整次统计(self, tmp_path):
        """进程被急停杀掉时，最后一行天然可能写了一半。"""
        traj = make_traj(tmp_path, "t1", [rejected(changed=False, step=1)])
        with (traj / "steps.jsonl").open("a", encoding="utf-8") as handle:
            handle.write('{"step": 2, "sub')
        assert len(analyze([traj]).rejections) == 1


class TestExportLabels:
    def test_只导出done时刻(self, tmp_path):
        """标准 3 要对照的是「模型说完成」这些时刻，其余步骤不用人看。"""
        traj = make_traj(
            tmp_path,
            "t1",
            [step(step=1), step(step=2, action_intent={"done": True})],
        )
        count = export_labels([traj], tmp_path / "labels.csv")
        assert count == 1

    def test_人工只需要填一列(self, tmp_path):
        """要标的东西越多，标注越容易半途而废——标准 3 的瓶颈就是人工。"""
        traj = make_traj(tmp_path, "t1", [step(step=1, action_intent={"done": True})])
        export_labels([traj], tmp_path / "labels.csv")
        header = (tmp_path / "labels.csv").read_text(encoding="utf-8-sig").splitlines()[0]
        assert header.count(",") == 5
        assert "人工判定" in header

    def test_带上截图路径(self, tmp_path):
        """不带路径的话，标注的人得自己去翻目录找图，标到一半就会放弃。"""
        traj = make_traj(
            tmp_path,
            "t1",
            [step(step=1, action_intent={"done": True}, screenshot_before="frames/x.png")],
        )
        export_labels([traj], tmp_path / "labels.csv")
        assert "frames/x.png" in (tmp_path / "labels.csv").read_text(encoding="utf-8-sig")

    def test_用带bom的utf8以便excel直接打开(self, tmp_path):
        """标注的人多半用 Excel 打开。不带 BOM 的话中文全是乱码。"""
        traj = make_traj(tmp_path, "t1", [step(step=1, action_intent={"done": True})])
        export_labels([traj], tmp_path / "labels.csv")
        assert (tmp_path / "labels.csv").read_bytes().startswith(b"\xef\xbb\xbf")


class TestFieldNamesMatchTheRealWriter:
    """**字段名必须跟着写入方走。**

    本项目反复踩的坑之一就是「没先查清字段的约定就直接读」——漏了
    `point_norm` 那次，下游读到的永远是空。这里把读写两端钉在一起：
    `core/reflector.py` 改了字段名或取值，这条测试先红，而不是等统计
    跑出一片 0 之后才发现。
    """

    def test_读的键名与Verdict写出来的一致(self):
        from core.reflector import Verdict

        payload = Verdict(verdict="reject", reason="没变化", level=1).as_dict()
        assert "verdict" in payload

    def test_否决的取值就是reject(self):
        from core.reflector import VERDICT_REJECT

        assert VERDICT_REJECT == "reject"

    def test_真实的否决记录能被统计到(self, tmp_path):
        """用 Reflector 真正写出来的那份 dict 造数据，不手搓。

        手搓的话，写入方改了字段名这条测试照样绿——那就等于没测。
        """
        from core.reflector import Verdict

        verdict = Verdict(verdict="reject", reason="屏幕没变化", level=1)
        make_traj(
            tmp_path,
            "t1",
            [
                step(
                    step=1,
                    meta={"reflector": verdict.as_dict(), "change": {"changed": False}},
                ),
                step(step=2, action_intent={"done": True}),
            ],
        )
        result = analyze([tmp_path / "t1"])
        assert len(result.rejections) == 1
        assert result.rejections[0].error_class == "B"
