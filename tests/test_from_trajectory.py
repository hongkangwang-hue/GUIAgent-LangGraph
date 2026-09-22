"""蒸馏样本的筛选规则 —— **每一条都对应一次实测教训，不能被悄悄改掉**。

这个文件守的是「教师轨迹转训练样本」这一步。它处在链条的上游：
这里漏进去一条脏样本，要等到训练完、跑完端到端才看得出来，而那时已经
烧掉几个小时 GPU 和一轮客机实测。

四条规则各自防的东西：

- **失败轮不要** —— 失败轨迹里的动作大多是错的
- **点空的步骤不要** —— M4 实测 38% 的动作「执行成功但屏幕没变」，
  它们在日志里和成功的点击长得一模一样
- **`done` 限量** —— 上一轮重训因 `done` 占 41.6%，误报率从 7.8% 涨到 36.5%
- **同图同答案去重** —— 一个任务跑 5 轮，开头几步常常逐像素相同
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from finetune.from_trajectory import (
    build_record,
    cap_done,
    collect,
    dedupe,
    done_record,
    image_ref,
    screen_changed,
    step_params,
    verified_trajectories,
    write_dataset,
)


def make_step(**overrides) -> dict:
    step = {
        "step": 1,
        "subtask_id": 0,
        "subtask": "点击任务栏上的 Edge 图标",
        "screenshot_before": "frames/step001-before.png",
        "screenshot_after": "frames/step001-after.png",
        "execution_status": "ok",
        "action_model_coords": {"action": "left_click", "x": 500, "y": 400},
        "action_intent": {"action_type": "left_click"},
    }
    step.update(overrides)
    return step


def write_png(path: Path, color: tuple[int, int, int], size=(8, 8)) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def make_trajectory(
    root: Path, traj_id: str, steps: list[dict], frames: dict | None = None
) -> Path:
    """造一条轨迹目录。`frames` 给 {相对路径: 颜色}，不给就按步骤自动造。"""
    traj = root / traj_id
    traj.mkdir(parents=True, exist_ok=True)
    (traj / "steps.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in steps) + "\n", encoding="utf-8"
    )
    if frames is None:
        frames = {}
        for index, step in enumerate(steps):
            # 默认前后不同色 = 屏幕变了
            if step.get("screenshot_before"):
                frames[step["screenshot_before"]] = (index * 10 % 250, 0, 0)
            if step.get("screenshot_after"):
                frames[step["screenshot_after"]] = (0, index * 10 % 250, 255)
    for rel, color in frames.items():
        write_png(traj / rel, color)
    return traj


class TestBuildRecord:
    def test_字段与基线训练格式对齐(self):
        """`train_lora` 只认 image / instruction / answer，`eval.action` 要按 action 分桶。

        多一份格式就多一处会漂的地方。
        """
        record = build_record(make_step(), "traj-1", "images/a.png")
        for key in ("image", "instruction", "answer", "action", "session_id"):
            assert key in record

    def test_坐标不做任何换算(self):
        """轨迹里的模型坐标已经是 1000×1000 空间，训练格式要的正是这个空间。

        **多一次缩放就多一个错位来源。**
        """
        record = build_record(
            make_step(action_model_coords={"action": "left_click", "x": 137, "y": 926}), "t", "i"
        )
        assert json.loads(record["answer"])["x"] == 137
        assert json.loads(record["answer"])["y"] == 926
        assert record["point_norm"] == [137, 926]

    def test_越界坐标直接丢掉(self):
        """越界的点在真机上会被安全层拦下，拿它当正确答案是在教模型犯错。"""
        step = make_step(action_model_coords={"action": "left_click", "x": 1500, "y": 10})
        assert build_record(step, "t", "i") is None

    def test_缺坐标的点击动作丢掉(self):
        step = make_step(action_model_coords={"action": "left_click"})
        assert build_record(step, "t", "i") is None

    def test_不需要坐标的动作照收(self):
        """`type` / `key` 没有坐标，但它们是训练集最缺的那一类。"""
        step = make_step(action_model_coords={"action": "type", "text": "你好世界"})
        record = build_record(step, "t", "i")
        assert json.loads(record["answer"]) == {"action": "type", "text": "你好世界"}

    def test_子任务为空的步骤丢掉(self):
        """`instruction` 就是子任务文本，空的样本没有监督信号。"""
        assert build_record(make_step(subtask="  "), "t", "i") is None

    def test_思考文本不进训练目标(self):
        """`reasoning` 是我们塞进去的，不是模型该输出的动作参数。"""
        params = step_params({"action": "left_click", "x": 1, "y": 2, "reasoning": "我觉得"})
        assert params == {}

    def test_按轨迹分组而不是按样本(self):
        """同一轮的步骤高度相关，划分时必须整条一起走，否则是泄漏。"""
        record = build_record(make_step(), "traj-9", "i")
        assert record["session_id"] == "traj-9"


class TestDoneRecord:
    def test_只认真正的完成信号(self):
        step = make_step(execution_status="no_action", action_intent={"done": True})
        record = done_record(step, "t", "i")
        assert json.loads(record["answer"]) == {"done": True}

    def test_不是完成信号的返回空(self):
        assert done_record(make_step(), "t", "i") is None


class TestCapDone:
    def test_把done占比压到上限以下(self):
        """训练集里 done 占 41.6% 那一轮，误报率从 7.8% 涨到 36.5%。"""
        records = [{"action": "left_click", "session_id": "t", "action_index": i} for i in range(8)]
        records += [
            {"action": "done", "session_id": "t", "action_index": 100 + i} for i in range(8)
        ]
        kept = cap_done(records, cap=0.2)
        dones = [r for r in kept if r["action"] == "done"]
        assert len(dones) / len(kept) <= 0.2

    def test_没有done时原样返回(self):
        records = [{"action": "type", "session_id": "t", "action_index": 1}]
        assert cap_done(records, cap=0.2) == records


class TestScreenChanged:
    def test_前后不同判为变了(self, tmp_path):
        write_png(tmp_path / "a.png", (0, 0, 0))
        write_png(tmp_path / "b.png", (255, 255, 255))
        assert screen_changed(tmp_path / "a.png", tmp_path / "b.png") is True

    def test_前后相同判为没变(self, tmp_path):
        write_png(tmp_path / "a.png", (10, 10, 10))
        write_png(tmp_path / "b.png", (10, 10, 10))
        assert screen_changed(tmp_path / "a.png", tmp_path / "b.png") is False

    def test_缺图返回None而不是False(self, tmp_path):
        """**判定不了**和**确认没变**必须分开。

        混为一谈的话，导出时漏掉几张图就会让一批正确的步骤被当成点空丢掉。
        """
        write_png(tmp_path / "a.png", (0, 0, 0))
        assert screen_changed(tmp_path / "a.png", tmp_path / "missing.png") is None


class TestVerifiedTrajectories:
    def _archive(self, tmp_path: Path, records: list[dict]) -> Path:
        path = tmp_path / "archive.json"
        path.write_text(json.dumps({"records": records}, ensure_ascii=False), encoding="utf-8")
        return path

    def test_只收判定通过的轮次(self, tmp_path):
        archive = self._archive(
            tmp_path,
            [
                {"trajectory_id": "ok-1", "verified": True, "precondition_ok": True},
                {"trajectory_id": "bad-1", "verified": False, "precondition_ok": True},
            ],
        )
        assert verified_trajectories(archive) == {"ok-1"}

    def test_起点没建立的不算(self, tmp_path):
        """起点没建立的轮次是无效轮，既不算成功也不算失败，更不能当示范。"""
        archive = self._archive(
            tmp_path, [{"trajectory_id": "x", "verified": True, "precondition_ok": False}]
        )
        assert verified_trajectories(archive) == set()

    def test_事后剔除的不算(self, tmp_path):
        archive = self._archive(
            tmp_path,
            [{"trajectory_id": "x", "verified": True, "precondition_ok": True, "excluded": True}],
        )
        assert verified_trajectories(archive) == set()

    def test_模型自报完成不作为依据(self, tmp_path):
        """实测一轮 25/25 自报完成而程序化判定 0/25 —— 拿它筛等于没筛。"""
        archive = self._archive(
            tmp_path,
            [
                {
                    "trajectory_id": "x",
                    "model_said_done": True,
                    "verified": False,
                    "precondition_ok": True,
                }
            ],
        )
        assert verified_trajectories(archive) == set()


class TestCollect:
    def test_失败轮整条不要(self, tmp_path):
        make_trajectory(tmp_path, "bad", [make_step()])
        records, stats = collect([tmp_path / "bad"], {"good"}, tmp_path / "out")
        assert records == []
        assert stats["轮次判定未通过"] == 1

    def test_点空的步骤被丢掉(self, tmp_path):
        """前后两帧一模一样 = 执行成功但屏幕没动。M4 实测占 38%。"""
        step = make_step()
        make_trajectory(
            tmp_path,
            "t1",
            [step],
            frames={step["screenshot_before"]: (7, 7, 7), step["screenshot_after"]: (7, 7, 7)},
        )
        records, stats = collect([tmp_path / "t1"], {"t1"}, tmp_path / "out")
        assert records == []
        assert stats["执行后屏幕没变（点空）"] == 1

    def test_执行失败的步骤被丢掉(self, tmp_path):
        make_trajectory(tmp_path, "t1", [make_step(execution_status="blocked")])
        records, stats = collect([tmp_path / "t1"], {"t1"}, tmp_path / "out")
        assert records == []
        assert stats["执行状态 blocked"] == 1

    def test_中途的done被丢掉只留收尾的(self, tmp_path):
        """子任务没做完就报完成，正是 3B 学坏的那个毛病，不能再教一遍。"""
        steps = [
            make_step(step=1, execution_status="no_action", action_intent={"done": True}),
            make_step(step=2),
            make_step(step=3, execution_status="no_action", action_intent={"done": True}),
        ]
        make_trajectory(tmp_path, "t1", steps)
        records, stats = collect([tmp_path / "t1"], {"t1"}, tmp_path / "out")
        assert stats["非收尾的 done"] == 1
        assert [r["action"] for r in records] == ["left_click", "done"]

    def test_帧被复制进输出目录(self, tmp_path):
        """训练可能在另一台机器上跑，输出目录必须自带全部素材。"""
        make_trajectory(tmp_path, "t1", [make_step()])
        records, _ = collect([tmp_path / "t1"], {"t1"}, tmp_path / "out")
        assert (tmp_path / "out" / "images" / "t1").is_dir()
        assert "\\" not in records[0]["image"], "反斜杠路径到 Linux 上一张图都打不开"

    def test_样本里的图片路径训练进程打得开(self, tmp_path):
        """`train_lora` 把 image 原样交给 process_vision_info，它按**当前工作
        目录**解析。写成相对输出目录的路径，训练启动后才会发现图找不到。
        """
        make_trajectory(tmp_path, "t1", [make_step()])
        records, _ = collect([tmp_path / "t1"], {"t1"}, tmp_path / "out")
        assert Path(records[0]["image"]).exists(), "从当前工作目录解析不到这张图"


class TestDedupe:
    def test_同图同答案只留一条(self, tmp_path):
        write_png(tmp_path / "images" / "a.png", (1, 2, 3))
        write_png(tmp_path / "images" / "b.png", (1, 2, 3))  # 内容相同，文件名不同
        records = [
            {"image": "images/a.png", "answer": '{"action":"left_click","x":1,"y":2}'},
            {"image": "images/b.png", "answer": '{"action":"left_click","x":1,"y":2}'},
        ]
        kept, dropped = dedupe(records, tmp_path)
        assert len(kept) == 1 and dropped == 1

    def test_同图不同答案都留(self, tmp_path):
        write_png(tmp_path / "images" / "a.png", (1, 2, 3))
        records = [
            {"image": "images/a.png", "answer": '{"action":"left_click","x":1,"y":2}'},
            {"image": "images/a.png", "answer": '{"action":"type","text":"x"}'},
        ]
        kept, dropped = dedupe(records, tmp_path)
        assert len(kept) == 2 and dropped == 0


class TestWriteDataset:
    def test_验证集原样复制不重新生成(self, tmp_path):
        """微调前后要在同一把尺子上比。验证集跟着变，两个数就不可比了。"""
        base = tmp_path / "base"
        base.mkdir()
        (base / "train.jsonl").write_text('{"action":"left_click"}\n', encoding="utf-8")
        (base / "val.jsonl").write_text('{"sample_id":"v1"}\n', encoding="utf-8")

        out = tmp_path / "out"
        write_dataset([{"action": "type", "session_id": "t"}], out, base)
        assert (out / "val.jsonl").read_text(encoding="utf-8") == (base / "val.jsonl").read_text(
            encoding="utf-8"
        )

    def test_基线与蒸馏样本合并且数得清(self, tmp_path):
        base = tmp_path / "base"
        base.mkdir()
        (base / "train.jsonl").write_text('{"action":"left_click"}\n' * 3, encoding="utf-8")
        (base / "val.jsonl").write_text("{}\n", encoding="utf-8")

        meta = write_dataset(
            [{"action": "type", "session_id": "t"}, {"action": "done", "session_id": "t"}],
            tmp_path / "out",
            base,
        )
        assert meta["base_train"] == 3
        assert meta["distilled"] == 2
        assert meta["train_total"] == 5

    def test_meta记下用了哪些轨迹(self, tmp_path):
        """事后要能回答「这份数据是从哪几轮来的」。记不下来就没法复现。"""
        base = tmp_path / "base"
        base.mkdir()
        (base / "train.jsonl").write_text("", encoding="utf-8")
        meta = write_dataset(
            [
                {"action": "type", "session_id": "traj-a"},
                {"action": "type", "session_id": "traj-b"},
            ],
            tmp_path / "out",
            base,
        )
        assert meta["distill_sessions"] == ["traj-a", "traj-b"]


class TestNoMixBase:
    def test_可以只写蒸馏样本(self, tmp_path):
        """用来单独看蒸馏数据的分布，不必每次都拖上 1783 条基线。"""
        base = tmp_path / "base"
        base.mkdir()
        (base / "train.jsonl").write_text('{"action":"left_click"}\n' * 5, encoding="utf-8")
        meta = write_dataset(
            [{"action": "type", "session_id": "t"}], tmp_path / "out", base, mix_base=False
        )
        assert meta["base_train"] == 0
        assert meta["train_total"] == 1


@pytest.mark.parametrize("cap", [0.0, 0.2, 0.5])
def test_done上限是可配的且边界不炸(cap):
    records = [{"action": "left_click", "session_id": "t", "action_index": i} for i in range(10)]
    records += [{"action": "done", "session_id": "t", "action_index": 50 + i} for i in range(10)]
    kept = cap_done(records, cap=cap)
    dones = sum(1 for r in kept if r["action"] == "done")
    assert dones / len(kept) <= cap + 1e-9


class TestImageRef:
    def test_在工作目录下时写相对路径(self, tmp_path, monkeypatch):
        """与基线数据同一形态（`data/raw/...`），整个仓库拷走仍然能用。"""
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "out" / "images" / "a.png"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"x")
        assert image_ref(target) == "out/images/a.png"

    def test_在工作目录之外时退回绝对路径(self, tmp_path, monkeypatch):
        """宁可写死绝对路径，也不要写一个训练时解析不到的相对路径。"""
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        outside = tmp_path / "elsewhere" / "a.png"
        outside.parent.mkdir(parents=True)
        outside.write_bytes(b"x")
        assert Path(image_ref(outside)).is_absolute()
