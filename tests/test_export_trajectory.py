"""导出扫描 —— **拦不住的东西要能被说清楚**。

这个脚本是「轨迹截图不出客机」这条规矩的唯一放行口。它的测试要守两件事：

1. 该拦的拦住，而且是**拒绝导出**，不是打印警告
2. 报告里必须写明它**扫不了截图内容**——否则读报告的人会以为数据已经安全了

第 2 条看起来像文档洁癖，其实是这个方案里最容易出事的地方：一份写着
「扫描通过」的报告，会让人跳过人工抽查。
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.export_trajectory import (
    export_one,
    mask,
    scan_text,
    scan_trajectory,
    write_report,
)


def make_traj(root: Path, name: str, steps_text: str = '{"step": 1}\n', frames: int = 2) -> Path:
    traj = root / name
    (traj / "frames").mkdir(parents=True, exist_ok=True)
    (traj / "steps.jsonl").write_text(steps_text, encoding="utf-8")
    (traj / "meta.json").write_text(f'{{"trajectory_id": "{name}"}}', encoding="utf-8")
    for i in range(frames):
        (traj / "frames" / f"step{i:03d}-before.png").write_bytes(b"\x89PNG\r\n\x1a\n fake")
    return traj


class TestScanText:
    def test_抓出api_key(self, tmp_path):
        path = tmp_path / "steps.jsonl"
        path.write_text('{"raw": "DASHSCOPE_API_KEY=sk-abcdefghijklmnop1234"}', encoding="utf-8")
        rules = {h.rule for h in scan_text(path)}
        assert "api_key" in rules or "dotenv" in rules

    def test_抓出邮箱(self, tmp_path):
        path = tmp_path / "steps.jsonl"
        path.write_text('{"ocr": "someone@example.com"}', encoding="utf-8")
        assert any(h.rule == "email" for h in scan_text(path))

    def test_抓出env片段(self, tmp_path):
        path = tmp_path / "steps.jsonl"
        path.write_text('{"raw": "LLM_PROVIDER=dashscope"}', encoding="utf-8")
        assert any(h.rule == "dotenv" for h in scan_text(path))

    def test_干净文本不报(self, tmp_path):
        path = tmp_path / "steps.jsonl"
        path.write_text('{"action": "left_click", "x": 500, "y": 400}', encoding="utf-8")
        assert scan_text(path) == []

    def test_读不出来的文件算命中而不是放行(self, tmp_path):
        """读不出来就放行是危险的默认值——改名的 dump 正好会走这条路。"""
        path = tmp_path / "weird.json"
        path.write_bytes(b"\xff\xfe\x00\x01\x02binary")
        assert any(h.rule == "unreadable" for h in scan_text(path))

    def test_命中片段做掩码(self):
        """报告本身会被传阅，不能把密钥原样抄进去。"""
        masked = mask("sk-abcdefghijklmnop1234")
        assert "abcdefghijklmnop" not in masked
        assert masked.startswith("sk-a")


class TestScanTrajectory:
    def test_只扫文本不扫图(self, tmp_path):
        """截图是图像，正则扫不了——这是这个方案的已知边界，不是遗漏。"""
        traj = make_traj(tmp_path, "t1")
        _, scanned = scan_trajectory(traj)
        assert scanned == 2  # steps.jsonl + meta.json，不含 png


class TestExportOne:
    def test_干净轨迹带截图导出(self, tmp_path):
        traj = make_traj(tmp_path / "src", "t1", frames=3)
        report = export_one(traj, tmp_path / "out")
        assert report.exported
        assert report.frames == 3
        assert (tmp_path / "out" / "t1" / "steps.jsonl").exists()
        assert (tmp_path / "out" / "t1" / "frames").is_dir()

    def test_命中就拒绝导出而不是警告(self, tmp_path):
        """**拒绝**是这个脚本的全部价值。改成警告就等于没有这一层。"""
        traj = make_traj(tmp_path / "src", "t2", steps_text='{"raw": "password: hunter2xyz"}\n')
        report = export_one(traj, tmp_path / "out")
        assert not report.exported
        assert not (tmp_path / "out" / "t2").exists()
        assert report.hits

    def test_可以只导出文本不带图(self, tmp_path):
        """平时的默认做法：只出 meta + steps，截图留在客机。"""
        traj = make_traj(tmp_path / "src", "t3", frames=2)
        report = export_one(traj, tmp_path / "out", include_frames=False)
        assert report.exported and report.frames == 0
        assert not (tmp_path / "out" / "t3" / "frames").exists()

    def test_缺steps的目录跳过(self, tmp_path):
        (tmp_path / "src" / "empty").mkdir(parents=True)
        report = export_one(tmp_path / "src" / "empty", tmp_path / "out")
        assert not report.exported


class TestReport:
    def test_报告写明扫不了截图(self, tmp_path):
        """一份只写「扫描通过」的报告，会让人跳过人工抽查。"""
        traj = make_traj(tmp_path / "src", "t1")
        reports = [export_one(traj, tmp_path / "out")]
        path = write_report(reports, tmp_path / "out", include_frames=True)
        text = path.read_text(encoding="utf-8")
        assert "不能" in text and "正则扫不了图像" in text

    def test_报告留出人工抽查栏位(self, tmp_path):
        traj = make_traj(tmp_path / "src", "t1")
        path = write_report([export_one(traj, tmp_path / "out")], tmp_path / "out", True)
        assert "人工抽查" in path.read_text(encoding="utf-8")

    def test_被拦下的轨迹在报告里列出原因(self, tmp_path):
        traj = make_traj(tmp_path / "src", "bad", steps_text='{"raw":"a@b.com"}\n')
        reports = [export_one(traj, tmp_path / "out")]
        path = write_report(reports, tmp_path / "out", True)
        text = path.read_text(encoding="utf-8")
        assert "被拦下的轨迹" in text and "bad" in text

    def test_机器可读的那份也写出来(self, tmp_path):
        """事后统计导出了多少轨迹、拦了多少，不该靠解析 Markdown。"""
        traj = make_traj(tmp_path / "src", "t1")
        write_report([export_one(traj, tmp_path / "out")], tmp_path / "out", True)
        payload = json.loads((tmp_path / "out" / "export-report.json").read_text(encoding="utf-8"))
        assert payload["include_frames"] is True
        assert payload["reports"][0]["trajectory"] == "t1"
