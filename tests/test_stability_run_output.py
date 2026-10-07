"""独立稳定性存档不能覆盖历史 M2 文件。"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from scripts.stability_run import Report, Sample, render


class StabilityOutputTests(unittest.TestCase):
    def test_custom_output_writes_matching_json_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "w6-stability.json"
            samples = [
                Sample(0, 100, 2, 3, 0),
                Sample(60, 101, 2, 3, 1),
                Sample(120, 101, 2, 3, 2),
            ]
            report = Report(started_at="test", minutes=2, aborted_reason="test-stop")
            with redirect_stdout(io.StringIO()):
                render(report, samples, output=output)

            stored = json.loads(output.read_text(encoding="utf-8"))
            markdown = output.with_suffix(".md").read_text(encoding="utf-8")
            self.assertEqual(stored["aborted_reason"], "test-stop")
            self.assertIn("test-stop", markdown)
            self.assertIn(str(output), markdown)


if __name__ == "__main__":
    unittest.main()
