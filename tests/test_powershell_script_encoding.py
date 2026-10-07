"""客机使用 Windows PowerShell 5.1，含中文的 .ps1 必须带 UTF-8 BOM。"""

from pathlib import Path


def test_guest_runner_has_utf8_bom() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_remaining_tests.ps1"
    assert script.read_bytes().startswith(b"\xef\xbb\xbf")
