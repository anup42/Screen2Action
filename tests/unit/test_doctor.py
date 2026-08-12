"""Environment doctor tests."""

from __future__ import annotations

import json
from pathlib import Path

from screen2action.cli import main
from screen2action.doctor import doctor_report


def test_cpu_doctor_is_secret_safe_and_actionable(tmp_path: Path) -> None:
    environment = {
        "SCREEN2ACTION_DATA_ROOT": str(tmp_path / "private-data"),
        "SCREEN2ACTION_CACHE_ROOT": str(tmp_path / "private-cache"),
        "SCREEN2ACTION_RUN_ROOT": str(tmp_path / "private-runs"),
        "HF_TOKEN": "must-not-appear",
    }

    report = doctor_report(Path.cwd(), device="cpu", environment=environment)

    serialized = json.dumps(report)
    assert str(tmp_path) not in serialized
    assert "must-not-appear" not in serialized
    assert report["requested_device"] == "cpu"
    assert report["roots"]["data"]["display_path"] == "${SCREEN2ACTION_DATA_ROOT}"
    assert report["recommended_next_command"]


def test_doctor_cli_emits_json(capsys: object) -> None:
    exit_code = main(["doctor", "--device", "cpu", "--json"])
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    payload = json.loads(captured.out)
    assert exit_code in {0, 1}
    assert payload["requested_device"] == "cpu"
    assert isinstance(payload["warnings"], list)
