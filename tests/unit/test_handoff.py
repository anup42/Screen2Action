"""Tests for portable handoff generation."""

from __future__ import annotations

import json
from pathlib import Path

from screen2action.handoff import build_handoff_snapshot, write_handoff_snapshot


def test_handoff_snapshot_uses_portable_paths(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "docs" / "plans").mkdir(parents=True)
    (root / "configs" / "models").mkdir(parents=True)
    (root / "docs" / "plans" / "complete_pipeline.md").write_text(
        "**Active milestone:** Milestone 0\n"
        "### Milestone alpha (COMPLETE)\n"
        "## Validation ledger\n"
        "- `python -m pytest` - exit 0\n",
        encoding="utf-8",
    )
    (root / "docs" / "OPEN_QUESTIONS.md").write_text(
        "1. What is unresolved across a wrapped\n   continuation line?\n\n",
        encoding="utf-8",
    )
    (root / "configs" / "models" / "registry.yaml").write_text(
        "schema_version: 1\n",
        encoding="utf-8",
    )
    data_root = tmp_path / "external-data"
    manifest = data_root / "normalized" / "v1" / "manifests" / "data.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")

    snapshot = build_handoff_snapshot(
        root,
        environment={"SCREEN2ACTION_DATA_ROOT": str(data_root)},
    )

    assert snapshot["plan"]["completed_milestones"] == ["Milestone alpha"]
    assert snapshot["data_manifests"][0]["path"] == (
        "${SCREEN2ACTION_DATA_ROOT}/normalized/v1/manifests/data.json"
    )
    serialized = json.dumps(snapshot)
    assert str(tmp_path) not in serialized
    assert snapshot["commands"]["last_successful"] == ["python -m pytest"]
    assert snapshot["unresolved_decisions"] == [
        "What is unresolved across a wrapped continuation line?"
    ]


def test_handoff_writes_both_formats(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "docs" / "plans").mkdir(parents=True)
    (root / "configs" / "models").mkdir(parents=True)
    (root / "docs" / "plans" / "complete_pipeline.md").write_text(
        "**Active milestone:** Milestone 0\n",
        encoding="utf-8",
    )
    (root / "configs" / "models" / "registry.yaml").write_text(
        "schema_version: 1\n",
        encoding="utf-8",
    )

    markdown, machine, snapshot = write_handoff_snapshot(root, output_directory=tmp_path / "out")

    assert markdown.name == "HANDOFF.md"
    assert machine.name == "handoff.json"
    assert "Screen2Action handoff" in markdown.read_text(encoding="utf-8")
    assert json.loads(machine.read_text(encoding="utf-8")) == snapshot
