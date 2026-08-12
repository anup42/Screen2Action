"""Configuration composition, validation, and snapshot tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from screen2action.config import load_config, write_resolved_config


def test_config_composition_environment_override_and_snapshot(tmp_path: Path) -> None:
    parent = tmp_path / "parent.yaml"
    child = tmp_path / "child.yaml"
    parent.write_text(
        "schema_version: 1\ndevice: cpu\nd: 64\nmodel:\n  dropout: 0.1\n",
        encoding="utf-8",
    )
    child.write_text(
        "extends: parent.yaml\nmodel:\n  cache: ${MODEL_CACHE}\nretrieval_top_k: 4\n",
        encoding="utf-8",
    )

    config = load_config(
        child,
        overrides=("model.dropout=0.25", "retrieval_top_k=8"),
        environment={"MODEL_CACHE": "portable-cache"},
    )

    assert config.values["device"] == "cpu"
    assert config.values["retrieval_top_k"] == 8
    assert config.values["model"] == {"dropout": 0.25, "cache": "portable-cache"}
    assert config.source_paths == ("parent.yaml", "child.yaml")
    snapshot = write_resolved_config(config, tmp_path / "run")
    payload = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
    assert payload["config_sha256"] == config.sha256
    assert payload["resolved"]["model"]["dropout"] == 0.25


def test_config_rejects_cycles_unset_environment_and_bad_values(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("extends: second.yaml\nschema_version: 1\nname: first\n", encoding="utf-8")
    second.write_text("extends: first.yaml\nschema_version: 1\nname: second\n", encoding="utf-8")
    with pytest.raises(ValueError, match="composition cycle"):
        load_config(first)

    unset = tmp_path / "unset.yaml"
    unset.write_text("schema_version: 1\npath: ${MISSING}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="MISSING"):
        load_config(unset, environment={})

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("schema_version: 1\nd: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="positive integer"):
        load_config(invalid)
