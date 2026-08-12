"""Unified CLI discovery and dry-run tests."""

from __future__ import annotations

import json
from argparse import Namespace

import pytest

from screen2action.cli import REQUIRED_COMMAND_PATHS, _parser, _precompute_shard, main


def test_every_required_command_has_help() -> None:
    parser = _parser()
    for path in REQUIRED_COMMAND_PATHS:
        with pytest.raises(SystemExit) as raised:
            parser.parse_args([*path.split(), "--help"])
        assert raised.value.code == 0


def test_configured_command_dry_run_is_deterministic(capsys: object) -> None:
    arguments = ["evaluate", "--dry-run", "--json", "--set", "ssb_budgets=[256]"]

    assert main(arguments) == 0
    first = capsys.readouterr().out  # type: ignore[attr-defined]
    assert main(arguments) == 0
    second = capsys.readouterr().out  # type: ignore[attr-defined]

    assert first == second
    payload = json.loads(first)
    assert payload["command"] == "evaluate"
    assert payload["status"] == "dry_run"
    assert payload["config_schema_version"] == 1


def test_registry_listing_is_real_read_only_command(capsys: object) -> None:
    assert main(["models", "list", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert "ui_detector_screenparser" in payload["roles"]


def test_mobilevit_shape_probe_has_a_no_weight_dry_run(capsys: object) -> None:
    assert (
        main(
            [
                "models",
                "probe-mobilevit",
                "--weight",
                "missing.safetensors",
                "--batch-size",
                "2",
                "--dry-run",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert payload["command"] == "models probe-mobilevit"
    assert payload["status"] == "dry_run"


def test_precompute_uses_torchrun_rank_for_process_safe_sharding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "2")
    assert _precompute_shard(Namespace(shard_count=None, shard_index=None)) == (2, 4)

    with pytest.raises(ValueError, match="WORLD_SIZE"):
        _precompute_shard(Namespace(shard_count=3, shard_index=None))
