from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from screen2action.config import load_config
from screen2action.export.runner import PARTITION_NAMES, run_partitioned_export
from screen2action.training.model_factory import build_screen2action_model
from screen2action.training.semantics import build_semantics_graph_model


def _fixture_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    config = load_config(Path("configs/export/tiny_cpu.yaml"))
    bundle = build_screen2action_model(
        config.values,
        tokenizer_vocab_size=2048,
        model_lock_path=None,
        cache_root=tmp_path / "models",
    )
    checkpoint = tmp_path / "downstream.pt"
    torch.save(
        {
            "format_version": 3,
            "model": bundle.model.state_dict(),
            "stage": "stage3_joint",
            "epoch": 1,
            "step": 2,
            "manifest_hash": "d" * 64,
            "model_lock_hash": bundle.model_lock_digest,
            "cache_manifest_hash": "c" * 64,
            "git_revision": "fixture",
        },
        checkpoint,
    )
    semantics = build_semantics_graph_model(
        config.values,
        model_lock_path=None,
        cache_root=tmp_path / "models",
    )
    visual = tmp_path / "visual-model.pt"
    torch.save(
        {
            "format_version": 1,
            "model": semantics.model.visual.state_dict(),
            "source_stage": "stage1_semantics_graph",
            "model_lock_digest": semantics.model_lock_digest,
        },
        visual,
    )
    return checkpoint, visual


@pytest.mark.integration_model
def test_tiny_partition_package_exports_all_four_subgraphs_with_cpu_parity(
    tmp_path: Path,
) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    checkpoint, visual = _fixture_artifacts(tmp_path)
    output = tmp_path / "exports"

    result = run_partitioned_export(
        load_config(Path("configs/export/tiny_cpu.yaml")),
        checkpoint=checkpoint,
        output_directory=output,
        profile="tiny_cpu",
        model_lock=None,
        model_cache_root=tmp_path / "models",
        visual_checkpoint=visual,
    )

    assert result.ok
    assert result.status == "complete"
    assert result.exported_partitions == PARTITION_NAMES
    package = output / "tiny_cpu"
    report = json.loads((package / "partition-report.json").read_text(encoding="utf-8"))
    assert report["truth_class"] == "cpu_reconstruction_export"
    assert report["mobile_runtime_measured"] is False
    assert report["metrics_may_merge_with_other_profiles"] is False
    assert all(partition["parity"]["passed"] for partition in report["partitions"])
    assert all((package / f"{name}.onnx").is_file() for name in PARTITION_NAMES)
    assert (package / "host-runtime-contract.json").is_file()
    assert (package / "shape-contract.json").is_file()
    assert (package / "resolved-config.yaml").is_file()


def test_fast_profile_emits_separate_unsupported_report_without_accurate_substitution(
    tmp_path: Path,
) -> None:
    checkpoint, _ = _fixture_artifacts(tmp_path)
    output = tmp_path / "exports"

    result = run_partitioned_export(
        load_config(Path("configs/export/fast_roi.yaml")),
        checkpoint=checkpoint,
        output_directory=output,
        profile="fast_roi",
    )

    assert not result.ok
    assert result.status == "unsupported"
    package = output / "fast_roi"
    report = json.loads((package / "partition-report.json").read_text(encoding="utf-8"))
    assert report["artifact_namespace"] == "fast_roi"
    assert all(partition["status"] == "unsupported" for partition in report["partitions"])
    assert not list(package.glob("*.onnx"))
    assert "not substituted" in report["partitions"][0]["reason"]
