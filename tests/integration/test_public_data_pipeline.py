from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from screen2action.cli import main
from screen2action.data.assets import register_local_source
from screen2action.data.dedup import deduplicate_canonical_dataset
from screen2action.data.layout import DataLayout
from screen2action.data.manifest import build_data_manifest, verify_data_manifest
from screen2action.data.references import annotate_public_weak_references
from screen2action.data.registry import load_source_registry
from screen2action.data.shards import build_webdataset_shards
from screen2action.data.splits import (
    create_app_disjoint_plan,
    load_split_assignments,
    write_split_plan,
)
from screen2action.data.storage import (
    normalize_registered_source,
    read_table_rows,
    validate_canonical_dataset,
)
from screen2action.perception.taxonomy import freeze_icon_taxonomy


def test_wave_ui_fixture_runs_through_frozen_manifest_and_shards(tmp_path: Path) -> None:
    registry = load_source_registry(Path("configs/data/sources.yaml"))
    revision = registry.source("wave_ui").revision
    source = tmp_path / "wave-source"
    source.mkdir()
    fixture = Path(__file__).parents[1] / "fixtures" / "data" / "wave_ui.json"
    rows = json.loads(fixture.read_text(encoding="utf-8"))
    (source / "sample.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    layout = DataLayout(tmp_path / "lake")
    register_local_source(
        registry,
        layout,
        "wave_ui",
        source,
        revision,
        accept_license=True,
    )
    result = normalize_registered_source(
        registry,
        layout,
        "wave_ui",
        revision,
        "fixture-v1",
    )
    assert (result.screens, result.commands, result.rejected) == (1, 1, 1)
    dataset = layout.normalized("fixture-v1")
    assert validate_canonical_dataset(dataset)["ok"]

    split_path = dataset / "audits" / "splits.json"
    plan = create_app_disjoint_plan(read_table_rows(dataset, "screens"), seed=3)
    write_split_plan(plan, split_path)
    dedup_path = dataset / "audits" / "dedup.json"
    deduplicate_canonical_dataset(
        dataset,
        load_split_assignments(split_path),
        dedup_path,
    )
    reference_path = dataset / "references" / "part-public-weak-v1.parquet"
    annotate_public_weak_references(dataset, reference_path)

    proposal = {
        "status": "ready_for_freeze",
        "classes": [
            {"proposed_id": index, "name": f"icon_{index:02d}", "source_aliases": []}
            for index in range(87)
        ],
        "unknown_policy": "masked",
    }
    frozen = freeze_icon_taxonomy(proposal, accept_reconstruction=True)
    taxonomy_path = tmp_path / "icon-taxonomy.yaml"
    taxonomy_path.write_text(yaml.safe_dump(frozen, sort_keys=False), encoding="utf-8")

    manifest_path = dataset / "manifests" / "fixture.json"
    manifest = build_data_manifest(
        dataset,
        manifest_path,
        split_plan=split_path,
        dedup_audit=dedup_path,
        taxonomy=taxonomy_path,
    )
    assert manifest["leakage_checks"]["screenspot_outside_test"] == 0
    verification = verify_data_manifest(manifest_path)
    assert verification.ok
    shard_result = build_webdataset_shards(manifest_path, tmp_path / "shards", max_samples=1)
    assert shard_result["screen_count"] == 1

    original = next((dataset / "commands").rglob("*.parquet"))
    extra = original.with_name("unfrozen.parquet")
    extra.write_bytes(original.read_bytes())
    with pytest.raises(ValueError, match="unfrozen metadata"):
        verify_data_manifest(manifest_path)
    extra.unlink()

    image = next((dataset / "images").rglob("*.png"))
    image.write_bytes(image.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="manifest file digest mismatch"):
        verify_data_manifest(manifest_path)


def test_cli_executes_fixture_data_workflow(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = load_source_registry(Path("configs/data/sources.yaml"))
    revision = registry.source("wave_ui").revision
    source = tmp_path / "source"
    source.mkdir()
    rows = json.loads(
        (Path(__file__).parents[1] / "fixtures" / "data" / "wave_ui.json").read_text(
            encoding="utf-8"
        )
    )
    (source / "sample.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    data_root = tmp_path / "lake"
    assert (
        main(
            [
                "data",
                "register-local",
                "wave_ui",
                str(source),
                "--revision",
                revision,
                "--accept-license",
                "--data-root",
                str(data_root),
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            [
                "data",
                "normalize",
                "wave_ui",
                "--revision",
                revision,
                "--dataset-version",
                "cli-fixture",
                "--data-root",
                str(data_root),
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()
    dataset = data_root / "normalized" / "cli-fixture"
    split = dataset / "audits" / "split.json"
    dedup = dataset / "audits" / "dedup.json"
    references = dataset / "references" / "part-public-weak-v1.parquet"
    manifest = dataset / "manifests" / "cli-fixture.json"
    shards = tmp_path / "shards"
    assert main(["data", "validate", str(dataset), "--json"]) == 0
    capsys.readouterr()
    assert main(["data", "split", "--dataset", str(dataset), "--output", str(split), "--json"]) == 0
    capsys.readouterr()
    assert (
        main(
            [
                "data",
                "dedup",
                "--dataset",
                str(dataset),
                "--split-plan",
                str(split),
                "--output",
                str(dedup),
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            [
                "data",
                "annotate-references",
                "--dataset",
                str(dataset),
                "--output",
                str(references),
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()
    proposal = {
        "status": "ready_for_freeze",
        "classes": [
            {"proposed_id": index, "name": f"icon_{index:02d}", "source_aliases": []}
            for index in range(87)
        ],
        "unknown_policy": "masked",
    }
    taxonomy = tmp_path / "taxonomy.yaml"
    taxonomy.write_text(
        yaml.safe_dump(freeze_icon_taxonomy(proposal, accept_reconstruction=True), sort_keys=False),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "data",
                "build-manifest",
                "--dataset",
                str(dataset),
                "--output",
                str(manifest),
                "--split-plan",
                str(split),
                "--dedup-audit",
                str(dedup),
                "--taxonomy",
                str(taxonomy),
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            [
                "data",
                "build-shards",
                "--manifest",
                str(manifest),
                "--output",
                str(shards),
                "--max-samples",
                "1",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["screen_count"] == 1
