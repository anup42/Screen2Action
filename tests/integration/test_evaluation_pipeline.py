from __future__ import annotations

import base64
import io
import json
from copy import deepcopy
from pathlib import Path
from typing import cast

import torch
import yaml
from PIL import Image

from screen2action.config import ResolvedConfig
from screen2action.data.assets import register_local_source
from screen2action.data.dedup import deduplicate_canonical_dataset
from screen2action.data.layout import DataLayout
from screen2action.data.manifest import build_data_manifest
from screen2action.data.registry import load_source_registry
from screen2action.data.schema import NodeRecord, NodeType
from screen2action.data.splits import (
    create_app_disjoint_plan,
    load_split_assignments,
    write_split_plan,
)
from screen2action.data.storage import normalize_registered_source, read_table_rows
from screen2action.eval.reporting import fit_validation_calibration_artifacts
from screen2action.eval.runner import run_evaluation
from screen2action.eval.sweep_runner import run_evaluation_sweep
from screen2action.perception.base import ImageFrame
from screen2action.perception.cache import ContentAddressedPerceptionCache, PerceptionCacheKey
from screen2action.perception.pipeline import (
    FullScreenPerceptionConfig,
    PerceptionFrame,
    perception_frame_to_payload,
)
from screen2action.perception.precompute import (
    PerceptionBundleSpec,
    bundle_cache,
    precompute_perception,
)
from screen2action.perception.taxonomy import freeze_icon_taxonomy
from screen2action.training.checkpoints import manifest_hash
from screen2action.training.data import load_training_corpus
from screen2action.training.model_factory import build_screen2action_model


def _image_base64() -> str:
    image = Image.new("RGB", (100, 50), color=(20, 30, 40))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _screenspot_manifest(tmp_path: Path) -> Path:
    registry = load_source_registry(Path("configs/data/sources.yaml"))
    revision = registry.source("screenspot").revision
    source = tmp_path / "s"
    source.mkdir()
    row = {
        "id": "screenspot-fixture",
        "img_filename": "desktop/example.png",
        "image_bytes_base64": _image_base64(),
        "instruction": "click search",
        "bbox": [20, 10, 80, 40],
        "image_size": {"width": 100, "height": 50},
        "data_type": "icon",
        "data_souce": "Windows",
        "app_id": "example-browser",
        "split": "train",
    }
    (source / "screenspot.json").write_text(json.dumps([row]), encoding="utf-8")
    layout = DataLayout(tmp_path / "l")
    register_local_source(
        registry,
        layout,
        "screenspot",
        source,
        revision,
        accept_license=True,
    )
    normalize_registered_source(registry, layout, "screenspot", revision, "e")
    dataset = layout.normalized("e")
    split = dataset / "audits" / "split.json"
    write_split_plan(create_app_disjoint_plan(read_table_rows(dataset, "screens")), split)
    dedup = dataset / "audits" / "dedup.json"
    deduplicate_canonical_dataset(
        dataset,
        load_split_assignments(split),
        dedup,
    )
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
        yaml.safe_dump(
            freeze_icon_taxonomy(proposal, accept_reconstruction=True),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    manifest = dataset / "manifests" / "screenspot.json"
    build_data_manifest(
        dataset,
        manifest,
        split_plan=split,
        dedup_audit=dedup,
        taxonomy=taxonomy,
    )
    return manifest


class _ScreenSpotPerception:
    def __init__(self, cache: ContentAddressedPerceptionCache) -> None:
        self.cache = cache
        self.model_bundle_lock_digest = "a" * 64
        self.config = FullScreenPerceptionConfig(visual_checkpoint_sha256="b" * 64)

    def key_for(self, image: ImageFrame) -> PerceptionCacheKey:
        return PerceptionCacheKey(
            image.sha256,
            self.model_bundle_lock_digest,
            self.config.digest,
        )

    def perceive(
        self,
        images: list[object] | tuple[object, ...],
        *,
        mode: str = "real",
    ) -> tuple[PerceptionFrame, ...]:
        assert mode == "real"
        frames = []
        for raw in images:
            image = cast(ImageFrame, raw)
            key = self.key_for(image)
            root = NodeRecord(
                0,
                NodeType.ROOT,
                (0.0, 0.0, 1.0, 1.0),
                mandatory=True,
                actionability_mask=(False, False, False, False),
                annotation_source="fixture",
            )
            target = NodeRecord(
                1,
                NodeType.ICON,
                (0.2, 0.2, 0.8, 0.8),
                detector_confidence=1.0,
                actionability_logits=(10.0, -10.0, -10.0, -10.0),
                annotation_source="fixture",
            )
            frame = PerceptionFrame(
                image.sha256,
                image.width,
                image.height,
                (root, target),
                (),
                0,
                "real",
                {"fixture": True},
                cache_key_digest=key.digest,
                raw_outputs={
                    "detector": {"items": []},
                    "ocr": {"items": []},
                    "icon_actionability": {"items": []},
                    "command_conditioned": False,
                },
            )
            self.cache.put(key, perception_frame_to_payload(frame))
            frames.append(frame)
        return tuple(frames)


def _config() -> ResolvedConfig:
    return ResolvedConfig(
        {
            "schema_version": 1,
            "name": "screenspot_eval_fixture",
            "model": {
                "profile": "tiny_cpu",
                "embedding_dim": 16,
                "vocab_size": 64,
                "graph_layers": 1,
                "graph_heads": 4,
                "command_layers": 1,
                "command_heads": 4,
                "command_ffn": 32,
                "command_max_length": 8,
                "node_text_length": 4,
                "top_k": 8,
                "ssb_budget": 512,
                "crop_size": 16,
                "crop_tokens": 4,
            },
            "evaluation": {
                "source": "screenspot",
                "split": "test",
                "batch_size": 1,
                "budgets": [512],
                "retrieval_top_k": [8],
            },
        },
        source_paths=("fixture",),
        overrides=(),
    )


def test_eval_reports_and_calibration_guard(
    tmp_path: Path,
) -> None:
    manifest = _screenspot_manifest(tmp_path)
    perception_config = FullScreenPerceptionConfig(visual_checkpoint_sha256="b" * 64)
    spec = PerceptionBundleSpec("a" * 64, perception_config.digest, "b" * 64, False)
    cache = bundle_cache(tmp_path / "cache", spec)
    precompute_perception(
        manifest,
        service=_ScreenSpotPerception(cache),
        cache=cache,
    )
    corpus = load_training_corpus(manifest)
    config = _config()
    bundle = build_screen2action_model(
        cast(dict[str, object], config.values),
        tokenizer_vocab_size=corpus.tokenizer.vocab_size,
        model_lock_path=None,
        cache_root=tmp_path / "models",
    )
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model": bundle.model.state_dict(),
            "manifest_hash": corpus.manifest_digest,
            "model_lock_hash": bundle.model_lock_digest,
            "cache_manifest_hash": manifest_hash(cache.root / "manifest.json"),
        },
        checkpoint,
    )
    result = run_evaluation(
        config,
        checkpoint=checkpoint,
        manifest=manifest,
        cache_manifest=cache.root / "manifest.json",
        output_directory=tmp_path / "evaluation",
    )

    assert result.status == "complete"
    assert result.metrics["examples"] == 1
    assert result.metrics["aggregation_policy"].endswith("no_independence_multiplication_v1")
    assert result.metrics["subsets"]["platform"]["Windows"]["examples"] == 1
    assert result.metrics["subsets"]["target_type"]["icon"]["examples"] == 1
    for name in (
        "metrics.json",
        "predictions.csv",
        "predictions.parquet",
        "report.md",
        "provenance.json",
        "resolved-config.json",
    ):
        assert (tmp_path / "evaluation" / name).is_file()
    try:
        fit_validation_calibration_artifacts(
            tmp_path / "evaluation" / "predictions.csv",
            tmp_path / "calibration",
        )
    except ValueError as error:
        assert "split=val" in str(error)
    else:
        raise AssertionError("ScreenSpot test predictions must not calibrate confidence")

    sweep_values = deepcopy(dict(config.values))
    sweep_values["evaluation"].update(
        {
            "mode": "sweep",
            "budgets": [512],
            "retrieval_top_k": [8],
            "ablations": [
                "learned_selector",
                "no_target_survival_loss",
                "roi_align_fast",
            ],
        }
    )
    sweep = run_evaluation_sweep(
        ResolvedConfig(sweep_values, source_paths=("sweep-fixture",), overrides=()),
        checkpoint=checkpoint,
        manifest=manifest,
        cache_manifest=cache.root / "manifest.json",
        output_directory=tmp_path / "sweep",
    )
    assert sweep["complete"] == 2
    assert sweep["requires_checkpoint"] == 1
    assert sweep["unsupported_optional"] == 1
    assert (tmp_path / "sweep" / "sweep.md").is_file()
