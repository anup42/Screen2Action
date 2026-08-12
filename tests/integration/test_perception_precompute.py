from __future__ import annotations

import base64
import importlib
import io
import json
import os
import socket
import sys
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

import pytest
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
from screen2action.perception.base import ImageFrame
from screen2action.perception.cache import ContentAddressedPerceptionCache, PerceptionCacheKey
from screen2action.perception.loader import CachedPerceptionLoader
from screen2action.perception.pipeline import (
    FullScreenPerceptionConfig,
    PerceptionFrame,
    perception_frame_to_payload,
)
from screen2action.perception.precompute import (
    PerceptionBundleSpec,
    bundle_cache,
    load_manifest_screens,
    perception_cache_stats,
    precompute_perception,
    validate_perception_cache,
)
from screen2action.perception.taxonomy import freeze_icon_taxonomy
from screen2action.training.runner import probe_batch_size, run_training_stage


def _png_base64(index: int) -> str:
    red = (index * 40) % 256
    image = Image.new("RGB", (4, 4), color=(red, 10, 255 - red))
    image.putpixel((index % 4, (index * 2) % 4), (255, 255, 255))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _manifest(tmp_path: Path, *, count: int = 4) -> Path:
    registry = load_source_registry(Path("configs/data/sources.yaml"))
    revision = registry.source("wave_ui").revision
    source = tmp_path / "source"
    source.mkdir()
    rows = [
        {
            "id": f"wave-{index}",
            "split": "train",
            "image_size": [4, 4],
            "image_bytes_base64": _png_base64(index),
            "instruction": f"open item {index}",
            "bbox": [0.1, 0.1, 0.9, 0.9],
            "source": "fixture",
            "source_license": "fixture-only",
            "platform": "web",
            "domain": f"app-{index}.example",
            "name": f"item {index}",
            "type": "button",
            "clickable": True,
            "long_clickable": True,
            "iconClass": f"fixture_icon_{index}",
            "icon_class_id": index,
        }
        for index in range(count)
    ]
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
    normalize_registered_source(registry, layout, "wave_ui", revision, "cache-fixture")
    dataset = layout.normalized("cache-fixture")
    split_path = dataset / "audits" / "split.json"
    write_split_plan(
        create_app_disjoint_plan(read_table_rows(dataset, "screens"), seed=5),
        split_path,
    )
    dedup_path = dataset / "audits" / "dedup.json"
    deduplicate_canonical_dataset(
        dataset,
        load_split_assignments(split_path),
        dedup_path,
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
    manifest = dataset / "manifests" / "fixture.json"
    build_data_manifest(
        dataset,
        manifest,
        split_plan=split_path,
        dedup_audit=dedup_path,
        taxonomy=taxonomy,
    )
    return manifest


class RetryFixturePerception:
    def __init__(self, cache: ContentAddressedPerceptionCache, fail_digest: str) -> None:
        self.cache = cache
        self.model_bundle_lock_digest = "a" * 64
        self.config = FullScreenPerceptionConfig(visual_checkpoint_sha256="b" * 64)
        self.fail_digest = fail_digest
        self.failure_calls = 0
        self.perceived: list[str] = []

    def key_for(self, image: ImageFrame) -> PerceptionCacheKey:
        return PerceptionCacheKey(
            image.sha256,
            self.model_bundle_lock_digest,
            self.config.digest,
        )

    def perceive(
        self,
        images: Sequence[object],
        *,
        mode: str = "real",
    ) -> tuple[PerceptionFrame, ...]:
        assert mode == "real"
        normalized = tuple(image for image in images if isinstance(image, ImageFrame))
        assert len(normalized) == len(images)
        if any(image.sha256 == self.fail_digest for image in normalized) and self.failure_calls < 2:
            self.failure_calls += 1
            raise RuntimeError("transient fixture inference failure")
        frames = []
        for image in normalized:
            key = self.key_for(image)
            self.perceived.append(image.sha256)
            root = NodeRecord(
                node_id=0,
                node_type=NodeType.ROOT,
                box_xyxy_norm=(0.0, 0.0, 1.0, 1.0),
                mandatory=True,
                retention_score=1.0,
                actionability_mask=(False, False, False, False),
                annotation_source="fixture",
            )
            control = NodeRecord(
                node_id=1,
                node_type=NodeType.CONTROL,
                box_xyxy_norm=(0.1, 0.1, 0.9, 0.9),
                detector_confidence=1.0,
                text="item",
                actionability_mask=(True, False, False, True),
                annotation_source="fixture",
            )
            frame = PerceptionFrame(
                screenshot_sha256=image.sha256,
                width=image.width,
                height=image.height,
                nodes=(root, control),
                edges=(),
                root_id=0,
                mode="real",
                diagnostics={"fixture": True},
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


def _tiny_stage2_config(*, epochs: int = 2) -> ResolvedConfig:
    return ResolvedConfig(
        values={
            "schema_version": 1,
            "name": "tiny_stage2_resume_fixture",
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
                "top_k": 1,
                "ssb_budget": 64,
                "crop_size": 16,
                "crop_tokens": 4,
            },
            "training": {
                "epochs": epochs,
                "global_batch_size": 1,
                "per_device_batch_size": 1,
                "seed": 23,
                "tensorboard": False,
                "checkpoint_interval_optimizer_steps": 1,
                "validation_interval_epochs": 1,
                "log_interval": 1,
            },
        },
        source_paths=("fixture",),
        overrides=(),
    )


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _stage_runner_worker(
    rank: int,
    world_size: int,
    port: int,
    manifest: str,
    cache_manifest: str,
    run_directory: str,
) -> None:
    os.environ.update(
        {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
        }
    )
    result = run_training_stage(
        _tiny_stage2_config(epochs=1),
        stage="stage2_selector_retrieval",
        manifest=Path(manifest),
        cache_manifest=Path(cache_manifest),
        run_directory=Path(run_directory),
        device="cpu",
    )
    Path(run_directory, f"rank-{rank}.json").write_text(
        json.dumps({"status": result.status, "step": result.global_step}),
        encoding="utf-8",
    )


def test_precompute_is_sharded_resumable_validated_and_cache_only(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    screen_set = load_manifest_screens(manifest)
    provisional_config = FullScreenPerceptionConfig(visual_checkpoint_sha256="b" * 64)
    spec = PerceptionBundleSpec(
        model_bundle_lock_digest="a" * 64,
        perception_config_digest=provisional_config.digest,
        visual_checkpoint_sha256="b" * 64,
        include_detector_roi_features=False,
    )
    cache = bundle_cache(tmp_path / "cache", spec)
    fail_image = screen_set.screens[0]
    fail_digest = (
        importlib.import_module("screen2action.perception.base")
        .normalize_image(fail_image.image_path)
        .sha256
    )
    service = RetryFixturePerception(cache, fail_digest)

    first = precompute_perception(
        manifest,
        service=service,
        cache=cache,
        shard_index=0,
        shard_count=2,
        batch_size=2,
        max_retries=3,
    )
    second = precompute_perception(
        manifest,
        service=service,
        cache=cache,
        shard_index=1,
        shard_count=2,
        batch_size=2,
        max_retries=3,
    )
    assert first.status == second.status == "complete"
    assert first.assigned + second.assigned == len(screen_set.screens)
    assert first.completed + second.completed == len(screen_set.screens)
    perceived_before_resume = len(service.perceived)

    resumed = precompute_perception(
        manifest,
        service=service,
        cache=cache,
        shard_index=0,
        shard_count=2,
        batch_size=2,
        max_retries=3,
    )
    assert resumed.status == "already_complete"
    assert len(service.perceived) == perceived_before_resume
    assert not tuple(cache.root.rglob("*.tmp"))

    validation = validate_perception_cache(cache.root / "manifest.json")
    assert validation.ok
    assert validation.entries == len(screen_set.screens)
    stats = perception_cache_stats(cache.root / "manifest.json")
    assert stats["entries"] == len(screen_set.screens)
    loader = CachedPerceptionLoader(
        cache.root / "manifest.json",
        data_manifest_digest=screen_set.manifest_digest,
        expected_model_lock_digest="a" * 64,
        expected_config_digest=provisional_config.digest,
    )
    assert len(loader) == len(screen_set.screens)
    frame = loader.frame_for_screen(loader.screen_ids[0])
    assert frame.raw_outputs["command_conditioned"] is False

    sys.modules.pop("screen2action.perception.screenparser", None)
    sys.modules.pop("screen2action.perception.doctr_crnn", None)
    import screen2action.perception.loader as cache_loader

    importlib.reload(cache_loader)
    assert "screen2action.perception.screenparser" not in sys.modules
    assert "screen2action.perception.doctr_crnn" not in sys.modules


def test_cache_validation_detects_entry_corruption(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    config = FullScreenPerceptionConfig(visual_checkpoint_sha256="b" * 64)
    spec = PerceptionBundleSpec("a" * 64, config.digest, "b" * 64, False)
    cache = bundle_cache(tmp_path / "cache", spec)
    service = RetryFixturePerception(cache, fail_digest="f" * 64)
    result = precompute_perception(manifest, service=service, cache=cache)
    assert result.status == "complete"
    entry = next((cache.root / "entries").rglob("*.json"))
    entry.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="compatibility mismatch"):
        validate_perception_cache(cache.root / "manifest.json")


def test_stage2_runner_exact_resume_matches_uninterrupted_cpu(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, count=8)
    config = FullScreenPerceptionConfig(visual_checkpoint_sha256="b" * 64)
    spec = PerceptionBundleSpec("a" * 64, config.digest, "b" * 64, False)
    cache = bundle_cache(tmp_path / "cache", spec)
    service = RetryFixturePerception(cache, fail_digest="f" * 64)
    precompute_perception(manifest, service=service, cache=cache)
    cache_manifest = cache.root / "manifest.json"
    resolved = _tiny_stage2_config()
    resumed_directory = tmp_path / "resumed-run"
    first = run_training_stage(
        resolved,
        stage="stage2_selector_retrieval",
        manifest=manifest,
        cache_manifest=cache_manifest,
        run_directory=resumed_directory,
        stop_after_optimizer_steps=1,
    )
    assert first.status == "stopped"
    assert first.latest_checkpoint is not None
    first_state = torch.load(Path(first.latest_checkpoint), map_location="cpu", weights_only=False)
    assert first_state["epoch"] == 0
    assert first_state["next_batch_index"] > 0
    resumed = run_training_stage(
        resolved,
        stage="stage2_selector_retrieval",
        manifest=manifest,
        cache_manifest=cache_manifest,
        run_directory=resumed_directory,
        resume=Path(first.latest_checkpoint),
    )
    uninterrupted = run_training_stage(
        resolved,
        stage="stage2_selector_retrieval",
        manifest=manifest,
        cache_manifest=cache_manifest,
        run_directory=tmp_path / "uninterrupted-run",
    )
    assert resumed.status == uninterrupted.status == "complete"
    resumed_state = torch.load(
        Path(resumed.latest_checkpoint), map_location="cpu", weights_only=False
    )
    uninterrupted_state = torch.load(
        Path(uninterrupted.latest_checkpoint), map_location="cpu", weights_only=False
    )
    assert resumed_state["step"] == uninterrupted_state["step"]
    assert resumed_state["optimizer_step"] == uninterrupted_state["optimizer_step"]
    for name, value in resumed_state["model"].items():
        assert torch.equal(value, uninterrupted_state["model"][name]), name


def test_probe_batch_reports_largest_observed_cpu_batch(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    perception_config = FullScreenPerceptionConfig(visual_checkpoint_sha256="b" * 64)
    spec = PerceptionBundleSpec("a" * 64, perception_config.digest, "b" * 64, False)
    cache = bundle_cache(tmp_path / "cache", spec)
    precompute_perception(
        manifest,
        service=RetryFixturePerception(cache, fail_digest="f" * 64),
        cache=cache,
    )
    result = probe_batch_size(
        _tiny_stage2_config(epochs=1),
        stage="stage2_selector_retrieval",
        manifest=manifest,
        cache_manifest=cache.root / "manifest.json",
        maximum_batch_size=2,
    )
    assert result.largest_successful_per_device_batch == 2
    assert result.peak_vram_bytes == 0
    assert result.recommended_accumulation_for_global_256 == 128


def test_two_process_cpu_ddp_runs_real_stage_runner(tmp_path: Path) -> None:
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("Torch Gloo distributed support is unavailable")
    manifest = _manifest(tmp_path)
    perception_config = FullScreenPerceptionConfig(visual_checkpoint_sha256="b" * 64)
    spec = PerceptionBundleSpec("a" * 64, perception_config.digest, "b" * 64, False)
    cache = bundle_cache(tmp_path / "cache", spec)
    precompute_perception(
        manifest,
        service=RetryFixturePerception(cache, fail_digest="f" * 64),
        cache=cache,
    )
    run_directory = tmp_path / "ddp-run"
    torch.multiprocessing.spawn(
        _stage_runner_worker,
        args=(
            2,
            _free_port(),
            str(manifest),
            str(cache.root / "manifest.json"),
            str(run_directory),
        ),
        nprocs=2,
        join=True,
    )
    rows = [
        json.loads((run_directory / f"rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(2)
    ]
    assert rows[0] == rows[1] == {"status": "complete", "step": 1}
    assert (run_directory / "checkpoints" / "latest.pt").is_file()


def test_stage1_semantics_and_stage3_joint_are_runnable_on_frozen_manifest(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    perception_config = FullScreenPerceptionConfig(visual_checkpoint_sha256="b" * 64)
    spec = PerceptionBundleSpec("a" * 64, perception_config.digest, "b" * 64, False)
    cache = bundle_cache(tmp_path / "cache", spec)
    precompute_perception(
        manifest,
        service=RetryFixturePerception(cache, fail_digest="f" * 64),
        cache=cache,
    )
    config = _tiny_stage2_config(epochs=1)
    stage1 = run_training_stage(
        config,
        stage="stage1_semantics_graph",
        manifest=manifest,
        run_directory=tmp_path / "stage1",
    )
    stage3 = run_training_stage(
        config,
        stage="stage3_joint",
        manifest=manifest,
        cache_manifest=cache.root / "manifest.json",
        run_directory=tmp_path / "stage3",
    )

    assert stage1.status == stage3.status == "complete"
    assert stage1.latest_checkpoint is not None
    assert (tmp_path / "stage1" / "checkpoints" / "visual-model.pt").is_file()
    assert stage3.latest_checkpoint is not None
    state = torch.load(Path(stage3.latest_checkpoint), map_location="cpu", weights_only=False)
    assert any(name.startswith("downstream.") for name in state["model"])
    assert any(name.startswith("semantics.") for name in state["model"])
    events = [
        json.loads(line)
        for line in (tmp_path / "stage3" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    epoch = next(event for event in events if event["event"] == "train_epoch")
    assert "stage3_semantic_perception_loss" in epoch
    assert "confidence_loss" in epoch

    stage4_values = deepcopy(dict(config.values))
    stage4_values["training"]["epochs"] = 1
    stage4_values["quantization"] = {
        "mode": "qat",
        "calibration_batches": 1,
        "per_channel_weights": True,
    }
    stage4_config = ResolvedConfig(
        stage4_values,
        source_paths=("fixture-stage4",),
        overrides=(),
    )
    stage4 = run_training_stage(
        stage4_config,
        stage="stage4_qat",
        manifest=manifest,
        cache_manifest=cache.root / "manifest.json",
        run_directory=tmp_path / "stage4",
        init_checkpoint=Path(stage3.latest_checkpoint),
    )
    assert stage4.status == "complete"
    quantization = json.loads(
        (tmp_path / "stage4" / "quantization-report.json").read_text(encoding="utf-8")
    )
    assert quantization["calibration_split"] == "train"
    assert quantization["comparison"]["same_checkpoint_and_validation_data"] is True
    assert set(quantization["comparison"]) >= {
        "fp32",
        "ptq_weight_only",
        "qat_fake_quant",
    }
    assert quantization["mobile_runtime_measured"] is False
