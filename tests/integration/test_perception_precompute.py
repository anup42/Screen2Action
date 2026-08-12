from __future__ import annotations

import base64
import importlib
import io
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml
from PIL import Image

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


def _png_base64(index: int) -> str:
    image = Image.new("RGB", (4, 4), color=(index * 40, 10, 255 - index * 40))
    image.putpixel((index % 4, (index * 2) % 4), (255, 255, 255))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _manifest(tmp_path: Path) -> Path:
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
        }
        for index in range(4)
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
            frame = PerceptionFrame(
                screenshot_sha256=image.sha256,
                width=image.width,
                height=image.height,
                nodes=(root,),
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
