"""Command-independent real/oracle/cached perception integration tests."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn

from screen2action.data.schema import ElementAnnotation, NodeType, ScreenRecord
from screen2action.perception.base import (
    ImageFrame,
    OcrResult,
    RawDetection,
    RawDetectionFrame,
    UiDetection,
)
from screen2action.perception.cache import ContentAddressedPerceptionCache
from screen2action.perception.icon_actionability import MobileNetV3IconActionability
from screen2action.perception.pipeline import FullScreenPerception
from screen2action.perception.screenparser import ScreenParserConfig, ScreenParserUiDetector


class CountingDetectorBackend:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def predict(
        self,
        images: Sequence[ImageFrame],
        config: ScreenParserConfig,
    ) -> Sequence[RawDetectionFrame]:
        self.batch_sizes.append(len(images))
        detections = (
            RawDetection(2, "Button", 0.95, (10.0, 8.0, 70.0, 32.0)),
            RawDetection(49, "Text", 0.9, (20.0, 12.0, 55.0, 24.0)),
        )
        return tuple(RawDetectionFrame(detections, 4, 2, True) for _ in images)


class FixtureRecognizer:
    def recognize(
        self,
        image: ImageFrame,
        detections: Sequence[UiDetection],
    ) -> tuple[OcrResult, ...]:
        text = next(item for item in detections if item.original_class_name == "Text")
        return (
            OcrResult(
                request_id=f"{image.sha256}:{text.node_id}",
                node_id=text.node_id,
                text="  Open   Item ",
                confidence=0.88,
                screen_box_xyxy_norm=text.box_xyxy_norm,
                status="ready",
            ),
        )


def _visual_model() -> MobileNetV3IconActionability:
    return MobileNetV3IconActionability(
        nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, padding=1), nn.ReLU()),
        backbone_dimension=8,
    )


def test_real_perception_deduplicates_batches_and_reuses_cache(tmp_path: Path) -> None:
    backend = CountingDetectorBackend()
    cache = ContentAddressedPerceptionCache(tmp_path / "cache")
    service = FullScreenPerception(
        detector=ScreenParserUiDetector(backend),
        recognizer=FixtureRecognizer(),
        visual_model=_visual_model(),
        cache=cache,
        model_bundle_lock_digest="d" * 64,
    )
    image = torch.zeros((3, 40, 80), dtype=torch.uint8)

    first = service.perceive((image, image), mode="real")

    assert backend.batch_sizes == [1]
    assert len(first) == 2
    assert first[0].screenshot_sha256 == first[1].screenshot_sha256
    assert first[0].mode == "real"
    by_type = {node.node_type: node for node in first[0].nodes}
    assert by_type[NodeType.ROOT].mandatory
    assert by_type[NodeType.TEXT].text == "  Open   Item "
    assert by_type[NodeType.CONTROL].text == "open item"
    assert len(by_type[NodeType.CONTROL].icon_probabilities) == 87
    assert len(by_type[NodeType.CONTROL].roi_visual_feature) == 256
    assert by_type[NodeType.CONTROL].visual_feature_confidence == 1.0
    assert by_type[NodeType.TEXT].parent_id == by_type[NodeType.CONTROL].node_id
    assert cache.stats()["entries"] == 1

    second = service.perceive((image,), mode="real")
    assert backend.batch_sizes == [1]
    assert second[0].mode == "cached"

    cache_only = FullScreenPerception(
        cache=cache,
        model_bundle_lock_digest="d" * 64,
    )
    cached = cache_only.perceive((image,), mode="cached")[0]
    assert cached.mode == "cached"
    assert cached.nodes == second[0].nodes
    assert "command" not in inspect.signature(FullScreenPerception.perceive).parameters


def test_oracle_mode_builds_a_rooted_graph_without_real_models(tmp_path: Path) -> None:
    image = torch.zeros((3, 20, 30), dtype=torch.uint8)
    screen = ScreenRecord(
        screen_id="screen-1",
        app_id="fixture-app",
        image_path="fixture.png",
        width=30,
        height=20,
        split="train",
        source="fixture",
        elements=(
            ElementAnnotation(
                "button-1",
                NodeType.CONTROL,
                (0.1, 0.1, 0.8, 0.8),
                text="Open",
                actionability_labels=(True, False, False, False),
            ),
        ),
    )
    service = FullScreenPerception(
        cache=ContentAddressedPerceptionCache(tmp_path / "oracle-cache"),
        model_bundle_lock_digest="e" * 64,
    )

    frame = service.perceive((image,), mode="oracle", oracle_screens=(screen,))[0]

    assert frame.mode == "oracle"
    assert any(node.node_type is NodeType.ROOT for node in frame.nodes)
    assert frame.root_id != 0
    assert frame.diagnostics["source"] == "oracle_annotations"
