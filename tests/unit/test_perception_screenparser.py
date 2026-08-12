"""Normalized image boundary and ScreenParser adapter tests."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
import torch

from screen2action.data.schema import NodeType
from screen2action.perception.base import ImageFrame, RawDetection, RawDetectionFrame
from screen2action.perception.screenparser import (
    SCREENPARSER_CLASS_MAPPING_VERSION,
    ContainerAugmentationPolicy,
    ScreenParserConfig,
    ScreenParserUiDetector,
)


class FakeScreenParserBackend:
    def __init__(self, detections: tuple[RawDetection, ...]) -> None:
        self.detections = detections
        self.batches: list[tuple[ImageFrame, ...]] = []

    def predict(
        self,
        images: Sequence[ImageFrame],
        config: ScreenParserConfig,
    ) -> Sequence[RawDetectionFrame]:
        assert config.device == "cpu"
        self.batches.append(tuple(images))
        return tuple(
            RawDetectionFrame(
                detections=self.detections,
                raw_proposal_count=len(self.detections) + 3,
                post_nms_count=len(self.detections),
                raw_diagnostics_supported=True,
            )
            for _ in images
        )


def test_screenparser_normalizes_all_input_kinds_and_preserves_provenance(
    tmp_path: Path,
) -> None:
    np = pytest.importorskip("numpy")
    image_module = pytest.importorskip("PIL.Image")
    array = np.zeros((3, 7, 3), dtype=np.uint8)
    array[:, :, 1] = 127
    pil = image_module.fromarray(array)
    image_path = tmp_path / "screen.png"
    pil.save(image_path)
    tensor = torch.zeros((3, 3, 7), dtype=torch.uint8)
    backend = FakeScreenParserBackend(
        (
            RawDetection(2, "Button", 0.9, (1.0, 1.0, 6.0, 3.0), (0.25, 0.75)),
            RawDetection(49, "Text", 0.8, (2.0, 1.0, 5.0, 2.0)),
        )
    )
    detector = ScreenParserUiDetector(backend, config=ScreenParserConfig(input_size=32))

    frames = detector.detect((tensor, pil, array, image_path))

    assert [frame.image.source_kind for frame in frames] == ["tensor", "pil", "numpy", "path"]
    assert {(frame.image.width, frame.image.height) for frame in frames} == {(7, 3)}
    assert len(backend.batches) == 1 and len(backend.batches[0]) == 4
    for frame in frames:
        root, button, text = frame.detections
        assert root.node_type is NodeType.ROOT
        assert root.box_xyxy_norm == (0.0, 0.0, 1.0, 1.0)
        assert button.node_type is NodeType.CONTROL
        assert button.original_class_id == 2
        assert button.original_class_feature == (0.25, 0.75)
        assert button.box_xyxy_norm == pytest.approx((1 / 7, 1 / 3, 6 / 7, 1.0))
        assert button.metadata["hierarchy_reliability"] == "leaf_detection_only"
        assert text.node_type is NodeType.TEXT
        assert "text" in text.semantic_roles
        assert text.preprocessing.original_width == 7
        assert text.preprocessing.original_height == 3
        assert frame.raw_proposal_count == 5
        assert frame.post_nms_count == 2
        assert frame.raw_diagnostics_supported
        assert frame.class_mapping_version == SCREENPARSER_CLASS_MAPPING_VERSION


def test_container_augmentation_is_explicit_ablatable_and_auditable() -> None:
    backend = FakeScreenParserBackend(
        tuple(
            RawDetection(2, "Button", 0.95, (10.0, float(top), 40.0, float(top + 8)))
            for top in (5, 20, 35)
        )
    )
    image = torch.zeros((3, 50, 100), dtype=torch.uint8)
    disabled = ScreenParserUiDetector(backend).detect((image,))[0]
    assert len(disabled.detections) == 4

    enabled = ScreenParserUiDetector(
        backend,
        container_policy=ContainerAugmentationPolicy(enabled=True),
    ).detect((image,))[0]
    generated = enabled.detections[-1]
    assert generated.node_type is NodeType.CONTAINER
    assert generated.annotation_source == "reconstruction_policy"
    assert generated.policy_version == "aligned_children_container_v1"
    assert generated.annotation_confidence == pytest.approx(0.65)
    assert generated.metadata["child_node_ids"] == "1,2,3"


def test_screenparser_rejects_a_class_contract_mismatch() -> None:
    backend = FakeScreenParserBackend((RawDetection(2, "Not Button", 0.9, (0, 0, 2, 2)),))
    detector = ScreenParserUiDetector(backend)

    with pytest.raises(ValueError, match="class-name mismatch"):
        detector.detect((torch.zeros((3, 4, 4), dtype=torch.uint8),))
