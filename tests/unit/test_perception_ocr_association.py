"""docTR crop policy and deterministic OCR association tests."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

from screen2action.data.schema import NodeRecord, NodeType
from screen2action.perception.association import associate_ocr
from screen2action.perception.base import (
    OcrResult,
    PreprocessingMetadata,
    UiDetection,
    normalize_image,
)
from screen2action.perception.doctr_crnn import (
    OCR_CROP_POLICY_VERSION,
    DoctrCrnnVgg16Recognizer,
    RawRecognition,
)

_PREPROCESSING = PreprocessingMetadata(200, 32, 224, 1.0, 0.0, 0.0, 200, 32, "test-v1")


def _detection(node_id: int, name: str, box: tuple[float, float, float, float]) -> UiDetection:
    return UiDetection(
        node_id=node_id,
        original_class_id=49 if name == "Text" else 2,
        original_class_name=name,
        confidence=0.9,
        box_xyxy_norm=box,
        node_type=NodeType.TEXT if name == "Text" else NodeType.CONTROL,
        semantic_roles=("text",),
        original_class_feature=(),
        annotation_source="fixture",
        policy_version="fixture-v1",
        annotation_confidence=0.9,
        preprocessing=_PREPROCESSING,
        metadata={},
    )


class PixelAwareRecognitionBackend:
    vocabulary = "greenwide"

    def __init__(self) -> None:
        self.batch_widths: list[list[int]] = []

    def recognize(self, crops: Sequence[torch.Tensor]) -> Sequence[RawRecognition]:
        self.batch_widths.append([int(crop.shape[-1]) for crop in crops])
        output = []
        for crop in crops:
            channel_means = crop.float().mean(dim=(1, 2))
            text = "wideΩ" if channel_means[0] > channel_means[1] else "green"
            output.append(RawRecognition(text, 0.9, ("fixture-token",)))
        return tuple(output)


def test_doctr_batches_by_aspect_without_reordering_and_reports_edge_cases() -> None:
    pixels = torch.zeros((3, 32, 1000), dtype=torch.uint8)
    pixels[0, :, :800] = 255
    pixels[1, :, 900:950] = 255
    image = normalize_image(pixels)
    detections = (
        _detection(8, "Text", (0.0, 0.0, 0.8, 1.0)),
        _detection(3, "Text", (0.9, 0.0, 0.95, 1.0)),
        _detection(9, "Text", (0.5, 0.5, 0.5, 0.7)),
        _detection(10, "Button", (0.0, 0.0, 0.2, 0.2)),
    )
    backend = PixelAwareRecognitionBackend()
    recognizer = DoctrCrnnVgg16Recognizer(backend, source_classes=frozenset({"Text"}))

    results = recognizer.recognize(image, detections)

    assert [result.node_id for result in results] == [8, 3, 9]
    assert [result.text for result in results] == ["wideΩ", "green", ""]
    assert results[0].status == "elongated_clamped"
    assert results[0].unknown_characters == ("Ω",)
    assert results[0].token_diagnostics == ("fixture-token",)
    assert results[1].status == "ready"
    assert results[2].status == "degenerate_box"
    assert all(result.request_id.startswith(image.sha256) for result in results)
    assert backend.batch_widths == [[50, 512]]
    assert OCR_CROP_POLICY_VERSION == "doctr_crnn_rgb_h32_aspect_v1"


def _ocr(node_id: int, text: str, confidence: float) -> OcrResult:
    return OcrResult(
        request_id=f"fixture:{node_id}",
        node_id=node_id,
        text=text,
        confidence=confidence,
        screen_box_xyxy_norm=(0.2, 0.2, 0.4, 0.3),
        status="ready",
    )


def test_ocr_association_keeps_text_nodes_and_audits_control_conflicts() -> None:
    nodes = (
        NodeRecord(1, NodeType.TEXT, (0.2, 0.2, 0.4, 0.3)),
        NodeRecord(2, NodeType.TEXT, (0.2, 0.32, 0.4, 0.4)),
        NodeRecord(3, NodeType.CONTROL, (0.1, 0.1, 0.5, 0.5)),
        NodeRecord(4, NodeType.CONTROL, (0.0, 0.0, 0.9, 0.9)),
        NodeRecord(5, NodeType.CONTROL, (0.6, 0.6, 0.9, 0.9)),
    )

    output = associate_ocr(nodes, (_ocr(1, "  SAVE   NOW ", 0.7), _ocr(2, "Cancel", 0.9)))
    by_id = {node.node_id: node for node in output.nodes}

    assert by_id[1].text == "  SAVE   NOW "
    assert by_id[2].text == "Cancel"
    assert by_id[3].text == "cancel"
    assert by_id[3].provenance["ocr_propagated_from"] == "2"
    assert by_id[4].text is None
    assert by_id[5].text is None
    propagated = [event for event in output.events if event.kind == "propagated_control_label"]
    assert len(propagated) == 2
    assert [(event.source_node_id, event.accepted) for event in propagated] == [
        (1, False),
        (2, True),
    ]
    assert propagated[0].reason == "lower_priority_conflict"
    assert propagated[1].reason == "deterministic_winner"


def test_ocr_association_rejects_missing_nodes() -> None:
    with pytest.raises(ValueError, match="missing node"):
        associate_ocr((), (_ocr(11, "missing", 0.5),))
