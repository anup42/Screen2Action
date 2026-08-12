"""Deterministic and auditable OCR-to-node association."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, replace

from screen2action.data.schema import NodeRecord, NodeType
from screen2action.perception.base import OcrResult
from screen2action.ssb.geometry import box_area, intersection_box


@dataclass(frozen=True, slots=True)
class OcrAssociationEvent:
    """One direct or propagated OCR assignment decision."""

    source_node_id: int
    destination_node_id: int
    kind: str
    confidence: float
    accepted: bool
    reason: str


@dataclass(frozen=True, slots=True)
class OcrAssociationOutput:
    """Updated nodes and complete deterministic decision audit."""

    nodes: tuple[NodeRecord, ...]
    events: tuple[OcrAssociationEvent, ...]


def normalize_control_label(text: str) -> str:
    """Normalize OCR text for a control label without discarding Unicode letters."""

    normalized = unicodedata.normalize("NFKC", text).casefold().strip()
    return re.sub(r"\s+", " ", normalized)


def _coverage(container: NodeRecord, child: NodeRecord) -> float:
    child_area = box_area(child.box_xyxy_norm)
    if child_area <= 0.0:
        return 0.0
    intersection = intersection_box(container.box_xyxy_norm, child.box_xyxy_norm)
    if intersection[0] > intersection[2] or intersection[1] > intersection[3]:
        return 0.0
    return box_area(intersection) / child_area


def associate_ocr(
    nodes: Sequence[NodeRecord],
    results: Sequence[OcrResult],
    *,
    propagation_coverage: float = 0.95,
) -> OcrAssociationOutput:
    """Assign OCR once directly and optionally to one smallest enclosing control."""

    if not 0.0 < propagation_coverage <= 1.0:
        raise ValueError("propagation_coverage must be in (0, 1]")
    node_map = {node.node_id: node for node in nodes}
    if len(node_map) != len(nodes):
        raise ValueError("node IDs must be unique")
    events: list[OcrAssociationEvent] = []
    candidates_by_control: dict[int, list[tuple[float, int, str]]] = {}
    for result in sorted(results, key=lambda item: item.node_id):
        source = node_map.get(result.node_id)
        if source is None:
            raise ValueError(f"OCR result references missing node {result.node_id}")
        accepted = bool(result.text.strip()) and result.status != "degenerate_box"
        if accepted:
            node_map[source.node_id] = replace(
                source,
                text=result.text,
                ocr_confidence=result.confidence,
                provenance={
                    **source.provenance,
                    "ocr_policy": "doctr_crnn_rgb_h32_aspect_v1",
                    "ocr_status": result.status,
                },
            )
        events.append(
            OcrAssociationEvent(
                source_node_id=source.node_id,
                destination_node_id=source.node_id,
                kind="direct_text_region",
                confidence=result.confidence,
                accepted=accepted,
                reason="recognized_text" if accepted else result.status,
            )
        )
        if not accepted:
            continue
        controls = [
            node
            for node in node_map.values()
            if node.node_id != source.node_id
            and node.node_type in {NodeType.CONTROL, NodeType.INPUT}
            and _coverage(node, source) >= propagation_coverage
        ]
        if controls:
            destination = min(
                controls, key=lambda node: (box_area(node.box_xyxy_norm), node.node_id)
            )
            candidates_by_control.setdefault(destination.node_id, []).append(
                (result.confidence, source.node_id, normalize_control_label(result.text))
            )
    for control_id, candidates in sorted(candidates_by_control.items()):
        winner = min(candidates, key=lambda item: (-item[0], item[1], item[2]))
        control = node_map[control_id]
        node_map[control_id] = replace(
            control,
            text=winner[2],
            ocr_confidence=winner[0],
            provenance={
                **control.provenance,
                "ocr_propagated_from": str(winner[1]),
                "ocr_association_policy": "smallest_control_highest_confidence_v1",
            },
        )
        for confidence, source_id, _ in candidates:
            accepted = source_id == winner[1]
            events.append(
                OcrAssociationEvent(
                    source_node_id=source_id,
                    destination_node_id=control_id,
                    kind="propagated_control_label",
                    confidence=confidence,
                    accepted=accepted,
                    reason="deterministic_winner" if accepted else "lower_priority_conflict",
                )
            )
    return OcrAssociationOutput(
        nodes=tuple(node_map[node_id] for node_id in sorted(node_map)),
        events=tuple(events),
    )
