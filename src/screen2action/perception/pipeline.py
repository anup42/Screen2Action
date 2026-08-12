"""Command-independent full-screen perception for real, oracle, and cached modes."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any, cast

import torch

from screen2action.data.schema import (
    EdgeRecord,
    NodeRecord,
    NodeType,
    RelationAxis,
    RelationDirection,
    RelationType,
    ScreenRecord,
)
from screen2action.perception.association import OcrAssociationEvent, associate_ocr
from screen2action.perception.base import (
    ImageFrame,
    TextRecognizer,
    UiDetection,
    UiDetector,
    normalize_image,
)
from screen2action.perception.cache import ContentAddressedPerceptionCache, PerceptionCacheKey
from screen2action.perception.icon_actionability import (
    IconActionabilityOutput,
    MobileNetV3IconActionability,
)
from screen2action.perception.oracle import OraclePerception
from screen2action.runtime.cropper import batch_crop_tensor
from screen2action.ssb.hierarchy import build_hierarchy
from screen2action.ssb.relations import build_ordinal_edges, build_proximity_edges


@dataclass(frozen=True, slots=True)
class FullScreenPerceptionConfig:
    """Versioned reconstruction controls affecting cache compatibility."""

    schema_version: int = 1
    node_crop_size: int = 224
    node_crop_margin: float = 0.0
    include_proximity: bool = True
    include_ordinal: bool = True
    ocr_control_propagation_coverage: float = 0.95

    def __post_init__(self) -> None:
        if self.schema_version <= 0 or self.node_crop_size <= 0:
            raise ValueError("schema and crop size must be positive")
        if not 0.0 <= self.node_crop_margin <= 1.0:
            raise ValueError("node_crop_margin must be in [0, 1]")
        if not 0.0 < self.ocr_control_propagation_coverage <= 1.0:
            raise ValueError("OCR propagation coverage must be in (0, 1]")

    @property
    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PerceptionFrame:
    """Reusable frame state produced once per unique screenshot."""

    screenshot_sha256: str
    width: int
    height: int
    nodes: tuple[NodeRecord, ...]
    edges: tuple[EdgeRecord, ...]
    root_id: int
    mode: str
    diagnostics: Mapping[str, object]
    ocr_associations: tuple[OcrAssociationEvent, ...] = ()
    cache_key_digest: str | None = None


def _node_to_dict(node: NodeRecord) -> dict[str, object]:
    return {
        "node_id": node.node_id,
        "node_type": node.node_type.value,
        "box_xyxy_norm": list(node.box_xyxy_norm),
        "detector_confidence": node.detector_confidence,
        "text": node.text,
        "text_token_ids": list(node.text_token_ids),
        "ocr_confidence": node.ocr_confidence,
        "icon_probabilities": list(node.icon_probabilities),
        "icon_confidence": node.icon_confidence,
        "roi_visual_feature": list(node.roi_visual_feature),
        "visual_feature_confidence": node.visual_feature_confidence,
        "actionability_logits": list(node.actionability_logits),
        "hierarchy_depth": node.hierarchy_depth,
        "parent_id": node.parent_id,
        "mandatory": node.mandatory,
        "is_modal": node.is_modal,
        "is_scroll_container": node.is_scroll_container,
        "retention_score": node.retention_score,
        "actionability_mask": list(node.actionability_mask),
        "annotation_source": node.annotation_source,
        "provenance": dict(node.provenance),
    }


def _edge_to_dict(edge: EdgeRecord) -> dict[str, object]:
    return {
        "src": edge.src,
        "dst": edge.dst,
        "relation": edge.relation.value,
        "direction": edge.direction.value,
        "relative_geometry": list(edge.relative_geometry),
        "axis": edge.axis.value if edge.axis is not None else None,
    }


def perception_frame_to_payload(frame: PerceptionFrame) -> dict[str, object]:
    """Serialize reusable state without screenshot bytes or raw OCR strings in diagnostics."""

    return {
        "screenshot_sha256": frame.screenshot_sha256,
        "width": frame.width,
        "height": frame.height,
        "nodes": [_node_to_dict(node) for node in frame.nodes],
        "edges": [_edge_to_dict(edge) for edge in frame.edges],
        "root_id": frame.root_id,
        "source_mode": frame.mode,
        "diagnostics": dict(frame.diagnostics),
        "ocr_associations": [asdict(event) for event in frame.ocr_associations],
    }


def _tuple_float(value: object, length: int, *, context: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{context} must be a list of length {length}")
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in result):
        raise ValueError(f"{context} must be finite")
    return result


def _int_value(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise ValueError(f"{context} must be an integer")
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{context} must be an integer") from error


def _float_value(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError(f"{context} must be numeric")
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(f"{context} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _node_from_dict(raw: object) -> NodeRecord:
    if not isinstance(raw, dict):
        raise ValueError("cached node must be a mapping")
    action_mask = raw.get("actionability_mask")
    if not isinstance(action_mask, list) or len(action_mask) != 4:
        raise ValueError("cached actionability mask must contain four values")
    provenance = raw.get("provenance", {})
    if not isinstance(provenance, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in provenance.items()
    ):
        raise ValueError("cached node provenance must be string-to-string")
    return NodeRecord(
        node_id=_int_value(raw["node_id"], context="node_id"),
        node_type=NodeType(str(raw["node_type"])),
        box_xyxy_norm=cast(Any, _tuple_float(raw["box_xyxy_norm"], 4, context="box")),
        detector_confidence=_float_value(
            raw.get("detector_confidence", 0.0), context="detector_confidence"
        ),
        text=str(raw["text"]) if raw.get("text") is not None else None,
        text_token_ids=tuple(
            _int_value(item, context="text_token_id")
            for item in cast(list[object], raw.get("text_token_ids", []))
        ),
        ocr_confidence=_float_value(raw.get("ocr_confidence", 0.0), context="ocr_confidence"),
        icon_probabilities=tuple(
            _float_value(item, context="icon_probability")
            for item in cast(list[object], raw.get("icon_probabilities", []))
        ),
        icon_confidence=_float_value(raw.get("icon_confidence", 0.0), context="icon_confidence"),
        roi_visual_feature=tuple(
            _float_value(item, context="roi_visual_feature")
            for item in cast(list[object], raw.get("roi_visual_feature", []))
        ),
        visual_feature_confidence=_float_value(
            raw.get("visual_feature_confidence", 0.0), context="visual_feature_confidence"
        ),
        actionability_logits=cast(
            Any,
            _tuple_float(raw.get("actionability_logits"), 4, context="actionability_logits"),
        ),
        hierarchy_depth=int(raw.get("hierarchy_depth", 0)),
        parent_id=int(raw["parent_id"]) if raw.get("parent_id") is not None else None,
        mandatory=bool(raw.get("mandatory", False)),
        is_modal=bool(raw.get("is_modal", False)),
        is_scroll_container=bool(raw.get("is_scroll_container", False)),
        retention_score=(
            float(raw["retention_score"]) if raw.get("retention_score") is not None else None
        ),
        actionability_mask=cast(Any, tuple(bool(item) for item in action_mask)),
        annotation_source=str(raw.get("annotation_source", "unknown")),
        provenance=cast(dict[str, str], provenance),
    )


def _edge_from_dict(raw: object) -> EdgeRecord:
    if not isinstance(raw, dict):
        raise ValueError("cached edge must be a mapping")
    axis = raw.get("axis")
    geometry = raw.get("relative_geometry", [])
    if not isinstance(geometry, list):
        raise ValueError("cached edge geometry must be a list")
    return EdgeRecord(
        src=int(raw["src"]),
        dst=int(raw["dst"]),
        relation=RelationType(str(raw["relation"])),
        direction=RelationDirection(str(raw["direction"])),
        relative_geometry=tuple(float(item) for item in geometry),
        axis=RelationAxis(str(axis)) if axis is not None else None,
    )


def perception_frame_from_payload(
    payload: Mapping[str, object],
    *,
    cache_key_digest: str,
) -> PerceptionFrame:
    """Validate and deserialize one cache payload."""

    nodes = payload.get("nodes")
    edges = payload.get("edges")
    associations = payload.get("ocr_associations", [])
    diagnostics = payload.get("diagnostics", {})
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("cached perception nodes/edges must be lists")
    if not isinstance(associations, list) or not isinstance(diagnostics, dict):
        raise ValueError("cached perception diagnostics are malformed")
    events = []
    for raw in associations:
        if not isinstance(raw, dict):
            raise ValueError("cached OCR association must be a mapping")
        events.append(
            OcrAssociationEvent(
                source_node_id=_int_value(raw["source_node_id"], context="source_node_id"),
                destination_node_id=_int_value(
                    raw["destination_node_id"], context="destination_node_id"
                ),
                kind=str(raw["kind"]),
                confidence=_float_value(raw["confidence"], context="confidence"),
                accepted=bool(raw["accepted"]),
                reason=str(raw["reason"]),
            )
        )
    return PerceptionFrame(
        screenshot_sha256=str(payload["screenshot_sha256"]),
        width=_int_value(payload["width"], context="width"),
        height=_int_value(payload["height"], context="height"),
        nodes=tuple(_node_from_dict(raw) for raw in nodes),
        edges=tuple(_edge_from_dict(raw) for raw in edges),
        root_id=_int_value(payload["root_id"], context="root_id"),
        mode="cached",
        diagnostics=cast(dict[str, object], diagnostics),
        ocr_associations=tuple(events),
        cache_key_digest=cache_key_digest,
    )


def _initial_node(detection: UiDetection) -> NodeRecord:
    return NodeRecord(
        node_id=detection.node_id,
        node_type=detection.node_type,
        box_xyxy_norm=detection.box_xyxy_norm,
        detector_confidence=detection.confidence,
        roi_visual_feature=detection.original_class_feature,
        mandatory=detection.node_type is NodeType.ROOT,
        is_modal="modal" in detection.semantic_roles,
        is_scroll_container="scroll" in detection.semantic_roles,
        retention_score=1.0 if detection.node_type is NodeType.ROOT else None,
        actionability_mask=(False, False, False, False),
        annotation_source=detection.annotation_source,
        provenance={
            **detection.metadata,
            "mapping_policy": detection.policy_version,
            "original_class_id": str(detection.original_class_id),
            "original_class_name": detection.original_class_name,
        },
    )


class FullScreenPerception:
    """Run expensive perception once per frame; no command input is accepted."""

    def __init__(
        self,
        *,
        detector: UiDetector | None = None,
        recognizer: TextRecognizer | None = None,
        visual_model: MobileNetV3IconActionability | None = None,
        oracle: OraclePerception | None = None,
        cache: ContentAddressedPerceptionCache | None = None,
        model_bundle_lock_digest: str,
        config: FullScreenPerceptionConfig | None = None,
    ) -> None:
        if len(model_bundle_lock_digest) != 64:
            raise ValueError("model_bundle_lock_digest must be SHA256")
        self.detector = detector
        self.recognizer = recognizer
        self.visual_model = visual_model
        self.oracle = oracle or OraclePerception()
        self.cache = cache
        self.model_bundle_lock_digest = model_bundle_lock_digest
        self.config = config or FullScreenPerceptionConfig()

    def _key(self, image: ImageFrame) -> PerceptionCacheKey:
        return PerceptionCacheKey(
            screenshot_sha256=image.sha256,
            model_bundle_lock_digest=self.model_bundle_lock_digest,
            perception_config_digest=self.config.digest,
        )

    def _graph(
        self,
        nodes: Sequence[NodeRecord],
    ) -> tuple[tuple[NodeRecord, ...], tuple[EdgeRecord, ...], int]:
        hierarchy = build_hierarchy(tuple(nodes))
        edges = list(hierarchy.containment_edges)
        if self.config.include_proximity:
            edges.extend(
                build_proximity_edges(
                    hierarchy.nodes,
                    parent_by_child=hierarchy.parent_by_child,
                )
            )
        if self.config.include_ordinal:
            edges.extend(
                build_ordinal_edges(
                    hierarchy.nodes,
                    parent_by_child=hierarchy.parent_by_child,
                )
            )
        unique = {
            (edge.src, edge.dst, edge.relation, edge.direction, edge.axis): edge for edge in edges
        }
        ordered = tuple(
            unique[key]
            for key in sorted(
                unique,
                key=lambda item: (
                    item[0],
                    item[1],
                    item[2].value,
                    item[3].value,
                    item[4].value if item[4] else "",
                ),
            )
        )
        return hierarchy.nodes, ordered, hierarchy.root_id

    def _attach_visual(
        self,
        image: ImageFrame,
        nodes: Sequence[NodeRecord],
    ) -> tuple[NodeRecord, ...]:
        if self.visual_model is None:
            return tuple(nodes)
        candidates = [node for node in nodes if node.node_type is not NodeType.ROOT]
        if not candidates:
            return tuple(nodes)
        crops, _ = batch_crop_tensor(
            image.pixels,
            [node.box_xyxy_norm for node in candidates],
            output_size=self.config.node_crop_size,
            margin_fraction=self.config.node_crop_margin,
        )
        parameter = next(self.visual_model.parameters())
        device = parameter.device
        output: IconActionabilityOutput = self.visual_model(crops.to(device) / 255.0)
        probabilities = torch.softmax(output.icon_logits, dim=-1).detach().cpu()
        action_logits = output.actionability_logits.detach().cpu()
        features = output.visual_features.detach().cpu()
        updates: dict[int, NodeRecord] = {}
        for index, node in enumerate(candidates):
            icon = probabilities[index]
            updates[node.node_id] = replace(
                node,
                icon_probabilities=tuple(float(value) for value in icon),
                icon_confidence=float(icon.max()),
                roi_visual_feature=tuple(float(value) for value in features[index]),
                visual_feature_confidence=1.0,
                actionability_logits=cast(
                    Any,
                    tuple(float(value) for value in action_logits[index]),
                ),
                actionability_mask=(True, True, True, True),
                provenance={**node.provenance, "visual_model": "mobilenet_v3_shared_v1"},
            )
        return tuple(updates.get(node.node_id, node) for node in nodes)

    def _real(self, images: Sequence[object]) -> tuple[PerceptionFrame, ...]:
        if self.detector is None or self.recognizer is None or self.visual_model is None:
            raise RuntimeError("real mode requires detector, recognizer, and visual model")
        normalized = tuple(normalize_image(image) for image in images)
        unique_images: dict[str, ImageFrame] = {}
        for image in normalized:
            unique_images.setdefault(image.sha256, image)
        outputs_by_digest: dict[str, PerceptionFrame] = {}
        misses: list[ImageFrame] = []
        for image in unique_images.values():
            key = self._key(image)
            cached = self.cache.get(key) if self.cache is not None else None
            if cached is None:
                misses.append(image)
            else:
                outputs_by_digest[image.sha256] = perception_frame_from_payload(
                    cached, cache_key_digest=key.digest
                )
        detected_frames = self.detector.detect(misses) if misses else ()
        expected_misses = {image.sha256 for image in misses}
        observed_misses = {frame.image.sha256 for frame in detected_frames}
        if observed_misses != expected_misses or len(detected_frames) != len(misses):
            raise RuntimeError("detector changed, omitted, duplicated, or reordered unique screens")
        for detected in detected_frames:
            key = self._key(detected.image)
            try:
                nodes = tuple(_initial_node(detection) for detection in detected.detections)
                ocr = self.recognizer.recognize(detected.image, detected.detections)
                associated = associate_ocr(
                    nodes,
                    ocr,
                    propagation_coverage=self.config.ocr_control_propagation_coverage,
                )
                visual_nodes = self._attach_visual(detected.image, associated.nodes)
                graph_nodes, edges, root_id = self._graph(visual_nodes)
                frame = PerceptionFrame(
                    screenshot_sha256=detected.image.sha256,
                    width=detected.image.width,
                    height=detected.image.height,
                    nodes=graph_nodes,
                    edges=edges,
                    root_id=root_id,
                    mode="real",
                    diagnostics={
                        "raw_proposal_count": detected.raw_proposal_count,
                        "post_nms_count": detected.post_nms_count,
                        "raw_diagnostics_supported": detected.raw_diagnostics_supported,
                        "class_mapping_version": detected.class_mapping_version,
                        "ocr_crop_count": len(ocr),
                        "node_count": len(graph_nodes),
                        "edge_count": len(edges),
                    },
                    ocr_associations=associated.events,
                    cache_key_digest=key.digest,
                )
                if self.cache is not None:
                    self.cache.put(key, perception_frame_to_payload(frame))
                outputs_by_digest[detected.image.sha256] = frame
            except Exception as error:
                if self.cache is not None:
                    self.cache.record_failure(
                        key,
                        error_type=type(error).__name__,
                        message=str(error),
                        attempt=1,
                    )
                raise
        return tuple(outputs_by_digest[image.sha256] for image in normalized)

    def _oracle(
        self,
        images: Sequence[object],
        screens: Sequence[ScreenRecord],
    ) -> tuple[PerceptionFrame, ...]:
        if len(images) != len(screens):
            raise ValueError("oracle screen count must match image count")
        outputs = []
        for image_input, screen in zip(images, screens, strict=True):
            image = normalize_image(image_input)
            oracle_frame = self.oracle.perceive(image.pixels.float() / 255.0, screen)
            graph_nodes, edges, root_id = self._graph(oracle_frame.nodes)
            key = self._key(image)
            frame = PerceptionFrame(
                screenshot_sha256=image.sha256,
                width=image.width,
                height=image.height,
                nodes=graph_nodes,
                edges=edges,
                root_id=root_id,
                mode="oracle",
                diagnostics={
                    "node_count": len(graph_nodes),
                    "edge_count": len(edges),
                    "source": "oracle_annotations",
                },
                cache_key_digest=key.digest,
            )
            if self.cache is not None:
                self.cache.put(key, perception_frame_to_payload(frame))
            outputs.append(frame)
        return tuple(outputs)

    def _cached(self, images: Sequence[object]) -> tuple[PerceptionFrame, ...]:
        if self.cache is None:
            raise RuntimeError("cached mode requires a perception cache")
        outputs = []
        for image_input in images:
            image = normalize_image(image_input)
            key = self._key(image)
            payload = self.cache.get(key)
            if payload is None:
                raise FileNotFoundError(f"perception cache miss for screenshot {image.sha256}")
            outputs.append(perception_frame_from_payload(payload, cache_key_digest=key.digest))
        return tuple(outputs)

    def perceive(
        self,
        images: Sequence[object],
        *,
        mode: str = "real",
        oracle_screens: Sequence[ScreenRecord] = (),
    ) -> tuple[PerceptionFrame, ...]:
        """Perceive unique screens without accepting or importing any command encoder."""

        if not images:
            raise ValueError("at least one image is required")
        if mode == "real":
            return self._real(images)
        if mode == "oracle":
            return self._oracle(images, oracle_screens)
        if mode == "cached":
            return self._cached(images)
        raise ValueError("mode must be real, oracle, or cached")
