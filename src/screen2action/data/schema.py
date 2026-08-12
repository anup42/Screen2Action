"""Typed records shared by dataset adapters, SSB code, and runtime code."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

type Box = tuple[float, float, float, float]
type Point = tuple[float, float]
type SerializedTokens = tuple[int, ...]


class NodeType(StrEnum):
    """Canonical semantic node types."""

    ROOT = "root"
    TEXT = "text"
    ICON = "icon"
    CONTROL = "control"
    INPUT = "input"
    IMAGE = "image"
    CONTAINER = "container"
    GESTURE_REGION = "gesture_region"
    OTHER = "other"


class ActionType(StrEnum):
    """Supported user action labels."""

    CLICK = "click"
    DRAG = "drag"
    SCROLL = "scroll"
    LONG_PRESS = "long_press"
    UNKNOWN = "unknown"


class RelationType(StrEnum):
    """Supported graph relation families."""

    CONTAINMENT = "containment"
    PROXIMITY = "proximity"
    ORDINAL = "ordinal"


class RelationDirection(StrEnum):
    """Direction or reference role attached to a directed edge."""

    INSIDE = "inside"
    CONTAINS = "contains"
    LEFT = "left"
    RIGHT = "right"
    ABOVE = "above"
    BELOW = "below"
    PREVIOUS = "previous"
    NEXT = "next"
    NONE = "none"


class RelationAxis(StrEnum):
    """Axis used by an ordinal relation, when applicable."""

    ROW = "row"
    COLUMN = "column"


def _require_finite(value: float, field_name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite, got {value!r}")


def _validate_box(box: Box, field_name: str = "box_xyxy_norm") -> None:
    if len(box) != 4:
        raise ValueError(f"{field_name} must have four coordinates")
    for index, value in enumerate(box):
        _require_finite(value, f"{field_name}[{index}]")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{field_name}[{index}] must be in [0, 1], got {value}")
    x1, y1, x2, y2 = box
    if x1 > x2 or y1 > y2:
        raise ValueError(f"{field_name} must be ordered as xyxy, got {box}")


def _validate_point(point: Point, field_name: str) -> None:
    if len(point) != 2:
        raise ValueError(f"{field_name} must have two coordinates")
    for index, value in enumerate(point):
        _require_finite(value, f"{field_name}[{index}]")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{field_name}[{index}] must be in [0, 1], got {value}")


@dataclass(frozen=True, slots=True)
class ElementAnnotation:
    """Source annotation for one visible screen element."""

    element_id: str
    node_type: NodeType
    box_xyxy_norm: Box
    text: str | None = None
    actionability_labels: tuple[bool | None, ...] = (None, None, None, None)
    icon_class_id: int | None = None
    parent_element_id: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_type", NodeType(self.node_type))
        _validate_box(self.box_xyxy_norm)
        if len(self.actionability_labels) != 4:
            raise ValueError("actionability_labels must contain four entries")
        if self.icon_class_id is not None and self.icon_class_id < 0:
            raise ValueError("icon_class_id must be non-negative")


@dataclass(frozen=True, slots=True)
class ScreenRecord:
    """Canonical screen record emitted by a dataset adapter."""

    screen_id: str
    app_id: str
    image_path: str
    width: int
    height: int
    split: str
    source: str
    elements: tuple[ElementAnnotation, ...] = ()

    def __post_init__(self) -> None:
        if not self.screen_id or not self.app_id:
            raise ValueError("screen_id and app_id are required")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("screen dimensions must be positive")
        if self.split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")


@dataclass(frozen=True, slots=True)
class CommandRecord:
    """Canonical command and target annotation."""

    command_id: str
    screen_id: str
    text: str
    action_type: ActionType
    target_box_xyxy_norm: Box
    target_point_xy_norm: Point | None = None
    action_parameters: Mapping[str, float] = field(default_factory=dict)
    relation_type: str = "direct"
    reference_element_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.command_id or not self.screen_id:
            raise ValueError("command_id and screen_id are required")
        if not self.text.strip():
            raise ValueError("command text cannot be empty")
        object.__setattr__(self, "action_type", ActionType(self.action_type))
        _validate_box(self.target_box_xyxy_norm, "target_box_xyxy_norm")
        if self.target_point_xy_norm is not None:
            _validate_point(self.target_point_xy_norm, "target_point_xy_norm")


@dataclass(frozen=True, slots=True)
class NodeRecord:
    """Perceived or oracle node in the normalized screen graph."""

    node_id: int
    node_type: NodeType
    box_xyxy_norm: Box
    detector_confidence: float = 0.0
    text: str | None = None
    text_token_ids: tuple[int, ...] = ()
    ocr_confidence: float = 0.0
    icon_probabilities: tuple[float, ...] = ()
    icon_confidence: float = 0.0
    roi_visual_feature: tuple[float, ...] = ()
    visual_feature_confidence: float = 0.0
    actionability_logits: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    hierarchy_depth: int = 0
    parent_id: int | None = None
    mandatory: bool = False
    is_modal: bool = False
    is_scroll_container: bool = False
    retention_score: float | None = None
    actionability_mask: tuple[bool, bool, bool, bool] = (True, True, True, True)
    annotation_source: str = "unknown"
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.node_id < 0:
            raise ValueError("node_id must be non-negative")
        object.__setattr__(self, "node_type", NodeType(self.node_type))
        _validate_box(self.box_xyxy_norm)
        for field_name, value in (
            ("detector_confidence", self.detector_confidence),
            ("ocr_confidence", self.ocr_confidence),
            ("icon_confidence", self.icon_confidence),
            ("visual_feature_confidence", self.visual_feature_confidence),
        ):
            _require_finite(value, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1]")
        if self.retention_score is not None:
            _require_finite(self.retention_score, "retention_score")
        if self.hierarchy_depth < 0:
            raise ValueError("hierarchy_depth must be non-negative")
        if len(self.actionability_logits) != 4:
            raise ValueError("actionability_logits must contain four values")
        if len(self.actionability_mask) != 4:
            raise ValueError("actionability_mask must contain four values")
        if not self.annotation_source:
            raise ValueError("annotation_source cannot be empty")
        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in self.provenance.items()
        ):
            raise ValueError("provenance must be string-to-string")
        if any(token_id < 0 for token_id in self.text_token_ids):
            raise ValueError("text_token_ids must be non-negative")
        if any(
            probability < 0.0 or not math.isfinite(probability)
            for probability in self.icon_probabilities
        ):
            raise ValueError("icon_probabilities must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class EdgeRecord:
    """Directed graph edge with optional ordinal axis."""

    src: int
    dst: int
    relation: RelationType
    direction: RelationDirection
    relative_geometry: tuple[float, ...] = ()
    axis: RelationAxis | None = None

    def __post_init__(self) -> None:
        if self.src < 0 or self.dst < 0:
            raise ValueError("edge endpoints must be non-negative")
        if self.src == self.dst:
            raise ValueError("self-edges are not allowed")
        object.__setattr__(self, "relation", RelationType(self.relation))
        object.__setattr__(self, "direction", RelationDirection(self.direction))
        if self.axis is not None:
            object.__setattr__(self, "axis", RelationAxis(self.axis))
        if any(not math.isfinite(value) for value in self.relative_geometry):
            raise ValueError("relative_geometry must contain finite values")


@dataclass(frozen=True, slots=True)
class SelectedNodeRecord:
    """A node retained by the inference-time SSB budget selector."""

    node_id: int
    token_cost: int
    serialized_index: int
    inserted_by_closure: bool = False

    def __post_init__(self) -> None:
        if self.node_id < 0 or self.token_cost <= 0 or self.serialized_index <= 0:
            raise ValueError("selected-node identifiers and costs must be positive")


@dataclass(frozen=True, slots=True)
class SerializedSsbRecord:
    """Portable record for a versioned SSB token stream."""

    version: int
    tokens: SerializedTokens
    node_ids: tuple[int, ...]
    total_cost: int

    def __post_init__(self) -> None:
        if self.version <= 0:
            raise ValueError("SSB version must be positive")
        if self.total_cost != len(self.tokens):
            raise ValueError("serialized SSB cost must equal token length")
        if len(self.node_ids) != len(set(self.node_ids)):
            raise ValueError("serialized SSB node IDs must be unique")


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    """One fixed-shape command retrieval result."""

    node_id: int
    score: float
    rank: int
    actionable: bool
    valid: bool = True

    def __post_init__(self) -> None:
        if self.node_id < 0 or self.rank < 0:
            raise ValueError("candidate node_id and rank must be non-negative")
        _require_finite(self.score, "candidate score")


@dataclass(frozen=True, slots=True)
class ConfidenceOutput:
    """Raw and optionally calibrated confidence for an action prediction."""

    logit: float
    probability: float
    calibrated: bool = False

    def __post_init__(self) -> None:
        _require_finite(self.logit, "confidence logit")
        _require_finite(self.probability, "confidence probability")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("confidence probability must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class GroundingOutput:
    """Canonical click/action prediction returned by the runtime."""

    node_id: int | None
    point_xy_norm: Point | None
    action_type: ActionType
    action_parameters: Mapping[str, float] = field(default_factory=dict)
    candidate_logits: tuple[float, ...] = ()
    confidence: ConfidenceOutput | None = None
    abstained: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_type", ActionType(self.action_type))
        if self.node_id is not None and self.node_id < 0:
            raise ValueError("node_id must be non-negative")
        if self.point_xy_norm is not None:
            _validate_point(self.point_xy_norm, "point_xy_norm")
        if any(not math.isfinite(value) for value in self.candidate_logits):
            raise ValueError("candidate logits must be finite")
