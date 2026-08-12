"""ScreenParser YOLO11-L adapter with versioned 55-class semantic mapping."""

from __future__ import annotations

import importlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import torch

from screen2action.data.schema import Box, NodeType
from screen2action.perception.base import (
    ImageFrame,
    PreprocessingMetadata,
    RawDetection,
    RawDetectionFrame,
    UiDetection,
    UiDetectionFrame,
    normalize_images,
    normalize_pixel_box,
)
from screen2action.ssb.geometry import box_area

SCREENPARSER_CLASS_MAPPING_VERSION = "screenparser_55_to_ssb_v1"
SCREENPARSER_CLASSES = (
    "Table",
    "Column/Browser",
    "Button",
    "Utility Button",
    "App Icon",
    "Navigation Bar",
    "Status Bar",
    "Search Field",
    "Toolbar",
    "Tooltip",
    "Video",
    "Tab Bar",
    "Side Bar",
    "Slider",
    "Picker",
    "ContextMenu",
    "DockMenu",
    "EditMenu",
    "Image",
    "Scroll",
    "Switch",
    "File Icon",
    "Chart",
    "Window",
    "Screen",
    "List",
    "List Item",
    "PopUp Menu",
    "Steppers",
    "Toggles",
    "Text Input",
    "Rating Indicator",
    "Checkbox",
    "Radiobox",
    "Select",
    "Avatar",
    "Badge",
    "Alert",
    "Progress bar",
    "Bottom navigation",
    "Breadcrumb",
    "Page control",
    "Link",
    "Menu",
    "Pagination",
    "Tab",
    "Search Bar",
    "Date-Time picker",
    "Calendar",
    "Text",
    "Heading",
    "Code snippet",
    "Carousel",
    "Notification",
    "Logo",
)

_TEXT = {"Text", "Heading", "Code snippet"}
_ICON = {"App Icon", "File Icon", "Logo", "Avatar", "Badge"}
_INPUT = {"Search Field", "Text Input", "Search Bar", "Picker", "Select"}
_IMAGE = {"Image", "Video", "Chart"}
_CONTAINER = {
    "Table",
    "Column/Browser",
    "Navigation Bar",
    "Status Bar",
    "Toolbar",
    "Tab Bar",
    "Side Bar",
    "ContextMenu",
    "DockMenu",
    "EditMenu",
    "Window",
    "Screen",
    "List",
    "PopUp Menu",
    "Bottom navigation",
    "Breadcrumb",
    "Menu",
    "Pagination",
    "Calendar",
    "Carousel",
    "Notification",
    "Alert",
}
_GESTURE = {"Scroll"}
_CONTROL = set(SCREENPARSER_CLASSES).difference(_TEXT, _ICON, _INPUT, _IMAGE, _CONTAINER, _GESTURE)
_OCR_SOURCE_CLASSES = frozenset(
    {"Text", "Heading", "Text Input", "Search Field", "Search Bar", "Button", "Link"}
)


def coarse_node_type(class_name: str) -> NodeType:
    """Map one original name to a stable coarse SSB type."""

    if class_name in _TEXT:
        return NodeType.TEXT
    if class_name in _ICON:
        return NodeType.ICON
    if class_name in _INPUT:
        return NodeType.INPUT
    if class_name in _IMAGE:
        return NodeType.IMAGE
    if class_name in _CONTAINER:
        return NodeType.CONTAINER
    if class_name in _GESTURE:
        return NodeType.GESTURE_REGION
    if class_name in _CONTROL:
        return NodeType.CONTROL
    return NodeType.OTHER


def semantic_roles(class_name: str) -> tuple[str, ...]:
    """Return explicit, non-exclusive semantic roles for one class."""

    roles: set[str] = set()
    if class_name in _OCR_SOURCE_CLASSES:
        roles.add("text")
    if class_name in _ICON:
        roles.add("icon")
    if class_name in _INPUT:
        roles.update(("input", "control"))
    if class_name in _CONTROL:
        roles.add("control")
    if class_name in _IMAGE:
        roles.add("image")
    if class_name in _CONTAINER:
        roles.add("container")
    if class_name in {"Alert", "PopUp Menu", "ContextMenu", "Window"}:
        roles.add("modal")
    if class_name in {"Scroll", "List", "Carousel"}:
        roles.add("scroll")
    return tuple(sorted(roles))


@dataclass(frozen=True, slots=True)
class ScreenParserConfig:
    """Reconstruction inference settings from the locked public contract."""

    input_size: int = 1280
    confidence_threshold: float = 0.10
    nms_iou_threshold: float = 0.10
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.input_size <= 0:
            raise ValueError("input_size must be positive")
        for name, value in (
            ("confidence_threshold", self.confidence_threshold),
            ("nms_iou_threshold", self.nms_iou_threshold),
        ):
            if not 0.0 <= value <= 1.0 or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and in [0, 1]")
        target = torch.device(self.device)
        if target.type not in {"cpu", "cuda"}:
            raise ValueError("device must resolve to CPU or CUDA")


class ScreenParserBackend(Protocol):
    """Injectable raw ScreenParser inference boundary."""

    def predict(
        self,
        images: Sequence[ImageFrame],
        config: ScreenParserConfig,
    ) -> Sequence[RawDetectionFrame]:
        """Return original-pixel detections after configured NMS."""


class UltralyticsScreenParserBackend:
    """Lazy YOLO backend that accepts only a locally locked weight file."""

    def __init__(self, weight_path: str | Path) -> None:
        path = Path(weight_path)
        if not path.is_file():
            raise FileNotFoundError(f"locked ScreenParser weight does not exist: {path}")
        self.weight_path = path
        self._model: object | None = None

    def _load(self) -> object:
        if self._model is None:
            try:
                module = importlib.import_module("ultralytics")
            except ImportError as error:
                raise RuntimeError("install the perception extra for ScreenParser") from error
            self._model = module.YOLO(str(self.weight_path))
        return self._model

    def predict(
        self,
        images: Sequence[ImageFrame],
        config: ScreenParserConfig,
    ) -> Sequence[RawDetectionFrame]:
        model = self._load()
        sources = [frame.pixels.permute(1, 2, 0).numpy() for frame in images]
        results = model.predict(  # type: ignore[attr-defined]
            source=sources,
            imgsz=config.input_size,
            conf=config.confidence_threshold,
            iou=config.nms_iou_threshold,
            device=config.device,
            verbose=False,
        )
        names = model.names  # type: ignore[attr-defined]
        frames: list[RawDetectionFrame] = []
        for result in results:
            detections: list[RawDetection] = []
            if result.boxes is not None:
                boxes = result.boxes.xyxy.detach().cpu().tolist()
                classes = result.boxes.cls.detach().cpu().tolist()
                confidences = result.boxes.conf.detach().cpu().tolist()
                for box, class_id, confidence in zip(boxes, classes, confidences, strict=True):
                    index = int(class_id)
                    name = names[index] if isinstance(names, dict) else names[index]
                    detections.append(
                        RawDetection(
                            class_id=index,
                            class_name=str(name),
                            confidence=float(confidence),
                            box_xyxy_px=cast(
                                tuple[float, float, float, float], tuple(map(float, box))
                            ),
                        )
                    )
            frames.append(
                RawDetectionFrame(
                    detections=tuple(detections),
                    raw_proposal_count=None,
                    post_nms_count=len(detections),
                    raw_diagnostics_supported=False,
                )
            )
        return tuple(frames)


@dataclass(frozen=True, slots=True)
class ContainerAugmentationPolicy:
    """Ablatable repeated-child container reconstruction."""

    enabled: bool = False
    policy_version: str = "aligned_children_container_v1"
    minimum_children: int = 3
    alignment_tolerance: float = 0.025
    margin: float = 0.01
    confidence: float = 0.65

    def __post_init__(self) -> None:
        if self.minimum_children < 3:
            raise ValueError("minimum_children must be at least three")
        if not 0.0 <= self.alignment_tolerance <= 0.2:
            raise ValueError("alignment_tolerance must be in [0, 0.2]")
        if not 0.0 <= self.margin <= 0.2:
            raise ValueError("margin must be in [0, 0.2]")

    def augment(self, detections: Sequence[UiDetection]) -> tuple[UiDetection, ...]:
        """Generate at most one deterministic aligned-child container."""

        if not self.enabled:
            return ()
        children = [
            detection
            for detection in detections
            if detection.node_type not in {NodeType.ROOT, NodeType.CONTAINER}
            and box_area(detection.box_xyxy_norm) > 0.0
        ]
        groups: dict[int, list[UiDetection]] = {}
        for child in children:
            bucket = round(child.box_xyxy_norm[0] / max(self.alignment_tolerance, 1e-6))
            groups.setdefault(bucket, []).append(child)
        candidates = [
            sorted(group, key=lambda item: (item.box_xyxy_norm[1], item.node_id))
            for group in groups.values()
            if len(group) >= self.minimum_children
        ]
        if not candidates:
            return ()
        group = min(candidates, key=lambda items: (-len(items), items[0].node_id))
        x1 = max(0.0, min(item.box_xyxy_norm[0] for item in group) - self.margin)
        y1 = max(0.0, min(item.box_xyxy_norm[1] for item in group) - self.margin)
        x2 = min(1.0, max(item.box_xyxy_norm[2] for item in group) + self.margin)
        y2 = min(1.0, max(item.box_xyxy_norm[3] for item in group) + self.margin)
        box: Box = (x1, y1, x2, y2)
        for existing in detections:
            if (
                existing.node_type is NodeType.CONTAINER
                and _coverage(box, existing.box_xyxy_norm) > 0.9
            ):
                return ()
        template = group[0]
        return (
            UiDetection(
                node_id=max(item.node_id for item in detections) + 1,
                original_class_id=-2,
                original_class_name="Generated aligned-child container",
                confidence=self.confidence,
                box_xyxy_norm=box,
                node_type=NodeType.CONTAINER,
                semantic_roles=("container",),
                original_class_feature=(),
                annotation_source="reconstruction_policy",
                policy_version=self.policy_version,
                annotation_confidence=self.confidence,
                preprocessing=template.preprocessing,
                metadata={
                    "child_node_ids": ",".join(str(item.node_id) for item in group),
                    "hierarchy_reliability": "heuristic_container_only",
                },
            ),
        )


def _coverage(container: Box, child: Box) -> float:
    x1 = max(container[0], child[0])
    y1 = max(container[1], child[1])
    x2 = min(container[2], child[2])
    y2 = min(container[3], child[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = max(0.0, child[2] - child[0]) * max(0.0, child[3] - child[1])
    return intersection / area if area else 0.0


class ScreenParserUiDetector:
    """Batched typed adapter preserving all original detector semantics."""

    def __init__(
        self,
        backend: ScreenParserBackend,
        *,
        config: ScreenParserConfig | None = None,
        container_policy: ContainerAugmentationPolicy | None = None,
    ) -> None:
        self.backend = backend
        self.config = config or ScreenParserConfig()
        self.container_policy = container_policy or ContainerAugmentationPolicy()

    def _preprocessing(self, frame: ImageFrame) -> PreprocessingMetadata:
        scale = min(self.config.input_size / frame.width, self.config.input_size / frame.height)
        resized_width = round(frame.width * scale)
        resized_height = round(frame.height * scale)
        return PreprocessingMetadata(
            original_width=frame.width,
            original_height=frame.height,
            input_size=self.config.input_size,
            scale=scale,
            pad_left=(self.config.input_size - resized_width) / 2.0,
            pad_top=(self.config.input_size - resized_height) / 2.0,
            resized_width=resized_width,
            resized_height=resized_height,
            policy_version="screenparser_ultralytics_letterbox_v1",
        )

    def detect(self, images: Sequence[object]) -> tuple[UiDetectionFrame, ...]:
        frames = normalize_images(images)
        raw_frames = tuple(self.backend.predict(frames, self.config))
        if len(raw_frames) != len(frames):
            raise RuntimeError("ScreenParser backend result count does not match input count")
        output: list[UiDetectionFrame] = []
        for frame, raw in zip(frames, raw_frames, strict=True):
            preprocessing = self._preprocessing(frame)
            root = UiDetection(
                node_id=0,
                original_class_id=-1,
                original_class_name="Synthetic full-screen root",
                confidence=1.0,
                box_xyxy_norm=(0.0, 0.0, 1.0, 1.0),
                node_type=NodeType.ROOT,
                semantic_roles=("container", "root"),
                original_class_feature=(),
                annotation_source="screen2action_root_policy",
                policy_version="full_screen_root_v1",
                annotation_confidence=1.0,
                preprocessing=preprocessing,
                metadata={"hierarchy_reliability": "root_only"},
            )
            mapped: list[UiDetection] = [root]
            ordered = sorted(
                raw.detections,
                key=lambda item: (
                    item.box_xyxy_px[1],
                    item.box_xyxy_px[0],
                    item.class_id,
                    -item.confidence,
                ),
            )
            for node_id, detection in enumerate(ordered, start=1):
                if detection.class_id < 0 or detection.class_id >= len(SCREENPARSER_CLASSES):
                    raise ValueError(f"ScreenParser class ID out of range: {detection.class_id}")
                expected = SCREENPARSER_CLASSES[detection.class_id]
                if detection.class_name != expected:
                    raise ValueError(
                        f"ScreenParser class-name mismatch for {detection.class_id}: "
                        f"expected {expected!r}, got {detection.class_name!r}"
                    )
                mapped.append(
                    UiDetection(
                        node_id=node_id,
                        original_class_id=detection.class_id,
                        original_class_name=detection.class_name,
                        confidence=detection.confidence,
                        box_xyxy_norm=normalize_pixel_box(
                            detection.box_xyxy_px,
                            width=frame.width,
                            height=frame.height,
                        ),
                        node_type=coarse_node_type(detection.class_name),
                        semantic_roles=semantic_roles(detection.class_name),
                        original_class_feature=detection.feature,
                        annotation_source="screenparser",
                        policy_version=SCREENPARSER_CLASS_MAPPING_VERSION,
                        annotation_confidence=detection.confidence,
                        preprocessing=preprocessing,
                        metadata={
                            "hierarchy_reliability": "leaf_detection_only",
                            "original_class_id": str(detection.class_id),
                            "original_class_name": detection.class_name,
                        },
                    )
                )
            mapped.extend(self.container_policy.augment(mapped))
            output.append(
                UiDetectionFrame(
                    image=frame,
                    detections=tuple(mapped),
                    raw_proposal_count=raw.raw_proposal_count,
                    post_nms_count=raw.post_nms_count,
                    raw_diagnostics_supported=raw.raw_diagnostics_supported,
                    class_mapping_version=SCREENPARSER_CLASS_MAPPING_VERSION,
                )
            )
        return tuple(output)
