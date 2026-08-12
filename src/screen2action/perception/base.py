"""Typed production perception protocols and one normalized image boundary."""

from __future__ import annotations

import hashlib
import importlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch

from screen2action.data.schema import Box, NodeType


@dataclass(frozen=True, slots=True)
class ImageFrame:
    """Canonical RGB uint8 screenshot and secret-safe source metadata."""

    pixels: torch.Tensor
    width: int
    height: int
    sha256: str
    source_kind: str

    def __post_init__(self) -> None:
        if self.pixels.shape != (3, self.height, self.width):
            raise ValueError("pixels must have shape [3, height, width]")
        if self.pixels.dtype is not torch.uint8 or self.pixels.device.type != "cpu":
            raise ValueError("canonical image pixels must be CPU uint8")
        if len(self.sha256) != 64:
            raise ValueError("image sha256 must have 64 hexadecimal characters")


@dataclass(frozen=True, slots=True)
class PreprocessingMetadata:
    """Detector resize/letterbox evidence for coordinate auditing."""

    original_width: int
    original_height: int
    input_size: int
    scale: float
    pad_left: float
    pad_top: float
    resized_width: int
    resized_height: int
    policy_version: str

    def __post_init__(self) -> None:
        if (
            min(
                self.original_width,
                self.original_height,
                self.input_size,
                self.resized_width,
                self.resized_height,
            )
            <= 0
        ):
            raise ValueError("preprocessing dimensions must be positive")
        if self.scale <= 0.0 or not math.isfinite(self.scale):
            raise ValueError("preprocessing scale must be finite and positive")
        if self.pad_left < 0.0 or self.pad_top < 0.0:
            raise ValueError("preprocessing padding cannot be negative")
        if not self.policy_version:
            raise ValueError("preprocessing policy version is required")


@dataclass(frozen=True, slots=True)
class RawDetection:
    """Backend detection in original-image pixel `xyxy`."""

    class_id: int
    class_name: str
    confidence: float
    box_xyxy_px: tuple[float, float, float, float]
    feature: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.class_id < 0 or not self.class_name:
            raise ValueError("raw detection class ID and name are required")
        if not 0.0 <= self.confidence <= 1.0 or not math.isfinite(self.confidence):
            raise ValueError("raw detection confidence must be finite and in [0, 1]")
        if len(self.box_xyxy_px) != 4 or any(
            not math.isfinite(value) for value in self.box_xyxy_px
        ):
            raise ValueError("raw detection box must contain four finite values")
        x1, y1, x2, y2 = self.box_xyxy_px
        if x1 > x2 or y1 > y2:
            raise ValueError("raw detection box must be ordered xyxy")
        if any(not math.isfinite(value) for value in self.feature):
            raise ValueError("raw detection features must be finite")


@dataclass(frozen=True, slots=True)
class RawDetectionFrame:
    """One backend result with proposal diagnostics."""

    detections: tuple[RawDetection, ...]
    raw_proposal_count: int | None
    post_nms_count: int
    raw_diagnostics_supported: bool

    def __post_init__(self) -> None:
        if self.post_nms_count < 0 or self.post_nms_count != len(self.detections):
            raise ValueError("post-NMS count must equal the detection count")
        if self.raw_proposal_count is not None:
            if self.raw_proposal_count < self.post_nms_count:
                raise ValueError("raw proposal count cannot be below post-NMS count")


@dataclass(frozen=True, slots=True)
class UiDetection:
    """Mapped detector proposal preserving original and coarse semantics."""

    node_id: int
    original_class_id: int
    original_class_name: str
    confidence: float
    box_xyxy_norm: Box
    node_type: NodeType
    semantic_roles: tuple[str, ...]
    original_class_feature: tuple[float, ...]
    annotation_source: str
    policy_version: str
    annotation_confidence: float
    preprocessing: PreprocessingMetadata
    metadata: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.node_id < 0 or not self.original_class_name:
            raise ValueError("mapped detection ID and class name are required")
        if not 0.0 <= self.confidence <= 1.0 or not math.isfinite(self.confidence):
            raise ValueError("mapped confidence must be finite and in [0, 1]")
        if not 0.0 <= self.annotation_confidence <= 1.0 or not math.isfinite(
            self.annotation_confidence
        ):
            raise ValueError("annotation confidence must be finite and in [0, 1]")
        if not self.annotation_source or not self.policy_version:
            raise ValueError("annotation source and policy version are required")
        if len(self.box_xyxy_norm) != 4:
            raise ValueError("mapped box must contain four values")
        x1, y1, x2, y2 = self.box_xyxy_norm
        if (
            x1 > x2
            or y1 > y2
            or any(
                not 0.0 <= value <= 1.0 or not math.isfinite(value) for value in self.box_xyxy_norm
            )
        ):
            raise ValueError("mapped box must be normalized ordered xyxy")
        if any(not math.isfinite(value) for value in self.original_class_feature):
            raise ValueError("mapped class features must be finite")
        if not all(
            isinstance(key, str) and isinstance(value, str) for key, value in self.metadata.items()
        ):
            raise ValueError("mapped metadata must be string-to-string")


@dataclass(frozen=True, slots=True)
class UiDetectionFrame:
    """Detections and auditable pre/post-NMS counts for one screenshot."""

    image: ImageFrame
    detections: tuple[UiDetection, ...]
    raw_proposal_count: int | None
    post_nms_count: int
    raw_diagnostics_supported: bool
    class_mapping_version: str


class UiDetector(Protocol):
    """Batched UI detector boundary."""

    def detect(self, images: Sequence[object]) -> tuple[UiDetectionFrame, ...]:
        """Normalize images and return mapped detections with one root each."""


@dataclass(frozen=True, slots=True)
class OcrCrop:
    """Normalized recognition crop with screen transform metadata."""

    request_id: str
    node_id: int
    pixels: torch.Tensor
    screen_box_xyxy_norm: Box
    pixel_box_xyxy: tuple[int, int, int, int]
    original_aspect_ratio: float
    normalization_policy_version: str
    status: str = "ready"


@dataclass(frozen=True, slots=True)
class OcrResult:
    """Recognition output aligned one-to-one with input requests."""

    request_id: str
    node_id: int
    text: str
    confidence: float
    screen_box_xyxy_norm: Box
    status: str
    unknown_characters: tuple[str, ...] = ()
    token_diagnostics: tuple[str, ...] = ()


class TextRecognizer(Protocol):
    """Recognition-only OCR boundary."""

    def recognize(
        self,
        image: ImageFrame,
        detections: Sequence[UiDetection],
    ) -> tuple[OcrResult, ...]:
        """Recognize configured text-like regions without changing order."""


class TextRegionDetector(Protocol):
    """Optional future DBNet-like region detector, disabled by default."""

    def detect_regions(self, images: Sequence[ImageFrame]) -> Sequence[Sequence[Box]]:
        """Return text boxes for each image."""


def _from_pil(image: object) -> torch.Tensor:
    converted = cast(Any, image).convert("RGB")
    width, height = converted.size
    buffer = bytearray(converted.tobytes())
    return torch.frombuffer(buffer, dtype=torch.uint8).clone().reshape(height, width, 3)


def _load_path(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"image does not exist: {path}")
    try:
        image_module = importlib.import_module("PIL.Image")
    except ImportError as error:
        raise RuntimeError("Pillow is required for image paths") from error
    with image_module.open(path) as image:
        return _from_pil(image)


def _to_tensor(value: object) -> tuple[torch.Tensor, str]:
    if isinstance(value, torch.Tensor):
        return value.detach(), "tensor"
    if isinstance(value, str | Path):
        return _load_path(Path(value)), "path"
    module = type(value).__module__.split(".", 1)[0]
    if module == "PIL" and hasattr(value, "convert"):
        return _from_pil(value), "pil"
    if module == "numpy" and hasattr(value, "shape"):
        return torch.as_tensor(value), "numpy"
    raise TypeError("image must be a tensor, PIL image, NumPy array, or path")


def normalize_image(value: object) -> ImageFrame:
    """Normalize all supported image inputs to contiguous CPU RGB uint8 CHW."""

    if isinstance(value, ImageFrame):
        return value
    tensor, source_kind = _to_tensor(value)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(-1)
    if tensor.ndim != 3:
        raise ValueError("image must have two or three dimensions")
    if source_kind == "tensor" and tensor.shape[0] in {1, 3, 4}:
        tensor = tensor.permute(1, 2, 0)
    elif tensor.shape[-1] not in {1, 3, 4}:
        raise ValueError("image must have one, three, or four channels")
    if tensor.shape[-1] == 1:
        tensor = tensor.expand(-1, -1, 3)
    elif tensor.shape[-1] == 4:
        tensor = tensor[..., :3]
    if tensor.shape[0] <= 0 or tensor.shape[1] <= 0:
        raise ValueError("image dimensions must be positive")
    if tensor.is_floating_point():
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("image contains non-finite values")
        maximum = float(tensor.max()) if tensor.numel() else 0.0
        minimum = float(tensor.min()) if tensor.numel() else 0.0
        if minimum < 0.0:
            raise ValueError("floating image values cannot be negative")
        tensor = tensor * 255.0 if maximum <= 1.0 else tensor
        tensor = tensor.round().clamp(0, 255).to(torch.uint8)
    else:
        tensor = tensor.clamp(0, 255).to(torch.uint8)
    pixels = tensor.permute(2, 0, 1).contiguous().cpu()
    height, width = int(pixels.shape[1]), int(pixels.shape[2])
    offset = pixels.storage_offset()
    storage = bytes(pixels.untyped_storage())
    digest = hashlib.sha256(storage[offset : offset + pixels.numel()]).hexdigest()
    return ImageFrame(pixels, width, height, digest, source_kind)


def normalize_images(values: Sequence[object]) -> tuple[ImageFrame, ...]:
    """Normalize a non-empty sequence while preserving caller order."""

    if not values:
        raise ValueError("at least one image is required")
    return tuple(normalize_image(value) for value in values)


def normalize_pixel_box(
    box: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
) -> Box:
    """Validate, clip, and convert original-image pixel `xyxy` to normalized coordinates."""

    if width <= 0 or height <= 0 or any(not math.isfinite(value) for value in box):
        raise ValueError("pixel box and image dimensions must be finite and positive")
    x1, y1, x2, y2 = box
    if x1 > x2 or y1 > y2:
        raise ValueError("pixel box must be ordered xyxy")
    return (
        min(max(x1 / width, 0.0), 1.0),
        min(max(y1 / height, 0.0), 1.0),
        min(max(x2 / width, 0.0), 1.0),
        min(max(y2 / height, 0.0), 1.0),
    )
