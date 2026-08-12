"""Recognition-only docTR CRNN adapter with deterministic crop normalization."""

from __future__ import annotations

import importlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from torch.nn import functional as F

from screen2action.perception.base import ImageFrame, OcrCrop, OcrResult, UiDetection

OCR_CROP_POLICY_VERSION = "doctr_crnn_rgb_h32_aspect_v1"
DEFAULT_OCR_SOURCE_CLASSES = frozenset(
    {"Text", "Heading", "Text Input", "Search Field", "Search Bar", "Button", "Link"}
)


@dataclass(frozen=True, slots=True)
class RawRecognition:
    """Backend text/confidence plus optional token-level diagnostics."""

    text: str
    confidence: float
    tokens: tuple[str, ...] = ()


class RecognitionBackend(Protocol):
    """Injectable batched recognition backend."""

    @property
    def vocabulary(self) -> str:
        """Return the model vocabulary when available."""

    def recognize(self, crops: Sequence[torch.Tensor]) -> Sequence[RawRecognition]:
        """Recognize normalized RGB crops in caller order."""


class DoctrCrnnBackend:
    """Lazy docTR predictor loaded from a locally locked pretrained state dict."""

    def __init__(
        self,
        weight_path: str | Path,
        *,
        device: str = "cpu",
        batch_size: int = 128,
    ) -> None:
        path = Path(weight_path)
        if not path.is_file():
            raise FileNotFoundError(f"locked docTR CRNN weight does not exist: {path}")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        target = torch.device(device)
        if target.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA OCR requested but torch.cuda.is_available() is false")
        self.weight_path = path
        self.device = target
        self.batch_size = batch_size
        self._predictor: object | None = None
        self._vocabulary = ""

    def _load(self) -> object:
        if self._predictor is None:
            try:
                models = importlib.import_module("doctr.models")
            except ImportError as error:
                raise RuntimeError("install the perception extra for docTR OCR") from error
            predictor = models.recognition_predictor(
                arch="crnn_vgg16_bn",
                pretrained=False,
                batch_size=self.batch_size,
            )
            state = torch.load(self.weight_path, map_location=self.device, weights_only=True)
            predictor.model.load_state_dict(state)
            predictor.model.to(self.device)
            predictor.model.eval()
            config = getattr(predictor.model, "cfg", {})
            self._vocabulary = str(config.get("vocab", "")) if isinstance(config, dict) else ""
            self._predictor = predictor
        return self._predictor

    @property
    def vocabulary(self) -> str:
        self._load()
        return self._vocabulary

    def recognize(self, crops: Sequence[torch.Tensor]) -> Sequence[RawRecognition]:
        predictor = self._load()
        arrays = [crop.permute(1, 2, 0).contiguous().cpu().numpy() for crop in crops]
        raw = predictor(arrays)  # type: ignore[operator]
        results: list[RawRecognition] = []
        for item in raw:
            if not isinstance(item, tuple | list) or len(item) < 2:
                raise RuntimeError("unexpected docTR recognition output")
            results.append(RawRecognition(str(item[0]), float(item[1])))
        return tuple(results)


def _pixel_bounds(
    detection: UiDetection,
    image: ImageFrame,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = detection.box_xyxy_norm
    left = max(0, min(image.width, math.floor(x1 * image.width)))
    top = max(0, min(image.height, math.floor(y1 * image.height)))
    right = max(0, min(image.width, math.ceil(x2 * image.width)))
    bottom = max(0, min(image.height, math.ceil(y2 * image.height)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def prepare_ocr_crop(
    image: ImageFrame,
    detection: UiDetection,
    *,
    target_height: int = 32,
    minimum_width: int = 8,
    maximum_width: int = 512,
) -> OcrCrop:
    """Crop and resize one region while preserving its screen transform."""

    if target_height <= 0 or minimum_width <= 0 or maximum_width < minimum_width:
        raise ValueError("invalid OCR normalization dimensions")
    request_id = f"{image.sha256}:{detection.node_id}"
    bounds = _pixel_bounds(detection, image)
    if bounds is None:
        return OcrCrop(
            request_id=request_id,
            node_id=detection.node_id,
            pixels=torch.empty((3, target_height, minimum_width), dtype=torch.uint8),
            screen_box_xyxy_norm=detection.box_xyxy_norm,
            pixel_box_xyxy=(0, 0, 0, 0),
            original_aspect_ratio=0.0,
            normalization_policy_version=OCR_CROP_POLICY_VERSION,
            status="degenerate_box",
        )
    left, top, right, bottom = bounds
    width = right - left
    height = bottom - top
    aspect = width / height
    target_width = max(minimum_width, min(maximum_width, round(target_height * aspect)))
    crop = image.pixels[:, top:bottom, left:right]
    resized = F.interpolate(
        crop.unsqueeze(0).float(),
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    status = "elongated_clamped" if round(target_height * aspect) > maximum_width else "ready"
    return OcrCrop(
        request_id=request_id,
        node_id=detection.node_id,
        pixels=resized.round().clamp(0, 255).to(torch.uint8),
        screen_box_xyxy_norm=detection.box_xyxy_norm,
        pixel_box_xyxy=bounds,
        original_aspect_ratio=aspect,
        normalization_policy_version=OCR_CROP_POLICY_VERSION,
        status=status,
    )


class DoctrCrnnVgg16Recognizer:
    """Aspect-batched OCR adapter preserving input/output alignment."""

    def __init__(
        self,
        backend: RecognitionBackend,
        *,
        source_classes: frozenset[str] = DEFAULT_OCR_SOURCE_CLASSES,
        aspect_batch_size: int = 64,
    ) -> None:
        if aspect_batch_size <= 0:
            raise ValueError("aspect_batch_size must be positive")
        self.backend = backend
        self.source_classes = source_classes
        self.aspect_batch_size = aspect_batch_size

    def recognize(
        self,
        image: ImageFrame,
        detections: Sequence[UiDetection],
    ) -> tuple[OcrResult, ...]:
        selected = [
            detection
            for detection in detections
            if detection.original_class_name in self.source_classes
        ]
        crops = [prepare_ocr_crop(image, detection) for detection in selected]
        results: list[OcrResult | None] = [None] * len(crops)
        ready_indices = [
            index for index, crop in enumerate(crops) if crop.status != "degenerate_box"
        ]
        ready_indices.sort(key=lambda index: (crops[index].original_aspect_ratio, index))
        vocabulary = set(self.backend.vocabulary)
        for offset in range(0, len(ready_indices), self.aspect_batch_size):
            indices = ready_indices[offset : offset + self.aspect_batch_size]
            recognized = tuple(self.backend.recognize([crops[index].pixels for index in indices]))
            if len(recognized) != len(indices):
                raise RuntimeError("OCR backend result count does not match crop count")
            for index, raw in zip(indices, recognized, strict=True):
                if not 0.0 <= raw.confidence <= 1.0 or not math.isfinite(raw.confidence):
                    raise ValueError("OCR confidence must be finite and in [0, 1]")
                crop = crops[index]
                unknown = tuple(
                    sorted(
                        {
                            character
                            for character in raw.text
                            if vocabulary and character not in vocabulary
                        }
                    )
                )
                results[index] = OcrResult(
                    request_id=crop.request_id,
                    node_id=crop.node_id,
                    text=raw.text,
                    confidence=raw.confidence,
                    screen_box_xyxy_norm=crop.screen_box_xyxy_norm,
                    status=crop.status,
                    unknown_characters=unknown,
                    token_diagnostics=raw.tokens,
                )
        for index, crop in enumerate(crops):
            if results[index] is None:
                results[index] = OcrResult(
                    request_id=crop.request_id,
                    node_id=crop.node_id,
                    text="",
                    confidence=0.0,
                    screen_box_xyxy_norm=crop.screen_box_xyxy_norm,
                    status=crop.status,
                )
        return tuple(result for result in results if result is not None)
