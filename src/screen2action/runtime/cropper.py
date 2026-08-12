"""Candidate crop extraction and normalized coordinate transforms."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from screen2action.data.schema import Box, Point
from screen2action.ssb.geometry import clip_box


@dataclass(frozen=True, slots=True)
class CropSpec:
    """Expanded normalized box and resized crop for one candidate."""

    source_box_xyxy_norm: Box
    expanded_box_xyxy_norm: Box
    pixel_box_xyxy: tuple[int, int, int, int]
    crop: torch.Tensor


def expand_box(box: Box, margin_fraction: float = 0.12) -> Box:
    """Expand each side by a fraction of the candidate width/height and clip."""

    if margin_fraction < 0.0 or not math.isfinite(margin_fraction):
        raise ValueError("margin_fraction must be finite and non-negative")
    width = box[2] - box[0]
    height = box[3] - box[1]
    return clip_box(
        (
            box[0] - width * margin_fraction,
            box[1] - height * margin_fraction,
            box[2] + width * margin_fraction,
            box[3] + height * margin_fraction,
        )
    )


def _pixel_bounds(box: Box, height: int, width: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(width - 1, math.floor(box[0] * width)))
    y1 = max(0, min(height - 1, math.floor(box[1] * height)))
    x2 = max(x1 + 1, min(width, math.ceil(box[2] * width)))
    y2 = max(y1 + 1, min(height, math.ceil(box[3] * height)))
    return x1, y1, x2, y2


def crop_tensor(
    screenshot: torch.Tensor,
    box: Box,
    *,
    output_size: int = 96,
    margin_fraction: float = 0.12,
) -> CropSpec:
    """Crop an expanded box from `[C,H,W]` pixels and resize it on CPU/GPU."""

    if screenshot.ndim != 3 or screenshot.shape[0] not in {1, 3}:
        raise ValueError("screenshot must have shape [C, H, W] with one or three channels")
    if output_size <= 0:
        raise ValueError("output_size must be positive")
    expanded = expand_box(box, margin_fraction=margin_fraction)
    height, width = int(screenshot.shape[-2]), int(screenshot.shape[-1])
    x1, y1, x2, y2 = _pixel_bounds(expanded, height, width)
    crop = screenshot[:, y1:y2, x1:x2]
    resized = F.interpolate(
        crop.unsqueeze(0).float(),
        size=(output_size, output_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return CropSpec(box, expanded, (x1, y1, x2, y2), resized)


def screen_to_crop_point(point: Point, expanded_box: Box) -> Point:
    """Map a normalized screen point to normalized coordinates in an expanded crop."""

    width = expanded_box[2] - expanded_box[0]
    height = expanded_box[3] - expanded_box[1]
    if width <= 0.0 or height <= 0.0:
        raise ValueError("expanded crop must have positive area")
    return (
        min(max((point[0] - expanded_box[0]) / width, 0.0), 1.0),
        min(max((point[1] - expanded_box[1]) / height, 0.0), 1.0),
    )


def crop_to_screen_point(point: Point, expanded_box: Box) -> Point:
    """Map normalized crop-local coordinates back to normalized screen space."""

    if not 0.0 <= point[0] <= 1.0 or not 0.0 <= point[1] <= 1.0:
        raise ValueError("crop-local point must be in [0, 1]")
    return (
        expanded_box[0] + point[0] * (expanded_box[2] - expanded_box[0]),
        expanded_box[1] + point[1] * (expanded_box[3] - expanded_box[1]),
    )


def batch_crop_tensor(
    screenshot: torch.Tensor,
    boxes: tuple[Box, ...] | list[Box],
    *,
    output_size: int = 96,
    margin_fraction: float = 0.12,
) -> tuple[torch.Tensor, tuple[CropSpec, ...]]:
    """Extract a fixed-size batch while preserving crop metadata."""

    specs = tuple(
        crop_tensor(
            screenshot,
            box,
            output_size=output_size,
            margin_fraction=margin_fraction,
        )
        for box in boxes
    )
    if not specs:
        return torch.empty((0, screenshot.shape[0], output_size, output_size)), specs
    return torch.stack([spec.crop for spec in specs]), specs
