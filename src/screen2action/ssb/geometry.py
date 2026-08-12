"""Normalized box geometry and coordinate quantization."""

from __future__ import annotations

import math

from screen2action.data.schema import Box, Point


def _validate_finite_box(box: Box) -> None:
    if len(box) != 4 or any(not math.isfinite(value) for value in box):
        raise ValueError(f"box must contain four finite values, got {box!r}")
    x1, y1, x2, y2 = box
    if x1 > x2 or y1 > y2:
        raise ValueError(f"box must be ordered as xyxy, got {box!r}")


def box_area(box: Box) -> float:
    """Return the non-negative area of a normalized `xyxy` box."""

    _validate_finite_box(box)
    return (box[2] - box[0]) * (box[3] - box[1])


def box_center(box: Box) -> Point:
    """Return the center point of a normalized box."""

    _validate_finite_box(box)
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def clip_box(box: Box, minimum: float = 0.0, maximum: float = 1.0) -> Box:
    """Clip an ordered box to an inclusive coordinate interval."""

    _validate_finite_box(box)
    if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum > maximum:
        raise ValueError("clip interval must be finite and ordered")
    clipped = tuple(min(max(value, minimum), maximum) for value in box)
    return clipped  # type: ignore[return-value]


def intersection_box(first: Box, second: Box) -> Box:
    """Return the intersection box, which may have zero area."""

    _validate_finite_box(first)
    _validate_finite_box(second)
    return (
        max(first[0], second[0]),
        max(first[1], second[1]),
        min(first[2], second[2]),
        min(first[3], second[3]),
    )


def iou(first: Box, second: Box) -> float:
    """Return intersection-over-union for two normalized boxes."""

    overlap = (
        box_area(intersection_box(first, second))
        if (
            max(first[0], second[0]) <= min(first[2], second[2])
            and max(first[1], second[1]) <= min(first[3], second[3])
        )
        else 0.0
    )
    union = box_area(first) + box_area(second) - overlap
    if union == 0.0:
        return 1.0 if first == second else 0.0
    return overlap / union


def point_in_box(point: Point, box: Box, *, inclusive: bool = True) -> bool:
    """Return whether a normalized point is inside a normalized box."""

    if len(point) != 2 or any(not math.isfinite(value) for value in point):
        raise ValueError(f"point must contain two finite values, got {point!r}")
    _validate_finite_box(box)
    x, y = point
    if inclusive:
        return box[0] <= x <= box[2] and box[1] <= y <= box[3]
    return box[0] < x < box[2] and box[1] < y < box[3]


def quantize_coordinate(value: float, levels: int = 1023) -> int:
    """Quantize a normalized coordinate using the paper's 10-bit rule."""

    if not math.isfinite(value):
        raise ValueError("coordinate must be finite")
    if levels <= 0:
        raise ValueError("levels must be positive")
    return int(round(min(max(value, 0.0), 1.0) * levels))


def dequantize_coordinate(value: int, levels: int = 1023) -> float:
    """Map an integer coordinate back to the normalized interval."""

    if levels <= 0:
        raise ValueError("levels must be positive")
    if not 0 <= value <= levels:
        raise ValueError(f"quantized coordinate must be in [0, {levels}]")
    return value / levels


def quantize_box(box: Box, levels: int = 1023) -> tuple[int, int, int, int]:
    """Quantize all four box coordinates."""

    _validate_finite_box(box)
    return tuple(quantize_coordinate(value, levels) for value in box)  # type: ignore[return-value]


def dequantize_box(values: tuple[int, int, int, int], levels: int = 1023) -> Box:
    """Dequantize an integer `xyxy` box."""

    if len(values) != 4:
        raise ValueError("quantized box must contain four values")
    box = tuple(dequantize_coordinate(value, levels) for value in values)
    _validate_finite_box(box)  # type: ignore[arg-type]
    return box  # type: ignore[return-value]
