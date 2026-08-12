"""Deterministic structured semantic bottleneck primitives."""

from screen2action.ssb.geometry import (
    box_area,
    box_center,
    clip_box,
    dequantize_coordinate,
    intersection_box,
    iou,
    point_in_box,
    quantize_coordinate,
)

__all__ = [
    "box_area",
    "box_center",
    "clip_box",
    "dequantize_coordinate",
    "intersection_box",
    "iou",
    "point_in_box",
    "quantize_coordinate",
]
