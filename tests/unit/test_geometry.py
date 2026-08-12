from __future__ import annotations

import pytest

from screen2action.ssb.geometry import (
    box_area,
    clip_box,
    dequantize_box,
    intersection_box,
    iou,
    point_in_box,
    quantize_box,
    quantize_coordinate,
)


def test_geometry_intersection_iou_and_point_membership() -> None:
    first = (0.1, 0.1, 0.5, 0.5)
    second = (0.3, 0.3, 0.7, 0.7)
    assert box_area(first) == pytest.approx(0.16)
    assert intersection_box(first, second) == (0.3, 0.3, 0.5, 0.5)
    assert iou(first, second) == pytest.approx(0.04 / 0.28)
    assert point_in_box((0.3, 0.3), first)
    assert not point_in_box((0.1, 0.1), first, inclusive=False)


def test_clip_and_quantization_follow_ten_bit_contract() -> None:
    assert clip_box((-0.2, 0.2, 1.2, 0.8)) == (0.0, 0.2, 1.0, 0.8)
    assert quantize_coordinate(-1.0) == 0
    assert quantize_coordinate(2.0) == 1023
    quantized = quantize_box((0.0, 0.25, 0.5, 1.0))
    assert quantized == (0, 256, 512, 1023)
    restored = dequantize_box(quantized)
    assert restored[0] == 0.0
    assert restored[3] == 1.0
    assert (
        max(abs(left - right) for left, right in zip(restored, (0.0, 0.25, 0.5, 1.0), strict=True))
        <= 1 / 1023
    )


def test_invalid_clip_interval_is_rejected() -> None:
    with pytest.raises(ValueError, match="clip interval"):
        clip_box((0.1, 0.1, 0.2, 0.2), minimum=1.0, maximum=0.0)
