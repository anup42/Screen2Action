"""ScreenSpot evaluation-only validation and official point-in-box scoring."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from screen2action.data.schema import Box, Point
from screen2action.ssb.geometry import point_in_box


def official_point_in_target(point: Point | None, target_box: Box) -> bool:
    """Apply ScreenSpot's point-grounding rule in normalized coordinates."""

    return point is not None and point_in_box(point, target_box, inclusive=True)


def require_evaluation_only_screenspot(
    screens: Sequence[Mapping[str, object]],
    *,
    split: str,
) -> None:
    """Reject any attempt to use ScreenSpot outside its frozen test split."""

    if split != "test":
        raise ValueError("ScreenSpot is evaluation-only and must use split=test")
    if not screens:
        raise ValueError("ScreenSpot evaluation selected no screens")
    for screen in screens:
        if str(screen.get("source_dataset", "")) != "screenspot":
            raise ValueError("ScreenSpot evaluation contains a non-ScreenSpot screen")
        if str(screen.get("split", "")) != "test":
            raise ValueError("ScreenSpot leakage guard found a non-test canonical row")


def target_type_for_screen(
    elements: Mapping[str, Mapping[str, object]],
    target_box: Box,
) -> str:
    """Recover the benchmark target type from the best-overlapping element row."""

    if not elements:
        return "unknown"

    def overlap(row: Mapping[str, object]) -> float:
        box = tuple(float(str(row[key])) for key in ("x1", "y1", "x2", "y2"))
        left = max(target_box[0], box[0])
        top = max(target_box[1], box[1])
        right = min(target_box[2], box[2])
        bottom = min(target_box[3], box[3])
        return max(0.0, right - left) * max(0.0, bottom - top)

    best = max(elements.values(), key=lambda row: (overlap(row), str(row["element_id"])))
    return str(best.get("node_type", "unknown"))
