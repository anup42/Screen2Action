"""Direct cascade metrics; no independence multiplication."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from screen2action.data.schema import Box, NodeRecord, Point
from screen2action.ssb.geometry import iou, point_in_box


def point_in_target_accuracy(points: Iterable[Point], targets: Iterable[Box]) -> float:
    """Compute direct point-in-target accuracy."""

    pairs = list(zip(points, targets, strict=True))
    if not pairs:
        return 0.0
    return sum(point_in_box(point, target) for point, target in pairs) / len(pairs)


def proposal_recall(
    target_boxes: Iterable[Box],
    proposals: Iterable[Iterable[NodeRecord]],
    *,
    iou_threshold: float = 0.5,
) -> float:
    """Measure whether each target has at least one matched proposal."""

    pairs = list(zip(target_boxes, proposals, strict=True))
    if not pairs:
        return 0.0
    return sum(
        any(iou(target, node.box_xyxy_norm) >= iou_threshold for node in nodes)
        for target, nodes in pairs
    ) / len(pairs)


def target_survival_rate(
    matched_proposal_ids: Iterable[int | None],
    selected_node_ids: Iterable[Iterable[int]],
) -> float:
    """Measure survival conditional on a matched proposal."""

    pairs = [
        (matched, set(selected))
        for matched, selected in zip(matched_proposal_ids, selected_node_ids, strict=True)
        if matched is not None
    ]
    if not pairs:
        return 0.0
    return sum(matched in selected for matched, selected in pairs) / len(pairs)


def retrieval_recall(
    target_node_ids: Iterable[int],
    candidate_node_ids: Iterable[Iterable[int]],
    *,
    k: int | None = None,
) -> float:
    """Compute R@K directly over candidate lists."""

    pairs = list(zip(target_node_ids, candidate_node_ids, strict=True))
    if not pairs:
        return 0.0
    return sum(
        target in tuple(candidates)[:k] if k is not None else target in candidates
        for target, candidates in pairs
    ) / len(pairs)


def selection_accuracy_when_retrieved(
    target_node_ids: Iterable[int],
    candidate_node_ids: Iterable[Iterable[int]],
    selected_node_ids: Iterable[int],
) -> float:
    """Measure candidate selection only on examples whose target was retrieved."""

    pairs = list(zip(target_node_ids, candidate_node_ids, selected_node_ids, strict=True))
    eligible = [
        (target, selected) for target, candidates, selected in pairs if target in candidates
    ]
    if not eligible:
        return 0.0
    return sum(target == selected for target, selected in eligible) / len(eligible)


def point_accuracy_given_candidate(
    points: Iterable[Point],
    targets: Iterable[Box],
    candidate_correct: Iterable[bool],
) -> float:
    """Measure point-in-target accuracy conditional on a correct candidate."""

    pairs = [
        (point, target)
        for point, target, correct in zip(points, targets, candidate_correct, strict=True)
        if correct
    ]
    if not pairs:
        return 0.0
    return sum(point_in_box(point, target) for point, target in pairs) / len(pairs)


def action_parameter_error(
    predicted: Iterable[Iterable[float]],
    targets: Iterable[Iterable[float]],
    masks: Iterable[Iterable[bool]] | None = None,
) -> float:
    """Return masked mean absolute action-parameter error."""

    predicted_rows = [list(row) for row in predicted]
    target_rows = [list(row) for row in targets]
    if masks is None:
        mask_rows = [[True] * len(row) for row in target_rows]
    else:
        mask_rows = [list(row) for row in masks]
    if not (len(predicted_rows) == len(target_rows) == len(mask_rows)):
        raise ValueError("parameter rows and masks must have equal lengths")
    errors: list[float] = []
    for predicted_row, target_row, mask_row in zip(
        predicted_rows, target_rows, mask_rows, strict=True
    ):
        if not (len(predicted_row) == len(target_row) == len(mask_row)):
            raise ValueError("parameter rows and masks must have equal widths")
        errors.extend(
            abs(float(predicted_value) - float(target_value))
            for predicted_value, target_value, valid in zip(
                predicted_row, target_row, mask_row, strict=True
            )
            if valid
        )
    return sum(errors) / len(errors) if errors else 0.0


def action_accuracy(predicted: Iterable[int], targets: Iterable[int]) -> float:
    """Compute action-type accuracy or zero for an empty set."""

    pairs = list(zip(predicted, targets, strict=True))
    return sum(prediction == target for prediction, target in pairs) / len(pairs) if pairs else 0.0


def ssb_statistics(
    lengths: Iterable[int], node_counts: Iterable[int], edge_counts: Iterable[int]
) -> Mapping[str, float]:
    """Return average and p90 token/node/edge statistics."""

    def percentile(values: list[int], fraction: float) -> float:
        if not values:
            return 0.0
        values.sort()
        index = min(len(values) - 1, int(round((len(values) - 1) * fraction)))
        return float(values[index])

    arrays = {
        "ssb_length": list(lengths),
        "node_count": list(node_counts),
        "edge_count": list(edge_counts),
    }
    return {
        f"{name}_mean": sum(values) / len(values) if values else 0.0
        for name, values in arrays.items()
    } | {f"{name}_p90": percentile(values, 0.90) for name, values in arrays.items()}


def stage_metrics(records: Sequence[Mapping[str, object]]) -> dict[str, float]:
    """Aggregate independently measured cascade fields from evaluation records."""

    if not records:
        return {}
    result: dict[str, float] = {}
    for key in (
        "proposal_hit",
        "target_survived",
        "reference_survived",
        "retrieved_at_k",
        "selected_correct",
        "point_correct_given_candidate",
        "point_correct_end_to_end",
    ):
        values = [
            float(value)
            for record in records
            if key in record
            for value in (record[key],)
            if isinstance(value, (int, float))
        ]
        if values:
            result[key] = sum(values) / len(values)
    return result
