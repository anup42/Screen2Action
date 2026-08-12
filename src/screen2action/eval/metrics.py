"""Direct cascade metrics; no independence multiplication."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

import torch

from screen2action.data.schema import Box, NodeRecord, Point
from screen2action.eval.calibration import calibration_metrics, risk_coverage
from screen2action.eval.records import EvaluationRecord
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
    """Return mean, median, and p90 token/node/edge statistics."""

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
    return (
        {
            f"{name}_mean": sum(values) / len(values) if values else 0.0
            for name, values in arrays.items()
        }
        | {f"{name}_median": percentile(values, 0.50) for name, values in arrays.items()}
        | {f"{name}_p90": percentile(values, 0.90) for name, values in arrays.items()}
    )


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


def _mean(values: Iterable[bool | float]) -> float:
    materialized = [float(value) for value in values]
    return sum(materialized) / len(materialized) if materialized else 0.0


def _percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _subset_metrics(records: Sequence[EvaluationRecord], field: str) -> dict[str, object]:
    grouped: dict[str, list[EvaluationRecord]] = {}
    for record in records:
        key = str(getattr(record, field))
        grouped.setdefault(key, []).append(record)
    return {
        key: {
            "examples": len(values),
            "proposal_recall": _mean(value.proposal_hit for value in values),
            "point_in_target_accuracy": _mean(value.point_correct_end_to_end for value in values),
        }
        for key, values in sorted(grouped.items())
    }


def aggregate_evaluation_records(
    records: Sequence[EvaluationRecord],
    *,
    recall_ks: Sequence[int],
    confidence_temperature: float = 1.0,
) -> dict[str, Any]:
    """Aggregate direct cascade observations without multiplying stage rates."""

    if not records:
        raise ValueError("evaluation records cannot be empty")
    if confidence_temperature <= 0.0:
        raise ValueError("confidence temperature must be positive")
    ks = tuple(sorted(set(int(value) for value in recall_ks)))
    if not ks or any(value <= 0 for value in ks):
        raise ValueError("recall_ks must contain positive values")

    matched = [record for record in records if record.target_survived is not None]
    survived = [record for record in records if record.target_survived is True]
    references = [record for record in records if record.reference_survived is not None]
    target_present = [record for record in records if record.selected_correct is not None]
    candidate_correct = [
        record for record in records if record.point_correct_given_candidate is not None
    ]
    action_labeled = [record for record in records if record.action_correct is not None]
    parameter_errors: list[float] = []
    for record in records:
        if record.parameter_absolute_error is not None:
            parameter_errors.append(record.parameter_absolute_error)

    retrieval: dict[str, dict[str, float | int]] = {}
    for k in ks:
        key = str(k)
        retrieval[key] = {
            "end_to_end": _mean(record.retrieval_hits.get(key, False) for record in records),
            "conditional_on_target_survival": _mean(
                record.retrieval_hits.get(key, False) for record in survived
            ),
            "conditional_denominator": len(survived),
        }

    confidence_rows = [
        record
        for record in records
        if record.confidence_logit is not None and record.confidence_probability is not None
    ]
    confidence: dict[str, object] | None = None
    if confidence_rows:
        logits = torch.tensor(
            [float(cast(float, record.confidence_logit)) for record in confidence_rows],
            dtype=torch.float64,
        )
        correct = torch.tensor(
            [record.point_correct_end_to_end for record in confidence_rows],
            dtype=torch.bool,
        )
        summary = calibration_metrics(
            logits,
            correct,
            temperature=confidence_temperature,
        )
        confidence = {
            "examples": len(confidence_rows),
            "temperature": confidence_temperature,
            "nll": summary.nll,
            "ece": summary.ece,
            "brier": summary.brier,
            "risk_coverage": [
                {"coverage": coverage, "risk": risk}
                for coverage, risk in risk_coverage(
                    torch.sigmoid(logits / confidence_temperature),
                    correct,
                )
            ],
        }

    latency_fields = {
        "data": "latency_data_ms",
        "graph_selector": "latency_graph_selector_ms",
        "retrieval_grounding": "latency_retrieval_grounding_ms",
        "postprocess": "latency_postprocess_ms",
    }
    latency = {
        name: {
            "p50_ms": _percentile((getattr(record, field) for record in records), 0.50),
            "p90_ms": _percentile((getattr(record, field) for record in records), 0.90),
        }
        for name, field in latency_fields.items()
    }
    latency["end_to_end"] = {
        "p50_ms": _percentile(
            (
                record.latency_data_ms
                + record.latency_graph_selector_ms
                + record.latency_retrieval_grounding_ms
                + record.latency_postprocess_ms
                for record in records
            ),
            0.50,
        ),
        "p90_ms": _percentile(
            (
                record.latency_data_ms
                + record.latency_graph_selector_ms
                + record.latency_retrieval_grounding_ms
                + record.latency_postprocess_ms
                for record in records
            ),
            0.90,
        ),
    }

    return {
        "examples": len(records),
        "cascade": {
            "target_proposal_recall": _mean(record.proposal_hit for record in records),
            "target_survival_conditional_on_matched_proposal": _mean(
                bool(record.target_survived) for record in matched
            ),
            "target_survival_denominator": len(matched),
            "relational_reference_survival": _mean(
                bool(record.reference_survived) for record in references
            ),
            "reference_survival_denominator": len(references),
            "retrieval_recall": retrieval,
            "candidate_selection_accuracy_when_target_present": _mean(
                bool(record.selected_correct) for record in target_present
            ),
            "candidate_selection_denominator": len(target_present),
            "point_accuracy_when_candidate_correct": _mean(
                bool(record.point_correct_given_candidate) for record in candidate_correct
            ),
            "point_given_candidate_denominator": len(candidate_correct),
            "point_in_target_accuracy_end_to_end": _mean(
                record.point_correct_end_to_end for record in records
            ),
        },
        "actions": {
            "accuracy": _mean(bool(record.action_correct) for record in action_labeled),
            "labeled_examples": len(action_labeled),
            "parameter_mae": _mean(float(value) for value in parameter_errors),
            "parameter_labeled_examples": len(parameter_errors),
        },
        "structure": dict(
            ssb_statistics(
                (record.ssb_length for record in records),
                (record.node_count for record in records),
                (record.edge_count for record in records),
            )
        ),
        "confidence": confidence,
        "abstention_rate": _mean(record.abstained for record in records),
        "latency": latency,
        "failures": dict(sorted(Counter(record.failure_reason for record in records).items())),
        "subsets": {
            "platform": _subset_metrics(records, "platform"),
            "target_type": _subset_metrics(records, "target_type"),
        },
        "aggregation_policy": "direct_observations_no_independence_multiplication_v1",
    }
