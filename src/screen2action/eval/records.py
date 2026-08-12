"""Typed, auditable per-command evaluation records."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from screen2action.data.schema import Box, Point


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    """One directly observed cascade outcome; conditional fields may be absent."""

    command_id: str
    screen_id: str
    source_dataset: str
    split: str
    platform: str
    target_type: str
    relation_type: str
    target_box_xyxy_norm: Box
    target_area: float
    proposal_node_id: int | None
    proposal_iou: float
    proposal_center_fallback: bool
    proposal_hit: bool
    selected_node_ids: tuple[int, ...]
    target_survived: bool | None
    reference_required: bool
    reference_survived: bool | None
    candidate_node_ids: tuple[int, ...]
    candidate_scores: tuple[float, ...]
    retrieval_hits: dict[str, bool]
    selected_candidate_id: int | None
    selected_candidate_rank: int | None
    selected_correct: bool | None
    predicted_point_xy_norm: Point | None
    point_correct_given_candidate: bool | None
    point_correct_end_to_end: bool
    predicted_action: str | None
    target_action: str
    action_labeled: bool
    action_correct: bool | None
    predicted_parameters: tuple[float, ...]
    target_parameters: tuple[float, ...]
    parameter_mask: tuple[bool, ...]
    parameter_absolute_error: float | None
    confidence_logit: float | None
    confidence_probability: float | None
    confidence_calibrated: bool
    abstained: bool
    ssb_length: int
    node_count: int
    edge_count: int
    latency_data_ms: float
    latency_graph_selector_ms: float
    latency_retrieval_grounding_ms: float
    latency_postprocess_ms: float
    failure_reason: str

    def as_dict(self) -> dict[str, Any]:
        """Return a stable JSON-safe mapping."""

        return asdict(self)

    def as_csv_row(self) -> dict[str, object]:
        """Flatten structured columns as canonical JSON strings for CSV."""

        row = self.as_dict()
        for key, value in tuple(row.items()):
            if isinstance(value, dict | list | tuple):
                row[key] = json.dumps(value, sort_keys=True, separators=(",", ":"))
        return row
