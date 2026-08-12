"""Positive proposal matching for target survival supervision and metrics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from screen2action.data.schema import Box, NodeRecord
from screen2action.ssb.geometry import box_center, iou, point_in_box


@dataclass(frozen=True, slots=True)
class ProposalMatch:
    """Best proposal and the fallback path used to obtain it."""

    proposal_id: int | None
    iou: float
    used_center_fallback: bool


def positive_proposals(
    target_box: Box,
    proposals: Iterable[NodeRecord],
    *,
    iou_threshold: float = 0.50,
) -> tuple[NodeRecord, ...]:
    """Return all proposals above IoU threshold, or center-containing fallbacks."""

    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    proposal_list = list(proposals)
    iou_matches = [
        proposal
        for proposal in proposal_list
        if iou(target_box, proposal.box_xyxy_norm) >= iou_threshold
    ]
    if iou_matches:
        return tuple(sorted(iou_matches, key=lambda proposal: proposal.node_id))
    center = box_center(target_box)
    return tuple(
        sorted(
            (
                proposal
                for proposal in proposal_list
                if point_in_box(center, proposal.box_xyxy_norm)
            ),
            key=lambda proposal: proposal.node_id,
        )
    )


def match_positive_proposal(
    target_box: Box,
    proposals: Iterable[NodeRecord],
    *,
    iou_threshold: float = 0.50,
) -> ProposalMatch:
    """Match the highest-IoU proposal, then use center containment as fallback."""

    proposal_list = list(proposals)
    exact = [
        (iou(target_box, proposal.box_xyxy_norm), proposal)
        for proposal in proposal_list
        if iou(target_box, proposal.box_xyxy_norm) >= iou_threshold
    ]
    if exact:
        best_iou, best = max(exact, key=lambda item: (item[0], -item[1].node_id))
        return ProposalMatch(best.node_id, best_iou, False)
    center = box_center(target_box)
    fallback = [
        (iou(target_box, proposal.box_xyxy_norm), proposal)
        for proposal in proposal_list
        if point_in_box(center, proposal.box_xyxy_norm)
    ]
    if fallback:
        best_iou, best = max(fallback, key=lambda item: (item[0], -item[1].node_id))
        return ProposalMatch(best.node_id, best_iou, True)
    return ProposalMatch(None, 0.0, True)
