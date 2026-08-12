from __future__ import annotations

from screen2action.ssb.matching import match_positive_proposal, positive_proposals


def test_highest_iou_is_selected(make_node) -> None:
    target = (0.2, 0.2, 0.4, 0.4)
    proposals = [
        make_node(2, (0.18, 0.18, 0.42, 0.42)),
        make_node(1, (0.2, 0.2, 0.4, 0.4)),
    ]
    result = match_positive_proposal(target, proposals)
    assert result.proposal_id == 1
    assert result.iou == 1.0
    assert not result.used_center_fallback
    assert [node.node_id for node in positive_proposals(target, proposals)] == [1, 2]


def test_center_containment_is_the_fallback() -> None:
    from screen2action.data.schema import NodeRecord, NodeType

    target = (0.4, 0.4, 0.6, 0.6)
    proposal = NodeRecord(4, NodeType.CONTROL, (0.3, 0.3, 0.7, 0.7), detector_confidence=0.4)
    result = match_positive_proposal(target, [proposal], iou_threshold=0.95)
    assert result.proposal_id == 4
    assert result.used_center_fallback
    assert positive_proposals(target, [proposal], iou_threshold=0.95) == (proposal,)


def test_missing_proposal_is_explicit() -> None:
    from screen2action.ssb.matching import ProposalMatch

    result = match_positive_proposal((0.1, 0.1, 0.2, 0.2), [])
    assert result == ProposalMatch(None, 0.0, True)
