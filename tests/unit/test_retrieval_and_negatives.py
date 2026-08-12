from __future__ import annotations

import torch

from screen2action.data.schema import EdgeRecord, RelationDirection, RelationType
from screen2action.models.hard_negatives import mine_same_screen_negatives
from screen2action.models.retriever import select_top_k_actionable


def test_fixed_k_retrieval_fills_with_valid_nodes(make_node) -> None:
    nodes = [
        make_node(0, (0.0, 0.0, 0.2, 0.2), confidence=0.8),
        make_node(1, (0.3, 0.0, 0.5, 0.2), confidence=0.7),
    ]
    scores = torch.tensor([0.9, 0.4])
    result = select_top_k_actionable(scores, nodes, k=4)
    assert result.valid.tolist() == [True, True, True, True]
    assert result.node_ids == (0, 1, 0, 0)
    assert all(candidate.valid for candidate in result.candidates)


def test_same_screen_negative_categories_and_weights(make_node) -> None:
    target = make_node(0, (0.1, 0.1, 0.2, 0.2), text_token_ids=(1,))
    same_text = make_node(1, (0.25, 0.1, 0.35, 0.2), text_token_ids=(1,))
    same_icon = make_node(2, (0.4, 0.1, 0.5, 0.2))
    edge = EdgeRecord(0, 1, RelationType.ORDINAL, RelationDirection.NEXT)
    negatives = mine_same_screen_negatives(
        "tap Wi-Fi",
        target,
        [target, same_text, same_icon],
        [edge],
    )
    assert any(negative.category == "ordinal" and negative.weight == 2.0 for negative in negatives)
