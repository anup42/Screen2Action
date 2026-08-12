from __future__ import annotations

import torch

from screen2action.data.schema import EdgeRecord, RelationDirection, RelationType
from screen2action.models.hard_negatives import mine_same_screen_negatives
from screen2action.models.relation_gat import RELATION_TO_ID
from screen2action.models.retriever import RelationAwareReranker, select_top_k_actionable


def test_fixed_k_retrieval_marks_padding_instead_of_duplicating_nodes(make_node) -> None:
    nodes = [
        make_node(0, (0.0, 0.0, 0.2, 0.2), confidence=0.8),
        make_node(1, (0.3, 0.0, 0.5, 0.2), confidence=0.7),
    ]
    scores = torch.tensor([0.9, 0.4])
    result = select_top_k_actionable(scores, nodes, k=4)
    assert result.valid.tolist() == [True, True, False, False]
    assert result.node_ids == (0, 1, -1, -1)
    assert [candidate.valid for candidate in result.candidates] == [True, True, False, False]


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


def test_same_screen_negative_mining_fills_four_unique_nodes_when_available(make_node) -> None:
    target = make_node(0, (0.0, 0.0, 0.1, 0.1))
    nodes = [target] + [
        make_node(index, (0.1 * index, 0.2, min(0.1 * index + 0.05, 1.0), 0.3))
        for index in range(1, 7)
    ]

    negatives = mine_same_screen_negatives("tap target", target, nodes)

    assert len({negative.node_id for negative in negatives}) >= 4
    assert all(negative.node_id != target.node_id for negative in negatives)


def test_dense_relation_reranker_matches_canonical_edge_equation() -> None:
    torch.manual_seed(21)
    reranker = RelationAwareReranker(embedding_dim=16, geometry_dim=12).eval()
    scores = torch.tensor([0.8, 0.4, -0.2, 0.1])
    nodes = torch.randn(4, 16)
    valid = torch.tensor([True, True, True, False])
    edges = [
        EdgeRecord(0, 1, RelationType.CONTAINMENT, RelationDirection.CONTAINS, (0.1,) * 12),
        EdgeRecord(0, 1, RelationType.ORDINAL, RelationDirection.NEXT, (0.15,) * 12),
        EdgeRecord(2, 1, RelationType.PROXIMITY, RelationDirection.LEFT, (0.2,) * 12),
        EdgeRecord(1, 2, RelationType.ORDINAL, RelationDirection.NEXT, (0.3,) * 12),
        EdgeRecord(2, 3, RelationType.PROXIMITY, RelationDirection.RIGHT, (0.4,) * 12),
    ]
    relation_mask = torch.zeros((1, 3, 4, 4), dtype=torch.bool)
    geometry = torch.zeros((1, 3, 4, 4, 12))
    for edge in edges:
        relation = RELATION_TO_ID[edge.relation]
        relation_mask[0, relation, edge.src, edge.dst] = True
        geometry[0, relation, edge.src, edge.dst] = torch.tensor(edge.relative_geometry)

    expected = reranker(scores, nodes, edges, valid)
    actual = reranker.forward_dense(
        scores.unsqueeze(0),
        nodes.unsqueeze(0),
        relation_mask,
        geometry,
        valid.unsqueeze(0),
    )[0]

    assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-6)
