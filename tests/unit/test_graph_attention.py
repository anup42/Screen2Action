from __future__ import annotations

import pytest
import torch

from screen2action.data.schema import EdgeRecord, RelationDirection, RelationType
from screen2action.models.relation_gat import (
    DenseRelationGraphAttention,
    RelationAwareGraphAttention,
    edge_tensors,
)


def test_variable_and_dense_relation_attention_have_parity() -> None:
    torch.manual_seed(3)
    nodes = torch.randn(4, 16, requires_grad=True)
    edges = [
        EdgeRecord(0, 1, RelationType.CONTAINMENT, RelationDirection.CONTAINS, (0.1,) * 12),
        EdgeRecord(1, 2, RelationType.PROXIMITY, RelationDirection.RIGHT, (0.2,) * 12),
        EdgeRecord(2, 3, RelationType.ORDINAL, RelationDirection.NEXT, (0.3,) * 12),
    ]
    edge_index, relation_ids, geometry = edge_tensors(edges)
    layer = RelationAwareGraphAttention(embedding_dim=16, heads=4, geometry_dim=12)
    dense = DenseRelationGraphAttention(layer)
    edge_output = layer(nodes, edge_index, relation_ids, geometry)
    dense_relation = torch.full((4, 4), -1, dtype=torch.long)
    dense_geometry = torch.zeros((4, 4, 12))
    for index, edge in enumerate(edges):
        dense_relation[edge.src, edge.dst] = relation_ids[index]
        dense_geometry[edge.src, edge.dst] = geometry[index]
    dense_output = dense(nodes, dense_relation, dense_geometry)
    assert torch.allclose(edge_output, dense_output, atol=1e-6, rtol=1e-6)
    edge_output.sum().backward()
    assert nodes.grad is not None and torch.isfinite(nodes.grad).all()


@pytest.mark.parametrize("variant", ["paper_eq_v1", "cpu_reconstruction_v1"])
def test_fixed_export_graph_equations_match_edge_path(variant: str) -> None:
    torch.manual_seed(13)
    nodes = torch.randn(3, 4, 16)
    valid = torch.tensor(
        [
            [True, True, True, False],
            [True, True, True, True],
            [True, True, True, True],
        ]
    )
    relation_mask = torch.zeros((3, 3, 4, 4), dtype=torch.bool)
    geometry = torch.zeros((3, 3, 4, 4, 12))
    for batch in range(2):
        relation_mask[batch, 0, 0, 1] = True
        relation_mask[batch, 2, 0, 1] = True
        relation_mask[batch, 1, 1, 2] = True
        geometry[batch, 0, 0, 1] = 0.1
        geometry[batch, 2, 0, 1] = 0.15
        geometry[batch, 1, 1, 2] = 0.2
    relation_mask[1, 2, 2, 3] = True
    geometry[1, 2, 2, 3] = 0.3
    layer = RelationAwareGraphAttention(
        embedding_dim=16,
        heads=4,
        geometry_dim=12,
        variant=variant,
    ).eval()

    expected_rows = []
    for batch in range(3):
        relation, source, destination = torch.nonzero(relation_mask[batch], as_tuple=True)
        expected_rows.append(
            layer(
                nodes[batch],
                torch.stack((source, destination)),
                relation,
                geometry[batch, relation, source, destination],
            )
            * valid[batch].unsqueeze(-1)
        )
    expected = torch.stack(expected_rows)
    actual = layer.forward_export_dense(nodes, relation_mask, geometry, valid)

    assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-6)
