from __future__ import annotations

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
