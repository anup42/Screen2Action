"""Relation-aware graph attention for variable and fixed-shape graph inputs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import torch
from torch import nn

from screen2action.data.schema import EdgeRecord, RelationType

RELATION_ORDER = (RelationType.CONTAINMENT, RelationType.PROXIMITY, RelationType.ORDINAL)
RELATION_TO_ID = {relation: index for index, relation in enumerate(RELATION_ORDER)}


def edge_tensors(
    edges: Iterable[EdgeRecord],
    *,
    device: torch.device | str | None = None,
    geometry_dim: int = 12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert canonical edges to `[2,E]`, relation IDs, and geometry tensors."""

    edge_list = list(edges)
    target_device = torch.device(device) if device is not None else None
    if not edge_list:
        return (
            torch.empty((2, 0), dtype=torch.long, device=target_device),
            torch.empty((0,), dtype=torch.long, device=target_device),
            torch.empty((0, geometry_dim), dtype=torch.float32, device=target_device),
        )
    edge_index = torch.tensor(
        [[edge.src for edge in edge_list], [edge.dst for edge in edge_list]],
        dtype=torch.long,
        device=target_device,
    )
    relation_ids = torch.tensor(
        [RELATION_TO_ID[edge.relation] for edge in edge_list],
        dtype=torch.long,
        device=target_device,
    )
    geometry = torch.zeros(
        (len(edge_list), geometry_dim), dtype=torch.float32, device=target_device
    )
    for index, edge in enumerate(edge_list):
        values = edge.relative_geometry[:geometry_dim]
        if values:
            geometry[index, : len(values)] = torch.tensor(
                values, dtype=torch.float32, device=target_device
            )
    return edge_index, relation_ids, geometry


class RelationAwareGraphAttention(nn.Module):
    """One relation-aware GAT layer with per-relation neighbor softmax."""

    def __init__(
        self,
        *,
        embedding_dim: int = 256,
        heads: int = 4,
        geometry_dim: int = 12,
        relation_count: int = len(RELATION_ORDER),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embedding_dim % heads != 0:
            raise ValueError("embedding_dim must be divisible by heads")
        if relation_count <= 0 or geometry_dim <= 0:
            raise ValueError("relation_count and geometry_dim must be positive")
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.head_dim = embedding_dim // heads
        self.relation_count = relation_count
        self.query = nn.ModuleList(
            nn.Linear(embedding_dim, embedding_dim) for _ in range(relation_count)
        )
        self.key = nn.ModuleList(
            nn.Linear(embedding_dim, embedding_dim) for _ in range(relation_count)
        )
        self.value = nn.ModuleList(
            nn.Linear(embedding_dim, embedding_dim) for _ in range(relation_count)
        )
        self.geometry = nn.ModuleList(
            nn.Linear(geometry_dim, embedding_dim) for _ in range(relation_count)
        )
        self.relation_embedding = nn.Parameter(torch.zeros(relation_count, embedding_dim))
        self.output = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embedding_dim)

    def _forward_edges(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        relation_ids: torch.Tensor,
        relative_geometry: torch.Tensor,
    ) -> torch.Tensor:
        if node_features.ndim != 2:
            raise ValueError("node_features must have shape [N, D]")
        if edge_index.shape[0] != 2 or edge_index.shape[1] != relation_ids.numel():
            raise ValueError("edge_index and relation_ids have incompatible shapes")
        if relative_geometry.shape[0] != relation_ids.numel():
            raise ValueError("relative_geometry and relation_ids have incompatible shapes")
        node_count = node_features.shape[0]
        aggregated = torch.zeros_like(node_features)
        if relation_ids.numel() == 0:
            return cast(torch.Tensor, self.norm(node_features))
        if relation_ids.min() < 0 or relation_ids.max() >= self.relation_count:
            raise ValueError("relation IDs are outside the configured range")
        for relation_id in range(self.relation_count):
            relation_positions = torch.nonzero(
                relation_ids == relation_id, as_tuple=False
            ).flatten()
            if relation_positions.numel() == 0:
                continue
            for destination in range(node_count):
                local_positions = relation_positions[
                    edge_index[1, relation_positions] == destination
                ]
                if local_positions.numel() == 0:
                    continue
                source = edge_index[0, local_positions]
                query = self.query[relation_id](node_features[destination]).view(
                    self.heads, self.head_dim
                )
                key = self.key[relation_id](node_features[source]).view(
                    -1, self.heads, self.head_dim
                )
                geometry = self.geometry[relation_id](relative_geometry[local_positions]).view(
                    -1, self.heads, self.head_dim
                )
                relation = self.relation_embedding[relation_id].view(1, self.heads, self.head_dim)
                scores = ((key + geometry + relation) * query.unsqueeze(0)).sum(dim=-1)
                scores = scores / (self.head_dim**0.5)
                weights = torch.softmax(scores, dim=0)
                values = self.value[relation_id](node_features[source]).view(
                    -1, self.heads, self.head_dim
                )
                message = (weights.unsqueeze(-1) * values).sum(dim=0).reshape(self.embedding_dim)
                aggregated[destination] = aggregated[destination] + message
        return cast(torch.Tensor, self.norm(node_features + self.dropout(self.output(aggregated))))

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        relation_ids: torch.Tensor,
        relative_geometry: torch.Tensor,
    ) -> torch.Tensor:
        """Update nodes from variable-length edge-list tensors."""

        return self._forward_edges(node_features, edge_index, relation_ids, relative_geometry)

    def forward_dense(
        self,
        node_features: torch.Tensor,
        relation_ids: torch.Tensor,
        relative_geometry: torch.Tensor,
        valid_nodes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Consume fixed `[N,N]` relation IDs and `[N,N,G]` geometry."""

        if relation_ids.ndim != 2 or relation_ids.shape[0] != relation_ids.shape[1]:
            raise ValueError("relation_ids must have square shape [N, N]")
        if relative_geometry.shape[:2] != relation_ids.shape:
            raise ValueError("dense geometry must start with relation_ids shape")
        valid = relation_ids >= 0
        if valid_nodes is not None:
            valid = valid & valid_nodes.view(-1, 1) & valid_nodes.view(1, -1)
        source, destination = torch.nonzero(valid, as_tuple=True)
        edge_index = torch.stack((source, destination), dim=0)
        return self._forward_edges(
            node_features,
            edge_index,
            relation_ids[source, destination],
            relative_geometry[source, destination],
        )


class DenseRelationGraphAttention(nn.Module):
    """Fixed-shape wrapper sharing the variable-edge-list layer parameters."""

    def __init__(self, layer: RelationAwareGraphAttention) -> None:
        super().__init__()
        self.layer = layer

    def forward(
        self,
        node_features: torch.Tensor,
        relation_ids: torch.Tensor,
        relative_geometry: torch.Tensor,
        valid_nodes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.layer.forward_dense(node_features, relation_ids, relative_geometry, valid_nodes)


class RelationGraphEncoder(nn.Module):
    """Stack the two paper-reference relation-aware graph layers."""

    def __init__(
        self,
        *,
        layers: int = 2,
        embedding_dim: int = 256,
        heads: int = 4,
        geometry_dim: int = 12,
        relation_count: int = len(RELATION_ORDER),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if layers <= 0:
            raise ValueError("layers must be positive")
        self.layers = nn.ModuleList(
            RelationAwareGraphAttention(
                embedding_dim=embedding_dim,
                heads=heads,
                geometry_dim=geometry_dim,
                relation_count=relation_count,
                dropout=dropout,
            )
            for _ in range(layers)
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        relation_ids: torch.Tensor,
        relative_geometry: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            node_features = layer(node_features, edge_index, relation_ids, relative_geometry)
        return node_features
