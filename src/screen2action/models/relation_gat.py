"""Relation-aware graph attention for variable and fixed-shape graph inputs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import torch
from torch import nn

from screen2action.data.schema import EdgeRecord, RelationType

RELATION_ORDER = (RelationType.CONTAINMENT, RelationType.PROXIMITY, RelationType.ORDINAL)
RELATION_TO_ID = {relation: index for index, relation in enumerate(RELATION_ORDER)}
GRAPH_ATTENTION_VARIANTS = ("paper_eq_v1", "cpu_reconstruction_v1")


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
        variant: str = "cpu_reconstruction_v1",
    ) -> None:
        super().__init__()
        if embedding_dim % heads != 0:
            raise ValueError("embedding_dim must be divisible by heads")
        if relation_count <= 0 or geometry_dim <= 0:
            raise ValueError("relation_count and geometry_dim must be positive")
        if variant not in GRAPH_ATTENTION_VARIANTS:
            raise ValueError(f"unknown graph-attention variant: {variant}")
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.head_dim = embedding_dim // heads
        self.relation_count = relation_count
        self.variant = variant
        self.paper_query = nn.Linear(embedding_dim, embedding_dim)
        self.paper_key = nn.Linear(embedding_dim, embedding_dim)
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
                if self.variant == "paper_eq_v1":
                    query = self.paper_query(node_features[destination]).view(
                        self.heads, self.head_dim
                    )
                    key = self.paper_key(node_features[source]).view(-1, self.heads, self.head_dim)
                else:
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

        if node_features.ndim == 3:
            if relation_ids.ndim != 3 or relative_geometry.ndim != 4:
                raise ValueError("batched dense relations must have shape [B,N,N] and [B,N,N,G]")
            if relation_ids.shape[0] != node_features.shape[0]:
                raise ValueError("batched dense graph count does not match node features")
            if valid_nodes is None:
                valid_nodes = torch.ones(
                    node_features.shape[:2], dtype=torch.bool, device=node_features.device
                )
            if valid_nodes.shape != node_features.shape[:2]:
                raise ValueError("valid_nodes must have shape [B,N]")
            outputs = [
                self.forward_dense(
                    node_features[index],
                    relation_ids[index],
                    relative_geometry[index],
                    valid_nodes[index],
                )
                for index in range(node_features.shape[0])
            ]
            return torch.stack(outputs) * valid_nodes.unsqueeze(-1).to(node_features.dtype)
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

    def forward_batched_edges(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        relation_ids: torch.Tensor,
        relative_geometry: torch.Tensor,
        *,
        valid_nodes: torch.Tensor,
        valid_edges: torch.Tensor,
    ) -> torch.Tensor:
        """Consume padded edge lists and return `[B,N,D]` with invalid nodes zeroed."""

        if node_features.ndim != 3 or edge_index.ndim != 3 or edge_index.shape[1] != 2:
            raise ValueError("batched nodes/edges must have shapes [B,N,D] and [B,2,E]")
        batch, node_count, _ = node_features.shape
        edge_count = edge_index.shape[-1]
        if relation_ids.shape != (batch, edge_count) or valid_edges.shape != (
            batch,
            edge_count,
        ):
            raise ValueError("relation IDs and edge mask must have shape [B,E]")
        if relative_geometry.shape[:2] != (batch, edge_count):
            raise ValueError("relative geometry must have shape [B,E,G]")
        if valid_nodes.shape != (batch, node_count):
            raise ValueError("valid node mask must have shape [B,N]")
        outputs = []
        for index in range(batch):
            edge_mask = valid_edges[index]
            local_edges = edge_index[index, :, edge_mask]
            if local_edges.numel() and (
                int(local_edges.min()) < 0 or int(local_edges.max()) >= node_count
            ):
                raise ValueError("padded edge index references an invalid node position")
            output = self._forward_edges(
                node_features[index],
                local_edges,
                relation_ids[index, edge_mask],
                relative_geometry[index, edge_mask],
            )
            outputs.append(output * valid_nodes[index].unsqueeze(-1).to(output.dtype))
        return torch.stack(outputs)


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
        variant: str = "cpu_reconstruction_v1",
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
                variant=variant,
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
