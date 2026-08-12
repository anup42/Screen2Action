"""Cosine retrieval, typed relation reranking, and fixed-K candidate selection."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from screen2action.data.schema import EdgeRecord, NodeRecord, RetrievalCandidate
from screen2action.models.relation_gat import RELATION_ORDER


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Fixed-shape retrieval output for one screen."""

    node_indices: torch.Tensor
    node_ids: tuple[int, ...]
    scores: torch.Tensor
    actionable: torch.Tensor
    valid: torch.Tensor
    candidates: tuple[RetrievalCandidate, ...]


def select_top_k_actionable(
    scores: torch.Tensor,
    nodes: Iterable[NodeRecord],
    *,
    k: int = 8,
    actionability_threshold: float = 0.5,
) -> RetrievalResult:
    """Select actionable nodes then fill fixed K from valid high-score nodes."""

    if scores.ndim != 1:
        raise ValueError("scores must have shape [N]")
    if k <= 0:
        raise ValueError("k must be positive")
    node_list = list(nodes)
    if len(node_list) != scores.numel():
        raise ValueError("scores and nodes have incompatible lengths")
    if not 0.0 <= actionability_threshold <= 1.0:
        raise ValueError("actionability_threshold must be in [0, 1]")
    actionability = torch.tensor(
        [
            float(torch.sigmoid(torch.tensor(node.actionability_logits)).max().item())
            >= actionability_threshold
            for node in node_list
        ],
        dtype=torch.bool,
        device=scores.device,
    )
    valid_order = sorted(
        range(len(node_list)),
        key=lambda index: (-float(scores[index].detach().cpu()), node_list[index].node_id),
    )
    actionable_order = [index for index in valid_order if bool(actionability[index])]
    chosen = actionable_order[:k]
    chosen.extend(index for index in valid_order if index not in chosen and len(chosen) < k)
    if chosen and len(chosen) < k:
        chosen.extend([chosen[0]] * (k - len(chosen)))
    indices = chosen + [-1] * (k - len(chosen))
    valid_flags = [index >= 0 for index in indices]
    valid = torch.tensor(valid_flags, dtype=torch.bool, device=scores.device)
    safe_indices = torch.tensor(
        [max(index, 0) for index in indices], dtype=torch.long, device=scores.device
    )
    selected_scores = scores[safe_indices]
    selected_actionable = actionability[safe_indices] & valid
    node_ids = tuple(node_list[index].node_id if index >= 0 else -1 for index in indices)
    candidates = tuple(
        RetrievalCandidate(
            node_id=node_id if node_id >= 0 else 0,
            score=float(selected_scores[rank].detach().cpu()) if is_valid else 0.0,
            rank=rank,
            actionable=bool(selected_actionable[rank]),
            valid=is_valid,
        )
        for rank, (node_id, is_valid) in enumerate(zip(node_ids, valid_flags, strict=True))
    )
    return RetrievalResult(
        node_indices=safe_indices,
        node_ids=node_ids,
        scores=selected_scores,
        actionable=selected_actionable,
        valid=valid,
        candidates=candidates,
    )


class CosineRetriever(nn.Module):
    """Project commands and graph nodes into a shared normalized space."""

    def __init__(self, input_dim: int = 256, retrieval_dim: int = 256) -> None:
        super().__init__()
        self.query_projection = nn.Linear(input_dim, retrieval_dim)
        self.node_projection = nn.Linear(input_dim, retrieval_dim)

    def forward(self, query: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        """Return cosine scores for one query and `[N,D]` node features."""

        if query.ndim == 2 and query.shape[0] == 1:
            query = query[0]
        if query.ndim != 1 or nodes.ndim != 2:
            raise ValueError("query must be [D] and nodes must be [N,D]")
        query_projected = F.normalize(self.query_projection(query), dim=-1)
        node_projected = F.normalize(self.node_projection(nodes), dim=-1)
        return node_projected @ query_projected


class RelationAwareReranker(nn.Module):
    """Propagate retrieval evidence across typed edges with normalized weights."""

    def __init__(
        self, embedding_dim: int = 256, geometry_dim: int = 12, lambda_ref: float = 0.30
    ) -> None:
        super().__init__()
        if lambda_ref < 0.0:
            raise ValueError("lambda_ref must be non-negative")
        self.lambda_ref = lambda_ref
        self.geometry_dim = geometry_dim
        self.relation_weights = nn.Parameter(torch.zeros(len(RELATION_ORDER)))
        self.compatibility = nn.ModuleList(nn.Linear(geometry_dim, 1) for _ in RELATION_ORDER)
        self.node_gate = nn.Linear(embedding_dim, 1)

    def forward(
        self,
        base_scores: torch.Tensor,
        node_features: torch.Tensor,
        edges: Iterable[EdgeRecord],
    ) -> torch.Tensor:
        """Return base scores plus relation-specific incoming evidence."""

        if base_scores.ndim != 1 or node_features.ndim != 2:
            raise ValueError("base_scores must be [N] and node_features must be [N,D]")
        if base_scores.numel() != node_features.shape[0]:
            raise ValueError("base_scores and node_features have incompatible lengths")
        output = base_scores.clone()
        edge_list = list(edges)
        if not edge_list:
            return output
        normalized_weights = torch.softmax(self.relation_weights, dim=0)
        for relation_index, relation in enumerate(RELATION_ORDER):
            relation_edges = [edge for edge in edge_list if edge.relation is relation]
            for destination in range(node_features.shape[0]):
                incoming = [edge for edge in relation_edges if edge.dst == destination]
                if not incoming:
                    continue
                source_scores = torch.stack([base_scores[edge.src] for edge in incoming])
                geometry = torch.zeros(
                    (len(incoming), self.geometry_dim),
                    dtype=node_features.dtype,
                    device=node_features.device,
                )
                for edge_index, edge in enumerate(incoming):
                    values = edge.relative_geometry[: self.geometry_dim]
                    if values:
                        geometry[edge_index, : len(values)] = torch.tensor(
                            values, dtype=geometry.dtype, device=geometry.device
                        )
                compatibility = self.compatibility[relation_index](geometry).squeeze(-1)
                neighbor_weights = torch.softmax(compatibility, dim=0)
                gate = torch.sigmoid(self.node_gate(node_features[destination])).squeeze(-1)
                output[destination] = (
                    output[destination]
                    + self.lambda_ref
                    * normalized_weights[relation_index]
                    * gate
                    * (neighbor_weights * source_scores).sum()
                )
        return output
