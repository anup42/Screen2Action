"""UI-aware command/node contrastive objective."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def ui_contrastive_loss(
    query: torch.Tensor,
    nodes: torch.Tensor,
    positive_index: torch.Tensor,
    *,
    temperature: float = 0.07,
    negative_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """InfoNCE over normalized command/node representations."""

    if query.ndim != 2 or nodes.ndim != 3 or positive_index.ndim != 1:
        raise ValueError("query, nodes, and positive_index have invalid ranks")
    if query.shape[0] != nodes.shape[0] or query.shape[0] != positive_index.shape[0]:
        raise ValueError("contrastive batch dimensions do not match")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    logits = torch.einsum("bd,bnd->bn", F.normalize(query, dim=-1), F.normalize(nodes, dim=-1))
    logits = logits / temperature
    if negative_weights is not None:
        if negative_weights.shape != logits.shape:
            raise ValueError("negative_weights must match contrastive logits")
        logits = logits + torch.log(negative_weights.clamp_min(1e-12))
    return F.cross_entropy(logits, positive_index)
