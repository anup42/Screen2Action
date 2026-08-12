"""Candidate and crop-local point losses."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def candidate_loss(
    candidate_logits: torch.Tensor,
    target_index: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy over the fixed candidate dimension."""

    if candidate_logits.ndim != 2 or target_index.ndim != 1:
        raise ValueError("candidate logits must be [B,K] and targets must be [B]")
    logits = candidate_logits
    if candidate_mask is not None:
        logits = logits.masked_fill(~candidate_mask.bool(), -1e9)
    if torch.any(target_index < 0) or torch.any(target_index >= logits.shape[1]):
        raise ValueError("candidate targets are outside the candidate dimension")
    return F.cross_entropy(logits, target_index)


def point_loss(
    point_local: torch.Tensor,
    target_index: torch.Tensor,
    target_point_local: torch.Tensor,
    *,
    beta: float = 1.0,
) -> torch.Tensor:
    """Smooth-L1 loss for the target candidate's crop-local point."""

    if point_local.ndim != 3 or point_local.shape[-1] != 2:
        raise ValueError("point_local must be [B,K,2]")
    if target_point_local.shape != (point_local.shape[0], 2):
        raise ValueError("target_point_local must be [B,2]")
    selected = point_local[
        torch.arange(point_local.shape[0], device=point_local.device), target_index
    ]
    return F.smooth_l1_loss(selected, target_point_local, beta=beta)
