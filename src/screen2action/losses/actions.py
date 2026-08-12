"""Masked action-type and action-parameter losses."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def masked_action_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    positive_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Class-balanced BCE that ignores unknown action labels."""

    if logits.shape != targets.shape or mask.shape != targets.shape:
        raise ValueError("action logits, targets, and mask must have identical shapes")
    values = F.binary_cross_entropy_with_logits(
        logits,
        targets.to(logits.dtype),
        pos_weight=positive_weight,
        reduction="none",
    )
    valid = mask.to(values.dtype)
    denominator = valid.sum().clamp_min(1.0)
    return (values * valid).sum() / denominator


def masked_parameter_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Smooth-L1 parameter loss with action-specific masking."""

    if predicted.shape != target.shape or mask.shape != target.shape:
        raise ValueError("parameter prediction, target, and mask must match")
    values = F.smooth_l1_loss(predicted, target, reduction="none")
    valid = mask.to(values.dtype)
    return (values * valid).sum() / valid.sum().clamp_min(1.0)
