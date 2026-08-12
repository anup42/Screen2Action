"""Target and relation-reference survival losses with empty-set handling."""

from __future__ import annotations

import torch


def _survival(logits: torch.Tensor, positive_mask: torch.Tensor, epsilon: float) -> torch.Tensor:
    if logits.shape != positive_mask.shape or logits.ndim != 2:
        raise ValueError("survival logits and mask must both be [B,N]")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    losses: list[torch.Tensor] = []
    for row_logits, row_mask in zip(logits, positive_mask.bool(), strict=True):
        if not bool(row_mask.any()):
            losses.append(row_logits.sum() * 0.0)
            continue
        total = torch.logsumexp(row_logits, dim=0)
        positive = torch.logsumexp(row_logits.masked_fill(~row_mask, -torch.inf), dim=0)
        losses.append(
            -(positive - total).clamp_min(torch.log(torch.tensor(epsilon, device=logits.device)))
        )
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def target_survival_loss(
    retention_logits: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Penalize probability mass assigned away from matched target proposals."""

    return _survival(retention_logits, positive_mask, epsilon)


def reference_survival_loss(
    retention_logits: torch.Tensor,
    reference_mask: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Penalize loss of relational reference proposals."""

    return _survival(retention_logits, reference_mask, epsilon)
