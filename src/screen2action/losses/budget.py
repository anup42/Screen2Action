"""Differentiable expected-token budget penalty."""

from __future__ import annotations

import torch


def budget_loss(
    retention_logits: torch.Tensor,
    token_costs: torch.Tensor,
    budget: float,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Penalize expected selected tokens above the configured budget."""

    if retention_logits.shape != token_costs.shape:
        raise ValueError("retention_logits and token_costs must have identical shapes")
    if budget <= 0.0 or epsilon <= 0.0:
        raise ValueError("budget and epsilon must be positive")
    expected_cost = (torch.sigmoid(retention_logits) * token_costs).sum()
    return torch.relu(expected_cost - budget) / max(float(budget), epsilon)
