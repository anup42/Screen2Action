"""Trainable command-independent node retention and survival/budget losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from screen2action.ssb.selector import straight_through_gumbel_mask


@dataclass(frozen=True, slots=True)
class RetentionOutput:
    logits: torch.Tensor
    scores: torch.Tensor
    selection_mask: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class RetentionLoss:
    total: torch.Tensor
    target_survival: torch.Tensor
    reference_survival: torch.Tensor
    budget: torch.Tensor
    target_count: int
    reference_count: int


class RetentionScorer(nn.Module):
    """Implement `r_i = sigmoid(w_rho^T h_i)` independently of commands."""

    def __init__(self, embedding_dim: int = 256) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.projection = nn.Linear(embedding_dim, 1)

    def forward(
        self,
        node_states: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        stochastic: bool = False,
        temperature: float = 1.0,
        mandatory_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> RetentionOutput:
        if node_states.ndim != 3 or valid_mask.shape != node_states.shape[:2]:
            raise ValueError("retention inputs must have shapes [B,N,D] and [B,N]")
        if mandatory_mask is None:
            mandatory_mask = torch.zeros_like(valid_mask)
        if mandatory_mask.shape != valid_mask.shape:
            raise ValueError("mandatory retention mask must have shape [B,N]")
        logits = self.projection(node_states).squeeze(-1)
        scores = torch.sigmoid(logits) * valid_mask.to(logits.dtype)
        if stochastic:
            selection = straight_through_gumbel_mask(
                logits,
                temperature,
                mandatory_mask=mandatory_mask | ~valid_mask,
                generator=generator,
            )
            selection = selection * valid_mask.to(selection.dtype)
        else:
            selection = valid_mask.to(logits.dtype)
        return RetentionOutput(logits, scores, selection, valid_mask)


def retention_loss(
    output: RetentionOutput,
    *,
    target_mask: torch.Tensor,
    reference_mask: torch.Tensor,
    token_costs: torch.Tensor,
    budgets: torch.Tensor,
    survival_weight: float = 0.5,
    budget_weight: float = 0.05,
) -> RetentionLoss:
    """Compute positive survival and differentiable expected-budget penalties."""

    shape = output.valid_mask.shape
    if target_mask.shape != shape or reference_mask.shape != shape:
        raise ValueError("target/reference retention masks must have shape [B,N]")
    if token_costs.shape != shape or budgets.shape != (shape[0],):
        raise ValueError("retention token costs/budgets have incompatible shapes")
    target = target_mask.bool() & output.valid_mask
    reference = reference_mask.bool() & output.valid_mask
    zero = output.logits.sum() * 0.0
    target_loss = (
        F.binary_cross_entropy_with_logits(
            output.logits[target], torch.ones_like(output.logits[target])
        )
        if bool(target.any())
        else zero
    )
    reference_loss = (
        F.binary_cross_entropy_with_logits(
            output.logits[reference], torch.ones_like(output.logits[reference])
        )
        if bool(reference.any())
        else zero
    )
    expected_cost = (output.scores * token_costs).sum(dim=1)
    budget_loss = (torch.relu(expected_cost - budgets) / budgets.clamp_min(1.0)).mean()
    total = survival_weight * (target_loss + reference_loss) + budget_weight * budget_loss
    return RetentionLoss(
        total,
        target_loss,
        reference_loss,
        budget_loss,
        int(target.sum()),
        int(reference.sum()),
    )
