"""Full configurable Screen2Action objective."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class TotalLossWeights:
    """Reference objective weights from the implementation contract."""

    candidate: float = 1.0
    point: float = 1.0
    action: float = 0.4
    ui: float = 0.2
    survive: float = 0.5
    budget: float = 0.05

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(value) or value < 0
            for value in (
                self.candidate,
                self.point,
                self.action,
                self.ui,
                self.survive,
                self.budget,
            )
        ):
            raise ValueError("loss weights must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class LossBreakdown:
    """Individual losses and weighted total."""

    candidate: torch.Tensor
    point: torch.Tensor
    action: torch.Tensor
    ui: torch.Tensor
    survive: torch.Tensor
    budget: torch.Tensor
    total: torch.Tensor


def total_objective(
    *,
    candidate: torch.Tensor,
    point: torch.Tensor,
    action: torch.Tensor,
    ui: torch.Tensor,
    survive: torch.Tensor,
    budget: torch.Tensor,
    weights: TotalLossWeights | None = None,
) -> LossBreakdown:
    """Combine finite scalar objective terms with explicit reference weights."""

    if weights is None:
        weights = TotalLossWeights()
    terms = (candidate, point, action, ui, survive, budget)
    if any(term.ndim != 0 or not torch.isfinite(term) for term in terms):
        raise ValueError("all objective terms must be finite scalar tensors")
    total = (
        weights.candidate * candidate
        + weights.point * point
        + weights.action * action
        + weights.ui * ui
        + weights.survive * survive
        + weights.budget * budget
    )
    return LossBreakdown(candidate, point, action, ui, survive, budget, total)
