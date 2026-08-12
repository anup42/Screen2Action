"""Validation-only temperature scaling and selective prediction metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class CalibrationMetrics:
    """NLL, ECE, and Brier metrics for binary confidence."""

    nll: float
    ece: float
    brier: float


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    max_iter: int = 50,
) -> float:
    """Fit one positive temperature on held-out validation logits."""

    if logits.ndim != 1 or labels.shape != logits.shape:
        raise ValueError("logits and labels must be one-dimensional and aligned")
    if logits.numel() == 0:
        raise ValueError("calibration set cannot be empty")
    parameter = nn.Parameter(torch.zeros((), dtype=logits.dtype, device=logits.device))
    optimizer = torch.optim.LBFGS(
        [parameter], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = parameter.exp().clamp_min(1e-4)
        loss = F.binary_cross_entropy_with_logits(logits / temperature, labels.float())
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(parameter.detach().exp().clamp_min(1e-4))


def calibration_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 1.0,
    bins: int = 16,
) -> CalibrationMetrics:
    """Report NLL, equal-width ECE, and Brier score."""

    if temperature <= 0.0 or bins <= 0:
        raise ValueError("temperature and bins must be positive")
    probabilities = torch.sigmoid(logits / temperature)
    labels_float = labels.float()
    nll = float(F.binary_cross_entropy(probabilities, labels_float).item())
    brier = float(((probabilities - labels_float) ** 2).mean().item())
    ece = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        mask = (probabilities >= lower) & (
            probabilities < upper if index < bins - 1 else probabilities <= upper
        )
        if bool(mask.any()):
            ece += float(mask.float().mean()) * abs(
                float(probabilities[mask].mean()) - float(labels_float[mask].mean())
            )
    return CalibrationMetrics(nll, ece, brier)


def risk_coverage(
    confidence: torch.Tensor,
    correct: torch.Tensor,
    *,
    points: int = 11,
) -> tuple[tuple[float, float], ...]:
    """Return risk at deterministic coverage levels sorted by confidence."""

    if confidence.ndim != 1 or correct.shape != confidence.shape or points < 2:
        raise ValueError("confidence/correct shape or points are invalid")
    if confidence.numel() == 0:
        raise ValueError("risk-coverage input cannot be empty")
    order = torch.argsort(confidence, descending=True)
    sorted_correct = correct.float()[order]
    result: list[tuple[float, float]] = []
    for index in range(1, points + 1):
        coverage = index / points
        count = max(1, min(len(sorted_correct), round(coverage * len(sorted_correct))))
        risk = 1.0 - float(sorted_correct[:count].mean())
        result.append((count / len(sorted_correct), risk))
    return tuple(result)
