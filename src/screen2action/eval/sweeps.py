"""Configuration-driven budget/K and ablation sweep helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SweepResult:
    """One evaluated configuration point."""

    settings: Mapping[str, object]
    metrics: Mapping[str, float]


def run_budget_k_sweep(
    budgets: tuple[int, ...],
    top_ks: tuple[int, ...],
    evaluate: Callable[[int, int], Mapping[str, float]],
) -> tuple[SweepResult, ...]:
    """Evaluate every configured B/K pair without hidden global state."""

    if not budgets or not top_ks or any(value <= 0 for value in (*budgets, *top_ks)):
        raise ValueError("budgets and top_ks must contain positive values")
    return tuple(
        SweepResult({"budget": budget, "top_k": top_k}, evaluate(budget, top_k))
        for budget in budgets
        for top_k in top_ks
    )


def standard_ablations(
    evaluate: Callable[[str], Mapping[str, float]],
) -> tuple[SweepResult, ...]:
    """Run the plan's selector/relation ablation names in stable order."""

    names = (
        "learned_selector",
        "confidence_only_selector",
        "no_target_survival_loss",
        "no_reference_survival_loss",
        "no_budget_loss",
        "no_relation_reranking",
        "no_graph_relations",
        "dense_no_ssb_baseline",
    )
    return tuple(SweepResult({"ablation": name}, evaluate(name)) for name in names)
