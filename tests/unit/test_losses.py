from __future__ import annotations

import torch

from screen2action.losses import (
    TotalLossWeights,
    budget_loss,
    candidate_loss,
    masked_action_loss,
    masked_parameter_loss,
    point_loss,
    reference_survival_loss,
    target_survival_loss,
    total_objective,
    ui_contrastive_loss,
)


def test_full_objective_terms_are_finite_and_backpropagate() -> None:
    logits = torch.randn(2, 4, requires_grad=True)
    points = torch.sigmoid(torch.randn(2, 4, 2, requires_grad=True))
    actions = torch.randn(2, 4, requires_grad=True)
    parameters = torch.randn(2, 4, requires_grad=True)
    query = torch.randn(2, 8, requires_grad=True)
    nodes = torch.randn(2, 4, 8, requires_grad=True)
    candidate = candidate_loss(logits, torch.tensor([1, 2]))
    point = point_loss(points, torch.tensor([1, 2]), torch.full((2, 2), 0.5))
    action = masked_action_loss(
        actions, torch.ones_like(actions), torch.ones_like(actions, dtype=torch.bool)
    )
    parameter = masked_parameter_loss(
        parameters, torch.zeros_like(parameters), torch.ones_like(parameters, dtype=torch.bool)
    )
    ui = ui_contrastive_loss(query, nodes, torch.tensor([1, 2]))
    survive = target_survival_loss(
        logits, torch.tensor([[True, False, False, False], [False, True, False, False]])
    )
    survive = survive + reference_survival_loss(logits, torch.zeros_like(logits, dtype=torch.bool))
    budget = budget_loss(logits, torch.ones_like(logits), 2.0)
    total = total_objective(
        candidate=candidate,
        point=point,
        action=action + parameter,
        ui=ui,
        survive=survive,
        budget=budget,
        weights=TotalLossWeights(),
    )
    assert torch.isfinite(total.total)
    total.total.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert query.grad is not None and torch.isfinite(query.grad).all()
