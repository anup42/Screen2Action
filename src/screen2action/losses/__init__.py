"""Screen2Action objective terms and finite masked losses."""

from screen2action.losses.actions import masked_action_loss, masked_parameter_loss
from screen2action.losses.budget import budget_loss
from screen2action.losses.contrastive import ui_contrastive_loss
from screen2action.losses.grounding import candidate_loss, point_loss
from screen2action.losses.survival import reference_survival_loss, target_survival_loss
from screen2action.losses.total import LossBreakdown, TotalLossWeights, total_objective

__all__ = [
    "LossBreakdown",
    "TotalLossWeights",
    "budget_loss",
    "candidate_loss",
    "masked_action_loss",
    "masked_parameter_loss",
    "point_loss",
    "reference_survival_loss",
    "target_survival_loss",
    "total_objective",
    "ui_contrastive_loss",
]
