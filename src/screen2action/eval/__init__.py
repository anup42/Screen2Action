"""Stage-wise metrics, calibration, and CPU latency helpers."""

from screen2action.eval.calibration import calibration_metrics, fit_temperature, risk_coverage
from screen2action.eval.latency import peak_python_memory
from screen2action.eval.metrics import (
    action_parameter_error,
    point_accuracy_given_candidate,
    point_in_target_accuracy,
    proposal_recall,
    retrieval_recall,
    selection_accuracy_when_retrieved,
    stage_metrics,
    target_survival_rate,
)
from screen2action.eval.sweeps import run_budget_k_sweep, standard_ablations

__all__ = [
    "calibration_metrics",
    "action_parameter_error",
    "fit_temperature",
    "point_accuracy_given_candidate",
    "point_in_target_accuracy",
    "peak_python_memory",
    "proposal_recall",
    "retrieval_recall",
    "selection_accuracy_when_retrieved",
    "risk_coverage",
    "run_budget_k_sweep",
    "stage_metrics",
    "standard_ablations",
    "target_survival_rate",
]
