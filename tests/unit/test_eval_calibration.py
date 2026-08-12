from __future__ import annotations

import pytest
import torch

from screen2action.eval.calibration import (
    calibration_metrics,
    fit_temperature,
    fit_threshold_policy,
    risk_coverage,
)
from screen2action.eval.latency import peak_python_memory
from screen2action.eval.metrics import (
    action_parameter_error,
    point_accuracy_given_candidate,
    point_in_target_accuracy,
    retrieval_recall,
    selection_accuracy_when_retrieved,
    target_survival_rate,
)


def test_metrics_are_direct_and_calibration_is_finite() -> None:
    logits = torch.tensor([-2.0, 2.0, 0.0, 1.0])
    labels = torch.tensor([0, 1, 1, 1])
    temperature = fit_temperature(logits, labels, max_iter=10)
    metrics = calibration_metrics(logits, labels, temperature=temperature)
    assert temperature > 0.0
    assert metrics.nll >= 0.0 and metrics.ece >= 0.0 and metrics.brier >= 0.0
    curve = risk_coverage(torch.sigmoid(logits), torch.tensor([True, True, False, True]))
    assert len(curve) == 11
    assert (
        point_in_target_accuracy(
            [(0.5, 0.5), (0.1, 0.1)], [(0.0, 0.0, 1.0, 1.0), (0.2, 0.2, 0.3, 0.3)]
        )
        == 0.5
    )
    assert retrieval_recall([2, 1], [[2, 3], [4, 1]], k=1) == 0.5
    assert target_survival_rate([2, None], [[2], [1]]) == 1.0
    assert selection_accuracy_when_retrieved((1, 2, 3), ((1, 4), (5,), (3, 7)), (1, 5, 7)) == 0.5
    assert (
        point_accuracy_given_candidate(
            ((0.5, 0.5), (0.1, 0.1)),
            ((0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 0.2, 0.2)),
            (True, False),
        )
        == 1.0
    )
    assert action_parameter_error(((1.0, 0.0),), ((0.0, 0.5),), ((True, False),)) == 1.0
    result, peak = peak_python_memory(lambda: [0] * 10)
    assert len(result) == 10 and peak > 0


def test_threshold_policy_respects_equal_confidence_groups() -> None:
    policy = fit_threshold_policy(
        torch.tensor([0.9, 0.8, 0.8, 0.1]),
        torch.tensor([True, True, False, False]),
        target_risk=0.34,
    )
    assert policy.threshold == pytest.approx(0.8)
    assert policy.accepted == 3
    assert policy.coverage == 0.75
