from __future__ import annotations

from screen2action.data.adapters.base import validate_adapter_output
from screen2action.data.adapters.synthetic import SyntheticAdapter
from screen2action.eval.sweeps import run_budget_k_sweep, standard_ablations
from screen2action.training.stages import forced_positive_probability, stage_spec


def test_synthetic_adapter_and_configured_sweeps() -> None:
    adapter = SyntheticAdapter()
    screens = adapter.screens()
    commands = adapter.commands()
    validate_adapter_output(screens, commands)
    assert adapter.audit().accepted_commands == len(commands)
    budget_results = run_budget_k_sweep(
        (128, 256), (4, 8), lambda budget, top_k: {"score": budget + top_k}
    )
    assert len(budget_results) == 4
    ablations = standard_ablations(lambda name: {"score": float(len(name))})
    assert len(ablations) == 8


def test_reference_stage_schedule_is_explicit() -> None:
    stage = stage_spec("stage2_selector_retrieval")
    assert stage.epochs == 8
    assert stage.weight_decay == 0.05
    assert forced_positive_probability(stage.name, 0) == 0.5
    assert forced_positive_probability(stage.name, 1) == 0.5
    assert forced_positive_probability(stage.name, 2) == 0.0
