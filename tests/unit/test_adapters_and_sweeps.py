from __future__ import annotations

from screen2action.config import load_config
from screen2action.data.adapters.base import validate_adapter_output
from screen2action.data.adapters.synthetic import SyntheticAdapter
from screen2action.eval.sweep_runner import build_sweep_jobs
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
    assert len(ablations) == 14


def test_required_experiment_sweep_expands_with_honest_checkpoint_boundaries() -> None:
    jobs = build_sweep_jobs(load_config("configs/experiments/budget_k_sweep.yaml"))
    budget_jobs = [job for job in jobs if job.family == "budget_k"]
    by_name = {job.name: job for job in jobs}

    assert len(budget_jobs) == 9
    assert by_name["no_target_survival_loss"].checkpoint_variant == ("no_target_survival_loss")
    assert by_name["quantization_qat_fake_quant"].checkpoint_variant == (
        "quantization_qat_fake_quant"
    )
    assert by_name["roi_align_fast"].supported is False


def test_reference_stage_schedule_is_explicit() -> None:
    stage = stage_spec("stage2_selector_retrieval")
    assert stage.epochs == 8
    assert stage.weight_decay == 0.05
    assert forced_positive_probability(stage.name, 0) == 0.5
    assert forced_positive_probability(stage.name, 1) == 0.5
    assert forced_positive_probability(stage.name, 2) == 0.0
