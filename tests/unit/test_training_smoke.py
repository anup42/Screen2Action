"""Device-safe training smoke tests."""

from pathlib import Path

from screen2action.training.smoke import run_training_smoke


def test_cpu_smoke_forward_backward_checkpoint_resume(tmp_path: Path) -> None:
    result = run_training_smoke(device="cpu", steps=2, run_directory=tmp_path)

    assert result.checkpoint_resume_parity
    assert result.resumed_step == 2
    assert result.checkpoint_path == "smoke-checkpoint.pt"
    assert (tmp_path / result.checkpoint_path).is_file()
    assert result.initial_loss > 0.0
    assert result.final_loss > 0.0
