from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

from screen2action.training.checkpoints import (
    configuration_hash,
    load_checkpoint,
    save_checkpoint,
)
from screen2action.training.engine import LossResult, Trainer, TrainerConfig
from screen2action.training.optimizer import accumulation_plan


class CountingLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)
        nn.init.zeros_(self.linear.weight)
        self.no_sync_calls = 0

    @contextmanager
    def no_sync(self):
        self.no_sync_calls += 1
        yield

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value)


def _regression_loss(model: nn.Module, batch: object, step: int) -> LossResult:
    del step
    features, targets = batch
    predicted = model(features)
    loss = torch.nn.functional.mse_loss(predicted, targets)
    return LossResult(loss, {"mse": loss.detach()}, len(features))


def test_final_partial_accumulation_flushes_and_uses_no_sync() -> None:
    model = CountingLinear()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    trainer = Trainer(
        model,
        optimizer,
        config=TrainerConfig(gradient_accumulation_steps=2),
    )
    zero = (torch.ones((1, 1)), torch.zeros((1, 1)))
    final = (torch.ones((1, 1)), torch.ones((1, 1)))

    result = trainer.train_epoch([zero, zero, zero, zero, final], _regression_loss)

    assert result.complete
    assert result.micro_batches == 5
    assert result.optimizer_steps == 3
    assert trainer.optimizer_step == 3
    assert model.no_sync_calls == 2
    assert float(model.linear.weight.detach()) > 0.0


def test_accumulation_plan_reports_ceiling_effective_batch() -> None:
    plan = accumulation_plan(global_batch_size=256, per_device_batch_size=24, world_size=2)
    assert plan.accumulation_steps == 6
    assert plan.effective_global_batch == 288


def test_v2_checkpoint_is_atomic_rank_zero_only_and_digest_strict(tmp_path: Path) -> None:
    model = CountingLinear()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    config = {"schema_version": 1, "name": "fixture"}
    checkpoint = tmp_path / "latest.pt"

    assert (
        save_checkpoint(
            tmp_path / "rank-one.pt",
            model,
            optimizer,
            epoch=2,
            step=7,
            rank=1,
        )
        is None
    )
    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        scheduler=scheduler,
        epoch=2,
        step=7,
        next_batch_index=3,
        optimizer_step=4,
        best_metric=0.25,
        sampler_state={"epoch": 2, "seed": 17},
        config=config,
        data_manifest_hash="data-v1",
        model_lock_hash="model-v1",
        cache_manifest_hash="cache-v1",
        run_manifest_hash="run-v1",
    )
    assert checkpoint.is_file()
    assert not tuple(tmp_path.glob("*.partial"))

    resumed = CountingLinear()
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.01)
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(resumed_optimizer, lambda _: 1.0)
    state = load_checkpoint(
        checkpoint,
        resumed,
        resumed_optimizer,
        scheduler=resumed_scheduler,
        expected_config_hash=configuration_hash(config),
        expected_manifest_hash="data-v1",
        expected_model_lock_hash="model-v1",
        expected_cache_manifest_hash="cache-v1",
    )
    assert (state.epoch, state.next_batch_index, state.optimizer_step) == (2, 3, 4)
    assert state.best_metric == 0.25
    assert state.sampler_state == {"epoch": 2, "seed": 17}
