"""Minimal resumable trainer usable for CPU smoke tests and later DDP."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    """Device-safe trainer controls."""

    device: str = "cpu"
    gradient_accumulation_steps: int = 1
    use_amp: bool = False
    grad_clip_norm: float = 1.0


class Trainer:
    """Run a model/loss callback with accumulation and optional CPU-safe AMP."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        config: TrainerConfig | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config or TrainerConfig()
        if self.config.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        amp_enabled = self.config.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    def train_epoch(
        self,
        batches: Iterable[object],
        loss_function: Callable[[nn.Module, object], torch.Tensor],
    ) -> float:
        """Run one epoch and return mean detached loss."""

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        values: list[float] = []
        for index, batch in enumerate(batches):
            with torch.autocast(
                device_type=self.device.type,
                enabled=self.config.use_amp and self.device.type == "cuda",
            ):
                loss = loss_function(self.model, batch)
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise ValueError("training loss must be a finite scalar")
            self.scaler.scale(loss / self.config.gradient_accumulation_steps).backward()
            if (index + 1) % self.config.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
            values.append(float(loss.detach().cpu()))
        if not values:
            raise ValueError("batches cannot be empty")
        return sum(values) / len(values)
