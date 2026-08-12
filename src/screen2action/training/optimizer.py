"""Reference AdamW parameter groups and warmup/cosine scheduling."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class AccumulationPlan:
    """Global-batch arithmetic reported before training starts."""

    desired_global_batch: int
    per_device_batch: int
    world_size: int
    accumulation_steps: int
    effective_global_batch: int


def build_adamw(
    model: nn.Module,
    *,
    pretrained_prefixes: tuple[str, ...] = (),
    new_head_lr: float = 2e-4,
    pretrained_lr: float = 2e-5,
    weight_decay: float = 0.05,
) -> torch.optim.AdamW:
    """Build separate LR groups without device-specific assumptions."""

    pretrained: list[nn.Parameter] = []
    new_heads: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (pretrained if name.startswith(pretrained_prefixes) else new_heads).append(parameter)
    groups = []
    if new_heads:
        groups.append({"params": new_heads, "lr": new_head_lr, "group_name": "new_heads"})
    if pretrained:
        groups.append(
            {
                "params": pretrained,
                "lr": pretrained_lr,
                "group_name": "pretrained_encoders",
            }
        )
    if not groups:
        raise ValueError("model has no trainable parameters")
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def accumulation_plan(
    *,
    global_batch_size: int,
    per_device_batch_size: int,
    world_size: int,
) -> AccumulationPlan:
    """Compute ceiling accumulation and expose the effective global batch."""

    if min(global_batch_size, per_device_batch_size, world_size) <= 0:
        raise ValueError("global, per-device, and world-size batch values must be positive")
    accumulation = math.ceil(global_batch_size / (per_device_batch_size * world_size))
    return AccumulationPlan(
        global_batch_size,
        per_device_batch_size,
        world_size,
        accumulation,
        accumulation * per_device_batch_size * world_size,
    )


def warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_fraction: float = 0.05,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a linear-warmup/cosine-decay scheduler."""

    if total_steps <= 0 or not 0.0 <= warmup_fraction < 1.0:
        raise ValueError("scheduler arguments are invalid")
    warmup_steps = max(1, round(total_steps * warmup_fraction))

    def scale(step: int) -> float:
        if step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-8)
        progress = min(max((step - warmup_steps) / max(total_steps - warmup_steps, 1), 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_fraction: float = 0.05,
    schedule: str = "cosine",
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a configurable warm-up schedule with a stable step contract."""

    if total_steps <= 0 or not 0.0 <= warmup_fraction < 1.0:
        raise ValueError("scheduler arguments are invalid")
    if schedule not in {"cosine", "linear", "constant"}:
        raise ValueError("scheduler must be cosine, linear, or constant")
    warmup_steps = max(1, round(total_steps * warmup_fraction))

    def scale(step: int) -> float:
        if step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-8)
        progress = min(
            max((step - warmup_steps) / max(total_steps - warmup_steps, 1), 0.0),
            1.0,
        )
        if schedule == "constant":
            return 1.0
        if schedule == "linear":
            return 1.0 - progress
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
