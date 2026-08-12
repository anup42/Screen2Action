"""Reference AdamW parameter groups and warmup/cosine scheduling."""

from __future__ import annotations

import math

import torch
from torch import nn


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
        groups.append({"params": new_heads, "lr": new_head_lr})
    if pretrained:
        groups.append({"params": pretrained, "lr": pretrained_lr})
    if not groups:
        raise ValueError("model has no trainable parameters")
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


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
