"""CPU dynamic PTQ helper for supported Linear modules."""

from __future__ import annotations

import copy
from typing import cast

import torch
from torch import nn


def dynamic_ptq(model: nn.Module) -> nn.Module:
    """Return a detached dynamically quantized CPU copy."""

    if next(model.parameters(), torch.empty(0)).is_cuda:
        raise ValueError("dynamic PTQ must be prepared from a CPU model")
    return cast(
        nn.Module,
        torch.ao.quantization.quantize_dynamic(
            copy.deepcopy(model).cpu().eval(), {nn.Linear}, dtype=torch.qint8
        ),
    )
