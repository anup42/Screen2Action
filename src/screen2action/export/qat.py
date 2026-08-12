"""Optional CPU QAT preparation."""

from __future__ import annotations

import copy
from typing import cast

import torch
from torch import nn


def prepare_qat_cpu(model: nn.Module) -> nn.Module:
    """Prepare a CPU model for fake-quantization fine-tuning."""

    prepared = copy.deepcopy(model).cpu().train()
    prepared.qconfig = torch.ao.quantization.get_default_qat_qconfig("fbgemm")
    return cast(nn.Module, torch.ao.quantization.prepare_qat(prepared, inplace=False))
