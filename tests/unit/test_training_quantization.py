from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from screen2action.training.quantization import (
    ActivationCalibrator,
    PerChannelPtqConv2d,
    PerChannelPtqLinear,
    PerChannelQatConv2d,
    PerChannelQatLinear,
    convert_per_channel_ptq,
    enable_per_channel_qat,
    quantization_inventory,
    set_qat_fake_quant,
)


def _model() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(4 * 8 * 8, 3),
    )


def test_per_channel_qat_ptq_and_calibration_are_real_and_finite() -> None:
    torch.manual_seed(7)
    original = _model().eval()
    value = torch.randn(2, 3, 8, 8)
    expected = original(value)
    qat = deepcopy(original)

    assert enable_per_channel_qat(qat) == 2
    assert isinstance(qat[0], PerChannelQatConv2d)
    assert isinstance(qat[3], PerChannelQatLinear)
    set_qat_fake_quant(qat, False)
    assert torch.allclose(qat(value), expected, atol=0.0, rtol=0.0)

    set_qat_fake_quant(qat, True)
    calibrator = ActivationCalibrator.attach(qat)
    output = qat(value)
    output.square().mean().backward()
    calibrator.close()
    assert torch.isfinite(output).all()
    assert qat[0].weight.grad is not None
    assert qat[3].weight.grad is not None
    assert calibrator.report()["activation_ranges"]
    inventory = quantization_inventory(qat)
    assert len(inventory["supported_weight_subgraphs"]) == 2
    assert inventory["mobile_runtime_measured"] is False

    ptq = deepcopy(qat).eval()
    assert convert_per_channel_ptq(ptq) == 2
    assert isinstance(ptq[0], PerChannelPtqConv2d)
    assert isinstance(ptq[3], PerChannelPtqLinear)
    ptq_output = ptq(value)
    assert ptq_output.shape == expected.shape
    assert torch.isfinite(ptq_output).all()
