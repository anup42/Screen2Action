"""Numerical PyTorch/ONNXRuntime parity checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


def onnx_parity(
    model: nn.Module,
    onnx_path: str | Path,
    inputs: tuple[torch.Tensor, ...],
    *,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> tuple[float, ...]:
    """Run CPU ONNXRuntime and return maximum absolute errors per output."""

    try:
        import onnxruntime as ort  # type: ignore[import-untyped]
    except ImportError as error:
        raise RuntimeError("onnxruntime is required for parity validation") from error
    model.eval()
    with torch.no_grad():
        torch_outputs = model(*inputs)
    if isinstance(torch_outputs, torch.Tensor):
        torch_outputs = (torch_outputs,)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_names = [value.name for value in session.get_inputs()]
    output_values = session.run(
        None,
        {
            name: tensor.detach().cpu().numpy()
            for name, tensor in zip(input_names, inputs, strict=True)
        },
    )
    errors: list[float] = []
    for expected, actual in zip(torch_outputs, output_values, strict=True):
        expected_np = expected.detach().cpu().numpy()
        if not np.allclose(expected_np, actual, atol=atol, rtol=rtol):
            raise AssertionError("ONNX output is outside numerical parity tolerance")
        errors.append(float(np.max(np.abs(expected_np - actual))))
    return tuple(errors)
