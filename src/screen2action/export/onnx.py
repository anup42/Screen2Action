"""Backend-neutral ONNX export entry point."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn


def export_onnx(
    model: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    destination: str | Path,
    *,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    opset_version: int = 17,
) -> Path:
    """Export a CPU-evaluable model with fixed example shapes."""

    if not example_inputs:
        raise ValueError("example_inputs cannot be empty")
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                example_inputs,
                destination_path,
                input_names=list(
                    input_names or (f"input_{index}" for index in range(len(example_inputs)))
                ),
                output_names=list(output_names or (f"output_{index}" for index in range(2))),
                opset_version=opset_version,
                do_constant_folding=True,
                dynamo=False,
            )
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)
    return destination_path
