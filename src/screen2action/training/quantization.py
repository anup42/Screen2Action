"""Explicit per-channel QAT/PTQ wrappers and representative activation calibration."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional as F


def _weight_scale(weight: torch.Tensor) -> torch.Tensor:
    flattened = weight.detach().abs().flatten(1)
    return (flattened.amax(dim=1) / 127.0).clamp_min(torch.finfo(torch.float32).eps)


def _fake_quantize_weight(weight: torch.Tensor) -> torch.Tensor:
    scale = _weight_scale(weight).to(weight.device, dtype=torch.float32)
    zero_point = torch.zeros(scale.shape, dtype=torch.int32, device=weight.device)
    return torch.fake_quantize_per_channel_affine(
        weight,
        scale,
        zero_point,
        0,
        -127,
        127,
    )


class PerChannelQatLinear(nn.Linear):
    """Trainable Linear using straight-through per-output-channel int8 weights."""

    fake_quant_enabled: bool

    @classmethod
    def from_float(cls, module: nn.Linear) -> PerChannelQatLinear:
        converted = cls(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        converted.load_state_dict(module.state_dict())
        converted.fake_quant_enabled = True
        return converted

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = _fake_quantize_weight(self.weight) if self.fake_quant_enabled else self.weight
        return F.linear(input, weight, self.bias)


class PerChannelQatConv2d(nn.Conv2d):
    """Trainable Conv2d using straight-through per-output-channel int8 weights."""

    fake_quant_enabled: bool

    @classmethod
    def from_float(cls, module: nn.Conv2d) -> PerChannelQatConv2d:
        converted = cls(
            module.in_channels,
            module.out_channels,
            cast(tuple[int, int], module.kernel_size),
            stride=cast(tuple[int, int], module.stride),
            padding=cast(Any, module.padding),
            dilation=cast(tuple[int, int], module.dilation),
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        converted.load_state_dict(module.state_dict())
        converted.fake_quant_enabled = True
        return converted

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = _fake_quantize_weight(self.weight) if self.fake_quant_enabled else self.weight
        return cast(torch.Tensor, self._conv_forward(input, weight, self.bias))


class PerChannelPtqLinear(nn.Module):
    quantized_weight: torch.Tensor
    weight_scale: torch.Tensor
    bias_value: torch.Tensor

    def __init__(self, module: nn.Linear) -> None:
        super().__init__()
        scale = _weight_scale(module.weight).to(module.weight.device)
        quantized = torch.round(module.weight.detach() / scale.unsqueeze(1)).clamp(-127, 127)
        self.register_buffer("quantized_weight", quantized.to(torch.int8))
        self.register_buffer("weight_scale", scale)
        self.register_buffer(
            "bias_value",
            module.bias.detach().clone() if module.bias is not None else torch.empty(0),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.quantized_weight.to(input.dtype) * self.weight_scale.to(
            input.dtype
        ).unsqueeze(1)
        bias = self.bias_value.to(input.dtype) if self.bias_value.numel() else None
        return F.linear(input, weight, bias)


class PerChannelPtqConv2d(nn.Module):
    quantized_weight: torch.Tensor
    weight_scale: torch.Tensor
    bias_value: torch.Tensor
    stride: tuple[int, int]
    padding: str | tuple[int, int]
    dilation: tuple[int, int]
    groups: int

    def __init__(self, module: nn.Conv2d) -> None:
        super().__init__()
        scale = _weight_scale(module.weight).to(module.weight.device)
        shape = (module.out_channels,) + (1,) * (module.weight.ndim - 1)
        quantized = torch.round(module.weight.detach() / scale.view(shape)).clamp(-127, 127)
        self.register_buffer("quantized_weight", quantized.to(torch.int8))
        self.register_buffer("weight_scale", scale)
        self.register_buffer(
            "bias_value",
            module.bias.detach().clone() if module.bias is not None else torch.empty(0),
        )
        self.stride = cast(tuple[int, int], module.stride)
        self.padding = cast(str | tuple[int, int], module.padding)
        self.dilation = cast(tuple[int, int], module.dilation)
        self.groups = module.groups

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        shape = (self.quantized_weight.shape[0],) + (1,) * (self.quantized_weight.ndim - 1)
        weight = self.quantized_weight.to(input.dtype) * self.weight_scale.to(input.dtype).view(
            shape
        )
        bias = self.bias_value.to(input.dtype) if self.bias_value.numel() else None
        return F.conv2d(
            input,
            weight,
            bias,
            stride=self.stride,
            padding=cast(Any, self.padding),
            dilation=self.dilation,
            groups=self.groups,
        )


def _replace_children(module: nn.Module, *, mode: str) -> int:
    if isinstance(module, nn.MultiheadAttention | nn.TransformerEncoderLayer):
        return 0
    replaced = 0
    for name, child in tuple(module.named_children()):
        replacement: nn.Module | None = None
        if (
            mode == "qat"
            and isinstance(child, nn.Linear)
            and not isinstance(child, PerChannelQatLinear)
        ):
            replacement = PerChannelQatLinear.from_float(child)
        elif (
            mode == "qat"
            and isinstance(child, nn.Conv2d)
            and not isinstance(child, PerChannelQatConv2d)
        ):
            replacement = PerChannelQatConv2d.from_float(child)
        elif mode == "ptq" and isinstance(child, nn.Linear):
            replacement = PerChannelPtqLinear(child)
        elif mode == "ptq" and isinstance(child, nn.Conv2d):
            replacement = PerChannelPtqConv2d(child)
        if replacement is not None:
            setattr(module, name, replacement)
            replaced += 1
        else:
            replaced += _replace_children(child, mode=mode)
    return replaced


def enable_per_channel_qat(model: nn.Module) -> int:
    """Replace supported layers in place and return the converted layer count."""

    return _replace_children(model, mode="qat")


def convert_per_channel_ptq(model: nn.Module) -> int:
    """Replace supported FP/QAT layers in place with weight-only int8 wrappers."""

    set_qat_fake_quant(model, False)
    return _replace_children(model, mode="ptq")


def set_qat_fake_quant(model: nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, PerChannelQatLinear | PerChannelQatConv2d):
            module.fake_quant_enabled = enabled


def quantization_inventory(model: nn.Module) -> dict[str, object]:
    """Report exactly converted layers and trainable operations left in FP."""

    supported: list[dict[str, object]] = []
    unsupported: list[dict[str, object]] = []
    for name, module in model.named_modules():
        if not name:
            continue
        if isinstance(module, PerChannelQatLinear | PerChannelQatConv2d):
            supported.append(
                {
                    "path": name,
                    "operation": type(module).__name__,
                    "weight_quantization": "symmetric_int8_per_output_channel",
                }
            )
            continue
        direct_parameters = tuple(module.parameters(recurse=False))
        if direct_parameters:
            unsupported.append(
                {
                    "path": name,
                    "operation": type(module).__name__,
                    "reason": "kept_fp32_or_non_weight_operation",
                }
            )
    return {
        "supported_weight_subgraphs": supported,
        "unsupported_or_fp_subgraphs": unsupported,
        "mobile_runtime_measured": False,
    }


@dataclass(slots=True)
class ActivationCalibrator:
    """Min/max observers for representative train-only executions."""

    ranges: dict[str, tuple[float, float]]
    _handles: list[torch.utils.hooks.RemovableHandle]

    @classmethod
    def attach(cls, model: nn.Module) -> ActivationCalibrator:
        calibrator = cls({}, [])
        supported = (
            PerChannelQatLinear,
            PerChannelQatConv2d,
            nn.Linear,
            nn.Conv2d,
        )
        for name, module in model.named_modules():
            if name and isinstance(module, supported):
                calibrator._handles.append(module.register_forward_hook(calibrator._hook(name)))
        if not calibrator._handles:
            raise ValueError("calibration found no supported Linear or Conv2d operations")
        return calibrator

    def _hook(
        self,
        name: str,
    ) -> Callable[[nn.Module, tuple[object, ...], object], None]:
        def observe(module: nn.Module, inputs: tuple[object, ...], output: object) -> None:
            del module, inputs
            tensors = tuple(_output_tensors(output))
            if not tensors:
                return
            minimum = min(float(tensor.detach().float().amin().cpu()) for tensor in tensors)
            maximum = max(float(tensor.detach().float().amax().cpu()) for tensor in tensors)
            previous = self.ranges.get(name)
            self.ranges[name] = (
                minimum if previous is None else min(previous[0], minimum),
                maximum if previous is None else max(previous[1], maximum),
            )

        return observe

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def report(self) -> Mapping[str, object]:
        return {
            "observer": "finite_minmax_v1",
            "activation_ranges": {
                name: {"minimum": values[0], "maximum": values[1]}
                for name, values in sorted(self.ranges.items())
            },
        }


def _output_tensors(value: object) -> Iterator[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for nested in value.values():
            yield from _output_tensors(nested)
    elif isinstance(value, tuple | list):
        for nested in value:
            yield from _output_tensors(nested)
