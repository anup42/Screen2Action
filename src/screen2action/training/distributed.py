"""Torchrun-compatible process/device lifecycle and metric synchronization."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any, cast

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True, slots=True)
class DistributedContext:
    """Validated rank, backend, and device for one training process."""

    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: str
    process_group_initialized: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    def barrier(self) -> None:
        if self.process_group_initialized:
            torch.distributed.barrier()

    def reduce_sum(self, value: torch.Tensor) -> torch.Tensor:
        output = value.detach().clone()
        if self.process_group_initialized:
            torch.distributed.all_reduce(output, op=torch.distributed.ReduceOp.SUM)
        return output

    def reduce_mean(self, value: torch.Tensor) -> torch.Tensor:
        return self.reduce_sum(value) / self.world_size

    def all_gather_mappings(
        self,
        value: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], ...]:
        """Gather small checkpoint metadata without tensor/device assumptions."""

        if not self.process_group_initialized:
            return (dict(value),)
        gathered: list[object | None] = [None] * self.world_size
        torch.distributed.all_gather_object(gathered, dict(value))
        if any(not isinstance(item, dict) for item in gathered):
            raise RuntimeError("distributed metadata gather returned a malformed value")
        return tuple(cast(dict[str, Any], item) for item in gathered)

    def broadcast_object(self, value: object | None, *, source: int = 0) -> object:
        """Broadcast one small launch/artifact value from the selected rank."""

        if not 0 <= source < self.world_size:
            raise ValueError("broadcast source rank is invalid")
        if not self.process_group_initialized:
            if value is None:
                raise ValueError("single-process broadcast value cannot be None")
            return value
        values = [value]
        torch.distributed.broadcast_object_list(values, src=source)
        if values[0] is None:
            raise RuntimeError("distributed object broadcast returned None")
        return values[0]

    def cleanup(self) -> None:
        if self.process_group_initialized and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def distributed_available() -> bool:
    """Return whether the installed Torch build exposes distributed support."""

    return torch.distributed.is_available()


def _environment_integer(name: str, default: int) -> int:
    raw = os.environ.get(name)
    try:
        value = default if raw is None else int(raw)
    except ValueError as error:
        raise RuntimeError(f"distributed variable {name} must be an integer") from error
    if value < 0:
        raise RuntimeError(f"distributed variable {name} cannot be negative")
    return value


def initialize_distributed(device: str = "cpu") -> DistributedContext:
    """Validate torchrun variables, choose Gloo/NCCL, and initialize once."""

    if device not in {"cpu", "cuda"}:
        raise ValueError("distributed device must be cpu or cuda")
    rank = _environment_integer("RANK", 0)
    local_rank = _environment_integer("LOCAL_RANK", 0)
    world_size = _environment_integer("WORLD_SIZE", 1)
    if world_size <= 0 or rank >= world_size:
        raise RuntimeError("distributed rank must be in [0, WORLD_SIZE)")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA training requested but torch.cuda.is_available() is false")
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError("LOCAL_RANK is outside the visible CUDA device range")
        torch.cuda.set_device(local_rank)
        target = torch.device("cuda", local_rank)
        backend = "nccl"
        if world_size > 1 and (
            not torch.distributed.is_nccl_available() or not torch.distributed.is_available()
        ):
            raise RuntimeError("multi-GPU training requires NCCL support")
    else:
        if local_rank >= world_size:
            raise RuntimeError("CPU LOCAL_RANK must be in [0, WORLD_SIZE)")
        target = torch.device("cpu")
        backend = "gloo"
    initialized = torch.distributed.is_initialized()
    if world_size > 1 and not initialized:
        if not distributed_available():
            raise RuntimeError("torch.distributed is unavailable in this build")
        required = ("MASTER_ADDR", "MASTER_PORT")
        missing = [name for name in required if name not in os.environ]
        if missing:
            raise RuntimeError(f"distributed launcher variables are missing: {missing}")
        torch.distributed.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
        )
        initialized = True
    if initialized:
        observed_rank = torch.distributed.get_rank()
        observed_world = torch.distributed.get_world_size()
        if (observed_rank, observed_world) != (rank, world_size):
            raise RuntimeError("initialized process group disagrees with launcher environment")
    return DistributedContext(rank, local_rank, world_size, target, backend, initialized)


def initialize_process_group(backend: str = "gloo") -> None:
    """Backward-compatible explicit process-group initialization."""

    if backend not in {"gloo", "nccl"}:
        raise ValueError("backend must be gloo or nccl")
    context = initialize_distributed("cuda" if backend == "nccl" else "cpu")
    if context.backend != backend:
        raise RuntimeError("requested backend does not match the selected device")


def _detach_diagnostics(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach() if value.requires_grad else value
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value,
            **{
                field.name: _detach_diagnostics(getattr(value, field.name))
                for field in fields(value)
            },
        )
    if isinstance(value, tuple):
        return tuple(_detach_diagnostics(item) for item in value)
    if isinstance(value, list):
        return [_detach_diagnostics(item) for item in value]
    if isinstance(value, dict):
        return {key: _detach_diagnostics(item) for key, item in value.items()}
    return value


class OptimizationOutputAdapter(nn.Module):
    """Expose only the optimized loss graph to DDP unused-parameter discovery.

    Structured stage outputs contain predictions from heads without labels on a
    particular rank. Their diagnostic tensors must not mark those heads as used.
    The underlying public model retains its ordinary differentiable API.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        output = self.module(*args, **kwargs)
        if is_dataclass(output) and not isinstance(output, type) and hasattr(output, "total_loss"):
            return replace(_detach_diagnostics(output), total_loss=output.total_loss)
        return output


def wrap_ddp(
    model: nn.Module,
    context: DistributedContext,
    *,
    find_unused_parameters: bool = False,
) -> nn.Module:
    """Move a model and wrap it exactly once when the world has multiple ranks."""

    model.to(context.device)
    if not context.distributed:
        return model
    if not context.process_group_initialized:
        raise RuntimeError("cannot wrap DDP without an initialized process group")
    if find_unused_parameters:
        model = OptimizationOutputAdapter(model)
    if context.device.type == "cuda":
        return DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
            find_unused_parameters=find_unused_parameters,
        )
    return DistributedDataParallel(
        model,
        find_unused_parameters=find_unused_parameters,
    )
