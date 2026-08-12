"""Stage-neutral training engine with exact accumulation and DDP synchronization."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import nn

from screen2action.training.distributed import DistributedContext


class NoSyncModule(Protocol):
    def no_sync(self) -> Any:
        """Suppress DDP gradient synchronization for one context."""


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    """Device, accumulation, precision, clipping, and logging controls."""

    device: str = "cpu"
    gradient_accumulation_steps: int = 1
    precision: str = "fp32"
    grad_clip_norm: float = 1.0
    log_interval: int = 10

    def __post_init__(self) -> None:
        if self.gradient_accumulation_steps <= 0 or self.grad_clip_norm <= 0.0:
            raise ValueError("accumulation and gradient clipping must be positive")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")


@dataclass(frozen=True, slots=True)
class LossResult:
    """Finite scalar loss plus detached component metrics and example count."""

    loss: torch.Tensor
    components: Mapping[str, torch.Tensor | float]
    examples: int
    data_time_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class EpochResult:
    """Aggregated epoch measurements and exact next cursor."""

    mean_loss: float
    micro_batches: int
    optimizer_steps: int
    examples: int
    next_batch_index: int
    complete: bool
    metrics: Mapping[str, float]


def select_precision(device: torch.device, requested: str = "auto") -> str:
    """Select BF16 when supported, otherwise FP16 on CUDA and FP32 on CPU."""

    if requested not in {"auto", "fp32", "fp16", "bf16"}:
        raise ValueError("precision must be auto, fp32, fp16, or bf16")
    if device.type == "cpu":
        if requested in {"fp16", "bf16"}:
            raise ValueError("CPU stage training supports fp32 precision")
        return "fp32"
    if requested != "auto":
        if requested == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 was requested but the selected GPU does not support it")
        return requested
    return "bf16" if torch.cuda.is_bf16_supported() else "fp16"


def _autocast_context(device: torch.device, precision: str) -> Any:
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


def _gradient_is_finite(model: nn.Module) -> bool:
    return all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def _component_value(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or not bool(torch.isfinite(value).all()):
            raise ValueError("loss components must be finite scalar tensors")
        return float(value.detach().cpu())
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("loss components must be finite")
    return result


class Trainer:
    """Run train/validation epochs with final flush, no-sync, AMP, and reductions."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        config: TrainerConfig | None = None,
        distributed: DistributedContext | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config or TrainerConfig()
        self.device = torch.device(self.config.device)
        self.distributed = distributed or DistributedContext(
            0,
            0,
            1,
            self.device,
            "gloo",
            False,
        )
        if self.distributed.device != self.device:
            raise ValueError("trainer and distributed devices differ")
        self.model.to(self.device)
        scaler_enabled = self.device.type == "cuda" and self.config.precision == "fp16"
        self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        self.global_step = 0
        self.optimizer_step = 0

    def _sync_context(self, synchronize: bool) -> Any:
        no_sync = getattr(self.model, "no_sync", None)
        if synchronize or not callable(no_sync):
            return nullcontext()
        return no_sync()

    def _peak_memory(self) -> int:
        if self.device.type != "cuda":
            return 0
        return int(torch.cuda.max_memory_allocated(self.device))

    def train_epoch(
        self,
        batches: Sequence[object],
        loss_function: Callable[[nn.Module, object, int], LossResult | torch.Tensor],
        *,
        start_batch: int = 0,
        max_optimizer_steps: int | None = None,
        event_callback: Callable[[Mapping[str, object]], None] | None = None,
        step_callback: Callable[[int, int], None] | None = None,
    ) -> EpochResult:
        """Train from an exact batch cursor and always flush the final partial group."""

        if not batches or not 0 <= start_batch <= len(batches):
            raise ValueError("training batches and start cursor are invalid")
        if max_optimizer_steps is not None and max_optimizer_steps <= 0:
            raise ValueError("max_optimizer_steps must be positive")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        loss_sum = 0.0
        example_sum = 0
        component_sums: dict[str, float] = {}
        consumed = 0
        optimizer_steps = 0
        started = time.perf_counter()
        data_mark = started
        index = start_batch
        while index < len(batches):
            group_remaining = len(batches) - index
            group_size = min(self.config.gradient_accumulation_steps, group_remaining)
            for group_index in range(group_size):
                batch_index = index + group_index
                batch = batches[batch_index]
                data_time = time.perf_counter() - data_mark
                synchronize = group_index == group_size - 1
                model_started = time.perf_counter()
                with (
                    self._sync_context(synchronize),
                    _autocast_context(
                        self.device,
                        self.config.precision,
                    ),
                ):
                    raw_result = loss_function(self.model, batch, self.global_step)
                    result = (
                        raw_result
                        if isinstance(raw_result, LossResult)
                        else LossResult(raw_result, {}, 1)
                    )
                    loss = result.loss
                    scaled_loss = loss / group_size
                if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"non-finite scalar loss at micro-batch {batch_index}")
                if result.examples <= 0:
                    raise ValueError("training loss result must count positive examples")
                self.scaler.scale(scaled_loss).backward()
                total_compute_time = time.perf_counter() - model_started
                data_time += result.data_time_seconds
                model_time = max(total_compute_time - result.data_time_seconds, 0.0)
                value = float(loss.detach().cpu())
                loss_sum += value * result.examples
                example_sum += result.examples
                for name, component in result.components.items():
                    component_sums[name] = component_sums.get(name, 0.0) + (
                        _component_value(component) * result.examples
                    )
                consumed += 1
                self.global_step += 1
                if event_callback is not None and (
                    self.global_step % self.config.log_interval == 0 or synchronize
                ):
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    remaining = len(batches) - (batch_index + 1)
                    event_callback(
                        {
                            "event": "train_micro_batch",
                            "global_step": self.global_step,
                            "batch_index": batch_index,
                            "loss": value,
                            "data_time_seconds": data_time,
                            "model_time_seconds": model_time,
                            "throughput_examples_per_second": example_sum / elapsed,
                            "eta_seconds": elapsed / consumed * remaining,
                            "peak_vram_bytes": self._peak_memory(),
                        }
                    )
                data_mark = time.perf_counter()
            self.scaler.unscale_(self.optimizer)
            if not _gradient_is_finite(self.model):
                raise FloatingPointError("non-finite gradients before optimizer step")
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip_norm,
                error_if_nonfinite=True,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            self.optimizer_step += 1
            index += group_size
            if event_callback is not None:
                event_callback(
                    {
                        "event": "optimizer_step",
                        "optimizer_step": self.optimizer_step,
                        "next_batch_index": index,
                        "gradient_norm": float(grad_norm.detach().cpu()),
                        "learning_rates": [
                            float(group["lr"]) for group in self.optimizer.param_groups
                        ],
                    }
                )
            if step_callback is not None:
                step_callback(index, self.optimizer_step)
            if max_optimizer_steps is not None and optimizer_steps >= max_optimizer_steps:
                break
        totals = torch.tensor(
            [loss_sum, float(example_sum), float(consumed), float(optimizer_steps)],
            dtype=torch.float64,
            device=self.device,
        )
        totals = self.distributed.reduce_sum(totals)
        global_examples = int(totals[1].item())
        if global_examples <= 0:
            raise ValueError("training epoch produced zero examples")
        reduced_components: dict[str, float] = {}
        for name, value in component_sums.items():
            reduced = self.distributed.reduce_sum(
                torch.tensor(value, dtype=torch.float64, device=self.device)
            )
            reduced_components[name] = float(reduced.item() / global_examples)
        complete = index >= len(batches)
        return EpochResult(
            mean_loss=float(totals[0].item() / global_examples),
            micro_batches=int(totals[2].item()),
            optimizer_steps=int(totals[3].item()),
            examples=global_examples,
            next_batch_index=0 if complete else index,
            complete=complete,
            metrics=reduced_components,
        )

    def validate(
        self,
        batches: Sequence[object],
        loss_function: Callable[[nn.Module, object, int], LossResult | torch.Tensor],
    ) -> EpochResult:
        """Run deterministic no-gradient validation with reduced direct metrics."""

        if not batches and not self.distributed.distributed:
            raise ValueError("validation batches cannot be empty")
        self.model.eval()
        loss_sum = 0.0
        examples = 0
        components: dict[str, float] = {}
        with torch.no_grad():
            for batch in batches:
                with _autocast_context(self.device, self.config.precision):
                    raw = loss_function(self.model, batch, self.global_step)
                result = raw if isinstance(raw, LossResult) else LossResult(raw, {}, 1)
                if result.loss.ndim != 0 or not bool(torch.isfinite(result.loss)):
                    raise FloatingPointError("validation loss is not a finite scalar")
                value = float(result.loss.detach().cpu())
                loss_sum += value * result.examples
                examples += result.examples
                for name, component in result.components.items():
                    components[name] = components.get(name, 0.0) + (
                        _component_value(component) * result.examples
                    )
        totals = self.distributed.reduce_sum(
            torch.tensor(
                [loss_sum, float(examples), float(len(batches))],
                dtype=torch.float64,
                device=self.device,
            )
        )
        global_examples = int(totals[1].item())
        if global_examples <= 0:
            raise ValueError("validation produced zero examples across all ranks")
        component_names = sorted(
            {
                name
                for mapping in self.distributed.all_gather_mappings(
                    {name: True for name in components}
                )
                for name in mapping
            }
        )
        reduced_components = {
            name: float(
                self.distributed.reduce_sum(
                    torch.tensor(
                        components.get(name, 0.0),
                        dtype=torch.float64,
                        device=self.device,
                    )
                ).item()
                / global_examples
            )
            for name in component_names
        }
        return EpochResult(
            float(totals[0].item() / global_examples),
            int(totals[2].item()),
            0,
            global_examples,
            0,
            True,
            reduced_components,
        )
