"""Atomic versioned CPU/GPU/DDP-safe checkpoint save and exact resume."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from screen2action.training.seed import capture_rng_state, restore_rng_state

CHECKPOINT_FORMAT_VERSION = 2


@dataclass(frozen=True, slots=True)
class CheckpointState:
    """Training cursor and compatibility metadata restored from a checkpoint."""

    epoch: int
    step: int
    config: Mapping[str, Any]
    manifest_hash: str
    git_revision: str
    next_batch_index: int = 0
    optimizer_step: int = 0
    best_metric: float | None = None
    sampler_state: Mapping[str, Any] | None = None
    config_hash: str = ""
    model_lock_hash: str = ""
    cache_manifest_hash: str = ""
    run_manifest_hash: str = ""


def manifest_hash(manifest: str | bytes | Path) -> str:
    """Hash a manifest string, bytes, or file without storing its contents."""

    if isinstance(manifest, Path):
        payload = manifest.read_bytes()
    elif isinstance(manifest, str):
        payload = manifest.encode("utf-8")
    else:
        payload = manifest
    return hashlib.sha256(payload).hexdigest()


def configuration_hash(config: Mapping[str, Any]) -> str:
    """Hash JSON-safe resolved configuration semantics."""

    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _unwrapped(model: nn.Module) -> nn.Module:
    while isinstance(candidate := getattr(model, "module", None), nn.Module):
        model = candidate
    return model


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    step: int,
    next_batch_index: int = 0,
    optimizer_step: int | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    config: Mapping[str, Any] | None = None,
    data_manifest_hash: str = "",
    model_lock_hash: str = "",
    cache_manifest_hash: str = "",
    run_manifest_hash: str = "",
    git_revision: str = "unknown",
    best_metric: float | None = None,
    sampler_state: Mapping[str, Any] | None = None,
    rng_states: Sequence[Mapping[str, Any]] | None = None,
    rank: int = 0,
) -> Path | None:
    """Atomically save complete state on rank zero; other ranks are no-ops."""

    if rank < 0:
        raise ValueError("rank must be non-negative")
    if rank != 0:
        return None
    if epoch < 0 or step < 0 or next_batch_index < 0:
        raise ValueError("epoch, step, and next batch must be non-negative")
    resolved_optimizer_step = step if optimizer_step is None else optimizer_step
    if resolved_optimizer_step < 0:
        raise ValueError("optimizer_step must be non-negative")
    config_values = dict(config or {})
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial")
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model": _unwrapped(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "step": step,
        "next_batch_index": next_batch_index,
        "optimizer_step": resolved_optimizer_step,
        "best_metric": best_metric,
        "sampler_state": dict(sampler_state or {}),
        "config": config_values,
        "config_hash": configuration_hash(config_values),
        "manifest_hash": data_manifest_hash,
        "model_lock_hash": model_lock_hash,
        "cache_manifest_hash": cache_manifest_hash,
        "run_manifest_hash": run_manifest_hash,
        "git_revision": git_revision,
        "rng": capture_rng_state(),
        "rng_by_rank": [dict(state) for state in rng_states] if rng_states is not None else None,
    }
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _require_expected(observed: str, expected: str | None, *, name: str) -> None:
    if expected is not None and observed != expected:
        raise ValueError(f"checkpoint {name} mismatch: expected {expected}, got {observed}")


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    device: torch.device | str = "cpu",
    restore_rng: bool = True,
    expected_config_hash: str | None = None,
    expected_manifest_hash: str | None = None,
    expected_model_lock_hash: str | None = None,
    expected_cache_manifest_hash: str | None = None,
    expected_run_manifest_hash: str | None = None,
    rank: int = 0,
) -> CheckpointState:
    """Load v1/v2 state with explicit device mapping and digest enforcement."""

    target = torch.device(device)
    # RNG byte tensors must stay on CPU even when restoring a CUDA optimizer.
    raw = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError("checkpoint payload must be a mapping")
    payload = cast(dict[str, Any], raw)
    version = payload.get("format_version")
    if version not in {1, CHECKPOINT_FORMAT_VERSION}:
        raise ValueError("unsupported checkpoint format")
    config = payload.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("checkpoint config must be a mapping")
    config_digest = str(payload.get("config_hash") or configuration_hash(config))
    manifest_digest = str(payload.get("manifest_hash", ""))
    model_digest = str(payload.get("model_lock_hash", ""))
    cache_digest = str(payload.get("cache_manifest_hash", ""))
    run_digest = str(payload.get("run_manifest_hash", ""))
    _require_expected(config_digest, expected_config_hash, name="configuration digest")
    _require_expected(manifest_digest, expected_manifest_hash, name="data manifest digest")
    _require_expected(model_digest, expected_model_lock_hash, name="model-lock digest")
    _require_expected(cache_digest, expected_cache_manifest_hash, name="cache manifest digest")
    _require_expected(run_digest, expected_run_manifest_hash, name="run manifest digest")
    _unwrapped(model).load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    _optimizer_to_device(optimizer, target)
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if rank < 0:
        raise ValueError("checkpoint rank must be non-negative")
    if restore_rng:
        rng_by_rank = payload.get("rng_by_rank")
        if rng_by_rank is not None:
            if not isinstance(rng_by_rank, list) or rank >= len(rng_by_rank):
                raise ValueError("checkpoint per-rank RNG state is malformed")
            rng = rng_by_rank[rank]
        else:
            rng = payload.get("rng")
        if not isinstance(rng, dict):
            raise ValueError("checkpoint RNG state is malformed")
        restore_rng_state(cast(dict[str, Any], rng))
    sampler = payload.get("sampler_state")
    if sampler is not None and not isinstance(sampler, dict):
        raise ValueError("checkpoint sampler state is malformed")
    best = payload.get("best_metric")
    return CheckpointState(
        epoch=int(payload["epoch"]),
        step=int(payload["step"]),
        config=cast(dict[str, Any], config),
        manifest_hash=manifest_digest,
        git_revision=str(payload.get("git_revision", "unknown")),
        next_batch_index=int(payload.get("next_batch_index", 0)),
        optimizer_step=int(payload.get("optimizer_step", payload["step"])),
        best_metric=float(best) if best is not None else None,
        sampler_state=cast(dict[str, Any] | None, sampler),
        config_hash=config_digest,
        model_lock_hash=model_digest,
        cache_manifest_hash=cache_digest,
        run_manifest_hash=run_digest,
    )


def load_model_weights(
    path: str | Path,
    model: nn.Module,
    *,
    device: torch.device | str = "cpu",
    expected_manifest_hash: str | None = None,
    expected_model_lock_hash: str | None = None,
    expected_cache_manifest_hash: str | None = None,
    source_prefixes: tuple[str, ...] = ("downstream.",),
) -> Mapping[str, Any]:
    """Load model-only initialization while preserving new optimizer/stage state."""

    raw = torch.load(Path(path), map_location=torch.device(device), weights_only=False)
    if not isinstance(raw, dict) or not isinstance(raw.get("model"), dict):
        raise ValueError("initial checkpoint payload is malformed")
    payload = cast(dict[str, Any], raw)
    _require_expected(
        str(payload.get("manifest_hash", "")),
        expected_manifest_hash,
        name="data manifest digest",
    )
    _require_expected(
        str(payload.get("model_lock_hash", "")),
        expected_model_lock_hash,
        name="model-lock digest",
    )
    _require_expected(
        str(payload.get("cache_manifest_hash", "")),
        expected_cache_manifest_hash,
        name="cache manifest digest",
    )
    state = cast(dict[str, torch.Tensor], payload["model"])
    target_keys = set(_unwrapped(model).state_dict())
    selected: dict[str, torch.Tensor] | None = None
    if set(state) == target_keys:
        selected = state
    else:
        for prefix in source_prefixes:
            candidate = {
                name.removeprefix(prefix): value
                for name, value in state.items()
                if name.startswith(prefix)
            }
            if set(candidate) == target_keys:
                selected = candidate
                break
    if selected is None:
        missing = sorted(target_keys - set(state))[:5]
        unexpected = sorted(set(state) - target_keys)[:5]
        raise ValueError(
            "initial checkpoint model topology mismatch; "
            f"missing={missing}, unexpected={unexpected}"
        )
    _unwrapped(model).load_state_dict(selected, strict=True)
    return {
        "source_checkpoint": Path(path).as_posix(),
        "source_epoch": int(payload.get("epoch", 0)),
        "source_step": int(payload.get("step", 0)),
        "source_git_revision": str(payload.get("git_revision", "unknown")),
    }
