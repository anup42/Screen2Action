"""Versioned CPU/GPU-safe checkpoint save and resume."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from screen2action.training.seed import capture_rng_state, restore_rng_state


@dataclass(frozen=True, slots=True)
class CheckpointState:
    """Metadata restored from a training checkpoint."""

    epoch: int
    step: int
    config: Mapping[str, Any]
    manifest_hash: str
    git_revision: str


def manifest_hash(manifest: str | bytes | Path) -> str:
    """Hash a manifest string, bytes, or file without storing its contents."""

    if isinstance(manifest, Path):
        payload = manifest.read_bytes()
    elif isinstance(manifest, str):
        payload = manifest.encode("utf-8")
    else:
        payload = manifest
    return hashlib.sha256(payload).hexdigest()


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    step: int,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    config: Mapping[str, Any] | None = None,
    data_manifest_hash: str = "",
    git_revision: str = "unknown",
) -> None:
    """Save model, optimizer, scheduler, scaler, RNG, and reproducibility metadata."""

    if epoch < 0 or step < 0:
        raise ValueError("epoch and step must be non-negative")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "step": step,
        "config": dict(config or {}),
        "manifest_hash": data_manifest_hash,
        "git_revision": git_revision,
        "rng": capture_rng_state(),
    }
    torch.save(payload, destination)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    device: torch.device | str = "cpu",
    restore_rng: bool = True,
) -> CheckpointState:
    """Load a checkpoint with explicit CPU-safe `map_location`."""

    payload = torch.load(Path(path), map_location=torch.device(device), weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if restore_rng:
        restore_rng_state(payload["rng"])
    return CheckpointState(
        epoch=int(payload["epoch"]),
        step=int(payload["step"]),
        config=payload.get("config", {}),
        manifest_hash=str(payload.get("manifest_hash", "")),
        git_revision=str(payload.get("git_revision", "unknown")),
    )
