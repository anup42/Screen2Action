"""Deterministic seed and RNG-state helpers."""

from __future__ import annotations

import random
from typing import Any

import torch


def seed_everything(seed: int) -> None:
    """Seed Python and Torch without assuming CUDA is installed."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        return


def capture_rng_state() -> dict[str, Any]:
    """Capture CPU and optional NumPy RNG states."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        state["numpy"] = None
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore a state captured by `capture_rng_state`."""

    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state.get("torch_cuda") is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if state.get("numpy") is not None:
        try:
            import numpy as np

            np.random.set_state(state["numpy"])
        except ImportError as error:
            raise RuntimeError(
                "checkpoint contains NumPy state but NumPy is unavailable"
            ) from error


def seed_worker(worker_id: int, *, base_seed: int, rank: int = 0) -> None:
    """Seed one data worker reproducibly without relying on process inheritance."""

    if worker_id < 0 or base_seed < 0 or rank < 0:
        raise ValueError("worker, base seed, and rank must be non-negative")
    seed_everything(base_seed + rank * 100_003 + worker_id)
