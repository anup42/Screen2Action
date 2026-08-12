"""Optional distributed setup kept out of CPU default paths."""

from __future__ import annotations

import os

import torch


def distributed_available() -> bool:
    """Return whether the installed Torch build exposes distributed support."""

    return torch.distributed.is_available()


def initialize_process_group(backend: str = "gloo") -> None:
    """Initialize a process group from standard launcher environment variables."""

    if not distributed_available():
        raise RuntimeError("torch.distributed is unavailable in this build")
    if torch.distributed.is_initialized():
        return
    required = ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(f"distributed launcher variables are missing: {missing}")
    torch.distributed.init_process_group(backend=backend)
