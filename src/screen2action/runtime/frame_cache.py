"""In-process command-independent frame cache."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class FrameCacheKey:
    """Robust screenshot identity including dimensions and orientation."""

    digest: str
    height: int
    width: int
    orientation: str


def frame_cache_key(screenshot: torch.Tensor, orientation: str = "unknown") -> FrameCacheKey:
    """Create a content hash without retaining screenshot pixels in logs."""

    if screenshot.ndim != 3:
        raise ValueError("screenshot must have shape [C,H,W]")
    payload = screenshot.detach().cpu().contiguous().numpy().tobytes()
    digest = hashlib.sha256(payload).hexdigest()
    return FrameCacheKey(digest, int(screenshot.shape[-2]), int(screenshot.shape[-1]), orientation)


class FrameCache[T]:
    """Bounded in-memory cache keyed by robust frame identity."""

    def __init__(self, max_entries: int = 8) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self._values: dict[FrameCacheKey, T] = {}

    def get(self, key: FrameCacheKey) -> T | None:
        """Return a cached state or None."""

        return self._values.get(key)

    def put(self, key: FrameCacheKey, value: T) -> None:
        """Store a state and evict the oldest inserted entry if necessary."""

        self._values[key] = value
        while len(self._values) > self.max_entries:
            self._values.pop(next(iter(self._values)))

    def clear(self) -> None:
        """Remove all cached states."""

        self._values.clear()
