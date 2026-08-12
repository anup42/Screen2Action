"""Stage-level CPU timing helpers."""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TimingSummary:
    """p50/p90/p99 timing summary in milliseconds."""

    count: int
    p50_ms: float
    p90_ms: float
    p99_ms: float


def time_callable[T](
    function: Callable[[], T], *, repeats: int = 20, warmup: int = 3
) -> tuple[T, TimingSummary]:
    """Measure a callable after deterministic warmup calls."""

    if repeats <= 0 or warmup < 0:
        raise ValueError("repeats must be positive and warmup cannot be negative")
    result: T | None = None
    for _ in range(warmup):
        result = function()
    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        timings.append((time.perf_counter() - start) * 1000.0)
    ordered = sorted(timings)

    def percentile(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))]

    if result is None:
        raise RuntimeError("callable did not produce a result")
    return result, TimingSummary(
        count=repeats,
        p50_ms=percentile(0.50),
        p90_ms=percentile(0.90),
        p99_ms=percentile(0.99),
    )


def peak_python_memory[T](function: Callable[[], T]) -> tuple[T, int]:
    """Measure peak Python-managed allocations for one CPU callable."""

    tracemalloc.start()
    try:
        result = function()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, int(peak)
