"""Atomic content-addressed perception cache tests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from screen2action.perception.cache import (
    ContentAddressedPerceptionCache,
    PerceptionCacheKey,
)


def _key() -> PerceptionCacheKey:
    return PerceptionCacheKey("a" * 64, "b" * 64, "c" * 64)


def test_cache_is_sharded_atomic_resumable_and_multiworker_safe(tmp_path: Path) -> None:
    cache = ContentAddressedPerceptionCache(tmp_path)
    key = _key()

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: cache.put(key, {"nodes": [1, 2, 3]}), range(8)))

    assert [result.status for result in results].count("written") == 1
    assert [result.status for result in results].count("already_complete") == 7
    assert all(result.relative_path.startswith(f"entries/{key.digest[:2]}/") for result in results)
    assert cache.get(key) == {"nodes": [1, 2, 3]}
    assert cache.stats() == {"entries": 1, "failures": 0, "temporary": 0, "locks": 0}


def test_failure_record_is_retryable_and_cleared_after_success(tmp_path: Path) -> None:
    cache = ContentAddressedPerceptionCache(tmp_path)
    key = _key()

    path = cache.record_failure(key, error_type="FixtureError", message="retry me", attempt=2)
    assert path.is_file()
    failure = cache.failure(key)
    assert failure is not None
    assert failure["retryable"] is True
    assert failure["attempt"] == 2

    result = cache.put(key, {"ok": True})
    assert result.status == "written"
    assert cache.failure(key) is None
