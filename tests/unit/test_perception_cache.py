"""Atomic content-addressed perception cache tests."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

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


def test_control_documents_are_atomic_immutable_and_path_safe(tmp_path: Path) -> None:
    cache = ContentAddressedPerceptionCache(tmp_path)
    path = cache.write_control("manifest.json", {"schema": 1}, immutable=True)
    assert path.is_file()
    assert cache.read_control("manifest.json") == {"schema": 1}
    assert cache.write_control("manifest.json", {"schema": 1}, immutable=True) == path

    with pytest.raises(ValueError, match="immutable"):
        cache.write_control("manifest.json", {"schema": 2}, immutable=True)
    with pytest.raises(ValueError, match="safe relative"):
        cache.write_control("../outside.json", {"schema": 1})
    assert not tuple(tmp_path.rglob("*.tmp"))


def test_abandoned_lock_and_temporary_are_recovered(tmp_path: Path) -> None:
    cache = ContentAddressedPerceptionCache(
        tmp_path,
        lock_timeout_seconds=0.5,
        stale_lock_seconds=0.1,
    )
    key = _key()
    lock = cache._lock_path(key.digest)
    lock.parent.mkdir(parents=True)
    lock.write_text("interrupted-worker", encoding="utf-8")
    stale = time.time() - 120.0
    os.utime(lock, (stale, stale))
    assert cache.put(key, {"recovered": True}).status == "written"

    temporary = tmp_path / "abandoned.tmp"
    temporary.write_text("partial", encoding="utf-8")
    os.utime(temporary, (stale, stale))
    assert cache.cleanup_stale_temporary(older_than_seconds=0.1) == 1
    assert not temporary.exists()
