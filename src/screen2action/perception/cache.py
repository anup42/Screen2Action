"""Atomic sharded content-addressed cache for command-independent perception."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from screen2action import PERCEPTION_CACHE_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class PerceptionCacheKey:
    """Every compatibility dimension that can change perceived frame state."""

    screenshot_sha256: str
    model_bundle_lock_digest: str
    perception_config_digest: str
    schema_version: int = PERCEPTION_CACHE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name, value in (
            ("screenshot_sha256", self.screenshot_sha256),
            ("model_bundle_lock_digest", self.model_bundle_lock_digest),
            ("perception_config_digest", self.perception_config_digest),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be lowercase SHA256")
        if self.schema_version <= 0:
            raise ValueError("schema_version must be positive")

    @property
    def digest(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, object]:
        return {
            "screenshot_sha256": self.screenshot_sha256,
            "model_bundle_lock_digest": self.model_bundle_lock_digest,
            "perception_config_digest": self.perception_config_digest,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class CacheWriteResult:
    """Observed cache mutation/resume status."""

    key_digest: str
    status: str
    relative_path: str


class ContentAddressedPerceptionCache:
    """JSON cache with atomic writes, sharding, worker locks, and failure logs."""

    def __init__(
        self,
        root: str | Path,
        *,
        lock_timeout_seconds: float = 10.0,
        stale_lock_seconds: float = 60.0,
    ) -> None:
        if lock_timeout_seconds <= 0.0 or stale_lock_seconds <= 0.0:
            raise ValueError("lock and stale-lock timeouts must be positive")
        self.root = Path(root)
        self.lock_timeout_seconds = lock_timeout_seconds
        self.stale_lock_seconds = stale_lock_seconds

    def _entry_path(self, digest: str) -> Path:
        return self.root / "entries" / digest[:2] / digest[2:4] / f"{digest}.json"

    def entry_path(self, key: PerceptionCacheKey) -> Path:
        """Return the deterministic entry path without reading or writing it."""

        return self._entry_path(key.digest)

    def _failure_path(self, digest: str) -> Path:
        return self.root / "failures" / digest[:2] / f"{digest}.json"

    def _lock_path(self, digest: str) -> Path:
        return self.root / "locks" / digest[:2] / f"{digest}.lock"

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    @contextmanager
    def _exclusive(self, digest: str) -> Iterator[None]:
        lock = self._lock_path(digest)
        lock.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        descriptor: int | None = None
        while descriptor is None:
            try:
                descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(descriptor, str(os.getpid()).encode("ascii"))
            except (FileExistsError, PermissionError) as error:
                # Windows may report a sharing violation as PermissionError while
                # another process still owns the O_EXCL lock file.
                if isinstance(error, PermissionError) and os.name != "nt":
                    raise
                try:
                    stale = time.time() - lock.stat().st_mtime >= self.stale_lock_seconds
                except FileNotFoundError:
                    continue
                if stale:
                    try:
                        lock.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                if time.monotonic() - started >= self.lock_timeout_seconds:
                    raise TimeoutError(
                        f"timed out waiting for perception cache key {digest}"
                    ) from None
                time.sleep(0.025)
        try:
            yield
        finally:
            assert descriptor is not None
            os.close(descriptor)
            try:
                lock.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _write_json_atomic(destination: Path, payload: Mapping[str, Any]) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(f".json.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)

    def _control_path(self, relative_path: str) -> Path:
        normalized = PurePosixPath(relative_path.replace("\\", "/"))
        if (
            normalized.is_absolute()
            or ".." in normalized.parts
            or not normalized.parts
            or normalized.suffix != ".json"
        ):
            raise ValueError("cache control path must be a safe relative JSON path")
        destination = (self.root / Path(*normalized.parts)).resolve()
        if not destination.is_relative_to(self.root.resolve()):
            raise ValueError("cache control path escapes cache root")
        return destination

    def read_control(self, relative_path: str) -> dict[str, Any] | None:
        """Read one cache manifest/status document from a safe relative path."""

        path = self._control_path(relative_path)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"cache control document is malformed: {relative_path}")
        return payload

    def write_control(
        self,
        relative_path: str,
        payload: Mapping[str, Any],
        *,
        immutable: bool = False,
    ) -> Path:
        """Atomically write a cross-process-safe bundle, run, or shard document."""

        destination = self._control_path(relative_path)
        lock_digest = hashlib.sha256(f"control:{relative_path}".encode()).hexdigest()
        with self._exclusive(lock_digest):
            existing = self.read_control(relative_path)
            if immutable and existing is not None:
                if existing != dict(payload):
                    raise ValueError(f"immutable cache control document differs: {relative_path}")
                return destination
            self._write_json_atomic(destination, payload)
        return destination

    def get(self, key: PerceptionCacheKey) -> dict[str, Any] | None:
        """Return a compatible payload, or `None` for a cache miss."""

        path = self._entry_path(key.digest)
        if not path.is_file():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("cache_key") != key.as_dict():
            raise ValueError(f"perception cache compatibility mismatch for key {key.digest}")
        payload = raw.get("payload")
        if not isinstance(payload, dict):
            raise ValueError(f"perception cache payload is malformed for key {key.digest}")
        return payload

    def put(
        self,
        key: PerceptionCacheKey,
        payload: Mapping[str, Any],
    ) -> CacheWriteResult:
        """Write once under a cross-process key lock; existing valid bytes resume."""

        destination = self._entry_path(key.digest)
        with self._exclusive(key.digest):
            if self.get(key) is not None:
                return CacheWriteResult(key.digest, "already_complete", self._relative(destination))
            document = {
                "cache_schema_version": PERCEPTION_CACHE_SCHEMA_VERSION,
                "cache_key": key.as_dict(),
                "key_digest": key.digest,
                "payload": dict(payload),
            }
            self._write_json_atomic(destination, document)
            failure = self._failure_path(key.digest)
            try:
                failure.unlink()
            except FileNotFoundError:
                pass
            return CacheWriteResult(key.digest, "written", self._relative(destination))

    def record_failure(
        self,
        key: PerceptionCacheKey,
        *,
        error_type: str,
        message: str,
        attempt: int,
    ) -> Path:
        """Record a retryable item failure without traceback, pixels, or paths."""

        if not error_type or not message or attempt <= 0:
            raise ValueError("failure records require error type, message, and positive attempt")
        destination = self._failure_path(key.digest)
        with self._exclusive(key.digest):
            self._write_json_atomic(
                destination,
                {
                    "cache_key": key.as_dict(),
                    "key_digest": key.digest,
                    "error_type": error_type,
                    "message": message,
                    "attempt": attempt,
                    "retryable": True,
                },
            )
        return destination

    def failure(self, key: PerceptionCacheKey) -> dict[str, Any] | None:
        """Read a failure record for retry orchestration."""

        path = self._failure_path(key.digest)
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("perception failure record is malformed")
        return value

    def stats(self) -> dict[str, int]:
        """Return bounded filesystem counts for operator diagnostics."""

        entries = sum(1 for _ in (self.root / "entries").glob("**/*.json"))
        failures = sum(1 for _ in (self.root / "failures").glob("**/*.json"))
        temporary = sum(1 for _ in self.root.glob("**/*.tmp"))
        locks = sum(1 for _ in (self.root / "locks").glob("**/*.lock"))
        return {"entries": entries, "failures": failures, "temporary": temporary, "locks": locks}

    def cleanup_stale_temporary(self, *, older_than_seconds: float = 60.0) -> int:
        """Remove only abandoned atomic-write temporaries below this cache root."""

        if older_than_seconds <= 0.0:
            raise ValueError("temporary-file age must be positive")
        removed = 0
        now = time.time()
        for path in self.root.glob("**/*.tmp"):
            try:
                if now - path.stat().st_mtime < older_than_seconds:
                    continue
                resolved = path.resolve()
                if not resolved.is_relative_to(self.root.resolve()):
                    raise ValueError("temporary cache path escapes cache root")
                path.unlink()
                removed += 1
            except FileNotFoundError:
                continue
        return removed
