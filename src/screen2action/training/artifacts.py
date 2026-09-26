"""Rank-zero structured metrics and immutable run provenance artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol


class ScalarWriter(Protocol):
    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        """Write one scalar event."""

    def flush(self) -> None:
        """Flush pending events."""

    def close(self) -> None:
        """Close the writer."""


def _canonical_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _git_output(root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def repository_code_identity(root: Path) -> dict[str, object]:
    """Describe committed and local code without embedding a patch in artifacts."""

    try:
        revision = _git_output(root, "rev-parse", "HEAD").decode().strip()
        status = _git_output(root, "status", "--porcelain=v1", "--untracked-files=all")
        tracked_diff = _git_output(root, "diff", "--binary", "HEAD", "--")
        untracked = (
            _git_output(root, "ls-files", "--others", "--exclude-standard").decode().splitlines()
        )
        digest = hashlib.sha256(tracked_diff)
        code_roots = {"src", "tests", "configs", "docs"}
        code_files = {"pyproject.toml", "Makefile", "AGENTS.md", "README.md"}
        for relative in sorted(untracked):
            portable = Path(relative)
            if portable.as_posix() not in code_files and (
                not portable.parts or portable.parts[0] not in code_roots
            ):
                continue
            path = (root / relative).resolve()
            if path.is_file() and path.is_relative_to(root.resolve()):
                digest.update(relative.encode())
                digest.update(path.read_bytes())
        return {
            "git_revision": revision,
            "dirty": bool(status.strip()),
            "working_tree_sha256": digest.hexdigest(),
        }
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return {
            "git_revision": "unknown",
            "dirty": True,
            "working_tree_sha256": "unknown",
        }


def write_run_manifest(
    run_directory: Path,
    *,
    root: Path,
    stage: str,
    config_sha256: str,
    data_manifest_digest: str,
    model_lock_digest: str,
    cache_manifest_sha256: str,
    world_size: int,
    device_type: str,
    initial_checkpoint_sha256: str | None = "",
) -> tuple[Path, str, str]:
    """Create or verify one immutable run manifest and return its file digest."""

    destination = run_directory / "run-manifest.json"
    if initial_checkpoint_sha256 is None:
        if not destination.is_file():
            raise ValueError("exact resume requires the original run manifest")
        previous = json.loads(destination.read_text(encoding="utf-8"))
        if not isinstance(previous, dict) or not isinstance(previous.get("identity"), dict):
            raise ValueError("existing run manifest is malformed")
        recorded_initialization = previous["identity"].get("initial_checkpoint_sha256")
        if not isinstance(recorded_initialization, str):
            raise ValueError("existing run manifest initialization digest is malformed")
        initial_checkpoint_sha256 = recorded_initialization
    identity: dict[str, object] = {
        "stage": stage,
        "config_sha256": config_sha256,
        "data_manifest_digest": data_manifest_digest,
        "model_lock_digest": model_lock_digest,
        "cache_manifest_sha256": cache_manifest_sha256,
        "initial_checkpoint_sha256": initial_checkpoint_sha256,
        "code": repository_code_identity(root),
        "world_size": world_size,
        "device_type": device_type,
    }
    identity_digest = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if not isinstance(existing, dict) or existing.get("identity_sha256") != identity_digest:
            raise ValueError("existing run manifest is incompatible with this launch")
    else:
        _atomic_json(
            destination,
            {
                "schema_version": 1,
                "created_at_utc": datetime.now(UTC).isoformat(),
                "identity": identity,
                "identity_sha256": identity_digest,
            },
        )
    file_digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    revision = str(cast_mapping(identity["code"])["git_revision"])
    return destination, file_digest, revision


def cast_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError("expected a mapping")
    return value


@dataclass(slots=True)
class RunArtifactWriter:
    """Append JSONL and mirror finite numeric values to TensorBoard."""

    run_directory: Path
    jsonl_path: Path
    tensorboard: ScalarWriter | None

    @classmethod
    def open(cls, run_directory: Path, *, tensorboard: bool) -> RunArtifactWriter:
        run_directory.mkdir(parents=True, exist_ok=True)
        writer: ScalarWriter | None = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as error:
                raise RuntimeError(
                    "TensorBoard logging requires the train extra: pip install -e .[train]"
                ) from error
            writer = SummaryWriter(log_dir=str(run_directory / "tensorboard"))
        return cls(run_directory, run_directory / "metrics.jsonl", writer)

    def write(self, event: Mapping[str, object]) -> None:
        record = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            **dict(event),
        }
        with self.jsonl_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            stream.flush()
        if self.tensorboard is None:
            return
        step_value = event.get("optimizer_step", event.get("global_step", event.get("epoch", 0)))
        step = int(step_value) if isinstance(step_value, int | float) else 0
        prefix = str(event.get("event", "metric"))
        for name, value in event.items():
            if name in {"event", "global_step", "optimizer_step", "epoch"}:
                continue
            if isinstance(value, int | float) and not isinstance(value, bool):
                self.tensorboard.add_scalar(f"{prefix}/{name}", float(value), step)

    def close(self) -> None:
        if self.tensorboard is not None:
            self.tensorboard.flush()
            self.tensorboard.close()
