"""Licensed, resumable, checksum-aware public data acquisition."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
import uuid
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from screen2action.data.layout import DataLayout
from screen2action.data.registry import SourceRegistry, SourceSpec

RAW_MANIFEST = ".screen2action-source.json"


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FileDigest:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class LicenseAcknowledgement:
    acknowledgement_id: str
    source: str
    revision: str
    license_identifier: str
    source_url: str
    accepted_at_utc: str


@dataclass(frozen=True, slots=True)
class RawSourceManifest:
    schema_version: int
    source: str
    revision: str
    provider: str
    upstream_id: str
    license_identifier: str
    license_acknowledgement_id: str
    sample_size: int | None
    files: tuple[FileDigest, ...]

    @property
    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DatasetProvider(Protocol):
    def resolve_revision(self, source: SourceSpec, requested_revision: str | None) -> str:
        """Resolve a requested/default revision to an immutable identifier."""

    def fetch(
        self,
        source: SourceSpec,
        revision: str,
        destination: Path,
        *,
        sample_size: int | None,
    ) -> None:
        """Fetch source bytes into an empty staging directory."""


def inventory_files(
    root: Path, *, exclude: Iterable[str] = (RAW_MANIFEST,)
) -> tuple[FileDigest, ...]:
    excluded = set(exclude)
    entries: list[FileDigest] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        entries.append(FileDigest(relative, path.stat().st_size, sha256_file(path)))
    return tuple(entries)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def record_license_acceptance(
    layout: DataLayout,
    source: SourceSpec,
    revision: str,
    *,
    accepted_at: datetime | None = None,
) -> LicenseAcknowledgement:
    """Persist a non-secret acknowledgement tied to exact source terms."""

    timestamp = (accepted_at or datetime.now(UTC)).astimezone(UTC).isoformat()
    identity = "\n".join(
        (source.name, revision, source.license_identifier, source.source_url, timestamp)
    )
    acknowledgement_id = f"data-license-{hashlib.sha256(identity.encode()).hexdigest()[:20]}"
    acknowledgement = LicenseAcknowledgement(
        acknowledgement_id=acknowledgement_id,
        source=source.name,
        revision=revision,
        license_identifier=source.license_identifier,
        source_url=source.source_url,
        accepted_at_utc=timestamp,
    )
    _atomic_json(
        layout.license_acknowledgements() / f"{acknowledgement_id}.json",
        asdict(acknowledgement),
    )
    return acknowledgement


def find_license_acceptance(
    layout: DataLayout,
    source: SourceSpec,
    revision: str,
) -> LicenseAcknowledgement | None:
    root = layout.license_acknowledgements()
    if not root.is_dir():
        return None
    for path in sorted(root.glob("data-license-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("source") == source.name
            and payload.get("revision") == revision
            and payload.get("license_identifier") == source.license_identifier
            and payload.get("source_url") == source.source_url
        ):
            return LicenseAcknowledgement(**payload)
    return None


def require_license_acceptance(
    layout: DataLayout,
    source: SourceSpec,
    revision: str,
    *,
    accept_license: bool,
) -> LicenseAcknowledgement:
    existing = find_license_acceptance(layout, source, revision)
    if existing is not None:
        return existing
    if accept_license:
        return record_license_acceptance(layout, source, revision)
    raise PermissionError(
        f"source {source.name!r} requires explicit license acknowledgement; "
        "review its source_url and rerun with --accept-license"
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, bytes | bytearray | memoryview):
        return {"__bytes_base64__": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Iterable):
        return [_json_safe(item) for item in value]
    if hasattr(value, "save"):
        with tempfile.SpooledTemporaryFile() as stream:
            value.save(stream, format="PNG")
            stream.seek(0)
            return {"__bytes_base64__": base64.b64encode(stream.read()).decode("ascii")}
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    return str(value)


class HuggingFaceDatasetProvider:
    """Official Hub API implementation; imports heavy packages only on use."""

    def resolve_revision(self, source: SourceSpec, requested_revision: str | None) -> str:
        from huggingface_hub import HfApi

        revision = requested_revision or source.revision
        info = HfApi().dataset_info(source.upstream_id, revision=revision)
        if not info.sha:
            raise RuntimeError(f"Hugging Face did not resolve {source.upstream_id!r} to a commit")
        return str(info.sha)

    def fetch(
        self,
        source: SourceSpec,
        revision: str,
        destination: Path,
        *,
        sample_size: int | None,
    ) -> None:
        if sample_size is None:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=source.upstream_id,
                repo_type="dataset",
                revision=revision,
                local_dir=destination,
            )
            return
        if sample_size <= 0:
            raise ValueError("sample size must be positive")
        if source.name == "amex":
            from huggingface_hub import snapshot_download

            destination.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=source.upstream_id,
                repo_type="dataset",
                revision=revision,
                local_dir=destination,
                allow_patterns=["sample/**", "README.md"],
            )
            _atomic_json(
                destination / "sample-metadata.json",
                {
                    "upstream_id": source.upstream_id,
                    "revision": revision,
                    "requested_sample_size": sample_size,
                    "official_sample_directory": True,
                },
            )
            return
        try:
            from datasets import (  # type: ignore[import-not-found]
                DatasetDict,
                IterableDatasetDict,
                load_dataset,
            )
        except ImportError as error:
            raise RuntimeError(
                "sample downloads require the 'data' extra: pip install -e .[data]"
            ) from error
        loaded = load_dataset(source.upstream_id, revision=revision, streaming=True)
        if isinstance(loaded, DatasetDict | IterableDatasetDict):
            split_name = "train" if "train" in loaded else sorted(loaded)[0]
            records = loaded[split_name]
        else:
            split_name = "train"
            records = loaded
        destination.mkdir(parents=True, exist_ok=True)
        output = destination / "sample.jsonl"
        with output.open("w", encoding="utf-8") as stream:
            for index, record in enumerate(records):
                if index >= sample_size:
                    break
                stream.write(json.dumps(_json_safe(record), sort_keys=True) + "\n")
            else:
                index = -1
        metadata = {
            "upstream_id": source.upstream_id,
            "revision": revision,
            "split": split_name,
            "requested_sample_size": sample_size,
            "stored_records": max(index + 1, 0),
        }
        _atomic_json(destination / "sample-metadata.json", metadata)


class GitRepositoryProvider:
    """Resumable immutable GitHub archive download."""

    def resolve_revision(self, source: SourceSpec, requested_revision: str | None) -> str:
        revision = requested_revision or source.revision
        if len(revision) < 12:
            raise ValueError("git source revisions must be immutable commit hashes")
        return revision

    def fetch(
        self,
        source: SourceSpec,
        revision: str,
        destination: Path,
        *,
        sample_size: int | None,
    ) -> None:
        if sample_size is not None:
            raise ValueError("git_repository sources do not support row sampling")
        destination.mkdir(parents=True, exist_ok=True)
        archive = destination / "source.zip"
        url = f"https://github.com/{source.upstream_id}/archive/{revision}.zip"
        download_resumable(url, archive)
        verify_zip(archive)
        safe_extract_zip(archive, destination / "source")


def provider_for(source: SourceSpec) -> DatasetProvider:
    if source.provider == "huggingface_dataset":
        return HuggingFaceDatasetProvider()
    if source.provider == "git_repository":
        return GitRepositoryProvider()
    if source.provider == "google_research_gcs":
        raise RuntimeError(
            "AndroidControl bulk download requires the official GCS path and local Google "
            "credentials/tooling; use data register-local after downloading the GZIP TFRecords"
        )
    if source.provider == "manual_official_archive":
        raise RuntimeError(
            f"{source.name} uses an official manually accepted archive; download from "
            f"{source.source_url} and use data register-local"
        )
    raise ValueError(f"unsupported dataset provider {source.provider!r}")


def download_resumable(url: str, destination: Path, *, chunk_size: int = 1024 * 1024) -> None:
    """Download with an HTTP Range request when a partial file exists."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    with urllib.request.urlopen(request) as response:  # noqa: S310 - explicit user source URL
        status = getattr(response, "status", 200)
        mode = "ab" if offset and status == 206 else "wb"
        with partial.open(mode) as stream:
            while chunk := response.read(chunk_size):
                stream.write(chunk)
    os.replace(partial, destination)


def verify_zip(path: Path) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
    except zipfile.BadZipFile as error:
        raise ValueError(f"invalid ZIP archive: {path}") from error
    if bad_member is not None:
        raise ValueError(f"ZIP CRC check failed for {bad_member!r} in {path}")


def safe_extract_zip(path: Path, destination: Path) -> None:
    """Extract a verified ZIP while rejecting traversal and absolute members."""

    verify_zip(path)
    destination.mkdir(parents=True, exist_ok=True)
    resolved_destination = destination.resolve()
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            member_path = (destination / member.filename).resolve()
            if not member_path.is_relative_to(resolved_destination):
                raise ValueError(f"ZIP member escapes extraction root: {member.filename!r}")
            if member.is_dir():
                member_path.mkdir(parents=True, exist_ok=True)
                continue
            member_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, member_path.open("wb") as target:
                shutil.copyfileobj(source, target)


def merge_multipart_zip(parts: Iterable[Path], output: Path) -> Path:
    """Concatenate ordered byte parts and require a valid merged ZIP."""

    ordered = sorted(Path(part) for part in parts)
    if len(ordered) < 2:
        raise ValueError("multipart ZIP merge requires at least two parts")
    if any(not part.is_file() for part in ordered):
        raise FileNotFoundError("one or more multipart ZIP components do not exist")
    temporary = output.with_suffix(output.suffix + ".partial")
    output.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as target:
        for part in ordered:
            with part.open("rb") as source:
                shutil.copyfileobj(source, target)
    verify_zip(temporary)
    os.replace(temporary, output)
    return output


def repair_amex_zip(source_zip: Path, output: Path) -> Path:
    """Run the AMEX-documented ``zip --fix`` workflow and verify the result."""

    executable = shutil.which("zip")
    if executable is None:
        raise RuntimeError(
            "AMEX split archive repair requires Info-ZIP 'zip'; run "
            "`zip --fix amex_v1.zip --out amex_v1_merged.zip` then register the result"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [executable, "--fix", str(source_zip), "--out", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"AMEX ZIP repair failed: {completed.stderr.strip()}")
    verify_zip(output)
    return output


def _prepare_registered_archives(source: SourceSpec, staging: Path) -> None:
    """Verify and extract registered archives before immutable finalization."""

    extraction_root = staging / "extracted"
    multipart_inputs: set[Path] = set()
    if source.multipart_repair == "zip_fix":
        for archive in sorted(staging.rglob("*.zip")):
            if any(archive.with_suffix(f".z{index:02d}").is_file() for index in range(1, 100)):
                merged = archive.with_name(f"{archive.stem}_merged.zip")
                repair_amex_zip(archive, merged)
                safe_extract_zip(merged, extraction_root)
                multipart_inputs.add(archive)
    for archive in sorted(staging.rglob("*.zip")):
        if archive in multipart_inputs or archive.name.endswith("_merged.zip"):
            continue
        safe_extract_zip(archive, extraction_root)


def _write_raw_manifest(
    destination: Path,
    source: SourceSpec,
    revision: str,
    acknowledgement: LicenseAcknowledgement,
    sample_size: int | None,
) -> RawSourceManifest:
    manifest = RawSourceManifest(
        schema_version=1,
        source=source.name,
        revision=revision,
        provider=source.provider,
        upstream_id=source.upstream_id,
        license_identifier=source.license_identifier,
        license_acknowledgement_id=acknowledgement.acknowledgement_id,
        sample_size=sample_size,
        files=inventory_files(destination),
    )
    payload = asdict(manifest)
    payload["manifest_digest"] = manifest.digest
    _atomic_json(destination / RAW_MANIFEST, payload)
    return manifest


def verify_raw_source(path: Path) -> RawSourceManifest:
    manifest_path = path / RAW_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(f"raw source manifest is missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded_digest = payload.pop("manifest_digest", None)
    manifest = RawSourceManifest(
        schema_version=int(payload["schema_version"]),
        source=str(payload["source"]),
        revision=str(payload["revision"]),
        provider=str(payload["provider"]),
        upstream_id=str(payload["upstream_id"]),
        license_identifier=str(payload["license_identifier"]),
        license_acknowledgement_id=str(payload["license_acknowledgement_id"]),
        sample_size=(
            int(payload["sample_size"]) if payload.get("sample_size") is not None else None
        ),
        files=tuple(FileDigest(**entry) for entry in payload["files"]),
    )
    if recorded_digest != manifest.digest:
        raise ValueError("raw source manifest digest mismatch")
    observed = inventory_files(path)
    if observed != manifest.files:
        raise ValueError("raw source files differ from the immutable registered inventory")
    return manifest


def register_local_source(
    registry: SourceRegistry,
    layout: DataLayout,
    source_name: str,
    source_path: Path,
    revision: str,
    *,
    accept_license: bool,
) -> RawSourceManifest:
    source = registry.source(source_name)
    acknowledgement = require_license_acceptance(
        layout, source, revision, accept_license=accept_license
    )
    origin = source_path.resolve()
    if not origin.exists():
        raise FileNotFoundError(f"local source path does not exist: {origin}")
    destination = layout.raw_revision(source.name, revision)
    if destination.exists():
        manifest = verify_raw_source(destination)
        if manifest.source != source.name or manifest.revision != revision:
            raise ValueError("existing raw directory belongs to a different source revision")
        return manifest
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{uuid.uuid4().hex}.partial")
    if origin.is_dir():
        shutil.copytree(origin, staging)
    else:
        staging.mkdir(parents=True)
        shutil.copy2(origin, staging / origin.name)
    _prepare_registered_archives(source, staging)
    manifest = _write_raw_manifest(staging, source, revision, acknowledgement, None)
    os.replace(staging, destination)
    return manifest


def download_registered_source(
    registry: SourceRegistry,
    layout: DataLayout,
    source_name: str,
    *,
    requested_revision: str | None,
    sample_size: int | None,
    accept_license: bool,
) -> RawSourceManifest:
    source = registry.source(source_name)
    provider = provider_for(source)
    revision = provider.resolve_revision(source, requested_revision)
    acknowledgement = require_license_acceptance(
        layout, source, revision, accept_license=accept_license
    )
    suffix = f"sample-{sample_size}" if sample_size is not None else "full"
    destination = layout.raw_revision(source.name, revision) / suffix
    if destination.exists():
        return verify_raw_source(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.partial")
    try:
        provider.fetch(source, revision, staging, sample_size=sample_size)
        if source.multipart_repair == "zip_fix" and sample_size is None:
            split_archives = sorted(
                path
                for path in staging.rglob("*.zip")
                if any(path.with_suffix(f".z{index:02d}").is_file() for index in range(1, 100))
            )
            if not split_archives:
                raise ValueError("AMEX full snapshot contains no multipart ZIP set")
            for archive in split_archives:
                merged = repair_amex_zip(archive, archive.with_name(f"{archive.stem}_merged.zip"))
                safe_extract_zip(merged, staging / "extracted")
        manifest = _write_raw_manifest(staging, source, revision, acknowledgement, sample_size)
        os.replace(staging, destination)
    except Exception:
        raise
    return manifest
