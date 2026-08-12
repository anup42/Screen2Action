"""Deterministic, resumable precomputation over unique canonical screens."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, cast

from screen2action import PERCEPTION_CACHE_SCHEMA_VERSION
from screen2action.data.assets import sha256_file
from screen2action.data.manifest import verify_data_manifest
from screen2action.data.schema import CANONICAL_SCHEMA_VERSION
from screen2action.data.storage import read_table_rows
from screen2action.perception.base import ImageFrame, normalize_image
from screen2action.perception.cache import ContentAddressedPerceptionCache, PerceptionCacheKey
from screen2action.perception.pipeline import (
    FullScreenPerceptionConfig,
    PerceptionFrame,
    perception_frame_from_payload,
)

CACHE_BUNDLE_MANIFEST_SCHEMA_VERSION = 1
PRECOMPUTE_RUN_SCHEMA_VERSION = 1
PRECOMPUTE_SHARD_SCHEMA_VERSION = 1


def _digest(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _document_with_digest(payload: Mapping[str, object]) -> dict[str, object]:
    document = dict(payload)
    document["document_digest"] = _digest(document)
    return document


def _verify_document_digest(payload: Mapping[str, object], *, context: str) -> str:
    recorded = payload.get("document_digest")
    if not isinstance(recorded, str):
        raise ValueError(f"{context} is missing document_digest")
    semantic = {key: value for key, value in payload.items() if key != "document_digest"}
    if _digest(semantic) != recorded:
        raise ValueError(f"{context} digest mismatch")
    return recorded


@dataclass(frozen=True, slots=True)
class ManifestScreen:
    """One unique content-addressed screen and all canonical aliases."""

    screenshot_sha256: str
    image_path: Path
    screen_ids: tuple[str, ...]
    width: int
    height: int
    splits: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ManifestScreenSet:
    """Verified manifest identity and deterministic unique-screen inventory."""

    manifest_digest: str
    dataset_version: str
    dataset_root: Path
    screens: tuple[ManifestScreen, ...]


@dataclass(frozen=True, slots=True)
class PerceptionBundleSpec:
    """Compatibility identity shared by every cache entry in one bundle."""

    model_bundle_lock_digest: str
    perception_config_digest: str
    visual_checkpoint_sha256: str
    include_detector_roi_features: bool
    cache_schema_version: int = PERCEPTION_CACHE_SCHEMA_VERSION
    canonical_schema_version: str = CANONICAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name, value in (
            ("model_bundle_lock_digest", self.model_bundle_lock_digest),
            ("perception_config_digest", self.perception_config_digest),
            ("visual_checkpoint_sha256", self.visual_checkpoint_sha256),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be lowercase SHA256")

    @property
    def digest(self) -> str:
        return _digest(self.as_dict())

    def as_dict(self) -> dict[str, object]:
        return {
            "model_bundle_lock_digest": self.model_bundle_lock_digest,
            "perception_config_digest": self.perception_config_digest,
            "visual_checkpoint_sha256": self.visual_checkpoint_sha256,
            "include_detector_roi_features": self.include_detector_roi_features,
            "cache_schema_version": self.cache_schema_version,
            "canonical_schema_version": self.canonical_schema_version,
            "command_conditioned": False,
        }

    @classmethod
    def from_service(cls, service: PerceptionService) -> PerceptionBundleSpec:
        return cls(
            model_bundle_lock_digest=service.model_bundle_lock_digest,
            perception_config_digest=service.config.digest,
            visual_checkpoint_sha256=service.config.visual_checkpoint_sha256,
            include_detector_roi_features=service.config.include_detector_roi_features,
        )


@dataclass(frozen=True, slots=True)
class PrecomputeResult:
    """One deterministic shard result suitable for CLI JSON output."""

    status: str
    bundle_digest: str
    data_manifest_digest: str
    shard_index: int
    shard_count: int
    assigned: int
    completed: int
    reused: int
    failed: int
    shard_status_path: str
    cache_manifest_path: str


@dataclass(frozen=True, slots=True)
class CacheValidation:
    """Observed validation counts across completed precompute shards."""

    ok: bool
    bundle_digest: str
    run_manifests: int
    expected_shards: int
    shard_documents: int
    complete_shards: int
    entries: int
    failures: int


class PerceptionService(Protocol):
    """Minimal command-free service contract used by the orchestrator."""

    model_bundle_lock_digest: str
    config: FullScreenPerceptionConfig
    cache: ContentAddressedPerceptionCache | None

    def key_for(self, image: ImageFrame) -> PerceptionCacheKey:
        """Return the cache key for one normalized screenshot."""

    def perceive(
        self,
        images: Sequence[object],
        *,
        mode: str = "real",
    ) -> tuple[PerceptionFrame, ...]:
        """Perceive command-independent screenshots in caller order."""


def _manifest_payload(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("data manifest must be a mapping")
    return cast(dict[str, object], payload)


def load_manifest_screens(path: Path, *, max_screens: int | None = None) -> ManifestScreenSet:
    """Verify a data manifest and return unique screens in stable hash order."""

    if max_screens is not None and max_screens <= 0:
        raise ValueError("max_screens must be positive when provided")
    verification = verify_data_manifest(path)
    payload = _manifest_payload(path)
    split = payload.get("split")
    if not isinstance(split, dict):
        raise ValueError("data manifest split declaration is missing")
    membership_name = split.get("membership_path")
    if not isinstance(membership_name, str):
        raise ValueError("data manifest membership path is missing")
    membership_path = (path.parent / membership_name).resolve()
    active_ids: set[str] = set()
    for line_number, line in enumerate(
        membership_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict) or not isinstance(raw.get("screen_id"), str):
            raise ValueError(f"invalid split membership row {line_number}")
        active_ids.add(str(raw["screen_id"]))
    dataset_root = Path(verification.dataset_root).resolve()
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in read_table_rows(dataset_root, "screens"):
        screen_id = str(row["screen_id"])
        if screen_id not in active_ids:
            continue
        digest = str(row["screen_sha256"])
        grouped.setdefault(digest, []).append(row)
    observed_ids = {str(row["screen_id"]) for rows in grouped.values() for row in rows}
    if observed_ids != active_ids:
        raise ValueError("manifest membership references missing canonical screens")

    screens: list[ManifestScreen] = []
    for digest, rows in sorted(grouped.items()):
        paths = {str(row["image_path"]) for row in rows}
        dimensions = {(int(str(row["width"])), int(str(row["height"]))) for row in rows}
        if len(paths) != 1 or len(dimensions) != 1:
            raise ValueError(f"content hash {digest} has conflicting image metadata")
        relative = Path(next(iter(paths)))
        image_path = (dataset_root / relative).resolve()
        if not image_path.is_relative_to(dataset_root):
            raise ValueError("canonical image path escapes dataset root")
        if sha256_file(image_path) != digest:
            raise ValueError(f"canonical image digest mismatch: {relative.as_posix()}")
        width, height = next(iter(dimensions))
        screens.append(
            ManifestScreen(
                screenshot_sha256=digest,
                image_path=image_path,
                screen_ids=tuple(sorted(str(row["screen_id"]) for row in rows)),
                width=width,
                height=height,
                splits=tuple(sorted({str(row["split"]) for row in rows})),
            )
        )
    if max_screens is not None:
        screens = screens[:max_screens]
    if not screens:
        raise ValueError("data manifest selected zero unique screens")
    dataset_version = payload.get("dataset_version")
    manifest_digest = payload.get("manifest_digest")
    if not isinstance(dataset_version, str) or not isinstance(manifest_digest, str):
        raise ValueError("data manifest identity is missing")
    return ManifestScreenSet(
        manifest_digest=manifest_digest,
        dataset_version=dataset_version,
        dataset_root=dataset_root,
        screens=tuple(screens),
    )


def assigned_screens(
    screens: Sequence[ManifestScreen],
    *,
    shard_index: int,
    shard_count: int,
) -> tuple[ManifestScreen, ...]:
    """Assign screens by stable content hash for torchrun/process-safe parallelism."""

    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("shard index must be in [0, shard_count)")
    return tuple(
        screen
        for screen in screens
        if int(screen.screenshot_sha256[:16], 16) % shard_count == shard_index
    )


def bundle_cache(cache_root: Path, spec: PerceptionBundleSpec) -> ContentAddressedPerceptionCache:
    """Resolve the content cache below `<root>/perception/<bundle-digest>`."""

    return ContentAddressedPerceptionCache(cache_root.resolve() / "perception" / spec.digest[:20])


def _bundle_manifest(spec: PerceptionBundleSpec) -> dict[str, object]:
    return _document_with_digest(
        {
            "cache_bundle_manifest_schema_version": CACHE_BUNDLE_MANIFEST_SCHEMA_VERSION,
            "bundle_digest": spec.digest,
            **spec.as_dict(),
        }
    )


def ensure_bundle_manifest(
    cache: ContentAddressedPerceptionCache,
    spec: PerceptionBundleSpec,
) -> Path:
    """Create or verify the immutable bundle compatibility manifest."""

    return cache.write_control("manifest.json", _bundle_manifest(spec), immutable=True)


def _run_root(data_manifest_digest: str, shard_count: int) -> str:
    return f"runs/{data_manifest_digest[:20]}/s{shard_count:05d}"


def _run_manifest(
    screen_set: ManifestScreenSet,
    spec: PerceptionBundleSpec,
    *,
    shard_count: int,
) -> dict[str, object]:
    screen_hashes = [screen.screenshot_sha256 for screen in screen_set.screens]
    return _document_with_digest(
        {
            "precompute_run_schema_version": PRECOMPUTE_RUN_SCHEMA_VERSION,
            "bundle_digest": spec.digest,
            "data_manifest_digest": screen_set.manifest_digest,
            "dataset_version": screen_set.dataset_version,
            "shard_count": shard_count,
            "unique_screen_count": len(screen_hashes),
            "screen_set_digest": _digest({"screen_sha256": screen_hashes}),
            "command_conditioned": False,
        }
    )


def _safe_error_message(message: str, dataset_root: Path) -> str:
    root = str(dataset_root)
    return message.replace(root, "${SCREEN2ACTION_DATA_ROOT}")[:1000]


def _reject_command_keys(value: object, *, path: str = "raw_outputs") -> None:
    forbidden = {"command", "command_tokens", "query", "query_tokens", "instruction"}
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if normalized in forbidden:
                raise ValueError(f"command-conditioned cache field is forbidden: {path}.{key}")
            _reject_command_keys(item, path=f"{path}.{key}")
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _reject_command_keys(item, path=f"{path}[{index}]")


def _validate_frame(frame: PerceptionFrame, key: PerceptionCacheKey) -> None:
    if frame.screenshot_sha256 != key.screenshot_sha256:
        raise ValueError("perception frame screenshot digest changed")
    if frame.cache_key_digest != key.digest:
        raise ValueError("perception frame cache key digest mismatch")
    if frame.raw_outputs.get("command_conditioned") is not False:
        raise ValueError("perception cache must explicitly declare command_conditioned=false")
    _reject_command_keys(frame.raw_outputs)
    for required in ("detector", "ocr", "icon_actionability"):
        if not isinstance(frame.raw_outputs.get(required), dict):
            raise ValueError(f"perception cache is missing raw {required} outputs")


def _entry_record(
    screen: ManifestScreen,
    frame: PerceptionFrame,
    key: PerceptionCacheKey,
    cache: ContentAddressedPerceptionCache,
    *,
    status: str,
    attempts: int,
) -> dict[str, object]:
    entry_path = cache.entry_path(key)
    if not entry_path.is_file():
        raise RuntimeError("perception service returned without atomically caching its frame")
    return {
        "canonical_screen_sha256": screen.screenshot_sha256,
        "perception_screenshot_sha256": frame.screenshot_sha256,
        "screen_ids": list(screen.screen_ids),
        "splits": list(screen.splits),
        "cache_key": key.as_dict(),
        "cache_key_digest": key.digest,
        "entry_path": entry_path.relative_to(cache.root).as_posix(),
        "entry_sha256": sha256_file(entry_path),
        "status": status,
        "attempts": attempts,
    }


def _shard_document(
    *,
    spec: PerceptionBundleSpec,
    screen_set: ManifestScreenSet,
    shard_index: int,
    shard_count: int,
    assigned: Sequence[ManifestScreen],
    entries: Sequence[Mapping[str, object]],
    failures: Sequence[Mapping[str, object]],
    status: str,
) -> dict[str, object]:
    return _document_with_digest(
        {
            "precompute_shard_schema_version": PRECOMPUTE_SHARD_SCHEMA_VERSION,
            "bundle_digest": spec.digest,
            "data_manifest_digest": screen_set.manifest_digest,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "assigned_screen_count": len(assigned),
            "assigned_screen_digest": _digest(
                {"screen_sha256": [screen.screenshot_sha256 for screen in assigned]}
            ),
            "status": status,
            "entries": list(entries),
            "failures": list(failures),
            "command_conditioned": False,
        }
    )


def _key_from_record(raw: object) -> PerceptionCacheKey:
    if not isinstance(raw, dict):
        raise ValueError("cached shard key must be a mapping")
    return PerceptionCacheKey(
        screenshot_sha256=str(raw["screenshot_sha256"]),
        model_bundle_lock_digest=str(raw["model_bundle_lock_digest"]),
        perception_config_digest=str(raw["perception_config_digest"]),
        schema_version=int(raw.get("schema_version", PERCEPTION_CACHE_SCHEMA_VERSION)),
    )


def _validate_shard_document(
    cache: ContentAddressedPerceptionCache,
    document: Mapping[str, object],
    *,
    expected_bundle_digest: str,
) -> tuple[int, int]:
    _verify_document_digest(document, context="precompute shard")
    if document.get("bundle_digest") != expected_bundle_digest:
        raise ValueError("precompute shard belongs to a different cache bundle")
    entries = document.get("entries")
    failures = document.get("failures")
    if not isinstance(entries, list) or not isinstance(failures, list):
        raise ValueError("precompute shard entries/failures are malformed")
    for raw in entries:
        if not isinstance(raw, dict):
            raise ValueError("precompute shard entry must be a mapping")
        key = _key_from_record(raw.get("cache_key"))
        if raw.get("cache_key_digest") != key.digest:
            raise ValueError("precompute shard cache key digest mismatch")
        payload = cache.get(key)
        if payload is None:
            raise FileNotFoundError(f"perception cache entry is missing: {key.digest}")
        frame = perception_frame_from_payload(payload, cache_key_digest=key.digest)
        _validate_frame(frame, key)
        entry_path = cache.entry_path(key)
        if raw.get("entry_sha256") != sha256_file(entry_path):
            raise ValueError(f"perception entry digest mismatch: {key.digest}")
    return len(entries), len(failures)


def _result_from_document(
    cache: ContentAddressedPerceptionCache,
    document: Mapping[str, object],
    *,
    status: str,
    shard_path: str,
) -> PrecomputeResult:
    entries = cast(list[dict[str, object]], document["entries"])
    failures = cast(list[dict[str, object]], document["failures"])
    return PrecomputeResult(
        status=status,
        bundle_digest=str(document["bundle_digest"]),
        data_manifest_digest=str(document["data_manifest_digest"]),
        shard_index=int(str(document["shard_index"])),
        shard_count=int(str(document["shard_count"])),
        assigned=int(str(document["assigned_screen_count"])),
        completed=len(entries),
        reused=sum(entry.get("status") == "already_complete" for entry in entries),
        failed=len(failures),
        shard_status_path=shard_path,
        cache_manifest_path=(cache.root / "manifest.json").as_posix(),
    )


def precompute_perception(
    manifest_path: Path,
    *,
    service: PerceptionService,
    cache: ContentAddressedPerceptionCache,
    shard_index: int = 0,
    shard_count: int = 1,
    batch_size: int = 4,
    max_retries: int = 3,
    max_screens: int | None = None,
) -> PrecomputeResult:
    """Precompute one deterministic shard with entry-level resume and retries."""

    if batch_size <= 0 or max_retries <= 0:
        raise ValueError("batch_size and max_retries must be positive")
    if service.cache is not cache:
        raise ValueError("perception service must write to the supplied bundle cache")
    spec = PerceptionBundleSpec.from_service(service)
    screen_set = load_manifest_screens(manifest_path, max_screens=max_screens)
    assigned = assigned_screens(
        screen_set.screens,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    cache.cleanup_stale_temporary()
    ensure_bundle_manifest(cache, spec)
    run_root = _run_root(screen_set.manifest_digest, shard_count)
    cache.write_control(
        f"{run_root}/manifest.json",
        _run_manifest(screen_set, spec, shard_count=shard_count),
        immutable=True,
    )
    shard_path = f"{run_root}/s/{shard_index:05d}.json"
    previous = cache.read_control(shard_path)
    if previous is not None and previous.get("status") == "complete":
        _validate_shard_document(cache, previous, expected_bundle_digest=spec.digest)
        return _result_from_document(
            cache,
            previous,
            status="already_complete",
            shard_path=shard_path,
        )

    previous_attempts: dict[str, int] = {}
    if previous is not None:
        _verify_document_digest(previous, context="previous precompute shard")
        previous_failures = previous.get("failures", [])
        if isinstance(previous_failures, list):
            for previous_failure in previous_failures:
                if isinstance(previous_failure, dict):
                    previous_attempts[str(previous_failure.get("canonical_screen_sha256", ""))] = (
                        int(str(previous_failure.get("attempts", 0)))
                    )

    entries: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    def checkpoint(status: str) -> None:
        cache.write_control(
            shard_path,
            _shard_document(
                spec=spec,
                screen_set=screen_set,
                shard_index=shard_index,
                shard_count=shard_count,
                assigned=assigned,
                entries=entries,
                failures=failures,
                status=status,
            ),
        )

    checkpoint("in_progress")
    for offset in range(0, len(assigned), batch_size):
        batch = assigned[offset : offset + batch_size]
        prepared: list[tuple[ManifestScreen, ImageFrame, PerceptionCacheKey]] = []
        for screen in batch:
            start = previous_attempts.get(screen.screenshot_sha256, 0)
            for local_attempt in range(1, max_retries + 1):
                attempt = start + local_attempt
                try:
                    image = normalize_image(screen.image_path)
                    if (image.width, image.height) != (screen.width, screen.height):
                        raise ValueError("canonical image dimensions differ from manifest")
                    key = service.key_for(image)
                    payload = cache.get(key)
                    if payload is not None:
                        frame = perception_frame_from_payload(
                            payload,
                            cache_key_digest=key.digest,
                        )
                        _validate_frame(frame, key)
                        entries.append(
                            _entry_record(
                                screen,
                                frame,
                                key,
                                cache,
                                status="already_complete",
                                attempts=start,
                            )
                        )
                    else:
                        prepared.append((screen, image, key))
                    break
                except Exception as error:
                    if local_attempt < max_retries:
                        continue
                    failures.append(
                        {
                            "canonical_screen_sha256": screen.screenshot_sha256,
                            "screen_ids": list(screen.screen_ids),
                            "error_type": type(error).__name__,
                            "message": _safe_error_message(
                                str(error),
                                screen_set.dataset_root,
                            ),
                            "attempts": attempt,
                            "retryable": True,
                        }
                    )
        if prepared:
            try:
                frames = service.perceive(tuple(item[1] for item in prepared), mode="real")
                if len(frames) != len(prepared):
                    raise RuntimeError("perception service changed batch cardinality")
                for (screen, _image, key), frame in zip(prepared, frames, strict=True):
                    _validate_frame(frame, key)
                    entries.append(
                        _entry_record(
                            screen,
                            frame,
                            key,
                            cache,
                            status="written" if frame.mode != "cached" else "already_complete",
                            attempts=previous_attempts.get(screen.screenshot_sha256, 0) + 1,
                        )
                    )
            except Exception:
                for screen, image, key in prepared:
                    failure: dict[str, object] | None = None
                    start = previous_attempts.get(screen.screenshot_sha256, 0)
                    for local_attempt in range(1, max_retries + 1):
                        attempt = start + local_attempt
                        try:
                            frame = service.perceive((image,), mode="real")[0]
                            _validate_frame(frame, key)
                            entries.append(
                                _entry_record(
                                    screen,
                                    frame,
                                    key,
                                    cache,
                                    status=(
                                        "written" if frame.mode != "cached" else "already_complete"
                                    ),
                                    attempts=attempt,
                                )
                            )
                            failure = None
                            break
                        except Exception as error:
                            message = _safe_error_message(str(error), screen_set.dataset_root)
                            cache.record_failure(
                                key,
                                error_type=type(error).__name__,
                                message=message,
                                attempt=attempt,
                            )
                            failure = {
                                "canonical_screen_sha256": screen.screenshot_sha256,
                                "screen_ids": list(screen.screen_ids),
                                "cache_key_digest": key.digest,
                                "error_type": type(error).__name__,
                                "message": message,
                                "attempts": attempt,
                                "retryable": True,
                            }
                    if failure is not None:
                        failures.append(failure)
        checkpoint("in_progress")

    entries.sort(key=lambda item: str(item["canonical_screen_sha256"]))
    failures.sort(key=lambda item: str(item["canonical_screen_sha256"]))
    status = "complete" if not failures else "partial_failure"
    checkpoint(status)
    document = cache.read_control(shard_path)
    assert document is not None
    if status == "complete":
        _validate_shard_document(cache, document, expected_bundle_digest=spec.digest)
    return _result_from_document(cache, document, status=status, shard_path=shard_path)


def validate_perception_cache(manifest_path: Path) -> CacheValidation:
    """Validate bundle/run/shard digests and every referenced entry offline."""

    cache = ContentAddressedPerceptionCache(manifest_path.resolve().parent)
    manifest = cache.read_control("manifest.json")
    if manifest is None:
        raise FileNotFoundError(f"perception cache manifest is missing: {manifest_path}")
    _verify_document_digest(manifest, context="perception cache manifest")
    bundle_digest = manifest.get("bundle_digest")
    if not isinstance(bundle_digest, str):
        raise ValueError("perception cache bundle digest is missing")
    semantic_spec = {
        key: value
        for key, value in manifest.items()
        if key
        not in {
            "cache_bundle_manifest_schema_version",
            "bundle_digest",
            "document_digest",
        }
    }
    if _digest(semantic_spec) != bundle_digest:
        raise ValueError("perception bundle identity digest mismatch")

    run_manifests = 0
    expected_shards = 0
    shard_documents = 0
    complete_shards = 0
    entries = 0
    failures = 0
    runs_complete = True
    for run_path in sorted(cache.root.glob("runs/*/s*/manifest.json")):
        run_relative = run_path.relative_to(cache.root).as_posix()
        run = cache.read_control(run_relative)
        assert run is not None
        _verify_document_digest(run, context="perception precompute run")
        if run.get("bundle_digest") != bundle_digest:
            raise ValueError("precompute run belongs to a different cache bundle")
        shard_count = int(str(run.get("shard_count", 0)))
        unique_screen_count = int(str(run.get("unique_screen_count", -1)))
        if shard_count <= 0 or unique_screen_count < 0:
            raise ValueError("precompute run counts are invalid")
        run_manifests += 1
        expected_shards += shard_count
        run_documents: list[dict[str, object]] = []
        for path in sorted(run_path.parent.glob("s/*.json")):
            relative = path.relative_to(cache.root).as_posix()
            document = cache.read_control(relative)
            assert document is not None
            entry_count, failure_count = _validate_shard_document(
                cache,
                document,
                expected_bundle_digest=bundle_digest,
            )
            run_documents.append(cast(dict[str, object], document))
            shard_documents += 1
            complete_shards += int(document.get("status") == "complete")
            entries += entry_count
            failures += failure_count
        run_is_complete = len(run_documents) == shard_count and all(
            document.get("status") == "complete" for document in run_documents
        )
        runs_complete = runs_complete and run_is_complete
        represented = sorted(
            str(item["canonical_screen_sha256"])
            for document in run_documents
            for field in ("entries", "failures")
            for item in cast(list[dict[str, object]], document.get(field, []))
        )
        if len(represented) != len(set(represented)):
            raise ValueError("precompute run represents one screen more than once")
        if len(represented) != unique_screen_count:
            runs_complete = False
        elif _digest({"screen_sha256": represented}) != run.get("screen_set_digest"):
            raise ValueError("precompute run screen-set digest mismatch")
    stats = cache.stats()
    if stats["temporary"] or stats["locks"]:
        raise ValueError("perception cache contains unfinished temporary files or locks")
    return CacheValidation(
        ok=bool(run_manifests and runs_complete and failures == 0),
        bundle_digest=bundle_digest,
        run_manifests=run_manifests,
        expected_shards=expected_shards,
        shard_documents=shard_documents,
        complete_shards=complete_shards,
        entries=entries,
        failures=failures,
    )


def perception_cache_stats(manifest_path: Path) -> dict[str, object]:
    """Return bounded entry/failure/run counts without loading model dependencies."""

    validation = validate_perception_cache(manifest_path)
    cache = ContentAddressedPerceptionCache(manifest_path.resolve().parent)
    return {**asdict(validation), **cache.stats()}
