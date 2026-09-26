"""Typed model registry, immutable lock, resumable fetch, and offline verification."""

from __future__ import annotations

import fnmatch
import hashlib
import importlib
import importlib.metadata
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Protocol, cast

import yaml

from screen2action import MODEL_LOCK_SCHEMA_VERSION

REQUIRED_MODEL_ROLES = frozenset(
    {
        "ui_detector_screenparser",
        "ocr_crnn_vgg16_bn",
        "icon_backbone_mobilenet_v3_small",
        "command_encoder_bert_l6_h256_a4",
        "crop_encoder_mobilevit_s",
    }
)


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be a string-keyed mapping")
    return cast(dict[str, object], value)


def _string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _optional_string(value: object, *, context: str) -> str | None:
    if value is None:
        return None
    return _string(value, context=context)


def _string_tuple(value: object, *, context: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{context} must be a list of non-empty strings")
    return tuple(cast(list[str], value))


def _safe_relative_path(value: str, *, context: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"{context} must be a safe relative path")
    return path.as_posix()


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Mutable registry intent for one logical model role."""

    role: str
    backend: str
    asset_provider: str
    upstream_identifier: str
    upstream_revision: str | None
    artifact_selector: str
    allow_patterns: tuple[str, ...]
    license_identifier: str
    license_note: str
    preprocessing: Mapping[str, object]
    output: Mapping[str, object]
    reconstruction_caveat: str


@dataclass(frozen=True, slots=True)
class ModelRegistry:
    """Validated registry plus its source digest."""

    schema_version: int
    registry_id: str
    digest: str
    path: str
    roles: Mapping[str, ModelSpec]


@dataclass(frozen=True, slots=True)
class RemoteFile:
    """One expected upstream artifact before local fetch."""

    remote_path: str
    cache_path: str
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        _safe_relative_path(self.cache_path, context="cache_path")
        _safe_relative_path(self.remote_path, context="remote_path")
        if self.source_sha256 is not None and len(self.source_sha256) != 64:
            raise ValueError("source_sha256 must contain 64 hexadecimal characters")


@dataclass(frozen=True, slots=True)
class ProviderResolution:
    """Immutable upstream resolution returned by a provider."""

    revision: str
    package_version: str | None
    files: tuple[RemoteFile, ...]


@dataclass(frozen=True, slots=True)
class LockedFile:
    """Expected and optionally verified local artifact."""

    remote_path: str
    path: str
    source_sha256: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None

    def __post_init__(self) -> None:
        _safe_relative_path(self.remote_path, context="remote_path")
        _safe_relative_path(self.path, context="path")
        for field_name, digest in (
            ("source_sha256", self.source_sha256),
            ("sha256", self.sha256),
        ):
            if digest is not None and (
                len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{field_name} must be lowercase SHA256")
        if self.size_bytes is not None and self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")


@dataclass(frozen=True, slots=True)
class LockedModel:
    """One immutable model resolution and its fetched-file evidence."""

    role: str
    backend: str
    asset_provider: str
    upstream_identifier: str
    resolved_revision: str
    package_version: str | None
    artifact_selector: str
    license_identifier: str
    license_note: str
    license_acknowledgement_id: str | None
    preprocessing: Mapping[str, object]
    output: Mapping[str, object]
    reconstruction_caveat: str
    files: tuple[LockedFile, ...]
    complete: bool = False


@dataclass(frozen=True, slots=True)
class ModelLock:
    """Portable immutable model bundle declaration."""

    schema_version: int
    registry_id: str
    registry_path: str
    registry_sha256: str
    generated_at_utc: str
    models: tuple[LockedModel, ...]

    @property
    def digest(self) -> str:
        payload = json.dumps(lock_to_dict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ModelAssetProvider(Protocol):
    """Network/package boundary injectable in offline tests."""

    def resolve(self, spec: ModelSpec) -> ProviderResolution:
        """Resolve mutable intent to an immutable revision and expected files."""

    def fetch(self, model: LockedModel, cache_root: Path) -> Sequence[Path]:
        """Fetch every expected file and return local paths."""


def load_model_registry(path: str | Path) -> ModelRegistry:
    """Load and validate all required logical model roles."""

    source = Path(path)
    payload = _json_mapping(
        yaml.safe_load(source.read_text(encoding="utf-8")),
        context="model registry",
    )
    version = payload.get("schema_version")
    if version != MODEL_LOCK_SCHEMA_VERSION:
        raise ValueError(f"model registry schema_version must be {MODEL_LOCK_SCHEMA_VERSION}")
    raw_roles = _json_mapping(payload.get("roles"), context="model registry roles")
    missing = REQUIRED_MODEL_ROLES.difference(raw_roles)
    if missing:
        raise ValueError(f"model registry is missing required roles: {sorted(missing)}")
    roles: dict[str, ModelSpec] = {}
    for role, raw in sorted(raw_roles.items()):
        entry = _json_mapping(raw, context=f"model role {role}")
        roles[role] = ModelSpec(
            role=role,
            backend=_string(entry.get("backend"), context=f"{role}.backend"),
            asset_provider=_string(entry.get("asset_provider"), context=f"{role}.asset_provider"),
            upstream_identifier=_string(
                entry.get("upstream_identifier"), context=f"{role}.upstream_identifier"
            ),
            upstream_revision=_optional_string(
                entry.get("upstream_revision"), context=f"{role}.upstream_revision"
            ),
            artifact_selector=_string(
                entry.get("artifact_selector"), context=f"{role}.artifact_selector"
            ),
            allow_patterns=_string_tuple(
                entry.get("allow_patterns"), context=f"{role}.allow_patterns"
            ),
            license_identifier=_string(
                entry.get("license_identifier"), context=f"{role}.license_identifier"
            ),
            license_note=_string(entry.get("license_note"), context=f"{role}.license_note"),
            preprocessing=_json_mapping(
                entry.get("preprocessing"), context=f"{role}.preprocessing"
            ),
            output=_json_mapping(entry.get("output"), context=f"{role}.output"),
            reconstruction_caveat=_string(
                entry.get("reconstruction_caveat"),
                context=f"{role}.reconstruction_caveat",
            ),
        )
    try:
        display_path = source.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        display_path = source.name
    return ModelRegistry(
        schema_version=version,
        registry_id=_string(payload.get("registry_id"), context="registry_id"),
        digest=sha256_file(source),
        path=display_path,
        roles=roles,
    )


def _locked_file_to_dict(file: LockedFile) -> dict[str, object]:
    return {
        "remote_path": file.remote_path,
        "path": file.path,
        "source_sha256": file.source_sha256,
        "sha256": file.sha256,
        "size_bytes": file.size_bytes,
    }


def _locked_model_to_dict(model: LockedModel) -> dict[str, object]:
    return {
        "role": model.role,
        "backend": model.backend,
        "asset_provider": model.asset_provider,
        "upstream_identifier": model.upstream_identifier,
        "resolved_revision": model.resolved_revision,
        "package_version": model.package_version,
        "artifact_selector": model.artifact_selector,
        "license_identifier": model.license_identifier,
        "license_note": model.license_note,
        "license_acknowledgement_id": model.license_acknowledgement_id,
        "preprocessing": dict(model.preprocessing),
        "output": dict(model.output),
        "reconstruction_caveat": model.reconstruction_caveat,
        "files": [_locked_file_to_dict(file) for file in model.files],
        "complete": model.complete,
    }


def lock_to_dict(lock: ModelLock) -> dict[str, object]:
    """Serialize a model lock using a stable public shape."""

    return {
        "schema_version": lock.schema_version,
        "registry_id": lock.registry_id,
        "registry_path": lock.registry_path,
        "registry_sha256": lock.registry_sha256,
        "generated_at_utc": lock.generated_at_utc,
        "models": [_locked_model_to_dict(model) for model in lock.models],
    }


def _locked_file_from_dict(raw: object, *, context: str) -> LockedFile:
    entry = _json_mapping(raw, context=context)
    size = entry.get("size_bytes")
    if size is not None and not isinstance(size, int):
        raise ValueError(f"{context}.size_bytes must be an integer or null")
    return LockedFile(
        remote_path=_string(entry.get("remote_path"), context=f"{context}.remote_path"),
        path=_string(entry.get("path"), context=f"{context}.path"),
        source_sha256=_optional_string(
            entry.get("source_sha256"), context=f"{context}.source_sha256"
        ),
        sha256=_optional_string(entry.get("sha256"), context=f"{context}.sha256"),
        size_bytes=size,
    )


def _locked_model_from_dict(raw: object, *, context: str) -> LockedModel:
    entry = _json_mapping(raw, context=context)
    raw_files = entry.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError(f"{context}.files must be a non-empty list")
    complete = entry.get("complete", False)
    if not isinstance(complete, bool):
        raise ValueError(f"{context}.complete must be boolean")
    return LockedModel(
        role=_string(entry.get("role"), context=f"{context}.role"),
        backend=_string(entry.get("backend"), context=f"{context}.backend"),
        asset_provider=_string(entry.get("asset_provider"), context=f"{context}.asset_provider"),
        upstream_identifier=_string(
            entry.get("upstream_identifier"), context=f"{context}.upstream_identifier"
        ),
        resolved_revision=_string(
            entry.get("resolved_revision"), context=f"{context}.resolved_revision"
        ),
        package_version=_optional_string(
            entry.get("package_version"), context=f"{context}.package_version"
        ),
        artifact_selector=_string(
            entry.get("artifact_selector"), context=f"{context}.artifact_selector"
        ),
        license_identifier=_string(
            entry.get("license_identifier"), context=f"{context}.license_identifier"
        ),
        license_note=_string(entry.get("license_note"), context=f"{context}.license_note"),
        license_acknowledgement_id=_optional_string(
            entry.get("license_acknowledgement_id"),
            context=f"{context}.license_acknowledgement_id",
        ),
        preprocessing=_json_mapping(entry.get("preprocessing"), context=f"{context}.preprocessing"),
        output=_json_mapping(entry.get("output"), context=f"{context}.output"),
        reconstruction_caveat=_string(
            entry.get("reconstruction_caveat"),
            context=f"{context}.reconstruction_caveat",
        ),
        files=tuple(
            _locked_file_from_dict(item, context=f"{context}.files[{index}]")
            for index, item in enumerate(raw_files)
        ),
        complete=complete,
    )


def load_model_lock(path: str | Path) -> ModelLock:
    """Load a lock without importing any model framework or accessing a network."""

    source = Path(path)
    payload = _json_mapping(json.loads(source.read_text(encoding="utf-8")), context="model lock")
    version = payload.get("schema_version")
    if version != MODEL_LOCK_SCHEMA_VERSION:
        raise ValueError(f"model lock schema_version must be {MODEL_LOCK_SCHEMA_VERSION}")
    raw_models = payload.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("model lock models must be a non-empty list")
    models = tuple(
        _locked_model_from_dict(item, context=f"models[{index}]")
        for index, item in enumerate(raw_models)
    )
    roles = {model.role for model in models}
    if roles != REQUIRED_MODEL_ROLES:
        raise ValueError("model lock roles do not match the required public bundle")
    return ModelLock(
        schema_version=version,
        registry_id=_string(payload.get("registry_id"), context="registry_id"),
        registry_path=_safe_relative_path(
            _string(payload.get("registry_path"), context="registry_path"),
            context="registry_path",
        ),
        registry_sha256=_string(payload.get("registry_sha256"), context="registry_sha256"),
        generated_at_utc=_string(payload.get("generated_at_utc"), context="generated_at_utc"),
        models=models,
    )


def write_model_lock(lock: ModelLock, path: str | Path) -> Path:
    """Atomically write a lock document."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(lock_to_dict(lock), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


class HuggingFaceProvider:
    """Resolve and fetch pinned Hub files using lazy official APIs."""

    def resolve(self, spec: ModelSpec) -> ProviderResolution:
        try:
            hub = importlib.import_module("huggingface_hub")
        except ImportError as error:
            raise RuntimeError(
                "install the perception extra to resolve Hugging Face assets"
            ) from error
        api = hub.HfApi()
        try:
            info = api.model_info(
                spec.upstream_identifier,
                revision=spec.upstream_revision,
                files_metadata=True,
            )
        except Exception as error:
            raise RuntimeError(
                f"failed to resolve Hub metadata for {spec.upstream_identifier}: {error}"
            ) from error
        revision = _string(getattr(info, "sha", None), context=f"{spec.role} Hub commit")
        selected: list[RemoteFile] = []
        for sibling in getattr(info, "siblings", ()):
            filename = str(sibling.rfilename)
            if spec.allow_patterns and not any(
                fnmatch.fnmatch(filename, pattern) for pattern in spec.allow_patterns
            ):
                continue
            lfs = getattr(sibling, "lfs", None)
            source_sha = (
                lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
            )
            selected.append(
                RemoteFile(
                    remote_path=filename,
                    cache_path=f"models/{spec.role}/files/{filename}",
                    source_sha256=source_sha,
                )
            )
        if not selected:
            raise RuntimeError(f"no Hub files matched {spec.role} allow_patterns")
        return ProviderResolution(
            revision=revision,
            package_version=importlib.metadata.version("huggingface-hub"),
            files=tuple(sorted(selected, key=lambda item: item.remote_path)),
        )

    def fetch(self, model: LockedModel, cache_root: Path) -> Sequence[Path]:
        try:
            hub = importlib.import_module("huggingface_hub")
        except ImportError as error:
            raise RuntimeError(
                "install the perception extra to fetch Hugging Face assets"
            ) from error
        downloaded: list[Path] = []
        local_directory = cache_root / "models" / model.role / "files"
        for file in model.files:
            path = Path(
                hub.hf_hub_download(
                    repo_id=model.upstream_identifier,
                    filename=file.remote_path,
                    revision=model.resolved_revision,
                    local_dir=local_directory,
                )
            )
            downloaded.append(path)
        return downloaded


class TorchvisionProvider:
    """Resolve a TorchVision enum and materialize its state dict."""

    def resolve(self, spec: ModelSpec) -> ProviderResolution:
        version = importlib.metadata.version("torchvision")
        return ProviderResolution(
            revision=(f"torchvision=={version}:MobileNet_V3_Small_Weights.IMAGENET1K_V1"),
            package_version=version,
            files=(
                RemoteFile(
                    remote_path="MobileNet_V3_Small_Weights.IMAGENET1K_V1",
                    cache_path=f"models/{spec.role}/weights.pt",
                ),
            ),
        )

    def fetch(self, model: LockedModel, cache_root: Path) -> Sequence[Path]:
        try:
            torch = importlib.import_module("torch")
            models = importlib.import_module("torchvision.models")
        except ImportError as error:
            raise RuntimeError("install CPU PyTorch and the perception extra") from error
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        module = models.mobilenet_v3_small(weights=weights)
        destination = cache_root / model.files[0].path
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".pt.partial")
        torch.save(module.state_dict(), temporary)
        temporary.replace(destination)
        return (destination,)


class DoctrProvider:
    """Resolve docTR's package-pinned CRNN and materialize its state dict."""

    def resolve(self, spec: ModelSpec) -> ProviderResolution:
        version = importlib.metadata.version("python-doctr")
        return ProviderResolution(
            revision=f"python-doctr=={version}:crnn_vgg16_bn:pretrained",
            package_version=version,
            files=(
                RemoteFile(
                    remote_path="crnn_vgg16_bn:pretrained",
                    cache_path=f"models/{spec.role}/weights.pt",
                ),
            ),
        )

    def fetch(self, model: LockedModel, cache_root: Path) -> Sequence[Path]:
        try:
            torch = importlib.import_module("torch")
            doctr_models = importlib.import_module("doctr.models")
        except ImportError as error:
            raise RuntimeError("install CPU PyTorch and the perception extra") from error
        module = doctr_models.crnn_vgg16_bn(pretrained=True)
        destination = cache_root / model.files[0].path
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".pt.partial")
        torch.save(module.state_dict(), temporary)
        temporary.replace(destination)
        return (destination,)


def default_model_providers() -> Mapping[str, ModelAssetProvider]:
    """Return lightweight provider objects; heavy modules remain lazy."""

    return {
        "huggingface": HuggingFaceProvider(),
        "torchvision": TorchvisionProvider(),
        "doctr": DoctrProvider(),
    }


def resolve_model_lock(
    registry: ModelRegistry,
    *,
    providers: Mapping[str, ModelAssetProvider] | None = None,
    generated_at_utc: str | None = None,
) -> ModelLock:
    """Resolve every registry role to immutable provider metadata."""

    available = default_model_providers() if providers is None else providers
    models: list[LockedModel] = []
    for role, spec in sorted(registry.roles.items()):
        try:
            provider = available[spec.asset_provider]
        except KeyError as error:
            raise ValueError(f"no model provider registered for {spec.asset_provider!r}") from error
        resolution = provider.resolve(spec)
        if not resolution.files:
            raise ValueError(f"provider returned no expected files for {role}")
        models.append(
            LockedModel(
                role=role,
                backend=spec.backend,
                asset_provider=spec.asset_provider,
                upstream_identifier=spec.upstream_identifier,
                resolved_revision=resolution.revision,
                package_version=resolution.package_version,
                artifact_selector=spec.artifact_selector,
                license_identifier=spec.license_identifier,
                license_note=spec.license_note,
                license_acknowledgement_id=None,
                preprocessing=spec.preprocessing,
                output=spec.output,
                reconstruction_caveat=spec.reconstruction_caveat,
                files=tuple(
                    LockedFile(
                        remote_path=file.remote_path,
                        path=file.cache_path,
                        source_sha256=file.source_sha256,
                    )
                    for file in resolution.files
                ),
            )
        )
    return ModelLock(
        schema_version=MODEL_LOCK_SCHEMA_VERSION,
        registry_id=registry.registry_id,
        registry_path=registry.path,
        registry_sha256=registry.digest,
        generated_at_utc=(
            generated_at_utc
            if generated_at_utc is not None
            else datetime.now(UTC).replace(microsecond=0).isoformat()
        ),
        models=tuple(models),
    )


def resolve_registry_to_file(
    registry_path: str | Path,
    lock_path: str | Path,
    *,
    providers: Mapping[str, ModelAssetProvider] | None = None,
) -> ModelLock:
    """Resolve once; refuse to replace a lock for changed registry intent."""

    registry = load_model_registry(registry_path)
    destination = Path(lock_path)
    if destination.exists():
        existing = load_model_lock(destination)
        if existing.registry_sha256 != registry.digest:
            raise ValueError(
                "existing lock belongs to different registry content; choose a new path"
            )
        return existing
    lock = resolve_model_lock(registry, providers=providers)
    write_model_lock(lock, destination)
    return lock


def _accepted(model: LockedModel, accepted: set[str]) -> bool:
    return bool(
        {"all", model.role, model.license_identifier}.intersection(accepted)
        or model.license_acknowledgement_id is not None
    )


def _model_files_valid(model: LockedModel, cache_root: Path) -> bool:
    if not model.complete or not model.files:
        return False
    for file in model.files:
        candidate = cache_root / file.path
        if file.sha256 is None or not candidate.is_file() or sha256_file(candidate) != file.sha256:
            return False
    return True


def fetch_locked_models(
    lock_path: str | Path,
    cache_root: str | Path,
    *,
    accepted_licenses: Sequence[str],
    roles: Sequence[str] = (),
    providers: Mapping[str, ModelAssetProvider] | None = None,
) -> dict[str, object]:
    """Fetch selected roles, finalizing and checkpointing lock digests per role."""

    lock = load_model_lock(lock_path)
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    available = default_model_providers() if providers is None else providers
    accepted = set(accepted_licenses)
    selected = set(roles) if roles else {model.role for model in lock.models}
    unknown = selected.difference(model.role for model in lock.models)
    if unknown:
        raise ValueError(f"unknown locked model roles: {sorted(unknown)}")
    statuses: dict[str, str] = {}
    current = lock
    for model in current.models:
        if model.role not in selected:
            statuses[model.role] = "not_selected"
            continue
        if _model_files_valid(model, root):
            statuses[model.role] = "already_verified"
            continue
        if not _accepted(model, accepted):
            raise PermissionError(
                f"license acknowledgement required for {model.role}; pass "
                f"--accept-license {model.role} after reviewing {model.license_identifier}"
            )
        try:
            provider = available[model.asset_provider]
        except KeyError as error:
            raise ValueError(
                f"no model provider registered for {model.asset_provider!r}"
            ) from error
        paths = tuple(Path(path).resolve() for path in provider.fetch(model, root))
        expected = tuple((root / file.path).resolve() for file in model.files)
        if set(paths) != set(expected):
            raise RuntimeError(f"provider file set mismatch for {model.role}")
        finalized_files: list[LockedFile] = []
        for declaration, path in zip(model.files, expected, strict=True):
            if not path.is_file():
                raise RuntimeError(f"provider did not create expected file for {model.role}")
            digest = sha256_file(path)
            if declaration.source_sha256 is not None and digest != declaration.source_sha256:
                raise RuntimeError(
                    f"upstream checksum mismatch for {model.role}/{declaration.remote_path}"
                )
            finalized_files.append(
                replace(declaration, sha256=digest, size_bytes=path.stat().st_size)
            )
        finalized = replace(
            model,
            files=tuple(finalized_files),
            complete=True,
            license_acknowledgement_id=(
                model.license_acknowledgement_id
                or f"accepted:{model.license_identifier}:{model.role}"
            ),
        )
        current = replace(
            current,
            models=tuple(finalized if item.role == model.role else item for item in current.models),
        )
        write_model_lock(current, lock_path)
        statuses[model.role] = "fetched_and_verified"
    return {"ok": True, "lock_sha256": current.digest, "roles": statuses}


def verify_locked_models(
    lock_path: str | Path,
    cache_root: str | Path,
    *,
    registry_path: str | Path | None = None,
) -> dict[str, object]:
    """Verify a complete bundle strictly offline using only local bytes."""

    lock = load_model_lock(lock_path)
    root = Path(cache_root)
    registry_match: bool | None = None
    if registry_path is not None:
        registry_match = load_model_registry(registry_path).digest == lock.registry_sha256
    reports: dict[str, dict[str, object]] = {}
    all_valid = registry_match is not False
    for model in lock.models:
        file_reports: list[dict[str, object]] = []
        model_valid = model.complete and model.license_acknowledgement_id is not None
        for file in model.files:
            candidate = root / file.path
            observed = sha256_file(candidate) if candidate.is_file() else None
            valid = bool(file.sha256 is not None and observed == file.sha256)
            model_valid = model_valid and valid
            file_reports.append(
                {
                    "path": file.path,
                    "exists": candidate.is_file(),
                    "expected_sha256": file.sha256,
                    "observed_sha256": observed,
                    "valid": valid,
                }
            )
        reports[model.role] = {
            "complete": model.complete,
            "valid": model_valid,
            "files": file_reports,
        }
        all_valid = all_valid and model_valid
    return {
        "ok": all_valid,
        "lock_sha256": lock.digest,
        "registry_match": registry_match,
        "models": reports,
    }


def configured_model_cache_root(
    repository_root: Path,
    explicit: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> Path:
    """Resolve model cache root without recording it in lock metadata."""

    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    env = os.environ if environment is None else environment
    configured = env.get("SCREEN2ACTION_CACHE_ROOT")
    return (
        Path(configured).expanduser().resolve()
        if configured
        else (repository_root / "cache").resolve()
    )
