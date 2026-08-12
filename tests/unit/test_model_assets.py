"""Offline registry, lock, fetch, and checksum tests."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from screen2action.model_assets import (
    LockedModel,
    ModelAssetProvider,
    ModelSpec,
    ProviderResolution,
    RemoteFile,
    fetch_locked_models,
    load_model_lock,
    load_model_registry,
    resolve_model_lock,
    verify_locked_models,
    write_model_lock,
)


class FakeProvider(ModelAssetProvider):
    """Deterministic fixture provider with no network or model import."""

    def resolve(self, spec: ModelSpec) -> ProviderResolution:
        return ProviderResolution(
            revision=f"immutable-{spec.role}",
            package_version="1.2.3",
            files=(
                RemoteFile(
                    remote_path="weights.bin",
                    cache_path=f"models/{spec.role}/weights.bin",
                ),
            ),
        )

    def fetch(self, model: LockedModel, cache_root: Path) -> Sequence[Path]:
        destination = cache_root / model.files[0].path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(f"fixture:{model.role}".encode())
        return (destination,)


def _lock(tmp_path: Path) -> tuple[Path, Path, Path]:
    registry_path = Path("configs/models/registry.yaml")
    registry = load_model_registry(registry_path)
    fake = FakeProvider()
    providers = {"huggingface": fake, "torchvision": fake, "doctr": fake}
    lock = resolve_model_lock(
        registry,
        providers=providers,
        generated_at_utc="2026-08-12T00:00:00+00:00",
    )
    lock_path = tmp_path / "lock.json"
    write_model_lock(lock, lock_path)
    return registry_path, lock_path, tmp_path / "cache"


def test_registry_has_required_provenance_contract() -> None:
    registry = load_model_registry("configs/models/registry.yaml")

    assert len(registry.roles) == 5
    for role, spec in registry.roles.items():
        assert role == spec.role
        assert spec.license_identifier
        assert spec.preprocessing
        assert spec.output
        assert spec.reconstruction_caveat


def test_mock_fetch_finalizes_lock_and_offline_verify_detects_corruption(
    tmp_path: Path,
) -> None:
    registry_path, lock_path, cache_root = _lock(tmp_path)
    fake = FakeProvider()
    providers = {"huggingface": fake, "torchvision": fake, "doctr": fake}

    result = fetch_locked_models(
        lock_path,
        cache_root,
        accepted_licenses=("all",),
        providers=providers,
    )

    assert result["ok"]
    finalized = load_model_lock(lock_path)
    assert all(model.complete for model in finalized.models)
    assert all(
        file.sha256 and file.size_bytes for model in finalized.models for file in model.files
    )
    verified = verify_locked_models(lock_path, cache_root, registry_path=registry_path)
    assert verified["ok"]

    first = finalized.models[0].files[0]
    (cache_root / first.path).write_bytes(b"corrupt")
    corrupted = verify_locked_models(lock_path, cache_root, registry_path=registry_path)
    assert not corrupted["ok"]
    model_reports = corrupted["models"]
    assert isinstance(model_reports, dict)
    assert not model_reports[finalized.models[0].role]["valid"]


def test_fetch_is_license_gated_and_role_selective(tmp_path: Path) -> None:
    _, lock_path, cache_root = _lock(tmp_path)
    fake = FakeProvider()
    providers = {"huggingface": fake, "torchvision": fake, "doctr": fake}
    with pytest.raises(PermissionError, match="license acknowledgement"):
        fetch_locked_models(
            lock_path,
            cache_root,
            accepted_licenses=(),
            roles=("ui_detector_screenparser",),
            providers=providers,
        )

    result = fetch_locked_models(
        lock_path,
        cache_root,
        accepted_licenses=("ui_detector_screenparser",),
        roles=("ui_detector_screenparser",),
        providers=providers,
    )
    role_statuses = result["roles"]
    assert isinstance(role_statuses, dict)
    assert role_statuses["ui_detector_screenparser"] == "fetched_and_verified"
