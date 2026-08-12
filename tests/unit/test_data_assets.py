from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from screen2action.data import assets
from screen2action.data.assets import (
    download_registered_source,
    merge_multipart_zip,
    register_local_source,
    safe_extract_zip,
    verify_raw_source,
)
from screen2action.data.layout import DataLayout
from screen2action.data.registry import load_source_registry


def test_local_registration_requires_ack_and_is_immutable(tmp_path: Path) -> None:
    source = tmp_path / "predownloaded"
    source.mkdir()
    (source / "sample.jsonl").write_text('{"id":"one"}\n', encoding="utf-8")
    layout = DataLayout(tmp_path / "lake")
    registry = load_source_registry(Path("configs/data/sources.yaml"))
    revision = registry.source("wave_ui").revision

    with pytest.raises(PermissionError, match="--accept-license"):
        register_local_source(
            registry,
            layout,
            "wave_ui",
            source,
            revision,
            accept_license=False,
        )

    manifest = register_local_source(
        registry,
        layout,
        "wave_ui",
        source,
        revision,
        accept_license=True,
    )
    destination = layout.raw_revision("wave_ui", revision)
    assert verify_raw_source(destination) == manifest

    (destination / "sample.jsonl").write_text('{"id":"changed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="immutable registered inventory"):
        verify_raw_source(destination)


def test_download_is_pinned_resumable_and_provider_mocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = load_source_registry(Path("configs/data/sources.yaml"))
    source = registry.source("wave_ui")
    calls: list[tuple[str, int | None]] = []

    class FakeProvider:
        def resolve_revision(self, source_spec: object, requested: str | None) -> str:
            assert source_spec == source
            return requested or source.revision

        def fetch(
            self,
            source_spec: object,
            revision: str,
            destination: Path,
            *,
            sample_size: int | None,
        ) -> None:
            assert source_spec == source
            calls.append((revision, sample_size))
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "sample.jsonl").write_text('{"id":"fixture"}\n', encoding="utf-8")

    monkeypatch.setattr(assets, "provider_for", lambda _: FakeProvider())
    layout = DataLayout(tmp_path / "lake")
    first = download_registered_source(
        registry,
        layout,
        "wave_ui",
        requested_revision=None,
        sample_size=1,
        accept_license=True,
    )
    second = download_registered_source(
        registry,
        layout,
        "wave_ui",
        requested_revision=None,
        sample_size=1,
        accept_license=False,
    )
    assert first == second
    assert calls == [(source.revision, 1)]


def test_multipart_zip_merge_requires_valid_crc(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("fixture.txt", "screen2action")
    payload = archive.read_bytes()
    midpoint = len(payload) // 2
    first = tmp_path / "archive.part01"
    second = tmp_path / "archive.part02"
    first.write_bytes(payload[:midpoint])
    second.write_bytes(payload[midpoint:])
    merged = merge_multipart_zip((second, first), tmp_path / "merged.zip")
    with zipfile.ZipFile(merged) as output:
        assert output.read("fixture.txt") == b"screen2action"


def test_safe_zip_extraction_rejects_parent_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../escaped.txt", "must not escape")

    with pytest.raises(ValueError, match="escapes extraction root"):
        safe_extract_zip(archive, tmp_path / "extracted")

    assert not (tmp_path / "escaped.txt").exists()
