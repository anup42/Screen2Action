"""Typed public-source registry with immutable revision requirements."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """One licensed upstream source and its immutable default revision."""

    name: str
    provider: str
    upstream_id: str
    source_url: str
    revision: str
    license_identifier: str
    license_note: str
    acceptance_required: bool
    train_eligible: bool
    evaluation_only: bool = False
    hard_leakage_guard: bool = False
    optional: bool = False
    data_url: str | None = None
    project_url: str | None = None
    subsets: tuple[str, ...] = ()
    multipart_repair: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.provider or not self.upstream_id:
            raise ValueError("source name, provider, and upstream_id are required")
        if not self.revision or self.revision == "unresolved":
            raise ValueError(f"source {self.name!r} must have an immutable default revision")
        if self.evaluation_only and self.train_eligible:
            raise ValueError(f"evaluation-only source {self.name!r} cannot be train eligible")
        if self.hard_leakage_guard and not self.evaluation_only:
            raise ValueError("hard_leakage_guard is only valid for evaluation-only sources")


@dataclass(frozen=True, slots=True)
class SourceRegistry:
    """Validated source registry loaded from YAML."""

    schema_version: int
    registry_id: str
    sources: Mapping[str, SourceSpec]

    def source(self, name: str) -> SourceSpec:
        try:
            return self.sources[name]
        except KeyError as error:
            names = ", ".join(sorted(self.sources))
            raise ValueError(
                f"unknown data source {name!r}; registered sources: {names}"
            ) from error


def _required_string(payload: Mapping[str, object], key: str, source: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"source {source!r} requires non-empty {key}")
    return value.strip()


def load_source_registry(path: str | Path) -> SourceRegistry:
    """Load and validate a public source registry."""

    registry_path = Path(path)
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("source registry must be a mapping")
    schema_version = payload.get("schema_version")
    registry_id = payload.get("registry_id")
    raw_sources = payload.get("sources")
    if not isinstance(schema_version, int) or schema_version <= 0:
        raise ValueError("source registry schema_version must be a positive integer")
    if not isinstance(registry_id, str) or not registry_id:
        raise ValueError("source registry registry_id is required")
    if not isinstance(raw_sources, dict) or not raw_sources:
        raise ValueError("source registry sources must be a non-empty mapping")

    sources: dict[str, SourceSpec] = {}
    for name, raw in raw_sources.items():
        if not isinstance(name, str) or not isinstance(raw, dict):
            raise ValueError("source registry entries must map names to mappings")
        subsets = raw.get("subsets", [])
        if not isinstance(subsets, list) or not all(isinstance(item, str) for item in subsets):
            raise ValueError(f"source {name!r} subsets must be a list of strings")
        sources[name] = SourceSpec(
            name=name,
            provider=_required_string(raw, "provider", name),
            upstream_id=_required_string(raw, "upstream_id", name),
            source_url=_required_string(raw, "source_url", name),
            revision=_required_string(raw, "revision", name),
            license_identifier=_required_string(raw, "license_identifier", name),
            license_note=_required_string(raw, "license_note", name),
            acceptance_required=bool(raw.get("acceptance_required", True)),
            train_eligible=bool(raw.get("train_eligible", False)),
            evaluation_only=bool(raw.get("evaluation_only", False)),
            hard_leakage_guard=bool(raw.get("hard_leakage_guard", False)),
            optional=bool(raw.get("optional", False)),
            data_url=str(raw["data_url"]) if raw.get("data_url") else None,
            project_url=str(raw["project_url"]) if raw.get("project_url") else None,
            subsets=tuple(subsets),
            multipart_repair=(
                str(raw["multipart_repair"]) if raw.get("multipart_repair") else None
            ),
            note=str(raw.get("note", "")),
        )
    return SourceRegistry(schema_version, registry_id, sources)
