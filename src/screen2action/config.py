"""Schema-versioned YAML composition and deterministic configuration snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from screen2action import CONFIG_SCHEMA_VERSION

type ConfigValue = None | bool | int | float | str | list[ConfigValue] | dict[str, ConfigValue]
type ConfigMapping = dict[str, ConfigValue]

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_POSITIVE_INTEGER_KEYS = {
    "d",
    "epochs",
    "global_batch_size",
    "command_layers",
    "graph_layers",
    "command_heads",
    "command_ffn",
    "command_max_tokens",
    "node_text_max_tokens",
    "retrieval_top_k",
    "ssb_budget",
    "crop_size",
    "crop_tokens",
}
_FRACTION_KEYS = {
    "warmup_fraction",
    "forced_positive_probability_first_two_epochs",
    "crop_expansion_fraction",
}


@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    """Validated configuration plus reproducibility metadata."""

    values: Mapping[str, ConfigValue]
    source_paths: tuple[str, ...]
    overrides: tuple[str, ...]
    schema_version: int = CONFIG_SCHEMA_VERSION

    @property
    def sha256(self) -> str:
        """Return a stable digest of semantic configuration values."""

        payload = json.dumps(self.values, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def snapshot(self) -> ConfigMapping:
        """Return the portable serialized snapshot document."""

        return {
            "config_schema_version": self.schema_version,
            "config_sha256": self.sha256,
            "source_paths": list(self.source_paths),
            "overrides": list(self.overrides),
            "resolved": dict(self.values),
        }


def _normalize_yaml(value: Any, *, context: str) -> ConfigValue:
    if value is None or isinstance(value, bool | int | float | str):
        return cast(ConfigValue, value)
    if isinstance(value, list):
        return [_normalize_yaml(item, context=context) for item in value]
    if isinstance(value, dict):
        normalized: ConfigMapping = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{context}: configuration keys must be strings")
            normalized[key] = _normalize_yaml(item, context=f"{context}.{key}")
        return normalized
    raise ValueError(f"{context}: unsupported YAML value {type(value).__name__}")


def _mapping(value: ConfigValue, *, context: str) -> ConfigMapping:
    if not isinstance(value, dict):
        raise ValueError(f"{context}: top-level YAML document must be a mapping")
    return value


def _expand_string(value: str, environment: Mapping[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2)
        if name in environment:
            return environment[name]
        if default is not None:
            return default
        raise ValueError(f"configuration references unset environment variable {name}")

    return _ENV_PATTERN.sub(replace, value)


def _expand_environment(value: ConfigValue, environment: Mapping[str, str]) -> ConfigValue:
    if isinstance(value, str):
        return _expand_string(value, environment)
    if isinstance(value, list):
        return [_expand_environment(item, environment) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item, environment) for key, item in value.items()}
    return value


def _deep_merge(base: ConfigMapping, update: ConfigMapping) -> ConfigMapping:
    merged = dict(base)
    for key, value in update.items():
        previous = merged.get(key)
        if isinstance(previous, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(previous, value)
        else:
            merged[key] = value
    return merged


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def _load_composed(
    path: Path,
    environment: Mapping[str, str],
    stack: tuple[Path, ...],
) -> tuple[ConfigMapping, list[str]]:
    resolved_path = path.resolve()
    if resolved_path in stack:
        chain = " -> ".join(item.name for item in (*stack, resolved_path))
        raise ValueError(f"configuration composition cycle: {chain}")
    if not resolved_path.is_file():
        raise FileNotFoundError(f"configuration file does not exist: {path}")
    loaded = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    raw = _mapping(_normalize_yaml(loaded, context=_display_path(path)), context=str(path))
    raw = _mapping(_expand_environment(raw, environment), context=str(path))
    extends = raw.pop("extends", None)
    parents: list[str]
    if extends is None:
        parents = []
    elif isinstance(extends, str):
        parents = [extends]
    elif isinstance(extends, list) and all(isinstance(item, str) for item in extends):
        parents = cast(list[str], extends)
    else:
        raise ValueError(f"{_display_path(path)}: extends must be a path or list of paths")
    result: ConfigMapping = {}
    sources: list[str] = []
    for parent in parents:
        parent_path = (resolved_path.parent / parent).resolve()
        parent_values, parent_sources = _load_composed(
            parent_path,
            environment,
            (*stack, resolved_path),
        )
        result = _deep_merge(result, parent_values)
        sources.extend(parent_sources)
    result = _deep_merge(result, raw)
    sources.append(_display_path(path))
    return result, sources


def _parse_override(raw: str) -> tuple[list[str], ConfigValue]:
    if "=" not in raw:
        raise ValueError(f"override must use dotted.key=value syntax: {raw!r}")
    key, serialized = raw.split("=", 1)
    parts = key.split(".")
    if not all(part and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", part) for part in parts):
        raise ValueError(f"invalid override key: {key!r}")
    parsed = yaml.safe_load(serialized)
    return parts, _normalize_yaml(parsed, context=f"override {key}")


def _apply_override(values: ConfigMapping, raw: str) -> None:
    parts, value = _parse_override(raw)
    cursor = values
    for part in parts[:-1]:
        existing = cursor.get(part)
        if existing is None:
            nested: ConfigMapping = {}
            cursor[part] = nested
            cursor = nested
        elif isinstance(existing, dict):
            cursor = existing
        else:
            raise ValueError(f"override {raw!r} crosses non-mapping key {part!r}")
    cursor[parts[-1]] = value


def _validate_node(value: ConfigValue, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_node(item, (*path, key))
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_node(item, (*path, str(index)))
        return
    key = path[-1] if path else ""
    dotted = ".".join(path)
    if key in _POSITIVE_INTEGER_KEYS and (not isinstance(value, int) or value <= 0):
        raise ValueError(f"{dotted} must be a positive integer")
    if key in _FRACTION_KEYS and (
        not isinstance(value, int | float) or isinstance(value, bool) or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{dotted} must be a number in [0, 1]")


def validate_config(values: Mapping[str, ConfigValue]) -> None:
    """Validate the stable generic schema and common semantic constraints."""

    version = values.get("schema_version")
    if version != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}, got {version!r}")
    if len(values) <= 1:
        raise ValueError("configuration must contain values beyond schema_version")
    device = values.get("device")
    if device is not None and device not in {"cpu", "cuda", "cuda_optional", "auto"}:
        raise ValueError("device must be cpu, cuda, cuda_optional, or auto")
    _validate_node(dict(values))


def load_config(
    path: str | Path,
    *,
    overrides: Sequence[str] = (),
    environment: Mapping[str, str] | None = None,
) -> ResolvedConfig:
    """Load YAML, recursively compose parents, expand environment, and validate."""

    env = os.environ if environment is None else environment
    values, sources = _load_composed(Path(path), env, ())
    for override in overrides:
        _apply_override(values, override)
    validate_config(values)
    return ResolvedConfig(
        values=values,
        source_paths=tuple(dict.fromkeys(sources)),
        overrides=tuple(overrides),
    )


def write_resolved_config(config: ResolvedConfig, run_directory: str | Path) -> Path:
    """Atomically emit `resolved-config.yaml` in a run directory."""

    directory = Path(run_directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "resolved-config.yaml"
    temporary = destination.with_suffix(".yaml.tmp")
    temporary.write_text(
        yaml.safe_dump(config.snapshot(), sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination
