"""Stage-2 cache loader with no detector or OCR model imports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from screen2action import PERCEPTION_CACHE_SCHEMA_VERSION
from screen2action.perception.cache import ContentAddressedPerceptionCache, PerceptionCacheKey
from screen2action.perception.pipeline import PerceptionFrame, perception_frame_from_payload
from screen2action.perception.precompute import validate_perception_cache


@dataclass(frozen=True, slots=True)
class CachedFrameIndexEntry:
    """Portable mapping from canonical screen identity to one cache key."""

    canonical_screen_sha256: str
    screen_ids: tuple[str, ...]
    key: PerceptionCacheKey


class CachedPerceptionLoader:
    """Load finalized frame state without constructing ScreenParser or docTR."""

    def __init__(
        self,
        cache_manifest: Path,
        *,
        data_manifest_digest: str | None = None,
        expected_model_lock_digest: str | None = None,
        expected_config_digest: str | None = None,
        validate: bool = True,
    ) -> None:
        manifest_path = cache_manifest.resolve()
        if validate:
            validation = validate_perception_cache(manifest_path)
            if not validation.ok:
                raise ValueError("perception cache has incomplete or failed precompute shards")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("command_conditioned") is not False:
            raise ValueError("invalid command-independent perception cache manifest")
        model_digest = str(payload.get("model_bundle_lock_digest", ""))
        config_digest = str(payload.get("perception_config_digest", ""))
        if expected_model_lock_digest is not None and model_digest != expected_model_lock_digest:
            raise ValueError("perception cache model-lock digest mismatch")
        if expected_config_digest is not None and config_digest != expected_config_digest:
            raise ValueError("perception cache configuration digest mismatch")
        self.cache = ContentAddressedPerceptionCache(manifest_path.parent)
        self.bundle_digest = str(payload["bundle_digest"])
        self.model_bundle_lock_digest = model_digest
        self.perception_config_digest = config_digest
        self._by_screen_id: dict[str, CachedFrameIndexEntry] = {}
        self._by_hash: dict[str, CachedFrameIndexEntry] = {}

        pattern = "runs/*/s*/s/*.json"
        for shard_path in sorted(self.cache.root.glob(pattern)):
            shard = json.loads(shard_path.read_text(encoding="utf-8"))
            if not isinstance(shard, dict) or shard.get("status") != "complete":
                continue
            if data_manifest_digest is not None and (
                shard.get("data_manifest_digest") != data_manifest_digest
            ):
                continue
            raw_entries = shard.get("entries")
            if not isinstance(raw_entries, list):
                raise ValueError("perception shard entries are malformed")
            for raw in raw_entries:
                if not isinstance(raw, dict):
                    raise ValueError("perception shard entry is malformed")
                key_raw = raw.get("cache_key")
                if not isinstance(key_raw, dict):
                    raise ValueError("perception shard cache key is malformed")
                key = PerceptionCacheKey(
                    screenshot_sha256=str(key_raw["screenshot_sha256"]),
                    model_bundle_lock_digest=str(key_raw["model_bundle_lock_digest"]),
                    perception_config_digest=str(key_raw["perception_config_digest"]),
                    schema_version=int(
                        key_raw.get("schema_version", PERCEPTION_CACHE_SCHEMA_VERSION)
                    ),
                )
                screen_ids_raw = raw.get("screen_ids")
                if not isinstance(screen_ids_raw, list) or not all(
                    isinstance(value, str) for value in screen_ids_raw
                ):
                    raise ValueError("perception shard screen IDs are malformed")
                entry = CachedFrameIndexEntry(
                    canonical_screen_sha256=str(raw["canonical_screen_sha256"]),
                    screen_ids=tuple(cast(list[str], screen_ids_raw)),
                    key=key,
                )
                previous_hash = self._by_hash.setdefault(entry.canonical_screen_sha256, entry)
                if previous_hash.key != entry.key:
                    raise ValueError("one canonical screen maps to incompatible cache keys")
                for screen_id in entry.screen_ids:
                    previous_id = self._by_screen_id.setdefault(screen_id, entry)
                    if previous_id.key != entry.key:
                        raise ValueError("one screen ID maps to incompatible cache keys")
        if not self._by_hash:
            raise ValueError("no completed perception-cache entries match the requested run")

    def __len__(self) -> int:
        return len(self._by_hash)

    @property
    def screen_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_screen_id))

    @property
    def canonical_screen_hashes(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_hash))

    def _load(self, entry: CachedFrameIndexEntry) -> PerceptionFrame:
        payload = self.cache.get(entry.key)
        if payload is None:
            raise FileNotFoundError(f"perception cache miss for {entry.key.digest}")
        raw_outputs = payload.get("raw_outputs")
        if not isinstance(raw_outputs, dict) or raw_outputs.get("command_conditioned") is not False:
            raise ValueError("cached frame contains command-conditioned state")
        return perception_frame_from_payload(payload, cache_key_digest=entry.key.digest)

    def frame_for_screen(self, screen_id: str) -> PerceptionFrame:
        """Load one frame by canonical screen record ID."""

        try:
            entry = self._by_screen_id[screen_id]
        except KeyError as error:
            raise KeyError(f"screen ID is absent from perception cache: {screen_id}") from error
        return self._load(entry)

    def frame_for_hash(self, canonical_screen_sha256: str) -> PerceptionFrame:
        """Load one frame by the canonical image-file SHA256."""

        try:
            entry = self._by_hash[canonical_screen_sha256]
        except KeyError as error:
            raise KeyError(
                f"screen hash is absent from perception cache: {canonical_screen_sha256}"
            ) from error
        return self._load(entry)
