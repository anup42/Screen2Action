"""Deterministic inspect/build/freeze workflow for the public icon taxonomy."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

ICON_TAXONOMY_SCHEMA_VERSION = 1
ICON_TAXONOMY_POLICY_VERSION = "public_icon_taxonomy_v1"


@dataclass(frozen=True, slots=True)
class LabelObservation:
    """One licensed source label count with optional reviewed canonical label."""

    source: str
    label: str
    count: int
    canonical: str | None = None

    def __post_init__(self) -> None:
        if not self.source or not self.label or self.count <= 0:
            raise ValueError("label observation requires source, label, and positive count")


def normalize_icon_label(label: str) -> str:
    """Normalize spacing/punctuation without asserting semantic equivalence."""

    value = unicodedata.normalize("NFKC", label).casefold().strip()
    value = re.sub(r"[^\w]+", "_", value, flags=re.UNICODE)
    return value.strip("_")


def _records_from_object(value: object, *, default_source: str) -> list[LabelObservation]:
    if isinstance(value, dict) and "labels" in value:
        source = str(value.get("source", default_source))
        return _records_from_object(value["labels"], default_source=source)
    if isinstance(value, dict):
        observations = []
        for label, count in value.items():
            if not isinstance(label, str) or not isinstance(count, int):
                raise ValueError("label-count mappings require string labels and integer counts")
            observations.append(LabelObservation(default_source, label, count))
        return observations
    if not isinstance(value, list):
        raise ValueError("taxonomy input must be a list or label-count mapping")
    observations = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"taxonomy row {index} must be a mapping")
        source = item.get("source", default_source)
        label = item.get("label")
        count = item.get("count", 1)
        canonical = item.get("canonical")
        if not isinstance(source, str) or not isinstance(label, str) or not isinstance(count, int):
            raise ValueError(f"taxonomy row {index} has invalid source, label, or count")
        if canonical is not None and not isinstance(canonical, str):
            raise ValueError(f"taxonomy row {index} canonical must be a string")
        observations.append(LabelObservation(source, label, count, canonical))
    return observations


def load_label_observations(paths: Sequence[str | Path]) -> tuple[LabelObservation, ...]:
    """Load JSON/YAML/CSV source-count fixtures without any dataset dependency."""

    observations: list[LabelObservation] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"taxonomy input does not exist: {path}")
        if path.suffix.casefold() == ".csv":
            with path.open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            value: object = [
                {
                    "source": row.get("source") or path.stem,
                    "label": row.get("label"),
                    "count": int(row.get("count") or "1"),
                    "canonical": row.get("canonical") or None,
                }
                for row in rows
            ]
        elif path.suffix.casefold() in {".yaml", ".yml"}:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
        observations.extend(_records_from_object(value, default_source=path.stem))
    if not observations:
        raise ValueError("taxonomy inspection requires at least one label observation")
    return tuple(observations)


def inspect_icon_taxonomy(observations: Sequence[LabelObservation]) -> dict[str, object]:
    """Report counts, normalization candidates, and unresolved collisions."""

    counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    normalized: dict[str, set[str]] = defaultdict(set)
    explicit: dict[str, set[str]] = defaultdict(set)
    for observation in observations:
        counts[observation.label] += observation.count
        source_counts[observation.source] += observation.count
        candidate = normalize_icon_label(observation.canonical or observation.label)
        normalized[candidate].add(observation.label)
        if observation.canonical is not None:
            explicit[candidate].add(observation.label)
    collisions = [
        {
            "candidate": candidate,
            "labels": sorted(labels),
            "explicitly_reviewed": labels == explicit[candidate],
        }
        for candidate, labels in sorted(normalized.items())
        if len(labels) > 1
    ]
    return {
        "schema_version": ICON_TAXONOMY_SCHEMA_VERSION,
        "policy_version": ICON_TAXONOMY_POLICY_VERSION,
        "source_counts": dict(sorted(source_counts.items())),
        "label_counts": dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))),
        "normalization_candidates": {
            candidate: sorted(labels) for candidate, labels in sorted(normalized.items())
        },
        "collision_report": collisions,
        "unresolved_collision_count": sum(
            1 for collision in collisions if not collision["explicitly_reviewed"]
        ),
    }


def build_icon_taxonomy(
    observations: Sequence[LabelObservation],
    *,
    expected_count: int = 87,
) -> dict[str, object]:
    """Build all candidates; never silently truncate to the requested count."""

    if expected_count <= 1:
        raise ValueError("expected_count must be greater than one")
    inspection = inspect_icon_taxonomy(observations)
    grouped: dict[str, list[LabelObservation]] = defaultdict(list)
    for observation in observations:
        canonical = normalize_icon_label(observation.canonical or observation.label)
        if not canonical:
            raise ValueError(f"label normalizes to empty: {observation.label!r}")
        grouped[canonical].append(observation)
    classes = []
    for canonical, rows in grouped.items():
        aliases = sorted({row.label for row in rows})
        explicitly_reviewed = len(aliases) == 1 or all(row.canonical is not None for row in rows)
        classes.append(
            {
                "name": canonical,
                "source_aliases": aliases,
                "source_counts": dict(
                    sorted(
                        Counter(
                            {
                                source: sum(row.count for row in rows if row.source == source)
                                for source in {row.source for row in rows}
                            }
                        ).items()
                    )
                ),
                "total_count": sum(row.count for row in rows),
                "alias_reviewed": explicitly_reviewed,
                "justification": "explicit_source_canonical"
                if len(aliases) > 1
                else "unique_normalized_label",
            }
        )
    classes.sort(key=lambda item: (-cast(int, item["total_count"]), cast(str, item["name"])))
    for index, item in enumerate(classes):
        item["proposed_id"] = index
    unresolved = [item["name"] for item in classes if not item["alias_reviewed"]]
    ready = len(classes) == expected_count and not unresolved
    return {
        "schema_version": ICON_TAXONOMY_SCHEMA_VERSION,
        "policy_version": ICON_TAXONOMY_POLICY_VERSION,
        "status": "ready_for_freeze" if ready else "review_required",
        "expected_count": expected_count,
        "observed_count": len(classes),
        "unknown_policy": "unknown labels are masked; no fabricated negative class",
        "classes": classes,
        "unresolved_alias_collisions": unresolved,
        "collision_report": inspection["collision_report"],
    }


def _semantic_digest(value: Mapping[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def freeze_icon_taxonomy(
    proposal: Mapping[str, object],
    *,
    expected_count: int = 87,
    accept_reconstruction: bool,
    reviewer: str = "cli-operator",
) -> dict[str, object]:
    """Freeze exactly the reviewed count with an immutable semantic digest."""

    if not accept_reconstruction:
        raise PermissionError("pass --accept-reconstruction after human review")
    if proposal.get("status") != "ready_for_freeze":
        raise ValueError("taxonomy proposal is not ready; resolve collisions and count mismatch")
    classes = proposal.get("classes")
    if not isinstance(classes, list) or len(classes) != expected_count:
        raise ValueError(f"taxonomy must contain exactly {expected_count} classes")
    names = [item.get("name") for item in classes if isinstance(item, dict)]
    if len(names) != expected_count or len(set(names)) != expected_count:
        raise ValueError("taxonomy class names must be unique and complete")
    frozen_classes = [
        {
            **cast(dict[str, object], item),
            "id": index,
        }
        for index, item in enumerate(classes)
        if isinstance(item, dict)
    ]
    for item in frozen_classes:
        item.pop("proposed_id", None)
    document: dict[str, object] = {
        "schema_version": ICON_TAXONOMY_SCHEMA_VERSION,
        "policy_version": ICON_TAXONOMY_POLICY_VERSION,
        "status": "frozen_public_reconstruction",
        "count": expected_count,
        "classes": frozen_classes,
        "unknown_policy": proposal.get("unknown_policy"),
        "human_review": {
            "accepted_reconstruction": True,
            "reviewer": reviewer,
            "record_version": 1,
        },
    }
    document["digest"] = _semantic_digest(document)
    return document


def write_taxonomy(document: Mapping[str, object], path: str | Path) -> Path:
    """Atomically emit a human-readable deterministic YAML taxonomy artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(dict(document), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def require_frozen_icon_taxonomy(document: Mapping[str, object], *, count: int = 87) -> str:
    """Fail training fast unless a frozen reconstruction manifest is internally valid."""

    if document.get("status") != "frozen_public_reconstruction" or document.get("count") != count:
        raise ValueError("training requires a frozen public icon taxonomy with exactly 87 classes")
    expected = document.get("digest")
    if not isinstance(expected, str):
        raise ValueError("frozen taxonomy digest is missing")
    payload = dict(document)
    payload.pop("digest", None)
    if _semantic_digest(payload) != expected:
        raise ValueError("frozen taxonomy digest mismatch")
    return expected
