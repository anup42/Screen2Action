"""Immutable data manifest creation and digest enforcement."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from screen2action.data.assets import FileDigest, sha256_file
from screen2action.data.schema import CANONICAL_SCHEMA_VERSION
from screen2action.data.splits import load_split_assignments
from screen2action.data.storage import (
    DATASET_METADATA,
    dataset_table_paths,
    read_table_rows,
    validate_canonical_dataset,
)
from screen2action.perception.taxonomy import require_frozen_icon_taxonomy

MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ManifestVerification:
    ok: bool
    manifest_digest: str
    dataset_root: str
    verified_files: int
    total_bytes: int


def _digest_payload(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _portable_relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    resolved_root = root.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"manifest file path escapes dataset root: {path}")
    return resolved.relative_to(resolved_root).as_posix()


def _file_digest(path: Path, root: Path) -> FileDigest:
    return FileDigest(
        path=_portable_relative(path, root),
        size=path.stat().st_size,
        sha256=sha256_file(path),
    )


def _load_taxonomy_digest(path: Path) -> tuple[str, str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("icon taxonomy must be a mapping")
    return require_frozen_icon_taxonomy(payload), path.name


def _load_dedup(path: Path) -> dict[str, object]:
    raw_payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, dict):
        raise ValueError("dedup audit must be a mapping")
    payload: dict[str, object] = {str(key): value for key, value in raw_payload.items()}
    if not isinstance(payload.get("survivors"), list) or not isinstance(payload.get("digest"), str):
        raise ValueError("invalid dedup audit")
    digest_payload = {
        "policy": payload["policy"],
        "groups": payload["groups"],
        "removals": payload["removals"],
        "survivors": payload["survivors"],
    }
    if _digest_payload(digest_payload) != payload["digest"]:
        raise ValueError("dedup audit digest mismatch")
    return payload


def _target_size(box: tuple[float, float, float, float]) -> str:
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    if area < 0.0025:
        return "tiny"
    if area < 0.01:
        return "small"
    if area < 0.05:
        return "medium"
    return "large"


def _write_immutable(path: Path, serialized: str) -> None:
    if path.is_file():
        if path.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(f"refusing to overwrite a different immutable artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, path)


def build_data_manifest(
    dataset_root: Path,
    output: Path,
    *,
    split_plan: Path,
    dedup_audit: Path,
    taxonomy: Path,
    preprocessing_policy: str = "canonical_rgb_png_v1",
) -> dict[str, object]:
    """Freeze source, schema, split, dedup, taxonomy, count, and file identity."""

    dataset_root = dataset_root.resolve()
    validation = validate_canonical_dataset(dataset_root)
    assignments = load_split_assignments(split_plan)
    dedup = _load_dedup(dedup_audit)
    taxonomy_digest, taxonomy_name = _load_taxonomy_digest(taxonomy)
    survivor_values = dedup["survivors"]
    if not isinstance(survivor_values, list):
        raise ValueError("dedup survivors must be a list")
    survivors = {str(value) for value in survivor_values}
    screens = {str(row["screen_id"]): row for row in read_table_rows(dataset_root, "screens")}
    if set(assignments) != set(screens):
        missing = sorted(set(screens) - set(assignments))
        extra = sorted(set(assignments) - set(screens))
        raise ValueError(f"split membership mismatch; missing={missing[:3]}, extra={extra[:3]}")
    if not survivors <= set(screens):
        raise ValueError("dedup survivors include unknown screens")
    active_screens = {screen_id: screens[screen_id] for screen_id in sorted(survivors)}
    if not active_screens:
        raise ValueError("data manifest cannot contain zero surviving screens")

    app_splits: defaultdict[str, set[str]] = defaultdict(set)
    hash_splits: defaultdict[str, set[str]] = defaultdict(set)
    source_counts: Counter[str] = Counter()
    platform_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    app_counts: Counter[str] = Counter()
    source_revisions: defaultdict[tuple[str, str, str], int] = defaultdict(int)
    screenspot_non_test = 0
    for screen_id, row in active_screens.items():
        split = assignments[screen_id]
        app = str(row["app_id_canonical"])
        digest = str(row["screen_sha256"])
        source = str(row["source_dataset"])
        app_splits[app].add(split)
        hash_splits[digest].add(split)
        source_counts[source] += 1
        platform_counts[str(row["platform"])] += 1
        split_counts[split] += 1
        app_counts[app] += 1
        source_revisions[
            (
                source,
                str(row["source_revision"]),
                str(row["source_license_ack_id"]),
            )
        ] += 1
        screenspot_non_test += int(source == "screenspot" and split != "test")
    app_conflicts = {key: sorted(value) for key, value in app_splits.items() if len(value) > 1}
    exact_conflicts = {key: sorted(value) for key, value in hash_splits.items() if len(value) > 1}
    if app_conflicts:
        raise ValueError(f"app-disjoint leakage in manifest: {app_conflicts}")
    if exact_conflicts:
        raise ValueError(f"exact duplicate leakage in manifest: {exact_conflicts}")
    if screenspot_non_test:
        raise ValueError("ScreenSpot leakage guard failed")

    action_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    target_size_counts: Counter[str] = Counter()
    command_count = 0
    for command in read_table_rows(dataset_root, "commands"):
        if str(command["screen_id"]) not in active_screens:
            continue
        command_count += 1
        action_counts[str(command["action_type"])] += 1
        relation_counts[str(command["relation_type"])] += 1
        box = (
            float(str(command["target_x1"])),
            float(str(command["target_y1"])),
            float(str(command["target_x2"])),
            float(str(command["target_y2"])),
        )
        target_size_counts[_target_size(box)] += 1
    element_count = sum(
        str(element["screen_id"]) in active_screens
        for element in read_table_rows(dataset_root, "elements")
    )
    reference_count = 0
    for reference in read_table_rows(dataset_root, "references"):
        if str(reference["screen_id"]) in active_screens:
            reference_count += 1
            relation_counts[str(reference["relation"])] += 1

    membership_path = output.with_suffix(output.suffix + ".splits.jsonl")
    membership_lines = "".join(
        json.dumps(
            {
                "screen_id": screen_id,
                "split": assignments[screen_id],
                "app_id_canonical": str(active_screens[screen_id]["app_id_canonical"]),
                "screen_sha256": str(active_screens[screen_id]["screen_sha256"]),
            },
            sort_keys=True,
        )
        + "\n"
        for screen_id in active_screens
    )
    _write_immutable(membership_path, membership_lines)

    metadata_paths = [dataset_root / DATASET_METADATA]
    for table in ("screens", "elements", "commands", "references"):
        metadata_paths.extend(dataset_table_paths(dataset_root, table))
    image_paths = sorted({dataset_root / str(row["image_path"]) for row in active_screens.values()})
    files = [_file_digest(path, dataset_root) for path in (*metadata_paths, *image_paths)]
    split_membership_digest = sha256_file(membership_path)
    counts = {
        "screens": len(active_screens),
        "elements": element_count,
        "commands": command_count,
        "references": reference_count,
        "by_source": dict(sorted(source_counts.items())),
        "by_app": dict(sorted(app_counts.items())),
        "by_platform": dict(sorted(platform_counts.items())),
        "by_action": dict(sorted(action_counts.items())),
        "by_relation": dict(sorted(relation_counts.items())),
        "by_target_size": dict(sorted(target_size_counts.items())),
        "by_split": dict(sorted(split_counts.items())),
    }
    dataset_root_relative = Path(os.path.relpath(dataset_root, output.parent.resolve())).as_posix()
    dedup_leakage = dedup.get("leakage_checks")
    if not isinstance(dedup_leakage, dict):
        raise ValueError("dedup leakage checks are missing")
    payload: dict[str, object] = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
        "dataset_version": validation["dataset_version"],
        "dataset_root_relative": dataset_root_relative,
        "source_revisions": [
            {
                "source": source,
                "revision": revision,
                "license_acknowledgement_id": acknowledgement,
                "screens": count,
            }
            for (source, revision, acknowledgement), count in sorted(source_revisions.items())
        ],
        "taxonomy": {"name": taxonomy_name, "digest": taxonomy_digest, "class_count": 87},
        "split": {
            "policy_version": "app_disjoint_hash_v1",
            "membership_path": membership_path.name,
            "membership_digest": split_membership_digest,
        },
        "dedup": {
            "policy": dedup["policy"],
            "digest": dedup["digest"],
            "removal_count": dedup["removal_count"],
        },
        "preprocessing_policy": preprocessing_policy,
        "counts": counts,
        "leakage_checks": {
            "app_cross_split": len(app_conflicts),
            "exact_hash_cross_split": len(exact_conflicts),
            "screenspot_outside_test": screenspot_non_test,
            "screenspot_train_near_duplicate_survivors": dedup_leakage.get(
                "screenspot_train_near_duplicate_survivors"
            ),
        },
        "files": [asdict(file) for file in sorted(files, key=lambda item: item.path)],
    }
    payload["manifest_digest"] = _digest_payload(payload)
    _write_immutable(output, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def verify_data_manifest(path: Path) -> ManifestVerification:
    """Fail if any frozen manifest, split membership, metadata, or image digest drifts."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("manifest_digest", None)
    computed = _digest_payload(payload)
    if recorded != computed:
        raise ValueError("data manifest digest mismatch")
    dataset_root = (path.parent / str(payload["dataset_root_relative"])).resolve()
    membership_path = path.parent / str(payload["split"]["membership_path"])
    if sha256_file(membership_path) != payload["split"]["membership_digest"]:
        raise ValueError("split membership digest mismatch")
    verified = 0
    total_bytes = 0
    for raw in payload["files"]:
        relative = Path(str(raw["path"]))
        target = (dataset_root / relative).resolve()
        if not target.is_relative_to(dataset_root):
            raise ValueError("manifest file path escapes dataset root")
        if not target.is_file():
            raise FileNotFoundError(f"manifest file is missing: {relative.as_posix()}")
        size = target.stat().st_size
        if size != int(raw["size"]) or sha256_file(target) != raw["sha256"]:
            raise ValueError(f"manifest file digest mismatch: {relative.as_posix()}")
        verified += 1
        total_bytes += size
    return ManifestVerification(True, str(recorded), dataset_root.as_posix(), verified, total_bytes)
