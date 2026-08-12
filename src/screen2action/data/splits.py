"""Application-disjoint split validation."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from screen2action.data.schema import ScreenRecord


def validate_app_disjoint(screens: Iterable[ScreenRecord]) -> None:
    """Raise when an app identity appears in more than one split."""

    apps: defaultdict[str, set[str]] = defaultdict(set)
    for screen in screens:
        apps[screen.app_id].add(screen.split)
    conflicts = {app: sorted(splits) for app, splits in apps.items() if len(splits) > 1}
    if conflicts:
        raise ValueError(f"application-disjoint split violation: {conflicts}")


@dataclass(frozen=True, slots=True)
class SplitPlan:
    """Frozen app-disjoint assignment with alias and exact-hash audits."""

    schema_version: int
    policy_version: str
    seed: int
    assignments: Mapping[str, str]
    app_assignments: Mapping[str, str]
    aliases: Mapping[str, tuple[str, ...]]
    collisions: Mapping[str, tuple[str, ...]]
    exact_hash_groups: Mapping[str, tuple[str, ...]]


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first: str, second: str) -> None:
        left, right = self.find(first), self.find(second)
        if left == right:
            return
        survivor, merged = sorted((left, right))
        self.parent[merged] = survivor


def _hash_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def canonicalize_app_id(value: str, aliases: Mapping[str, str] | None = None) -> str:
    """Canonicalize package/domain identity with an optional reviewed alias map."""

    normalized = value.casefold().strip().replace(" ", "_").removeprefix("www.")
    alias_map = {key.casefold().strip(): target for key, target in (aliases or {}).items()}
    return alias_map.get(normalized, normalized)


def create_app_disjoint_plan(
    rows: Iterable[Mapping[str, object]],
    *,
    seed: int = 7,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    alias_map: Mapping[str, str] | None = None,
) -> SplitPlan:
    """Create splits by canonical app, joining exact-image duplicate app groups."""

    if len(ratios) != 3 or any(value < 0.0 for value in ratios):
        raise ValueError("split ratios must contain three non-negative values")
    total = sum(ratios)
    if total <= 0.0:
        raise ValueError("at least one split ratio must be positive")
    normalized_ratios = tuple(value / total for value in ratios)
    records = list(rows)
    if not records:
        raise ValueError("cannot split an empty dataset")
    screen_ids = [str(row["screen_id"]) for row in records]
    if len(screen_ids) != len(set(screen_ids)):
        raise ValueError("split input contains duplicate screen IDs")
    apps = {canonicalize_app_id(str(row["app_id_canonical"]), alias_map) for row in records}
    union = _UnionFind(apps)
    exact: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
    raw_aliases: defaultdict[str, set[str]] = defaultdict(set)
    raw_to_canonical: defaultdict[str, set[str]] = defaultdict(set)
    for row in records:
        canonical = canonicalize_app_id(str(row["app_id_canonical"]), alias_map)
        raw = str(row.get("app_id_raw") or canonical)
        raw_aliases[canonical].add(raw)
        raw_to_canonical[raw.casefold().strip()].add(canonical)
        digest = str(row.get("screen_sha256") or "")
        if digest:
            exact[digest].append(row)
    for duplicate_rows in exact.values():
        group_apps = {
            canonicalize_app_id(str(row["app_id_canonical"]), alias_map) for row in duplicate_rows
        }
        ordered = sorted(group_apps)
        for app in ordered[1:]:
            union.union(ordered[0], app)

    fixed_by_group: defaultdict[str, set[str]] = defaultdict(set)
    for row in records:
        app = canonicalize_app_id(str(row["app_id_canonical"]), alias_map)
        split_origin = str(row.get("split_origin", ""))
        if row.get("source_dataset") == "screenspot" or split_origin.startswith("official_"):
            fixed_by_group[union.find(app)].add(str(row["split"]))
    conflicts = {
        group: tuple(sorted(splits)) for group, splits in fixed_by_group.items() if len(splits) > 1
    }
    if conflicts:
        raise ValueError(f"official split or evaluation leakage conflict: {conflicts}")

    group_split: dict[str, str] = {}
    train_end = normalized_ratios[0]
    val_end = train_end + normalized_ratios[1]
    for app in sorted(apps):
        app_group = union.find(app)
        if app_group in group_split:
            continue
        fixed = fixed_by_group.get(app_group)
        if fixed:
            group_split[app_group] = next(iter(fixed))
            continue
        fraction = _hash_fraction(app_group, seed)
        group_split[app_group] = (
            "train" if fraction < train_end else "val" if fraction < val_end else "test"
        )
    app_assignments = {app: group_split[union.find(app)] for app in sorted(apps)}
    assignments: dict[str, str] = {}
    for row in records:
        screen_id = str(row["screen_id"])
        parent_id = str(row.get("synthetic_parent_id") or "")
        app = canonicalize_app_id(str(row["app_id_canonical"]), alias_map)
        assignments[screen_id] = app_assignments[app]
        if parent_id and parent_id in assignments:
            assignments[screen_id] = assignments[parent_id]
    for row in records:
        parent_id = str(row.get("synthetic_parent_id") or "")
        if parent_id:
            if parent_id not in assignments:
                raise ValueError(f"synthetic parent {parent_id!r} is absent from split input")
            assignments[str(row["screen_id"])] = assignments[parent_id]
    app_splits: defaultdict[str, set[str]] = defaultdict(set)
    for row in records:
        app = canonicalize_app_id(str(row["app_id_canonical"]), alias_map)
        app_splits[app].add(assignments[str(row["screen_id"])])
    violations = {app: sorted(values) for app, values in app_splits.items() if len(values) > 1}
    if violations:
        raise ValueError(f"application-disjoint split creation failed: {violations}")
    exact_groups = {
        digest: tuple(sorted(str(row["screen_id"]) for row in duplicate_rows))
        for digest, duplicate_rows in exact.items()
        if len(duplicate_rows) > 1
    }
    for digest, duplicate_ids in exact_groups.items():
        splits = {assignments[screen_id] for screen_id in duplicate_ids}
        if len(splits) > 1:
            raise ValueError(f"exact duplicate group crosses splits: {digest}")
    return SplitPlan(
        schema_version=1,
        policy_version="app_disjoint_hash_v1",
        seed=seed,
        assignments=dict(sorted(assignments.items())),
        app_assignments=dict(sorted(app_assignments.items())),
        aliases={key: tuple(sorted(values)) for key, values in sorted(raw_aliases.items())},
        collisions={
            key: tuple(sorted(values))
            for key, values in sorted(raw_to_canonical.items())
            if len(values) > 1
        },
        exact_hash_groups=exact_groups,
    )


def write_split_plan(plan: SplitPlan, path: Path) -> Path:
    """Atomically freeze a split plan and refuse a different overwrite."""

    payload = {
        "schema_version": plan.schema_version,
        "policy_version": plan.policy_version,
        "seed": plan.seed,
        "assignments": dict(plan.assignments),
        "app_assignments": dict(plan.app_assignments),
        "aliases": {key: list(value) for key, value in plan.aliases.items()},
        "collisions": {key: list(value) for key, value in plan.collisions.items()},
        "exact_hash_groups": {key: list(value) for key, value in plan.exact_hash_groups.items()},
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(f"refusing to overwrite a different split plan: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, path)
    return path


def load_split_assignments(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assignments = payload.get("assignments")
    if not isinstance(assignments, dict) or not all(
        isinstance(key, str) and value in {"train", "val", "test"}
        for key, value in assignments.items()
    ):
        raise ValueError("invalid split plan assignments")
    return dict(assignments)
