"""Masked public weak reference reconstruction with auditable precedence."""

from __future__ import annotations

import json
import math
import os
import re
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from screen2action.data.storage import read_table_rows

PUBLIC_WEAK_REFERENCE_POLICY = "public_weak_reference_v1"


@dataclass(frozen=True, slots=True)
class WeakReference:
    command_id: str
    screen_id: str
    target_element_id: str
    reference_element_id: str
    relation: str
    confidence: float
    annotation_source: str
    matching_method: str
    loss_mask: bool


def _float(value: object) -> float:
    return float(str(value))


def _box(
    row: Mapping[str, object],
    keys: tuple[str, str, str, str],
) -> tuple[float, float, float, float]:
    return (
        _float(row[keys[0]]),
        _float(row[keys[1]]),
        _float(row[keys[2]]),
        _float(row[keys[3]]),
    )


def _center(element: Mapping[str, object]) -> tuple[float, float]:
    return (
        (_float(element["x1"]) + _float(element["x2"])) / 2.0,
        (_float(element["y1"]) + _float(element["y2"])) / 2.0,
    )


def _area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    intersection = max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )
    union = _area(first) + _area(second) - intersection
    return intersection / union if union > 0.0 else 0.0


def _target_element(
    command: Mapping[str, object],
    elements: Iterable[Mapping[str, object]],
) -> Mapping[str, object] | None:
    candidates = list(elements)
    target = _box(command, ("target_x1", "target_y1", "target_x2", "target_y2"))
    point = None
    if bool(command["has_target_point"]):
        point = (_float(command["target_point_x"]), _float(command["target_point_y"]))
    scored: list[tuple[float, float, str, Mapping[str, object]]] = []
    for element in candidates:
        box = _box(element, ("x1", "y1", "x2", "y2"))
        contains = bool(
            point is not None and box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]
        )
        scored.append(
            (
                1.0 if contains else _iou(target, box),
                -_area(box),
                str(element["element_id"]),
                element,
            )
        )
    if not scored:
        return None
    best = max(scored, key=lambda item: (item[0], item[1], item[2]))
    return best[3] if best[0] > 0.0 else None


_RELATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("inside", re.compile(r"\b(?:inside|within|in)\b", re.I)),
    ("below", re.compile(r"\b(?:below|under|beneath)\b", re.I)),
    ("above", re.compile(r"\b(?:above|over)\b", re.I)),
    ("left", re.compile(r"\b(?:left of|to the left)\b", re.I)),
    ("right", re.compile(r"\b(?:right of|to the right)\b", re.I)),
    ("ordinal", re.compile(r"\b(?:first|second|third|last)\b", re.I)),
)


def _parsed_relation(text: str) -> str | None:
    matches = [relation for relation, pattern in _RELATION_PATTERNS if pattern.search(text)]
    return matches[0] if len(matches) == 1 else None


def _geometric_reference(
    target: Mapping[str, object],
    elements: Iterable[Mapping[str, object]],
    relation: str,
) -> Mapping[str, object] | None:
    candidates = [
        element for element in elements if str(element["element_id"]) != str(target["element_id"])
    ]
    target_x, target_y = _center(target)
    if relation == "inside":
        target_box = _box(target, ("x1", "y1", "x2", "y2"))
        containing = [
            element
            for element in candidates
            if _float(element["x1"]) <= target_box[0]
            and _float(element["y1"]) <= target_box[1]
            and _float(element["x2"]) >= target_box[2]
            and _float(element["y2"]) >= target_box[3]
        ]
        return min(
            containing,
            key=lambda item: (
                _area(_box(item, ("x1", "y1", "x2", "y2"))),
                str(item["element_id"]),
            ),
            default=None,
        )
    filtered: list[tuple[float, str, Mapping[str, object]]] = []
    for element in candidates:
        x, y = _center(element)
        valid = (
            (relation == "below" and y < target_y)
            or (relation == "above" and y > target_y)
            or (relation == "left" and x > target_x)
            or (relation == "right" and x < target_x)
            or relation == "ordinal"
        )
        if valid:
            filtered.append(
                (math.hypot(x - target_x, y - target_y), str(element["element_id"]), element)
            )
    return min(filtered, key=lambda item: (item[0], item[1]), default=(0.0, "", None))[2]


def infer_weak_reference(
    command: Mapping[str, object],
    elements: Iterable[Mapping[str, object]],
    *,
    minimum_confidence: float,
) -> WeakReference | None:
    """Apply source, hierarchy, geometry, then command-parse precedence."""

    element_list = list(elements)
    by_id = {str(element["element_id"]): element for element in element_list}
    target = _target_element(command, element_list)
    if target is None:
        return None
    target_id = str(target["element_id"])
    source_refs = json.loads(str(command["reference_element_ids_json"]))
    if isinstance(source_refs, list) and len(source_refs) == 1 and str(source_refs[0]) in by_id:
        candidate = WeakReference(
            str(command["command_id"]),
            str(command["screen_id"]),
            target_id,
            str(source_refs[0]),
            str(command.get("relation_type") or "source"),
            1.0,
            "source_reference",
            "source_provided_reference",
            True,
        )
        return candidate
    parent_id = str(target.get("parent_element_id") or "")
    relation = _parsed_relation(str(command["text"]))
    if relation == "inside" and parent_id in by_id:
        confidence = 0.95
        candidate = WeakReference(
            str(command["command_id"]),
            str(command["screen_id"]),
            target_id,
            parent_id,
            "inside",
            confidence,
            PUBLIC_WEAK_REFERENCE_POLICY,
            "source_hierarchy_parent",
            confidence >= minimum_confidence,
        )
        return candidate
    if relation is None:
        return None
    reference = _geometric_reference(target, element_list, relation)
    if reference is None:
        return None
    confidence = 0.85 if relation != "ordinal" else 0.80
    return WeakReference(
        str(command["command_id"]),
        str(command["screen_id"]),
        target_id,
        str(reference["element_id"]),
        relation,
        confidence,
        PUBLIC_WEAK_REFERENCE_POLICY,
        "deterministic_geometry_plus_command_parse",
        confidence >= minimum_confidence,
    )


def annotate_public_weak_references(
    dataset_root: Path,
    output: Path,
    *,
    minimum_confidence: float = 0.8,
    review_limit: int = 100,
) -> dict[str, object]:
    """Write a Parquet reference partition and a bounded JSON review sample."""

    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum confidence must be in [0, 1]")
    try:
        import pyarrow as pa
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("reference Parquet output requires: pip install -e .[data]") from error
    elements: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for element in read_table_rows(dataset_root, "elements"):
        elements[str(element["screen_id"])].append(element)
    references: list[WeakReference] = []
    unmasked_count = 0
    for command in read_table_rows(dataset_root, "commands"):
        reference = infer_weak_reference(
            command,
            elements.get(str(command["screen_id"]), ()),
            minimum_confidence=minimum_confidence,
        )
        if reference is None:
            continue
        references.append(reference)
        unmasked_count += int(reference.loss_mask)
    rows = [
        {
            **asdict(reference),
            "policy_version": PUBLIC_WEAK_REFERENCE_POLICY,
        }
        for reference in references
    ]
    schema = pa.schema(
        [
            pa.field("command_id", pa.string(), nullable=False),
            pa.field("screen_id", pa.string(), nullable=False),
            pa.field("target_element_id", pa.string(), nullable=False),
            pa.field("reference_element_id", pa.string(), nullable=False),
            pa.field("relation", pa.string(), nullable=False),
            pa.field("confidence", pa.float64(), nullable=False),
            pa.field("annotation_source", pa.string(), nullable=False),
            pa.field("matching_method", pa.string(), nullable=False),
            pa.field("loss_mask", pa.bool_(), nullable=False),
            pa.field("policy_version", pa.string(), nullable=False),
        ],
        metadata={b"screen2action_reference_policy": PUBLIC_WEAK_REFERENCE_POLICY.encode()},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.partial")
    parquet.write_table(pa.Table.from_pylist(rows, schema=schema), temporary, compression="zstd")
    os.replace(temporary, output)
    review_path = output.with_suffix(output.suffix + ".review.json")
    review = {
        "schema_version": 1,
        "policy_version": PUBLIC_WEAK_REFERENCE_POLICY,
        "minimum_confidence": minimum_confidence,
        "reference_count": len(references),
        "unmasked_reference_count": unmasked_count,
        "relation_counts": dict(sorted(Counter(item.relation for item in references).items())),
        "sample": rows[:review_limit],
    }
    review_path.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**review, "output": output.as_posix(), "review_path": review_path.as_posix()}
