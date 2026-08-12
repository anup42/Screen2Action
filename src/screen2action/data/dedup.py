"""Configurable duplicate detection and auditable removal records."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from screen2action.data.schema import ScreenRecord


@dataclass(frozen=True, slots=True)
class DuplicatePair:
    """One duplicate decision and the filter that caused it."""

    first_id: str
    second_id: str
    reason: str
    score: float


@dataclass(frozen=True, slots=True)
class ScreenFingerprint:
    """Staged production duplicate index entry."""

    screen_id: str
    source: str
    split: str
    exact_hash: str
    perceptual_hash: int
    ocr_tokens: tuple[str, ...] = ()
    embedding: tuple[float, ...] = ()
    annotation_confidence: float = 0.0


@dataclass(frozen=True, slots=True)
class DuplicateGroup:
    group_id: str
    survivor_id: str
    member_ids: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RemovalRecord:
    screen_id: str
    survivor_id: str
    group_id: str
    source_split: str
    reason: str


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, first: str, second: str) -> None:
        left, right = self.find(first), self.find(second)
        if left != right:
            survivor, merged = sorted((left, right))
            self.parent[merged] = survivor


@dataclass(slots=True)
class _BkNode:
    value: int
    children: dict[int, _BkNode]


class _HammingBkTree:
    """Sub-quadratic candidate index for 64-bit perceptual hashes."""

    def __init__(self) -> None:
        self.root: _BkNode | None = None

    def add(self, value: int) -> None:
        if self.root is None:
            self.root = _BkNode(value, {})
            return
        node = self.root
        while True:
            distance = (value ^ node.value).bit_count()
            if distance == 0:
                return
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = _BkNode(value, {})
                return
            node = child

    def query(self, value: int, radius: int) -> Iterator[int]:
        if self.root is None:
            return
        pending = [self.root]
        while pending:
            node = pending.pop()
            distance = (value ^ node.value).bit_count()
            if distance <= radius:
                yield node.value
            lower, upper = distance - radius, distance + radius
            pending.extend(child for edge, child in node.children.items() if lower <= edge <= upper)


def exact_image_hash(image: bytes | torch.Tensor) -> str:
    """Hash bytes or CPU tensor content deterministically."""

    if isinstance(image, torch.Tensor):
        payload = image.detach().cpu().contiguous().numpy().tobytes()
    else:
        payload = image
    return hashlib.sha256(payload).hexdigest()


def perceptual_hash(image: torch.Tensor, *, size: int = 8) -> int:
    """Return a small average-hash integer for optional duplicate filtering."""

    if image.ndim != 3 or image.shape[0] not in {1, 3}:
        raise ValueError("image must have shape [C,H,W]")
    gray = image.float().mean(dim=0, keepdim=True).unsqueeze(0)
    resized = torch.nn.functional.interpolate(
        gray, size=(size, size), mode="bilinear", align_corners=False
    )[0, 0]
    threshold = resized.mean()
    bits = (resized >= threshold).flatten().to(torch.int64)
    value = 0
    for bit in bits.tolist():
        value = (value << 1) | int(bit)
    return value


def perceptual_hash_path(path: Path, *, size: int = 8) -> int:
    """Compute average hash through Pillow without loading Torch image models."""

    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("perceptual hashing requires: pip install -e .[data]") from error
    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((size, size)).tobytes())
    threshold = sum(pixels) / max(len(pixels), 1)
    result = 0
    for pixel in pixels:
        result = (result << 1) | int(pixel >= threshold)
    return result


def ocr_token_jaccard(first: Iterable[str], second: Iterable[str]) -> float:
    """Compute case-folded OCR token Jaccard similarity."""

    left = {token.casefold() for token in first}
    right = {token.casefold() for token in second}
    if not left and not right:
        return 1.0
    return len(left & right) / max(len(left | right), 1)


def audit_duplicate_screens(
    screens: Iterable[ScreenRecord],
    *,
    image_hashes: Mapping[str, str] | None = None,
    perceptual_hashes: Mapping[str, int] | None = None,
    ocr_tokens: Mapping[str, tuple[str, ...]] | None = None,
    embedding_similarities: Mapping[tuple[str, str], float] | None = None,
    jaccard_threshold: float = 0.90,
    perceptual_hash_distance_threshold: int = 4,
    embedding_similarity_threshold: float = 0.98,
) -> tuple[DuplicatePair, ...]:
    """Produce auditable exact, perceptual, OCR, and embedding duplicate pairs.

    Embedding similarities are supplied by an optional offline feature job so
    normal CPU tests never download a model or compute an expensive index.
    When that mapping is provided, an OCR match must also meet the embedding
    threshold before it is reported as a combined duplicate.
    """

    if not 0.0 <= jaccard_threshold <= 1.0:
        raise ValueError("jaccard_threshold must be in [0, 1]")
    if perceptual_hash_distance_threshold < 0:
        raise ValueError("perceptual hash distance threshold must be non-negative")
    if not 0.0 <= embedding_similarity_threshold <= 1.0:
        raise ValueError("embedding similarity threshold must be in [0, 1]")
    screen_list = list(screens)
    image_hashes = image_hashes or {}
    perceptual_hashes = perceptual_hashes or {}
    ocr_tokens = ocr_tokens or {}
    embedding_similarities = embedding_similarities or {}
    result: list[DuplicatePair] = []
    for index, first in enumerate(screen_list):
        for second in screen_list[index + 1 :]:
            if image_hashes.get(first.screen_id) and image_hashes.get(
                first.screen_id
            ) == image_hashes.get(second.screen_id):
                result.append(
                    DuplicatePair(first.screen_id, second.screen_id, "exact_image_hash", 1.0)
                )
                continue
            if first.screen_id in perceptual_hashes and second.screen_id in perceptual_hashes:
                distance = (
                    perceptual_hashes[first.screen_id] ^ perceptual_hashes[second.screen_id]
                ).bit_count()
                if distance <= perceptual_hash_distance_threshold:
                    result.append(
                        DuplicatePair(
                            first.screen_id,
                            second.screen_id,
                            "perceptual_hash",
                            1.0 - distance / 64.0,
                        )
                    )
                    continue
            if first.screen_id in ocr_tokens and second.screen_id in ocr_tokens:
                score = ocr_token_jaccard(ocr_tokens[first.screen_id], ocr_tokens[second.screen_id])
                if score >= jaccard_threshold:
                    pair = (first.screen_id, second.screen_id)
                    reverse_pair = (second.screen_id, first.screen_id)
                    embedding_score = embedding_similarities.get(
                        pair, embedding_similarities.get(reverse_pair)
                    )
                    if embedding_similarities and (
                        embedding_score is None
                        or not math.isfinite(embedding_score)
                        or embedding_score < embedding_similarity_threshold
                    ):
                        continue
                    reason = (
                        "ocr_jaccard+embedding_similarity"
                        if embedding_score is not None
                        else "ocr_jaccard"
                    )
                    result.append(
                        DuplicatePair(
                            first.screen_id,
                            second.screen_id,
                            reason,
                            min(score, embedding_score) if embedding_score is not None else score,
                        )
                    )
    return tuple(result)


def _cosine(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    if len(first) != len(second) or not first:
        return -1.0
    numerator = sum(left * right for left, right in zip(first, second, strict=True))
    left_norm = math.sqrt(sum(value * value for value in first))
    right_norm = math.sqrt(sum(value * value for value in second))
    if left_norm == 0.0 or right_norm == 0.0:
        return -1.0
    return numerator / (left_norm * right_norm)


def _survivor_key(fingerprint: ScreenFingerprint) -> tuple[float | int | str, ...]:
    evaluation_priority = 0 if fingerprint.source == "screenspot" else 1
    split_priority = {"test": 0, "val": 1, "train": 2}.get(fingerprint.split, 3)
    return (
        evaluation_priority,
        split_priority,
        -fingerprint.annotation_confidence,
        fingerprint.source,
        fingerprint.screen_id,
    )


def group_duplicate_fingerprints(
    fingerprints: Iterable[ScreenFingerprint],
    *,
    perceptual_hash_distance_threshold: int = 4,
    ocr_jaccard_threshold: float = 0.90,
    embedding_similarity_threshold: float = 0.98,
) -> tuple[tuple[DuplicateGroup, ...], tuple[RemovalRecord, ...]]:
    """Group duplicates through exact hash, BK-tree pHash, OCR, and embeddings."""

    records = list(fingerprints)
    by_id = {record.screen_id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("duplicate fingerprint screen IDs")
    union = _UnionFind(by_id)
    reasons: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    by_exact: defaultdict[str, list[str]] = defaultdict(list)
    by_phash: defaultdict[int, list[str]] = defaultdict(list)
    for record in records:
        by_exact[record.exact_hash].append(record.screen_id)
        by_phash[record.perceptual_hash].append(record.screen_id)
    for digest, exact_members in by_exact.items():
        if not digest or len(exact_members) < 2:
            continue
        exact_first = min(exact_members)
        for member in exact_members:
            union.union(exact_first, member)
            pair = (min(exact_first, member), max(exact_first, member))
            reasons[pair].add("exact_image_hash")

    tree = _HammingBkTree()
    for value in sorted(by_phash):
        tree.add(value)
    seen_hash_pairs: set[tuple[int, int]] = set()
    seen_screen_pairs: set[tuple[str, str]] = set()
    for value in sorted(by_phash):
        for candidate in tree.query(value, perceptual_hash_distance_threshold):
            hash_pair = (min(value, candidate), max(value, candidate))
            if hash_pair in seen_hash_pairs:
                continue
            seen_hash_pairs.add(hash_pair)
            for first_id in by_phash[value]:
                for second_id in by_phash[candidate]:
                    screen_pair = (min(first_id, second_id), max(first_id, second_id))
                    if (
                        first_id == second_id
                        or screen_pair in seen_screen_pairs
                        or union.find(first_id) == union.find(second_id)
                    ):
                        continue
                    seen_screen_pairs.add(screen_pair)
                    first_record, second_record = by_id[first_id], by_id[second_id]
                    distance = (
                        first_record.perceptual_hash ^ second_record.perceptual_hash
                    ).bit_count()
                    if distance > perceptual_hash_distance_threshold:
                        continue
                    if first_record.ocr_tokens or second_record.ocr_tokens:
                        if (
                            ocr_token_jaccard(first_record.ocr_tokens, second_record.ocr_tokens)
                            < ocr_jaccard_threshold
                        ):
                            continue
                    if first_record.embedding or second_record.embedding:
                        if (
                            _cosine(first_record.embedding, second_record.embedding)
                            < embedding_similarity_threshold
                        ):
                            continue
                        reason = "perceptual_hash+ocr_jaccard+embedding_cosine"
                    elif first_record.ocr_tokens or second_record.ocr_tokens:
                        reason = "perceptual_hash+ocr_jaccard"
                    else:
                        reason = "perceptual_hash"
                    union.union(first_id, second_id)
                    reasons[(first_id, second_id)].add(reason)

    grouped: defaultdict[str, list[ScreenFingerprint]] = defaultdict(list)
    for record in records:
        grouped[union.find(record.screen_id)].append(record)
    groups: list[DuplicateGroup] = []
    removals: list[RemovalRecord] = []
    for duplicate_members in grouped.values():
        if len(duplicate_members) < 2:
            continue
        ordered = sorted(duplicate_members, key=_survivor_key)
        survivor = ordered[0]
        member_ids = tuple(sorted(record.screen_id for record in duplicate_members))
        group_id = f"dup-{hashlib.sha256(chr(10).join(member_ids).encode()).hexdigest()[:16]}"
        group_reasons = sorted(
            {
                reason
                for pair, pair_reasons in reasons.items()
                if pair[0] in member_ids and pair[1] in member_ids
                for reason in pair_reasons
            }
        )
        groups.append(
            DuplicateGroup(group_id, survivor.screen_id, member_ids, tuple(group_reasons))
        )
        for record in ordered[1:]:
            removals.append(
                RemovalRecord(
                    screen_id=record.screen_id,
                    survivor_id=survivor.screen_id,
                    group_id=group_id,
                    source_split=record.split,
                    reason="+".join(group_reasons),
                )
            )
    return (
        tuple(sorted(groups, key=lambda item: item.group_id)),
        tuple(sorted(removals, key=lambda item: item.screen_id)),
    )


_TOKEN = re.compile(r"[\w]+", re.UNICODE)


def deduplicate_canonical_dataset(
    dataset_root: Path,
    split_assignments: Mapping[str, str],
    output: Path,
    *,
    perceptual_hash_distance_threshold: int = 4,
    ocr_jaccard_threshold: float = 0.90,
    embedding_similarity_threshold: float = 0.98,
    embeddings: Mapping[str, tuple[float, ...]] | None = None,
) -> dict[str, object]:
    """Build a scalable duplicate audit directly from canonical metadata/images."""

    from screen2action.data.storage import read_table_rows

    text_by_screen: defaultdict[str, list[str]] = defaultdict(list)
    for element in read_table_rows(dataset_root, "elements"):
        if bool(element["has_text"]):
            text_by_screen[str(element["screen_id"])].extend(
                token.casefold() for token in _TOKEN.findall(str(element["text"]))
            )
    fingerprints: list[ScreenFingerprint] = []
    for screen in read_table_rows(dataset_root, "screens"):
        screen_id = str(screen["screen_id"])
        if screen_id not in split_assignments:
            raise ValueError(f"split plan is missing screen {screen_id!r}")
        image_path = dataset_root / str(screen["image_path"])
        fingerprints.append(
            ScreenFingerprint(
                screen_id=screen_id,
                source=str(screen["source_dataset"]),
                split=split_assignments[screen_id],
                exact_hash=str(screen["screen_sha256"]),
                perceptual_hash=perceptual_hash_path(image_path),
                ocr_tokens=tuple(text_by_screen.get(screen_id, ())),
                embedding=(embeddings or {}).get(screen_id, ()),
                annotation_confidence=float(str(screen.get("annotation_confidence") or 0.0)),
            )
        )
    groups, removals = group_duplicate_fingerprints(
        fingerprints,
        perceptual_hash_distance_threshold=perceptual_hash_distance_threshold,
        ocr_jaccard_threshold=ocr_jaccard_threshold,
        embedding_similarity_threshold=embedding_similarity_threshold,
    )
    removed_ids = {record.screen_id for record in removals}
    survivor_ids = sorted(
        record.screen_id for record in fingerprints if record.screen_id not in removed_ids
    )
    by_id = {record.screen_id: record for record in fingerprints}
    for group in groups:
        surviving = [screen_id for screen_id in group.member_ids if screen_id not in removed_ids]
        if len(surviving) != 1:
            raise ValueError(f"duplicate group {group.group_id} has {len(surviving)} survivors")
        sources = {by_id[screen_id].source for screen_id in group.member_ids}
        splits = {by_id[screen_id].split for screen_id in group.member_ids}
        if (
            "screenspot" in sources
            and "train" in splits
            and by_id[surviving[0]].source != "screenspot"
        ):
            raise ValueError("ScreenSpot near duplicate survived outside the evaluation source")
    policy = {
        "version": "staged_dedup_v1",
        "perceptual_hash_distance_threshold": perceptual_hash_distance_threshold,
        "ocr_jaccard_threshold": ocr_jaccard_threshold,
        "embedding_similarity_threshold": embedding_similarity_threshold,
        "embedding_enabled": embeddings is not None,
        "survivor_policy": "screenspot_then_test_then_val_then_train_then_confidence_then_id",
    }
    digest_payload = {
        "policy": policy,
        "groups": [asdict(group) for group in groups],
        "removals": [asdict(record) for record in removals],
        "survivors": survivor_ids,
    }
    digest = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload = {
        "schema_version": 1,
        "policy": policy,
        "digest": digest,
        "input_count": len(fingerprints),
        "survivor_count": len(survivor_ids),
        "removal_count": len(removals),
        "groups": [asdict(group) for group in groups],
        "removals": [asdict(record) for record in removals],
        "survivors": survivor_ids,
        "leakage_checks": {"screenspot_train_near_duplicate_survivors": 0},
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output.is_file() and output.read_text(encoding="utf-8") != serialized:
        raise FileExistsError(f"refusing to overwrite a different dedup audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.partial")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, output)
    return payload
