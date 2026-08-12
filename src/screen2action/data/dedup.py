"""Configurable duplicate detection and auditable removal records."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch

from screen2action.data.schema import ScreenRecord


@dataclass(frozen=True, slots=True)
class DuplicatePair:
    """One duplicate decision and the filter that caused it."""

    first_id: str
    second_id: str
    reason: str
    score: float


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
