from __future__ import annotations

import pytest

from screen2action.data.dedup import audit_duplicate_screens, ocr_token_jaccard
from screen2action.data.splits import validate_app_disjoint
from screen2action.data.synthetic import make_synthetic_examples


def test_synthetic_fixture_covers_required_edge_cases() -> None:
    examples = make_synthetic_examples()
    assert len(examples) >= 6
    assert any(example.command.relation_type == "ordinal" for example in examples)
    assert any(example.command.relation_type == "containment" for example in examples)
    assert any(example.command.relation_type == "proximity" for example in examples)
    assert any(element.text is None for element in examples[0].screen.elements)
    assert any(element.icon_class_id is None for element in examples[0].screen.elements)
    assert ocr_token_jaccard(("Wi-Fi", "settings"), ("wi-fi", "settings")) == 1.0


def test_split_and_duplicate_audits_are_explicit(make_node) -> None:
    examples = make_synthetic_examples()
    screen = examples[0].screen
    duplicate = type(screen)(
        screen_id="other",
        app_id=screen.app_id,
        image_path="<synthetic>",
        width=screen.width,
        height=screen.height,
        split="val",
        source="synthetic",
        elements=screen.elements,
    )
    with pytest.raises(ValueError, match="split violation"):
        validate_app_disjoint([screen, duplicate])
    pairs = audit_duplicate_screens(
        [screen, duplicate],
        image_hashes={screen.screen_id: "same", duplicate.screen_id: "same"},
    )
    assert pairs[0].reason == "exact_image_hash"

    perceptual_pairs = audit_duplicate_screens(
        [screen, duplicate],
        perceptual_hashes={screen.screen_id: 0, duplicate.screen_id: 3},
        perceptual_hash_distance_threshold=2,
    )
    assert perceptual_pairs[0].reason == "perceptual_hash"
    embedding_pairs = audit_duplicate_screens(
        [screen, duplicate],
        ocr_tokens={screen.screen_id: ("settings",), duplicate.screen_id: ("settings",)},
        embedding_similarities={(screen.screen_id, duplicate.screen_id): 0.99},
    )
    assert embedding_pairs[0].reason == "ocr_jaccard+embedding_similarity"
