from __future__ import annotations

from screen2action.data.dedup import ScreenFingerprint, group_duplicate_fingerprints
from screen2action.data.splits import create_app_disjoint_plan


def _screen(
    screen_id: str,
    app: str,
    digest: str,
    *,
    source: str = "wave_ui",
    split: str = "train",
    split_origin: str = "source",
    parent: str = "",
) -> dict[str, object]:
    return {
        "screen_id": screen_id,
        "app_id_canonical": app,
        "app_id_raw": app,
        "screen_sha256": digest,
        "source_dataset": source,
        "split": split,
        "split_origin": split_origin,
        "synthetic_parent_id": parent,
    }


def test_split_creation_joins_exact_duplicates_and_inherits_synthetic_parent() -> None:
    rows = [
        _screen("a", "app.one", "same"),
        _screen("b", "APP TWO", "same"),
        _screen("a-synthetic", "app.one", "different", parent="a"),
        _screen(
            "eval",
            "eval.app",
            "eval-hash",
            source="screenspot",
            split="test",
            split_origin="official_evaluation_only",
        ),
    ]
    plan = create_app_disjoint_plan(rows, seed=11)
    assert plan.assignments["a"] == plan.assignments["b"]
    assert plan.assignments["a-synthetic"] == plan.assignments["a"]
    assert plan.assignments["eval"] == "test"


def test_staged_dedup_keeps_evaluation_survivor_and_audits_removal() -> None:
    records = [
        ScreenFingerprint("train", "wave_ui", "train", "a", 0, ("settings",)),
        ScreenFingerprint("eval", "screenspot", "test", "b", 1, ("settings",)),
        ScreenFingerprint("other", "wave_ui", "train", "c", (1 << 63), ("different",)),
    ]
    groups, removals = group_duplicate_fingerprints(records, perceptual_hash_distance_threshold=1)
    assert len(groups) == 1
    assert groups[0].survivor_id == "eval"
    assert removals[0].screen_id == "train"
    assert "perceptual_hash" in removals[0].reason
