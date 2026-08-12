from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from screen2action.data.tokenizer import VocabularyTokenizer
from screen2action.training.data import CanonicalTrainingCorpus, TrainingCommand
from screen2action.training.detector_native import generate_yolo_dataset
from screen2action.training.semantics import (
    SemanticBatchBuilder,
    build_semantic_corpus,
    build_semantics_graph_model,
)


def _corpus(tmp_path: Path) -> CanonicalTrainingCorpus:
    screens: dict[str, dict[str, object]] = {}
    elements: dict[str, dict[str, dict[str, object]]] = {}
    commands = []
    for index, split in enumerate(("train", "val")):
        screen_id = f"screen-{split}"
        image_path = Path("images") / f"{screen_id}.png"
        destination = tmp_path / image_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 24), color=(20 + index * 30, 80, 140)).save(destination)
        screens[screen_id] = {
            "screen_id": screen_id,
            "image_path": image_path.as_posix(),
            "screen_sha256": f"{index + 1:064x}",
        }
        elements[screen_id] = {
            f"element-{element}": {
                "element_id": f"element-{element}",
                "node_type": "control",
                "x1": 0.05 + element * 0.5,
                "y1": 0.15,
                "x2": 0.45 + element * 0.5,
                "y2": 0.85,
                "text": f"item {element}",
                "has_text": True,
                "actionability_labels_json": json.dumps([True, False, False, True]),
                "icon_class_id": 3 + element,
                "metadata_json": json.dumps({"original_class_id": 7 + element}),
                "element_annotation_source": "fixture",
                "element_label_masks_json": json.dumps({"icon": True, "actionability": True}),
            }
            for element in range(2)
        }
        commands.append(
            TrainingCommand(
                f"command-{split}",
                screen_id,
                split,
                {"text": "open item"},
            )
        )
    return CanonicalTrainingCorpus(
        "d" * 64,
        tmp_path,
        screens,
        tuple(commands),
        elements,
        {},
        VocabularyTokenizer.from_corpus(("open item",)),
    )


def test_stage1_semantics_trains_visual_and_relation_graph_parameters(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    semantic = build_semantic_corpus(corpus)
    bundle = build_semantics_graph_model(
        {
            "model": {
                "profile": "tiny_cpu",
                "embedding_dim": 16,
                "graph_layers": 1,
                "graph_heads": 4,
                "semantic_crop_size": 16,
            }
        },
        model_lock_path=None,
        cache_root=tmp_path / "models",
    )
    batch = SemanticBatchBuilder(device=torch.device("cpu"), crop_size=16).build(
        semantic.by_split["train"]
    )
    output = bundle.model(batch)
    output.total_loss.backward()

    assert output.icon_count == 2
    assert output.actionability_count == 8
    assert any(parameter.grad is not None for parameter in bundle.model.visual.parameters())
    assert any(parameter.grad is not None for parameter in bundle.model.graph_encoder.parameters())
    assert bundle.model.graph_icon_head.weight.grad is not None
    assert bundle.model.graph_actionability_head.weight.grad is not None


def test_detector_dataset_uses_only_original_screenparser_ids(tmp_path: Path) -> None:
    result = generate_yolo_dataset(_corpus(tmp_path), tmp_path / "yolo")

    assert result.train_images == result.validation_images == 1
    assert result.labeled_boxes == 4
    assert result.skipped_unmapped_elements == 0
    train_label = next((tmp_path / "yolo" / "labels" / "train").glob("*.txt"))
    rows = train_label.read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("7 ")
    assert rows[1].startswith("8 ")
