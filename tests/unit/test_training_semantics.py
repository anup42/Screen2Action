from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from screen2action.data.schema import NodeRecord, NodeType
from screen2action.data.tokenizer import VocabularyTokenizer
from screen2action.models.screen2action_model import Screen2ActionModel
from screen2action.perception.base import normalize_image
from screen2action.perception.pipeline import PerceptionFrame
from screen2action.training.data import (
    CachedScreenBatchBuilder,
    CanonicalTrainingCorpus,
    TrainingCommand,
)
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


def test_cached_drag_labels_preserve_duration_and_mask_pseudo_points(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    image = normalize_image(tmp_path / str(corpus.screens["screen-train"]["image_path"]))
    nodes = (
        NodeRecord(1, NodeType.CONTROL, (0.05, 0.15, 0.45, 0.85)),
        NodeRecord(2, NodeType.CONTROL, (0.55, 0.15, 0.95, 0.85)),
    )
    frame = PerceptionFrame(image.sha256, image.width, image.height, nodes, (), 0, "fixture", {})
    loader = SimpleNamespace(frame_for_screen=lambda _: frame)
    builder = CachedScreenBatchBuilder(corpus, loader, Screen2ActionModel())
    row = {
        "text": "drag item",
        "action_type": "drag",
        "target_x1": 0.05,
        "target_y1": 0.15,
        "target_x2": 0.45,
        "target_y2": 0.85,
        "has_target_point": True,
        "target_point_x": 0.25,
        "target_point_y": 0.5,
        "target_point_source": "true",
        "command_label_masks_json": json.dumps({"target": True, "point": True, "parameters": True}),
        "action_parameters_json": json.dumps(
            {"destination_x": 0.75, "destination_y": 0.5, "duration": 0.4}
        ),
    }
    command = replace(corpus.commands[0], row=row)
    batch = builder.build((command,)).commands
    torch.testing.assert_close(
        batch.parameter_targets[0, 2:7], torch.tensor([0.25, 0.5, 0.75, 0.5, 0.4])
    )
    assert batch.parameter_mask[0, 2:7].all()
    assert batch.parameter_node_mask[0].all()
    pseudo = replace(command, row={**row, "target_point_source": "pseudo"})
    assert not builder.build((pseudo,)).commands.parameter_mask[0, 2:4].any()
