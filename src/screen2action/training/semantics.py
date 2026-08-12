"""Stage-1 source-supervised MobileNet/icon/actionability/graph training."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from screen2action.data.schema import EdgeRecord, NodeRecord, NodeType
from screen2action.model_assets import load_model_lock
from screen2action.models.relation_gat import RelationGraphEncoder, edge_tensors
from screen2action.perception.base import normalize_image
from screen2action.perception.icon_actionability import (
    MobileNetV3IconActionability,
    masked_icon_actionability_loss,
)
from screen2action.runtime.cropper import batch_crop_tensor
from screen2action.ssb.relations import (
    build_containment_edges,
    build_ordinal_edges,
    build_proximity_edges,
)
from screen2action.training.data import CanonicalTrainingCorpus
from screen2action.training.model_factory import _locked_model, _verified_paths, _weight_file


def _mapping(value: object, *, context: str) -> Mapping[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    return cast(dict[str, object], value)


def _integer(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")
    if value <= 0:
        raise ValueError(f"{context} must be positive")
    return value


def _json_list(value: object, *, context: str) -> list[object]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as error:
        raise ValueError(f"{context} is not valid JSON") from error
    if not isinstance(parsed, list):
        raise ValueError(f"{context} must be a JSON list")
    return parsed


def _json_mapping(value: object, *, context: str) -> Mapping[str, object]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as error:
        raise ValueError(f"{context} is not valid JSON") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{context} must be a JSON mapping")
    return cast(dict[str, object], parsed)


@dataclass(frozen=True, slots=True)
class SemanticScreenExample:
    """One source-labeled screen graph without decoded OCR training state."""

    screen_id: str
    image_path: Path
    nodes: tuple[NodeRecord, ...]
    edges: tuple[EdgeRecord, ...]
    icon_targets: tuple[int, ...]
    icon_mask: tuple[bool, ...]
    actionability_targets: tuple[tuple[float, float, float, float], ...]
    actionability_mask: tuple[tuple[bool, bool, bool, bool], ...]


@dataclass(frozen=True, slots=True)
class SemanticTrainingCorpus:
    """Stage-1 examples split with the same immutable membership as commands."""

    by_split: Mapping[str, tuple[SemanticScreenExample, ...]]

    def batches(
        self,
        split: str,
        *,
        epoch: int,
        seed: int,
        rank: int,
        world_size: int,
        per_device_batch_size: int,
        training: bool,
        max_screens: int | None = None,
    ) -> tuple[tuple[SemanticScreenExample, ...], ...]:
        if split not in {"train", "val", "test"}:
            raise ValueError("semantic split must be train, val, or test")
        values = list(self.by_split.get(split, ()))
        if max_screens is not None:
            values = values[:max_screens]
        if not values:
            raise ValueError(f"frozen manifest has no semantic screens in split {split!r}")
        if training:
            random.Random(seed + epoch).shuffle(values)
        global_batch = per_device_batch_size * world_size
        padded_count = math.ceil(len(values) / global_batch) * global_batch
        values.extend(values[index % len(values)] for index in range(padded_count - len(values)))
        local: list[SemanticScreenExample] = []
        for offset in range(0, len(values), global_batch):
            rank_offset = offset + rank * per_device_batch_size
            local.extend(values[rank_offset : rank_offset + per_device_batch_size])
        return tuple(
            tuple(local[offset : offset + per_device_batch_size])
            for offset in range(0, len(local), per_device_batch_size)
        )


def build_semantic_corpus(corpus: CanonicalTrainingCorpus) -> SemanticTrainingCorpus:
    """Build only honest source-supervised labels; absent labels remain masked."""

    grouped: dict[str, list[SemanticScreenExample]] = {"train": [], "val": [], "test": []}
    split_by_screen = {command.screen_id: command.split for command in corpus.commands}
    for screen_id in sorted(corpus.screens):
        split = split_by_screen.get(screen_id)
        if split is None:
            continue
        rows = tuple(
            corpus.elements.get(screen_id, {}).get(element_id)
            for element_id in sorted(corpus.elements.get(screen_id, {}))
        )
        elements = tuple(row for row in rows if row is not None)
        if not elements:
            continue
        nodes: list[NodeRecord] = []
        icons: list[int] = []
        icon_masks: list[bool] = []
        actions: list[tuple[float, float, float, float]] = []
        action_masks: list[tuple[bool, bool, bool, bool]] = []
        for index, row in enumerate(elements):
            labels = _json_list(
                row["actionability_labels_json"],
                context="element actionability labels",
            )
            if len(labels) != 4:
                raise ValueError("element actionability labels must contain four entries")
            masks = _json_mapping(
                row["element_label_masks_json"],
                context="element label masks",
            )
            action_enabled = bool(masks.get("actionability", True))
            action_mask = cast(
                tuple[bool, bool, bool, bool],
                tuple(value is not None and action_enabled for value in labels),
            )
            action_target = cast(
                tuple[float, float, float, float],
                tuple(float(bool(value)) if value is not None else 0.0 for value in labels),
            )
            icon = int(str(row["icon_class_id"]))
            icon_mask = icon >= 0 and bool(masks.get("icon", True))
            node_type = NodeType(str(row["node_type"]))
            nodes.append(
                NodeRecord(
                    node_id=index,
                    node_type=node_type,
                    box_xyxy_norm=cast(
                        tuple[float, float, float, float],
                        tuple(float(str(row[key])) for key in ("x1", "y1", "x2", "y2")),
                    ),
                    actionability_mask=action_mask,
                    annotation_source=str(row["element_annotation_source"]),
                )
            )
            icons.append(max(icon, 0))
            icon_masks.append(icon_mask)
            actions.append(action_target)
            action_masks.append(action_mask)
        if not any(icon_masks) and not any(any(mask) for mask in action_masks):
            continue
        node_tuple = tuple(nodes)
        edges = tuple(
            dict.fromkeys(
                (
                    *build_containment_edges(node_tuple),
                    *build_proximity_edges(node_tuple),
                    *build_ordinal_edges(node_tuple),
                )
            )
        )
        grouped[split].append(
            SemanticScreenExample(
                screen_id,
                corpus.dataset_root / str(corpus.screens[screen_id]["image_path"]),
                node_tuple,
                edges,
                tuple(icons),
                tuple(icon_masks),
                tuple(actions),
                tuple(action_masks),
            )
        )
    if not grouped["train"]:
        raise ValueError("stage1_semantics_graph requires source icon or actionability labels")
    return SemanticTrainingCorpus({name: tuple(values) for name, values in grouped.items()})


@dataclass(frozen=True, slots=True)
class SemanticBatch:
    crops: torch.Tensor
    edge_index: torch.Tensor
    relation_ids: torch.Tensor
    relative_geometry: torch.Tensor
    icon_targets: torch.Tensor
    icon_mask: torch.Tensor
    actionability_targets: torch.Tensor
    actionability_mask: torch.Tensor
    screen_count: int


class SemanticBatchBuilder:
    """Load tight source crops and disjoint graph edges on demand."""

    def __init__(self, *, device: torch.device, crop_size: int = 224) -> None:
        if crop_size <= 0:
            raise ValueError("semantic crop size must be positive")
        self.device = device
        self.crop_size = crop_size

    def build(self, examples: Sequence[SemanticScreenExample]) -> SemanticBatch:
        if not examples:
            raise ValueError("semantic batch cannot be empty")
        crop_parts: list[torch.Tensor] = []
        edges: list[EdgeRecord] = []
        icons: list[int] = []
        icon_masks: list[bool] = []
        actions: list[tuple[float, float, float, float]] = []
        action_masks: list[tuple[bool, bool, bool, bool]] = []
        offset = 0
        for example in examples:
            image = normalize_image(example.image_path)
            crops, _ = batch_crop_tensor(
                image.pixels,
                [node.box_xyxy_norm for node in example.nodes],
                output_size=self.crop_size,
                margin_fraction=0.0,
            )
            crop_parts.append(crops / 255.0)
            edges.extend(
                replace(edge, src=edge.src + offset, dst=edge.dst + offset)
                for edge in example.edges
            )
            offset += len(example.nodes)
            icons.extend(example.icon_targets)
            icon_masks.extend(example.icon_mask)
            actions.extend(example.actionability_targets)
            action_masks.extend(example.actionability_mask)
        edge_index, relation_ids, geometry = edge_tensors(edges, device=self.device)
        return SemanticBatch(
            torch.cat(crop_parts).to(self.device),
            edge_index,
            relation_ids,
            geometry,
            torch.tensor(icons, dtype=torch.long, device=self.device),
            torch.tensor(icon_masks, dtype=torch.bool, device=self.device),
            torch.tensor(actions, dtype=torch.float32, device=self.device),
            torch.tensor(action_masks, dtype=torch.bool, device=self.device),
            len(examples),
        )


@dataclass(frozen=True, slots=True)
class SemanticsGraphOutput:
    total_loss: torch.Tensor
    direct_icon_loss: torch.Tensor
    direct_actionability_loss: torch.Tensor
    graph_icon_loss: torch.Tensor
    graph_actionability_loss: torch.Tensor
    icon_count: int
    actionability_count: int


class _TinyVisualBackbone(nn.Module):
    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.Hardswish(),
            nn.Conv2d(16, channels, 3, stride=2, padding=1),
            nn.Hardswish(),
        )

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.layers(crops))


class SemanticsGraphModel(nn.Module):
    """Shared visual backbone plus direct and post-GAT semantic heads."""

    def __init__(
        self,
        visual: MobileNetV3IconActionability,
        *,
        embedding_dim: int,
        graph_layers: int,
        graph_heads: int,
        graph_variant: str,
        icon_class_count: int = 87,
    ) -> None:
        super().__init__()
        self.visual = visual
        self.node_projection = nn.Sequential(
            nn.Linear(visual.visual_dimension, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        self.graph_encoder = RelationGraphEncoder(
            layers=graph_layers,
            embedding_dim=embedding_dim,
            heads=graph_heads,
            variant=graph_variant,
        )
        self.graph_icon_head = nn.Linear(embedding_dim, icon_class_count)
        self.graph_actionability_head = nn.Linear(embedding_dim, 4)

    def forward(self, batch: SemanticBatch) -> SemanticsGraphOutput:
        visual = self.visual(batch.crops)
        direct = masked_icon_actionability_loss(
            visual,
            icon_targets=batch.icon_targets,
            icon_mask=batch.icon_mask,
            actionability_targets=batch.actionability_targets,
            actionability_mask=batch.actionability_mask,
        )
        node_states = self.node_projection(visual.visual_features)
        graph_states = self.graph_encoder(
            node_states,
            batch.edge_index,
            batch.relation_ids,
            batch.relative_geometry,
        )
        graph_icon_logits = self.graph_icon_head(graph_states)
        graph_action_logits = self.graph_actionability_head(graph_states)
        zero = graph_states.sum() * 0.0
        graph_icon = (
            F.cross_entropy(
                graph_icon_logits[batch.icon_mask],
                batch.icon_targets[batch.icon_mask],
            )
            if bool(batch.icon_mask.any())
            else zero
        )
        graph_action = (
            F.binary_cross_entropy_with_logits(
                graph_action_logits[batch.actionability_mask],
                batch.actionability_targets[batch.actionability_mask],
            )
            if bool(batch.actionability_mask.any())
            else zero
        )
        total = direct.total + graph_icon + graph_action
        return SemanticsGraphOutput(
            total,
            direct.icon,
            direct.actionability,
            graph_icon,
            graph_action,
            direct.icon_count,
            direct.actionability_count,
        )


@dataclass(frozen=True, slots=True)
class SemanticsModelBundle:
    model: SemanticsGraphModel
    profile: str
    model_lock_digest: str
    pretrained_prefixes: tuple[str, ...]
    crop_size: int


def build_semantics_graph_model(
    config_values: Mapping[str, object],
    *,
    model_lock_path: Path | None,
    cache_root: Path,
) -> SemanticsModelBundle:
    """Build the tiny fixture path or locked MobileNetV3 paper path."""

    values = _mapping(config_values.get("model"), context="model")
    profile = str(values.get("profile", "tiny_cpu"))
    if profile not in {"tiny_cpu", "paper_reference"}:
        raise ValueError("model.profile must be tiny_cpu or paper_reference")
    embedding_dim = _integer(
        values.get("embedding_dim", 64 if profile == "tiny_cpu" else 256),
        context="model.embedding_dim",
    )
    visual_feature_dim = _integer(
        values.get("visual_feature_dim", 16 if profile == "tiny_cpu" else 256),
        context="model.visual_feature_dim",
    )
    graph_layers = _integer(
        values.get("graph_layers", 1 if profile == "tiny_cpu" else 2),
        context="model.graph_layers",
    )
    graph_heads = _integer(values.get("graph_heads", 4), context="model.graph_heads")
    graph_variant = str(
        values.get(
            "graph_variant",
            "cpu_reconstruction_v1" if profile == "tiny_cpu" else "paper_eq_v1",
        )
    )
    crop_size = _integer(
        values.get("semantic_crop_size", 64 if profile == "tiny_cpu" else 224),
        context="model.semantic_crop_size",
    )
    if profile == "tiny_cpu":
        visual = MobileNetV3IconActionability(
            _TinyVisualBackbone(),
            backbone_dimension=32,
            visual_dimension=visual_feature_dim,
        )
        return SemanticsModelBundle(
            SemanticsGraphModel(
                visual,
                embedding_dim=embedding_dim,
                graph_layers=graph_layers,
                graph_heads=graph_heads,
                graph_variant=graph_variant,
            ),
            profile,
            "fixture-no-model-lock",
            (),
            crop_size,
        )
    if model_lock_path is None:
        raise ValueError("paper_reference semantics training requires --model-lock")
    lock = load_model_lock(model_lock_path)
    paths = _verified_paths(
        _locked_model(lock, "icon_backbone_mobilenet_v3_small"),
        cache_root,
    )
    visual = MobileNetV3IconActionability.from_locked_torchvision(
        _weight_file(paths, role="icon_backbone_mobilenet_v3_small"),
        visual_dimension=embedding_dim,
    )
    return SemanticsModelBundle(
        SemanticsGraphModel(
            visual,
            embedding_dim=embedding_dim,
            graph_layers=graph_layers,
            graph_heads=graph_heads,
            graph_variant=graph_variant,
        ),
        profile,
        lock.digest,
        ("visual.backbone.",),
        crop_size,
    )
