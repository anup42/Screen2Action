"""Manifest-enforced canonical command batches backed by cached perception."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import torch

from screen2action.data.manifest import verify_data_manifest
from screen2action.data.schema import ActionType, Box, NodeRecord, NodeType
from screen2action.data.storage import read_table_rows
from screen2action.data.tokenizer import VocabularyTokenizer
from screen2action.models.batching import CommandBatch
from screen2action.models.bert_adapter import CompactBertAdapter
from screen2action.models.screen2action_model import Screen2ActionBatch, Screen2ActionModel
from screen2action.perception.loader import CachedPerceptionLoader
from screen2action.perception.pipeline import PerceptionFrame
from screen2action.ssb.matching import match_positive_proposal

ACTION_TO_INDEX = {
    ActionType.CLICK.value: 0,
    ActionType.DRAG.value: 1,
    ActionType.SCROLL.value: 2,
    ActionType.LONG_PRESS.value: 3,
}


def _json_mapping(value: object, *, context: str) -> dict[str, object]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as error:
        raise ValueError(f"{context} is not valid JSON") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{context} must be a JSON mapping")
    return cast(dict[str, object], parsed)


def _json_list(value: object, *, context: str) -> list[object]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as error:
        raise ValueError(f"{context} is not valid JSON") from error
    if not isinstance(parsed, list):
        raise ValueError(f"{context} must be a JSON list")
    return parsed


def _box(row: Mapping[str, object], prefix: str = "target_") -> Box:
    return cast(
        Box,
        tuple(float(str(row[f"{prefix}{axis}"])) for axis in ("x1", "y1", "x2", "y2")),
    )


@dataclass(frozen=True, slots=True)
class TrainingCommand:
    """One canonical command row plus its frozen split identity."""

    command_id: str
    screen_id: str
    split: str
    row: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class CanonicalTrainingCorpus:
    """Verified survivor screens, commands, references, and train-only vocabulary."""

    manifest_digest: str
    dataset_root: Path
    screens: Mapping[str, Mapping[str, object]]
    commands: tuple[TrainingCommand, ...]
    elements: Mapping[str, Mapping[str, Mapping[str, object]]]
    references: Mapping[str, tuple[Mapping[str, object], ...]]
    tokenizer: VocabularyTokenizer

    def split_commands(self, split: str) -> tuple[TrainingCommand, ...]:
        if split not in {"train", "val", "test"}:
            raise ValueError("training split must be train, val, or test")
        return tuple(command for command in self.commands if command.split == split)

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
        max_commands: int | None = None,
    ) -> tuple[tuple[TrainingCommand, ...], ...]:
        """Build equal-count train batches or unpadded validation rank shards."""

        if min(epoch, seed, rank) < 0 or world_size <= 0 or rank >= world_size:
            raise ValueError("epoch/seed/rank/world-size values are invalid")
        if per_device_batch_size <= 0:
            raise ValueError("per-device command batch must be positive")
        commands = list(self.split_commands(split))
        if max_commands is not None:
            if max_commands <= 0:
                raise ValueError("max_commands must be positive")
            commands = commands[:max_commands]
        if not commands:
            raise ValueError(f"frozen manifest has no commands in split {split!r}")
        grouped: defaultdict[str, list[TrainingCommand]] = defaultdict(list)
        for command in commands:
            grouped[command.screen_id].append(command)
        screen_ids = sorted(grouped)
        if training:
            random.Random(seed + epoch).shuffle(screen_ids)
        ordered = [
            command
            for screen_id in screen_ids
            for command in sorted(grouped[screen_id], key=lambda item: item.command_id)
        ]
        if training:
            global_batch = per_device_batch_size * world_size
            padded_count = math.ceil(len(ordered) / global_batch) * global_batch
            ordered.extend(
                ordered[index % len(ordered)] for index in range(padded_count - len(ordered))
            )
            rank_values: list[TrainingCommand] = []
            for offset in range(0, len(ordered), global_batch):
                rank_offset = offset + rank * per_device_batch_size
                rank_values.extend(ordered[rank_offset : rank_offset + per_device_batch_size])
        else:
            rank_values = ordered[rank::world_size]
        return tuple(
            tuple(rank_values[offset : offset + per_device_batch_size])
            for offset in range(0, len(rank_values), per_device_batch_size)
            if rank_values[offset : offset + per_device_batch_size]
        )


def _membership(path: Path, manifest: Mapping[str, object]) -> dict[str, str]:
    split = manifest.get("split")
    if not isinstance(split, dict) or not isinstance(split.get("membership_path"), str):
        raise ValueError("data manifest split membership is malformed")
    membership_path = (path.parent / str(split["membership_path"])).resolve()
    result: dict[str, str] = {}
    for line_number, line in enumerate(
        membership_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"split membership row {line_number} is malformed")
        screen_id = str(raw.get("screen_id", ""))
        split_name = str(raw.get("split", ""))
        if not screen_id or split_name not in {"train", "val", "test"}:
            raise ValueError(f"split membership row {line_number} is invalid")
        if screen_id in result:
            raise ValueError(f"duplicate split membership for screen {screen_id}")
        result[screen_id] = split_name
    return result


def load_training_corpus(manifest_path: Path) -> CanonicalTrainingCorpus:
    """Enforce a frozen manifest and materialize only canonical metadata."""

    verification = verify_data_manifest(manifest_path)
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw_manifest, dict):
        raise ValueError("data manifest must be a mapping")
    split_by_screen = _membership(manifest_path, raw_manifest)
    dataset_root = Path(verification.dataset_root)
    screens = {
        str(row["screen_id"]): row
        for row in read_table_rows(dataset_root, "screens")
        if str(row["screen_id"]) in split_by_screen
    }
    if set(screens) != set(split_by_screen):
        raise ValueError("data manifest membership and canonical screens differ")
    for screen_id, row in screens.items():
        if row["source_dataset"] == "screenspot" and split_by_screen[screen_id] != "test":
            raise ValueError("ScreenSpot leakage guard blocked a training/validation screen")
    commands = tuple(
        TrainingCommand(
            str(row["command_id"]),
            str(row["screen_id"]),
            split_by_screen[str(row["screen_id"])],
            row,
        )
        for row in read_table_rows(dataset_root, "commands")
        if str(row["screen_id"]) in screens
    )
    if not commands:
        raise ValueError("frozen manifest contains no surviving commands")
    elements: defaultdict[str, dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in read_table_rows(dataset_root, "elements"):
        screen_id = str(row["screen_id"])
        if screen_id in screens:
            elements[screen_id][str(row["element_id"])] = row
    references: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in read_table_rows(dataset_root, "references"):
        if bool(row.get("loss_mask", False)):
            references[str(row["command_id"])].append(row)
    tokenizer = VocabularyTokenizer.from_corpus(
        str(command.row["text"]) for command in commands if command.split == "train"
    )
    return CanonicalTrainingCorpus(
        verification.manifest_digest,
        dataset_root,
        screens,
        commands,
        {screen_id: dict(values) for screen_id, values in elements.items()},
        {
            command_id: tuple(sorted(values, key=lambda row: str(row["reference_element_id"])))
            for command_id, values in references.items()
        },
        tokenizer,
    )


def _proposal_nodes(nodes: Sequence[NodeRecord]) -> tuple[NodeRecord, ...]:
    return tuple(node for node in nodes if node.node_type is not NodeType.ROOT)


def _element_box(row: Mapping[str, object]) -> Box:
    return cast(Box, tuple(float(str(row[key])) for key in ("x1", "y1", "x2", "y2")))


def _parameter_value(parameters: Mapping[str, object], key: str) -> tuple[float, bool]:
    value = parameters.get(key)
    if value is None:
        return 0.0, False
    result = float(str(value))
    if not math.isfinite(result):
        raise ValueError(f"action parameter {key} must be finite")
    return result, True


class CachedScreenBatchBuilder:
    """Convert metadata batches into model tensors without loading perception models."""

    def __init__(
        self,
        corpus: CanonicalTrainingCorpus,
        cache_loader: CachedPerceptionLoader,
        model: Screen2ActionModel,
    ) -> None:
        self.corpus = corpus
        self.cache_loader = cache_loader
        self.model = model

    def _tokens(self, texts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        encoder = self.model.command_encoder
        if isinstance(encoder, CompactBertAdapter):
            return encoder.tokenize(texts, device=self.model.device)
        values = self.corpus.tokenizer.encode_batch(
            texts,
            max_length=self.model.config.command_max_length,
        )
        input_ids = torch.tensor(values, dtype=torch.long, device=self.model.device)
        return input_ids, input_ids != self.corpus.tokenizer.pad_id

    def _frame(self, screen_id: str) -> PerceptionFrame:
        frame = self.cache_loader.frame_for_screen(screen_id)
        if isinstance(self.model.command_encoder, CompactBertAdapter):
            return frame
        nodes = []
        for node in frame.nodes:
            if not node.text or node.text_token_ids:
                nodes.append(node)
                continue
            encoded = self.corpus.tokenizer.encode(
                node.text,
                max_length=self.model.config.node_text_length,
            )
            tokens = tuple(
                token_id for token_id in encoded if token_id != self.corpus.tokenizer.pad_id
            )
            nodes.append(replace(node, text_token_ids=tokens))
        return replace(frame, nodes=tuple(nodes))

    def build(self, commands: Sequence[TrainingCommand]) -> Screen2ActionBatch:
        if not commands:
            raise ValueError("cannot build an empty Screen2Action training batch")
        screen_ids = tuple(dict.fromkeys(command.screen_id for command in commands))
        screen_index = {screen_id: index for index, screen_id in enumerate(screen_ids)}
        frames = tuple(self._frame(screen_id) for screen_id in screen_ids)
        images = tuple(
            self.corpus.dataset_root / str(self.corpus.screens[screen_id]["image_path"])
            for screen_id in screen_ids
        )
        input_ids, attention = self._tokens([str(command.row["text"]) for command in commands])
        count = len(commands)
        target_ids = torch.zeros(count, dtype=torch.long, device=self.model.device)
        target_mask = torch.zeros(count, dtype=torch.bool, device=self.model.device)
        target_boxes = torch.zeros((count, 4), dtype=torch.float32, device=self.model.device)
        target_box_mask = torch.zeros(count, dtype=torch.bool, device=self.model.device)
        target_points = torch.zeros((count, 2), dtype=torch.float32, device=self.model.device)
        target_point_mask = torch.zeros(count, dtype=torch.bool, device=self.model.device)
        action_types = torch.zeros(count, dtype=torch.long, device=self.model.device)
        action_mask = torch.zeros(count, dtype=torch.bool, device=self.model.device)
        parameter_targets = torch.zeros((count, 9), dtype=torch.float32, device=self.model.device)
        parameter_mask = torch.zeros((count, 9), dtype=torch.bool, device=self.model.device)
        parameter_node_ids = torch.zeros((count, 2), dtype=torch.long, device=self.model.device)
        parameter_node_mask = torch.zeros((count, 2), dtype=torch.bool, device=self.model.device)
        reference_matches: list[list[int]] = []

        for index, command in enumerate(commands):
            row = command.row
            frame = frames[screen_index[command.screen_id]]
            proposals = _proposal_nodes(frame.nodes)
            box = _box(row)
            target_boxes[index] = torch.tensor(box, device=self.model.device)
            masks = _json_mapping(row["command_label_masks_json"], context="command masks")
            target_box_mask[index] = bool(masks.get("target", True))
            target = match_positive_proposal(box, proposals)
            if target.proposal_id is not None and bool(masks.get("target", True)):
                target_ids[index] = target.proposal_id
                target_mask[index] = True
            if bool(row["has_target_point"]):
                target_points[index] = torch.tensor(
                    [float(str(row["target_point_x"])), float(str(row["target_point_y"]))],
                    device=self.model.device,
                )
                target_point_mask[index] = (
                    bool(masks.get("point", False)) and str(row["target_point_source"]) == "true"
                )
            action = str(row["action_type"])
            if action in ACTION_TO_INDEX and bool(masks.get("action", True)):
                action_types[index] = ACTION_TO_INDEX[action]
                action_mask[index] = True
            parameters = _json_mapping(
                row["action_parameters_json"],
                context="action parameters",
            )
            dx, has_dx = _parameter_value(parameters, "dx")
            dy, has_dy = _parameter_value(parameters, "dy")
            parameter_targets[index, :2] = torch.tensor(
                [dx, dy], dtype=torch.float32, device=self.model.device
            )
            if bool(masks.get("parameters", False)):
                parameter_mask[index, :2] = torch.tensor(
                    [has_dx, has_dy], dtype=torch.bool, device=self.model.device
                )
            if (
                action in {ActionType.SCROLL.value, ActionType.DRAG.value}
                and target.proposal_id is not None
            ):
                parameter_node_ids[index, 0] = target.proposal_id
                parameter_node_mask[index, 0] = True
            if action == ActionType.DRAG.value and bool(row["has_target_point"]):
                source_x = float(str(row["target_point_x"]))
                source_y = float(str(row["target_point_y"]))
                destination_x, has_destination_x = _parameter_value(parameters, "destination_x")
                destination_y, has_destination_y = _parameter_value(parameters, "destination_y")
                parameter_targets[index, 2:6] = torch.tensor(
                    [source_x, source_y, destination_x, destination_y],
                    device=self.model.device,
                )
                duration, has_duration = _parameter_value(parameters, "duration")
                parameter_targets[index, 6] = duration
                parameter_mask[index, 6] = has_duration and bool(masks.get("parameters", False))
                if bool(masks.get("parameters", False)):
                    parameter_mask[index, 2:6] = torch.tensor(
                        [
                            bool(target_point_mask[index]),
                            bool(target_point_mask[index]),
                            has_destination_x,
                            has_destination_y,
                        ],
                        device=self.model.device,
                    )
                if has_destination_x and has_destination_y:
                    destination_box: Box = (
                        destination_x,
                        destination_y,
                        destination_x,
                        destination_y,
                    )
                    destination = match_positive_proposal(destination_box, proposals)
                    if destination.proposal_id is not None:
                        parameter_node_ids[index, 1] = destination.proposal_id
                        parameter_node_mask[index, 1] = True
            if action == ActionType.LONG_PRESS.value and bool(row["has_target_point"]):
                parameter_targets[index, 7:9] = target_points[index]
                parameter_mask[index, 7:9] = target_point_mask[index]

            local_references: list[int] = []
            elements = self.corpus.elements.get(command.screen_id, {})
            for reference in self.corpus.references.get(command.command_id, ()):
                element = elements.get(str(reference["reference_element_id"]))
                if element is None:
                    continue
                matched = match_positive_proposal(_element_box(element), proposals)
                if matched.proposal_id is not None:
                    local_references.append(matched.proposal_id)
            reference_matches.append(sorted(set(local_references)))

        reference_slots = max(1, max(map(len, reference_matches), default=0))
        reference_ids = torch.zeros(
            (count, reference_slots), dtype=torch.long, device=self.model.device
        )
        reference_mask = torch.zeros(
            (count, reference_slots), dtype=torch.bool, device=self.model.device
        )
        for index, values in enumerate(reference_matches):
            if values:
                reference_ids[index, : len(values)] = torch.tensor(
                    values, dtype=torch.long, device=self.model.device
                )
                reference_mask[index, : len(values)] = True
        command_batch = CommandBatch(
            input_ids=input_ids,
            attention_mask=attention,
            screen_indices=torch.tensor(
                [screen_index[command.screen_id] for command in commands],
                dtype=torch.long,
                device=self.model.device,
            ),
            target_node_ids=target_ids,
            target_mask=target_mask,
            reference_node_ids=reference_ids,
            reference_mask=reference_mask,
            action_types=action_types,
            action_mask=action_mask,
            target_boxes=target_boxes,
            target_box_mask=target_box_mask,
            target_points=target_points,
            target_point_mask=target_point_mask,
            parameter_targets=parameter_targets,
            parameter_mask=parameter_mask,
            parameter_node_ids=parameter_node_ids,
            parameter_node_mask=parameter_node_mask,
        )
        return Screen2ActionBatch(
            images=images,
            commands=command_batch,
            perceived_frames=frames,
            perception_mode="cached",
        )
