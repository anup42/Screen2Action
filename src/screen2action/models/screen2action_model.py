"""Unified trainable Screen2Action model with separable frame and command paths."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import cast

import torch
from torch import nn

from screen2action.data.schema import EdgeRecord, NodeRecord, ScreenRecord
from screen2action.models.batching import CommandBatch
from screen2action.models.bert_adapter import CompactBertAdapter
from screen2action.models.command_encoder import CommandEncoder, CommandEncoding
from screen2action.models.crop_encoder import CropTokenEncoder
from screen2action.models.grounding_losses import (
    ConfidenceReconstructionConfig,
    GroundingLoss,
    GroundingSupervision,
    grounding_loss,
)
from screen2action.models.node_encoder import NodeEncoder, node_batch_from_records
from screen2action.models.relation_gat import RelationGraphEncoder, edge_tensors
from screen2action.models.retention import (
    RetentionLoss,
    RetentionOutput,
    RetentionScorer,
    retention_loss,
)
from screen2action.models.retriever import (
    CosineRetriever,
    RelationAwareReranker,
    select_top_k_actionable,
)
from screen2action.models.sparse_grounder import GroundingTensorOutput, SparseCandidateGrounder
from screen2action.perception.base import normalize_image
from screen2action.perception.pipeline import FullScreenPerception, PerceptionFrame
from screen2action.runtime.cropper import batch_crop_tensor
from screen2action.ssb.codec import graph_fixed_token_cost, node_token_cost
from screen2action.ssb.selector import SelectionResult, budget_select


@dataclass(frozen=True, slots=True)
class Screen2ActionModelConfig:
    """Trainable model dimensions; paper and CPU variants are named explicitly."""

    embedding_dim: int = 64
    vocab_size: int = 16_384
    node_text_length: int = 16
    icon_class_count: int = 87
    visual_feature_dim: int = 16
    graph_layers: int = 1
    graph_heads: int = 4
    graph_variant: str = "cpu_reconstruction_v1"
    command_layers: int = 2
    command_heads: int = 4
    command_ffn: int = 256
    command_max_length: int = 64
    node_text_strategy: str = "shared_wordpiece"
    top_k: int = 4
    ssb_budget: int = 256
    crop_size: int = 64
    crop_tokens: int = 16
    crop_margin_fraction: float = 0.12
    actionability_threshold: float = 0.5
    relation_lambda: float = 0.30
    selector_temperature: float = 1.0

    def __post_init__(self) -> None:
        positive = (
            self.embedding_dim,
            self.vocab_size,
            self.node_text_length,
            self.icon_class_count,
            self.visual_feature_dim,
            self.graph_layers,
            self.graph_heads,
            self.command_layers,
            self.command_heads,
            self.command_ffn,
            self.command_max_length,
            self.top_k,
            self.ssb_budget,
            self.crop_size,
            self.crop_tokens,
        )
        if min(positive) <= 0:
            raise ValueError("all Screen2Action model dimensions must be positive")
        if self.embedding_dim % self.graph_heads or self.embedding_dim % self.command_heads:
            raise ValueError("embedding dimension must divide graph and command heads")
        if self.node_text_strategy not in {"shared_wordpiece", "independent"}:
            raise ValueError("node_text_strategy must be shared_wordpiece or independent")
        if not 0.0 <= self.actionability_threshold <= 1.0:
            raise ValueError("actionability threshold must be in [0, 1]")
        if self.crop_margin_fraction < 0.0 or self.selector_temperature <= 0.0:
            raise ValueError("crop margin and selector temperature are invalid")

    @classmethod
    def tiny_cpu(cls) -> Screen2ActionModelConfig:
        return cls()

    @classmethod
    def paper_reference(cls) -> Screen2ActionModelConfig:
        """Paper dimensions; public locked backbones are supplied to the constructor."""

        return cls(
            embedding_dim=256,
            visual_feature_dim=256,
            graph_layers=2,
            graph_variant="paper_eq_v1",
            command_layers=6,
            command_ffn=1024,
            top_k=8,
            ssb_budget=512,
            crop_size=192,
            crop_tokens=144,
        )


@dataclass(frozen=True, slots=True)
class EncodedFrameBatch:
    """Padded reusable frame tensors and all masks required downstream."""

    node_states: torch.Tensor
    valid_node_mask: torch.Tensor
    valid_edge_mask: torch.Tensor
    edge_index: torch.Tensor
    relation_ids: torch.Tensor
    relative_geometry: torch.Tensor
    valid_text_token_mask: torch.Tensor
    selected_node_mask: torch.Tensor
    selection_weights: torch.Tensor
    retention: RetentionOutput
    retention_token_costs: torch.Tensor
    budgets: torch.Tensor
    records: tuple[tuple[NodeRecord, ...], ...]
    edges: tuple[tuple[EdgeRecord, ...], ...]
    selections: tuple[SelectionResult | None, ...]
    screenshots: tuple[torch.Tensor, ...]


@dataclass(frozen=True, slots=True)
class GroundCommandOutput:
    """Grounder outputs plus retrieval/crop masks and reversible boxes."""

    tensors: GroundingTensorOutput
    candidate_node_positions: torch.Tensor
    candidate_node_ids: torch.Tensor
    candidate_valid_mask: torch.Tensor
    valid_crop_token_mask: torch.Tensor
    candidate_boxes: torch.Tensor
    expanded_crop_boxes: torch.Tensor
    retrieval_scores: torch.Tensor


@dataclass(frozen=True, slots=True)
class ModelLossInputs:
    grounding: GroundingSupervision
    retention_target_mask: torch.Tensor
    retention_reference_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class Screen2ActionBatch:
    images: Sequence[object]
    commands: CommandBatch
    perceived_frames: Sequence[PerceptionFrame] = ()
    oracle_screens: Sequence[ScreenRecord] = ()
    perception_mode: str = "real"


@dataclass(frozen=True, slots=True)
class Screen2ActionModelOutput:
    perceived_frames: tuple[PerceptionFrame, ...]
    encoded_frames: EncodedFrameBatch
    grounded_commands: GroundCommandOutput
    loss_inputs: ModelLossInputs
    retention_loss: RetentionLoss
    grounding_loss: GroundingLoss
    total_loss: torch.Tensor
    stage: str


def _position_edges(
    edges: Sequence[EdgeRecord],
    nodes: Sequence[NodeRecord],
) -> tuple[EdgeRecord, ...]:
    positions = {node.node_id: index for index, node in enumerate(nodes)}
    return tuple(
        replace(edge, src=positions[edge.src], dst=positions[edge.dst])
        for edge in edges
        if edge.src in positions and edge.dst in positions
    )


class Screen2ActionModel(nn.Module):
    """One trainable model; callers control mode, freezing, and gradient context."""

    def __init__(
        self,
        *,
        config: Screen2ActionModelConfig | None = None,
        perception: FullScreenPerception | None = None,
        command_encoder: nn.Module | None = None,
        crop_encoder: nn.Module | None = None,
        confidence_config: ConfidenceReconstructionConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or Screen2ActionModelConfig.tiny_cpu()
        self.perception_service = perception
        self.command_encoder = command_encoder or CommandEncoder(
            vocab_size=self.config.vocab_size,
            embedding_dim=self.config.embedding_dim,
            layers=self.config.command_layers,
            heads=self.config.command_heads,
            feedforward_dim=self.config.command_ffn,
            max_length=self.config.command_max_length,
            dropout=0.0,
        )
        shared_embedding: nn.Embedding | None = None
        if self.config.node_text_strategy == "shared_wordpiece":
            if isinstance(self.command_encoder, CompactBertAdapter):
                shared_embedding = self.command_encoder.wordpiece_embeddings
            else:
                candidate = getattr(self.command_encoder, "embedding", None)
                if isinstance(candidate, nn.Embedding):
                    shared_embedding = candidate
            if shared_embedding is None:
                raise ValueError("shared_wordpiece requires an exposed command embedding")
        self.node_encoder = NodeEncoder(
            embedding_dim=self.config.embedding_dim,
            vocab_size=self.config.vocab_size,
            icon_class_count=self.config.icon_class_count,
            visual_feature_dim=self.config.visual_feature_dim,
            text_embedding=shared_embedding,
        )
        self.graph_encoder = RelationGraphEncoder(
            layers=self.config.graph_layers,
            embedding_dim=self.config.embedding_dim,
            heads=self.config.graph_heads,
            variant=self.config.graph_variant,
        )
        self.retention_scorer = RetentionScorer(self.config.embedding_dim)
        self.retriever = CosineRetriever(self.config.embedding_dim, self.config.embedding_dim)
        self.reranker = RelationAwareReranker(
            self.config.embedding_dim,
            lambda_ref=self.config.relation_lambda,
        )
        self.crop_encoder = crop_encoder or CropTokenEncoder(
            embedding_dim=self.config.embedding_dim,
            token_count=self.config.crop_tokens,
            tiny=True,
        )
        self.grounder = SparseCandidateGrounder(
            embedding_dim=self.config.embedding_dim,
            heads=self.config.command_heads,
            blocks=2,
        )
        self.confidence_config = confidence_config or ConfidenceReconstructionConfig()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def perceive(
        self,
        images: Sequence[object],
        *,
        mode: str = "real",
        oracle_screens: Sequence[ScreenRecord] = (),
    ) -> tuple[PerceptionFrame, ...]:
        """Run the configured typed perception boundary without command inputs."""

        if self.perception_service is None:
            raise RuntimeError("perceive requires a configured FullScreenPerception service")
        return self.perception_service.perceive(
            images,
            mode=mode,
            oracle_screens=oracle_screens,
        )

    def _node_records_with_text_ids(
        self,
        nodes: Sequence[NodeRecord],
    ) -> tuple[NodeRecord, ...]:
        missing = [node for node in nodes if node.text and not node.text_token_ids]
        if not missing:
            return tuple(nodes)
        if self.config.node_text_strategy != "shared_wordpiece" or not isinstance(
            self.command_encoder, CompactBertAdapter
        ):
            raise ValueError(
                "raw OCR text requires shared CompactBert tokenization or pre-tokenized nodes"
            )
        input_ids, attention = self.command_encoder.tokenize(
            [node.text or "" for node in missing],
            device=self.device,
        )
        replacements: dict[int, NodeRecord] = {}
        for row, node in enumerate(missing):
            token_ids = tuple(
                int(token)
                for token in input_ids[row, attention[row]][: self.config.node_text_length]
            )
            replacements[node.node_id] = replace(node, text_token_ids=token_ids)
        return tuple(replacements.get(node.node_id, node) for node in nodes)

    def encode_frames(
        self,
        perceived_frames: Sequence[PerceptionFrame],
        *,
        screenshots: Sequence[object] = (),
        stochastic_selector: bool = False,
        generator: torch.Generator | None = None,
    ) -> EncodedFrameBatch:
        """Encode each unique frame once and pad nodes/edges with explicit masks."""

        if not perceived_frames:
            raise ValueError("encode_frames requires at least one perceived frame")
        if screenshots and len(screenshots) != len(perceived_frames):
            raise ValueError("screenshot count must match perceived frame count")
        frame_records = tuple(
            self._node_records_with_text_ids(frame.nodes) for frame in perceived_frames
        )
        max_nodes = max(len(nodes) for nodes in frame_records)
        max_edges = max((len(frame.edges) for frame in perceived_frames), default=0)
        screen_count = len(perceived_frames)
        states = torch.zeros(
            (screen_count, max_nodes, self.config.embedding_dim), device=self.device
        )
        valid_nodes = torch.zeros((screen_count, max_nodes), dtype=torch.bool, device=self.device)
        valid_text = torch.zeros(
            (screen_count, max_nodes, self.config.node_text_length),
            dtype=torch.bool,
            device=self.device,
        )
        edge_index_batch = torch.zeros(
            (screen_count, 2, max_edges), dtype=torch.long, device=self.device
        )
        relation_batch = torch.zeros(
            (screen_count, max_edges), dtype=torch.long, device=self.device
        )
        geometry_batch = torch.zeros(
            (screen_count, max_edges, 12), dtype=torch.float32, device=self.device
        )
        valid_edges = torch.zeros((screen_count, max_edges), dtype=torch.bool, device=self.device)
        mandatory = torch.zeros_like(valid_nodes)
        token_costs = torch.zeros(
            (screen_count, max_nodes), dtype=torch.float32, device=self.device
        )
        positioned_edges: list[tuple[EdgeRecord, ...]] = []
        for screen_index, (frame, nodes) in enumerate(
            zip(perceived_frames, frame_records, strict=True)
        ):
            node_count = len(nodes)
            valid_nodes[screen_index, :node_count] = True
            mandatory[screen_index, :node_count] = torch.tensor(
                [node.mandatory for node in nodes], dtype=torch.bool, device=self.device
            )
            token_costs[screen_index, :node_count] = torch.tensor(
                [node_token_cost(node) for node in nodes],
                dtype=torch.float32,
                device=self.device,
            )
            node_batch = node_batch_from_records(
                nodes,
                text_length=self.config.node_text_length,
                icon_class_count=self.config.icon_class_count,
                visual_feature_dim=self.config.visual_feature_dim,
                device=self.device,
            )
            valid_text[screen_index, :node_count] = node_batch.text_mask
            node_states = self.node_encoder(node_batch)
            local_edges = _position_edges(frame.edges, nodes)
            positioned_edges.append(local_edges)
            edge_index, relation_ids, geometry = edge_tensors(local_edges, device=self.device)
            node_states = self.graph_encoder(
                node_states,
                edge_index,
                relation_ids,
                geometry,
            )
            states[screen_index, :node_count] = node_states
            edge_count = edge_index.shape[1]
            if edge_count:
                edge_index_batch[screen_index, :, :edge_count] = edge_index
                relation_batch[screen_index, :edge_count] = relation_ids
                geometry_batch[screen_index, :edge_count] = geometry
                valid_edges[screen_index, :edge_count] = True
        retention = self.retention_scorer(
            states,
            valid_nodes,
            stochastic=stochastic_selector,
            temperature=self.config.selector_temperature,
            mandatory_mask=mandatory,
            generator=generator,
        )
        selected = torch.zeros_like(valid_nodes)
        selection_weights = retention.selection_mask.clone()
        selections: list[SelectionResult | None] = []
        if stochastic_selector:
            selected = (selection_weights.detach() >= 0.5) & valid_nodes
            selected |= mandatory
            selections = [None] * screen_count
        else:
            selection_weights.zero_()
            for screen_index, (frame, nodes) in enumerate(
                zip(perceived_frames, frame_records, strict=True)
            ):
                values = {
                    node.node_id: float(retention.scores[screen_index, index].detach().cpu())
                    for index, node in enumerate(nodes)
                }
                selection = budget_select(
                    nodes,
                    frame.edges,
                    budget=self.config.ssb_budget,
                    values=values,
                )
                selections.append(selection)
                selected_ids = set(selection.selected_node_ids)
                for index, node in enumerate(nodes):
                    if node.node_id in selected_ids:
                        selected[screen_index, index] = True
                        selection_weights[screen_index, index] = 1.0
        encoded_states = states * selection_weights.unsqueeze(-1)
        screenshots_out: tuple[torch.Tensor, ...] = ()
        if screenshots:
            normalized = tuple(normalize_image(image) for image in screenshots)
            for image, frame in zip(normalized, perceived_frames, strict=True):
                if image.sha256 != frame.screenshot_sha256:
                    raise ValueError("screenshot bytes do not match perceived frame digest")
            screenshots_out = tuple(image.pixels.float() / 255.0 for image in normalized)
        budgets = torch.full(
            (screen_count,),
            float(self.config.ssb_budget - graph_fixed_token_cost()),
            dtype=torch.float32,
            device=self.device,
        )
        return EncodedFrameBatch(
            encoded_states,
            valid_nodes,
            valid_edges,
            edge_index_batch,
            relation_batch,
            geometry_batch,
            valid_text,
            selected,
            selection_weights,
            retention,
            token_costs,
            budgets,
            frame_records,
            tuple(positioned_edges),
            tuple(selections),
            screenshots_out,
        )

    def _command_encoding(self, commands: CommandBatch) -> CommandEncoding:
        encoded = self.command_encoder(commands.input_ids, commands.attention_mask)
        if not isinstance(encoded, CommandEncoding):
            raise TypeError("command encoder must return CommandEncoding")
        return encoded

    def ground_commands(
        self,
        encoded_frames: EncodedFrameBatch,
        command_batch: CommandBatch,
    ) -> GroundCommandOutput:
        """Fan command retrieval/crops/grounding out from reusable frame states."""

        if not encoded_frames.screenshots:
            raise ValueError("ground_commands requires screenshots for candidate crops")
        command_count = command_batch.input_ids.shape[0]
        if command_count == 0:
            raise ValueError("ground_commands requires at least one command")
        if int(command_batch.screen_indices.max()) >= len(encoded_frames.records):
            raise ValueError("command references a screen outside the encoded batch")
        command_encoding = self._command_encoding(command_batch)
        candidate_positions: list[torch.Tensor] = []
        candidate_ids: list[list[int]] = []
        candidate_valid: list[torch.Tensor] = []
        retrieval_scores: list[torch.Tensor] = []
        candidate_boxes: list[torch.Tensor] = []
        expanded_boxes: list[torch.Tensor] = []
        crop_batches: list[torch.Tensor] = []
        node_tokens: list[torch.Tensor] = []
        for command_index in range(command_count):
            screen_index = int(command_batch.screen_indices[command_index])
            nodes = encoded_frames.records[screen_index]
            node_count = len(nodes)
            features = encoded_frames.node_states[screen_index, :node_count]
            selected_mask = encoded_frames.selected_node_mask[screen_index, :node_count]
            base = self.retriever(command_encoding.pooled[command_index], features)
            reranked = self.reranker(
                base,
                features,
                encoded_frames.edges[screen_index],
                selected_mask,
            )
            requested_action = (
                int(command_batch.action_types[command_index])
                if bool(command_batch.action_mask[command_index])
                else None
            )
            retrieval = select_top_k_actionable(
                reranked,
                nodes,
                k=self.config.top_k,
                actionability_threshold=self.config.actionability_threshold,
                requested_action_index=requested_action,
                valid_mask=selected_mask,
            )
            positions = retrieval.node_indices
            valid = retrieval.valid
            candidate_positions.append(positions)
            candidate_ids.append(
                [
                    nodes[int(position)].node_id if bool(valid[rank]) else -1
                    for rank, position in enumerate(positions)
                ]
            )
            candidate_valid.append(valid)
            retrieval_scores.append(retrieval.scores)
            boxes = [
                nodes[int(position)].box_xyxy_norm if bool(valid[rank]) else (0.0, 0.0, 1.0, 1.0)
                for rank, position in enumerate(positions)
            ]
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32, device=self.device)
            candidate_boxes.append(boxes_tensor)
            screenshot = encoded_frames.screenshots[screen_index]
            crops, specs = batch_crop_tensor(
                screenshot,
                boxes,
                output_size=self.config.crop_size,
                margin_fraction=self.config.crop_margin_fraction,
            )
            crop_batches.append(crops)
            expanded_boxes.append(
                torch.tensor(
                    [spec.expanded_box_xyxy_norm for spec in specs],
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            node_tokens.append(features[positions])
        crops_tensor = torch.stack(crop_batches).to(self.device)
        flat_crops = crops_tensor.flatten(0, 1)
        crop_tokens = cast(torch.Tensor, self.crop_encoder(flat_crops))
        expected_shape = (
            command_count * self.config.top_k,
            self.config.crop_tokens,
            self.config.embedding_dim,
        )
        if crop_tokens.shape != expected_shape:
            raise ValueError(
                f"crop encoder output must be {expected_shape}, got {tuple(crop_tokens.shape)}"
            )
        crop_tokens = crop_tokens.reshape(
            command_count,
            self.config.top_k,
            self.config.crop_tokens,
            self.config.embedding_dim,
        )
        valid_candidates = torch.stack(candidate_valid)
        crop_mask = valid_candidates.unsqueeze(-1).expand(-1, -1, self.config.crop_tokens)
        boxes_tensor = torch.stack(candidate_boxes)
        grounded = self.grounder(
            command_encoding.token_states,
            crop_tokens,
            torch.stack(node_tokens),
            command_mask=command_batch.attention_mask,
            candidate_mask=valid_candidates,
            crop_mask=crop_mask,
            candidate_boxes=boxes_tensor,
        )
        return GroundCommandOutput(
            grounded,
            torch.stack(candidate_positions),
            torch.tensor(candidate_ids, dtype=torch.long, device=self.device),
            valid_candidates,
            crop_mask,
            boxes_tensor,
            torch.stack(expanded_boxes),
            torch.stack(retrieval_scores),
        )

    def _retention_supervision(
        self,
        encoded: EncodedFrameBatch,
        commands: CommandBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target = torch.zeros_like(encoded.valid_node_mask)
        reference = torch.zeros_like(encoded.valid_node_mask)
        id_maps = [
            {node.node_id: position for position, node in enumerate(nodes)}
            for nodes in encoded.records
        ]
        for command_index in range(commands.input_ids.shape[0]):
            screen = int(commands.screen_indices[command_index])
            if bool(commands.target_mask[command_index]):
                node_id = int(commands.target_node_ids[command_index])
                if node_id in id_maps[screen]:
                    target[screen, id_maps[screen][node_id]] = True
            for slot in range(commands.reference_node_ids.shape[1]):
                if bool(commands.reference_mask[command_index, slot]):
                    node_id = int(commands.reference_node_ids[command_index, slot])
                    if node_id in id_maps[screen]:
                        reference[screen, id_maps[screen][node_id]] = True
        return target, reference

    @staticmethod
    def _candidate_rank(node_ids: torch.Tensor, node_id: int) -> tuple[int, bool]:
        matches = torch.nonzero(node_ids == node_id, as_tuple=False).flatten()
        return (int(matches[0]), True) if matches.numel() else (0, False)

    def _grounding_supervision(
        self,
        grounded: GroundCommandOutput,
        commands: CommandBatch,
    ) -> GroundingSupervision:
        count = commands.input_ids.shape[0]
        device = self.device
        candidate_indices = torch.zeros(count, dtype=torch.long, device=device)
        candidate_mask = torch.zeros(count, dtype=torch.bool, device=device)
        point_local = torch.zeros((count, 2), dtype=torch.float32, device=device)
        point_mask = torch.zeros(count, dtype=torch.bool, device=device)
        long_point = torch.zeros((count, 2), dtype=torch.float32, device=device)
        long_mask = torch.zeros(count, dtype=torch.bool, device=device)
        scroll_indices = torch.zeros(count, dtype=torch.long, device=device)
        scroll_mask = torch.zeros(count, dtype=torch.bool, device=device)
        drag_source = torch.zeros(count, dtype=torch.long, device=device)
        drag_destination = torch.zeros(count, dtype=torch.long, device=device)
        drag_mask = torch.zeros(count, dtype=torch.bool, device=device)
        drag_source_point = torch.zeros((count, 2), dtype=torch.float32, device=device)
        drag_destination_point = torch.zeros((count, 2), dtype=torch.float32, device=device)
        drag_duration = torch.zeros((count, 1), dtype=torch.float32, device=device)
        drag_duration_mask = torch.zeros(count, dtype=torch.bool, device=device)
        for index in range(count):
            if bool(commands.target_mask[index]):
                rank, found = self._candidate_rank(
                    grounded.candidate_node_ids[index], int(commands.target_node_ids[index])
                )
                candidate_indices[index] = rank
                candidate_mask[index] = found
                if found and bool(commands.target_point_mask[index]):
                    box = grounded.expanded_crop_boxes[index, rank]
                    width = (box[2] - box[0]).clamp_min(1e-6)
                    height = (box[3] - box[1]).clamp_min(1e-6)
                    point_local[index] = torch.stack(
                        (
                            ((commands.target_points[index, 0] - box[0]) / width).clamp(0, 1),
                            ((commands.target_points[index, 1] - box[1]) / height).clamp(0, 1),
                        )
                    )
                    point_mask[index] = True
            if commands.parameter_targets.shape[1] >= 9:
                long_point[index] = commands.parameter_targets[index, 7:9]
                long_mask[index] = bool(commands.parameter_mask[index, 7:9].all())
            if bool(commands.parameter_node_mask[index, 0]):
                rank, found = self._candidate_rank(
                    grounded.candidate_node_ids[index], int(commands.parameter_node_ids[index, 0])
                )
                scroll_indices[index] = rank
                scroll_mask[index] = found and bool(commands.parameter_mask[index, :2].all())
                drag_source[index] = rank
                drag_mask[index] = found
            if bool(commands.parameter_node_mask[index, 1]):
                rank, found = self._candidate_rank(
                    grounded.candidate_node_ids[index], int(commands.parameter_node_ids[index, 1])
                )
                drag_destination[index] = rank
                drag_mask[index] &= found
            if commands.parameter_targets.shape[1] >= 7:
                drag_source_point[index] = commands.parameter_targets[index, 2:4]
                drag_destination_point[index] = commands.parameter_targets[index, 4:6]
                drag_duration[index] = commands.parameter_targets[index, 6:7]
                drag_mask[index] &= bool(commands.parameter_mask[index, 2:6].all())
                drag_duration_mask[index] = drag_mask[index] and bool(
                    commands.parameter_mask[index, 6]
                )
        return GroundingSupervision(
            candidate_indices,
            candidate_mask,
            point_local,
            point_mask,
            commands.action_types,
            commands.action_mask,
            long_point,
            long_mask,
            scroll_indices,
            scroll_mask,
            commands.parameter_targets[:, :2],
            drag_source,
            drag_destination,
            drag_mask,
            drag_source_point,
            drag_destination_point,
            drag_duration,
            drag_duration_mask,
            commands.target_boxes,
            commands.target_box_mask,
        )

    def forward(
        self,
        batch: Screen2ActionBatch,
        *,
        stage: str = "stage3_joint",
        step: int = 0,
        generator: torch.Generator | None = None,
    ) -> Screen2ActionModelOutput:
        """Run a declared stage and return predictions plus typed loss inputs."""

        allowed_stages = {
            "stage1_perception_graph",
            "stage2_grounding",
            "stage3_joint",
            "inference",
        }
        if stage not in allowed_stages:
            raise ValueError(f"unknown model stage: {stage}")
        frames = (
            tuple(batch.perceived_frames)
            if batch.perceived_frames
            else self.perceive(
                batch.images,
                mode=batch.perception_mode,
                oracle_screens=batch.oracle_screens,
            )
        )
        stochastic = self.training and stage != "inference"
        encoded = self.encode_frames(
            frames,
            screenshots=batch.images,
            stochastic_selector=stochastic,
            generator=generator,
        )
        grounded = self.ground_commands(encoded, batch.commands)
        target_retention, reference_retention = self._retention_supervision(encoded, batch.commands)
        retention_losses = retention_loss(
            encoded.retention,
            target_mask=target_retention,
            reference_mask=reference_retention,
            token_costs=encoded.retention_token_costs,
            budgets=encoded.budgets,
        )
        grounding_supervision = self._grounding_supervision(grounded, batch.commands)
        grounding_losses = grounding_loss(
            grounded.tensors,
            grounding_supervision,
            candidate_boxes=grounded.expanded_crop_boxes,
            confidence_config=self.confidence_config,
            step=step,
        )
        total = grounding_losses.total + retention_losses.total
        return Screen2ActionModelOutput(
            frames,
            encoded,
            grounded,
            ModelLossInputs(
                grounding_supervision,
                target_retention,
                reference_retention,
            ),
            retention_losses,
            grounding_losses,
            total,
            stage,
        )
