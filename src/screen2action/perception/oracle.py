"""Ground-truth/cached perception adapter for CPU architecture tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch

from screen2action.data.schema import NodeRecord, ScreenRecord
from screen2action.data.tokenizer import VocabularyTokenizer


@dataclass(frozen=True, slots=True)
class OracleFrame:
    """Perception output that can be consumed by graph construction."""

    screenshot: torch.Tensor
    nodes: tuple[NodeRecord, ...]
    element_to_node: dict[str, int]


@dataclass(slots=True)
class OraclePerception:
    """Convert source annotations into realistic typed node records."""

    icon_class_count: int = 87
    detector_confidence: float = 0.99
    tokenizer: VocabularyTokenizer | None = None

    def perceive(self, screenshot: torch.Tensor, screen: ScreenRecord) -> OracleFrame:
        """Return deterministic oracle nodes without touching network/model weights."""

        if screenshot.ndim != 3 or screenshot.shape[0] != 3:
            raise ValueError("screenshot must have shape [3, height, width]")
        if screenshot.shape[-1] != screen.width or screenshot.shape[-2] != screen.height:
            raise ValueError("screenshot dimensions do not match ScreenRecord")
        nodes: list[NodeRecord] = []
        element_to_node: dict[str, int] = {}
        for node_id, element in enumerate(screen.elements):
            labels = element.actionability_labels
            actionability_logits = cast(
                tuple[float, float, float, float],
                tuple(
                    3.0 if label is True else -3.0 if label is False else 0.0 for label in labels
                ),
            )
            actionability_mask = cast(
                tuple[bool, bool, bool, bool], tuple(label is not None for label in labels)
            )
            icon_probabilities: tuple[float, ...] = ()
            if element.icon_class_id is not None:
                if element.icon_class_id >= self.icon_class_count:
                    raise ValueError("icon class exceeds configured oracle manifest")
                probabilities = [0.0] * self.icon_class_count
                probabilities[element.icon_class_id] = 1.0
                icon_probabilities = tuple(probabilities)
            text_token_ids: tuple[int, ...] = ()
            if element.text is not None and self.tokenizer is not None:
                text_token_ids = tuple(
                    token_id
                    for token_id in self.tokenizer.encode(element.text, max_length=16)
                    if token_id != self.tokenizer.pad_id
                )
            metadata = {key.casefold(): value.casefold() for key, value in element.metadata.items()}
            node = NodeRecord(
                node_id=node_id,
                node_type=element.node_type,
                box_xyxy_norm=element.box_xyxy_norm,
                detector_confidence=self.detector_confidence,
                text=element.text,
                text_token_ids=text_token_ids,
                ocr_confidence=self.detector_confidence if element.text is not None else 0.0,
                icon_probabilities=icon_probabilities,
                icon_confidence=self.detector_confidence if icon_probabilities else 0.0,
                actionability_logits=actionability_logits,
                actionability_mask=actionability_mask,
                is_scroll_container=metadata.get("is_scroll_container") == "true",
                retention_score=0.95 if any(label is True for label in labels) else 0.70,
            )
            nodes.append(node)
            element_to_node[element.element_id] = node_id
        return OracleFrame(
            screenshot=screenshot, nodes=tuple(nodes), element_to_node=element_to_node
        )
