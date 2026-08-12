"""Deterministic same-screen hard-negative mining."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from screen2action.data.schema import EdgeRecord, NodeRecord, RelationType
from screen2action.ssb.geometry import box_center


@dataclass(frozen=True, slots=True)
class HardNegative:
    """One negative category and its contrastive weight."""

    node_id: int
    category: str
    weight: float


def _command_words(command: str) -> set[str]:
    return set(re.findall(r"\w+", command.casefold()))


def mine_same_screen_negatives(
    command: str,
    target: NodeRecord,
    nodes: Iterable[NodeRecord],
    edges: Iterable[EdgeRecord] = (),
    *,
    max_per_category: int = 4,
    hard_weight: float = 2.0,
    minimum_negatives: int = 4,
) -> tuple[HardNegative, ...]:
    """Find text/icon/ordinal/geometry confusables without cross-screen data."""

    if max_per_category <= 0 or hard_weight <= 0.0 or minimum_negatives < 0:
        raise ValueError("negative-mining limits and weight must be positive")
    node_list = [node for node in nodes if node.node_id != target.node_id]
    command_words = _command_words(command)
    target_icon = (
        max(range(len(target.icon_probabilities)), key=target.icon_probabilities.__getitem__)
        if target.icon_probabilities
        else None
    )
    target_center = box_center(target.box_xyxy_norm)
    ordinal_neighbors = {
        edge.dst
        for edge in edges
        if edge.src == target.node_id and edge.relation is RelationType.ORDINAL
    } | {
        edge.src
        for edge in edges
        if edge.dst == target.node_id and edge.relation is RelationType.ORDINAL
    }
    categories: dict[str, list[NodeRecord]] = {
        "text": [],
        "icon": [],
        "ordinal": [],
        "geometry": [],
        "dimensions": [],
        "role": [],
    }
    target_width = target.box_xyxy_norm[2] - target.box_xyxy_norm[0]
    target_height = target.box_xyxy_norm[3] - target.box_xyxy_norm[1]
    for node in node_list:
        words = _command_words(node.text or "")
        if command_words & words:
            categories["text"].append(node)
        icon = (
            max(range(len(node.icon_probabilities)), key=node.icon_probabilities.__getitem__)
            if node.icon_probabilities
            else None
        )
        if target_icon is not None and icon == target_icon:
            categories["icon"].append(node)
        if node.node_id in ordinal_neighbors:
            categories["ordinal"].append(node)
        center = box_center(node.box_xyxy_norm)
        distance = (
            (center[0] - target_center[0]) ** 2 + (center[1] - target_center[1]) ** 2
        ) ** 0.5
        if distance <= 0.20:
            categories["geometry"].append(node)
        width = node.box_xyxy_norm[2] - node.box_xyxy_norm[0]
        height = node.box_xyxy_norm[3] - node.box_xyxy_norm[1]
        if abs(width - target_width) <= 0.05 and abs(height - target_height) <= 0.05:
            categories["dimensions"].append(node)
        if node.node_type is target.node_type:
            categories["role"].append(node)
    priority = ("text", "icon", "ordinal", "geometry", "dimensions", "role")
    category_ids = {
        category: {node.node_id for node in values} for category, values in categories.items()
    }
    chosen: dict[int, HardNegative] = {}
    category_counts = {category: 0 for category in priority}
    for category in priority:
        for node_id in sorted(category_ids[category]):
            if node_id in chosen or category_counts[category] >= max_per_category:
                continue
            chosen[node_id] = HardNegative(
                node_id,
                category,
                hard_weight if category in {"text", "icon", "ordinal"} else 1.0,
            )
            category_counts[category] += 1
    if len(chosen) < minimum_negatives:
        for node in sorted(node_list, key=lambda item: item.node_id):
            if node.node_id not in chosen:
                chosen[node.node_id] = HardNegative(node.node_id, "same_screen_fallback", 1.0)
            if len(chosen) >= minimum_negatives:
                break
    return tuple(chosen[node_id] for node_id in sorted(chosen))
