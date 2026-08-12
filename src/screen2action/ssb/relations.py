"""Deterministic containment, proximity, and ordinal graph construction."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping

from screen2action.data.schema import (
    EdgeRecord,
    NodeRecord,
    RelationAxis,
    RelationDirection,
    RelationType,
)
from screen2action.ssb.geometry import box_area, box_center, intersection_box, iou


def _nodes_by_id(nodes: Iterable[NodeRecord]) -> dict[int, NodeRecord]:
    result: dict[int, NodeRecord] = {}
    for node in nodes:
        if node.node_id in result:
            raise ValueError(f"duplicate node ID {node.node_id}")
        result[node.node_id] = node
    return result


def _coverage(container: NodeRecord, child: NodeRecord) -> float:
    child_area = box_area(child.box_xyxy_norm)
    if child_area == 0.0:
        return (
            1.0
            if intersection_box(container.box_xyxy_norm, child.box_xyxy_norm) == child.box_xyxy_norm
            else 0.0
        )
    overlap = intersection_box(container.box_xyxy_norm, child.box_xyxy_norm)
    if overlap[0] > overlap[2] or overlap[1] > overlap[3]:
        return 0.0
    return box_area(overlap) / child_area


def choose_containment_parents(
    nodes: Iterable[NodeRecord],
    *,
    min_child_area_coverage: float = 0.80,
) -> dict[int, int]:
    """Choose the smallest qualifying parent for every contained node.

    The returned map excludes nodes without a qualifying parent. A parent must
    cover at least `min_child_area_coverage` of the child's area and have a
    strictly larger area, except that callers may add a synthetic root later.
    """

    if not 0.0 < min_child_area_coverage <= 1.0:
        raise ValueError("min_child_area_coverage must be in (0, 1]")
    node_map = _nodes_by_id(nodes)
    parents: dict[int, int] = {}
    for child in sorted(node_map.values(), key=lambda item: item.node_id):
        candidates: list[tuple[float, int]] = []
        child_area = box_area(child.box_xyxy_norm)
        for parent in node_map.values():
            if parent.node_id == child.node_id:
                continue
            parent_area = box_area(parent.box_xyxy_norm)
            if parent_area <= child_area:
                continue
            if _coverage(parent, child) >= min_child_area_coverage:
                candidates.append((parent_area, parent.node_id))
        if candidates:
            parents[child.node_id] = min(candidates)[1]
    return parents


def _direction_for_delta(dx: float, dy: float) -> RelationDirection | None:
    if dx == 0.0 and dy == 0.0:
        return None
    if abs(dx) >= abs(dy):
        return RelationDirection.RIGHT if dx > 0.0 else RelationDirection.LEFT
    return RelationDirection.BELOW if dy > 0.0 else RelationDirection.ABOVE


def _relative_geometry(
    source: NodeRecord,
    destination: NodeRecord,
    direction: RelationDirection,
) -> tuple[float, ...]:
    """Return the configured 12-value reconstruction geometry feature."""

    source_center = box_center(source.box_xyxy_norm)
    destination_center = box_center(destination.box_xyxy_norm)
    dx = destination_center[0] - source_center[0]
    dy = destination_center[1] - source_center[1]
    distance = math.hypot(dx, dy)
    source_width = max(source.box_xyxy_norm[2] - source.box_xyxy_norm[0], 1e-6)
    source_height = max(source.box_xyxy_norm[3] - source.box_xyxy_norm[1], 1e-6)
    destination_width = max(destination.box_xyxy_norm[2] - destination.box_xyxy_norm[0], 1e-6)
    destination_height = max(destination.box_xyxy_norm[3] - destination.box_xyxy_norm[1], 1e-6)
    overlap = iou(source.box_xyxy_norm, destination.box_xyxy_norm)
    source_area = box_area(source.box_xyxy_norm)
    destination_area = box_area(destination.box_xyxy_norm)
    intersection = intersection_box(source.box_xyxy_norm, destination.box_xyxy_norm)
    overlap_area = (
        box_area(intersection)
        if intersection[0] <= intersection[2] and intersection[1] <= intersection[3]
        else 0.0
    )
    source_coverage = overlap_area / source_area if source_area else 0.0
    destination_coverage = overlap_area / destination_area if destination_area else 0.0
    one_hot = tuple(
        float(direction is candidate)
        for candidate in (
            RelationDirection.LEFT,
            RelationDirection.RIGHT,
            RelationDirection.ABOVE,
            RelationDirection.BELOW,
        )
    )
    return (
        dx,
        dy,
        distance,
        math.log(destination_width / source_width),
        math.log(destination_height / source_height),
        overlap,
        source_coverage,
        destination_coverage,
        *one_hot,
    )


def build_containment_edges(
    nodes: Iterable[NodeRecord],
    *,
    min_child_area_coverage: float = 0.80,
) -> tuple[EdgeRecord, ...]:
    """Build directed parent-to-child containment edges."""

    node_map = _nodes_by_id(nodes)
    parents = choose_containment_parents(
        node_map.values(), min_child_area_coverage=min_child_area_coverage
    )
    edges = []
    for child_id, parent_id in sorted(parents.items()):
        parent = node_map[parent_id]
        child = node_map[child_id]
        edges.append(
            EdgeRecord(
                src=parent_id,
                dst=child_id,
                relation=RelationType.CONTAINMENT,
                direction=RelationDirection.CONTAINS,
                relative_geometry=_relative_geometry(parent, child, RelationDirection.CONTAINS),
            )
        )
    return tuple(edges)


def _ancestor_pairs(parent_by_child: Mapping[int, int]) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for child in parent_by_child:
        current = child
        visited: set[int] = set()
        while current in parent_by_child:
            if current in visited:
                raise ValueError("containment parent map contains a cycle")
            visited.add(current)
            parent = parent_by_child[current]
            pairs.add((parent, child))
            current = parent
    return pairs


def build_proximity_edges(
    nodes: Iterable[NodeRecord],
    *,
    parent_by_child: Mapping[int, int] | None = None,
    max_neighbors_per_node: int = 4,
) -> tuple[EdgeRecord, ...]:
    """Keep at most one nearest non-ancestor neighbor per cardinal direction."""

    if max_neighbors_per_node != 4:
        raise ValueError("the deterministic SSB profile requires four cardinal neighbors")
    node_map = _nodes_by_id(nodes)
    parent_map = dict(parent_by_child or choose_containment_parents(node_map.values()))
    ancestors = _ancestor_pairs(parent_map)
    result: list[EdgeRecord] = []
    for source in sorted(node_map.values(), key=lambda item: item.node_id):
        by_direction: dict[RelationDirection, tuple[float, int, NodeRecord]] = {}
        source_center = box_center(source.box_xyxy_norm)
        for destination in sorted(node_map.values(), key=lambda item: item.node_id):
            if destination.node_id == source.node_id:
                continue
            if (source.node_id, destination.node_id) in ancestors or (
                destination.node_id,
                source.node_id,
            ) in ancestors:
                continue
            destination_center = box_center(destination.box_xyxy_norm)
            direction = _direction_for_delta(
                destination_center[0] - source_center[0],
                destination_center[1] - source_center[1],
            )
            if direction is None:
                continue
            distance = math.hypot(
                destination_center[0] - source_center[0],
                destination_center[1] - source_center[1],
            )
            candidate = (distance, destination.node_id, destination)
            current = by_direction.get(direction)
            if current is None or candidate[:2] < current[:2]:
                by_direction[direction] = candidate
        for direction in sorted(by_direction, key=lambda value: value.value):
            destination = by_direction[direction][2]
            result.append(
                EdgeRecord(
                    src=source.node_id,
                    dst=destination.node_id,
                    relation=RelationType.PROXIMITY,
                    direction=direction,
                    relative_geometry=_relative_geometry(source, destination, direction),
                )
            )
    return tuple(sorted(result, key=lambda edge: (edge.src, edge.direction.value, edge.dst)))


class _DisjointSet:
    def __init__(self, values: Iterable[int]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            parent = self.find(parent)
            self.parent[value] = parent
        return parent

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


def _interval_overlap_ratio(first: tuple[float, float], second: tuple[float, float]) -> float:
    overlap = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    denominator = min(first[1] - first[0], second[1] - second[0])
    if denominator <= 0.0:
        return 1.0 if first == second else 0.0
    return overlap / denominator


def _ordinal_groups(
    nodes: list[NodeRecord],
    *,
    axis: RelationAxis,
    min_overlap_ratio: float,
) -> list[list[NodeRecord]]:
    disjoint = _DisjointSet(node.node_id for node in nodes)
    for index, first in enumerate(nodes):
        first_interval = (
            (
                first.box_xyxy_norm[1],
                first.box_xyxy_norm[3],
            )
            if axis is RelationAxis.ROW
            else (
                first.box_xyxy_norm[0],
                first.box_xyxy_norm[2],
            )
        )
        for second in nodes[index + 1 :]:
            second_interval = (
                (
                    second.box_xyxy_norm[1],
                    second.box_xyxy_norm[3],
                )
                if axis is RelationAxis.ROW
                else (
                    second.box_xyxy_norm[0],
                    second.box_xyxy_norm[2],
                )
            )
            if _interval_overlap_ratio(first_interval, second_interval) >= min_overlap_ratio:
                disjoint.union(first.node_id, second.node_id)
    groups: dict[int, list[NodeRecord]] = defaultdict(list)
    for node in nodes:
        groups[disjoint.find(node.node_id)].append(node)
    return [
        sorted(group, key=lambda node: (box_center(node.box_xyxy_norm), node.node_id))
        for group in sorted(groups.values(), key=lambda group: min(node.node_id for node in group))
    ]


def build_ordinal_edges(
    nodes: Iterable[NodeRecord],
    *,
    min_overlap_ratio: float = 0.50,
    parent_by_child: Mapping[int, int] | None = None,
) -> tuple[EdgeRecord, ...]:
    """Build previous/next edges for overlap-aware row and column groups."""

    if not 0.0 < min_overlap_ratio <= 1.0:
        raise ValueError("min_overlap_ratio must be in (0, 1]")
    node_map = _nodes_by_id(nodes)
    parent_map = dict(parent_by_child or choose_containment_parents(node_map.values()))
    ancestors = _ancestor_pairs(parent_map)
    result: list[EdgeRecord] = []
    for axis in (RelationAxis.ROW, RelationAxis.COLUMN):
        groups = _ordinal_groups(
            sorted(node_map.values(), key=lambda node: node.node_id),
            axis=axis,
            min_overlap_ratio=min_overlap_ratio,
        )
        for group in groups:
            if len(group) < 2:
                continue
            if axis is RelationAxis.ROW:
                ordered = sorted(
                    group,
                    key=lambda node: (
                        box_center(node.box_xyxy_norm)[0],
                        box_center(node.box_xyxy_norm)[1],
                        node.node_id,
                    ),
                )
            else:
                ordered = sorted(
                    group,
                    key=lambda node: (
                        box_center(node.box_xyxy_norm)[1],
                        box_center(node.box_xyxy_norm)[0],
                        node.node_id,
                    ),
                )
            for previous, following in zip(ordered, ordered[1:], strict=False):
                if (previous.node_id, following.node_id) in ancestors or (
                    following.node_id,
                    previous.node_id,
                ) in ancestors:
                    continue
                result.extend(
                    (
                        EdgeRecord(
                            src=previous.node_id,
                            dst=following.node_id,
                            relation=RelationType.ORDINAL,
                            direction=RelationDirection.NEXT,
                            axis=axis,
                            relative_geometry=_relative_geometry(
                                previous, following, RelationDirection.NEXT
                            ),
                        ),
                        EdgeRecord(
                            src=following.node_id,
                            dst=previous.node_id,
                            relation=RelationType.ORDINAL,
                            direction=RelationDirection.PREVIOUS,
                            axis=axis,
                            relative_geometry=_relative_geometry(
                                following, previous, RelationDirection.PREVIOUS
                            ),
                        ),
                    )
                )
    return tuple(
        sorted(
            result,
            key=lambda edge: (
                edge.src,
                edge.relation.value,
                edge.axis.value if edge.axis else "",
                {RelationDirection.PREVIOUS: 0, RelationDirection.NEXT: 1}.get(edge.direction, 2),
                edge.dst,
            ),
        )
    )
