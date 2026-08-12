"""Root insertion, hierarchy depth, and deterministic containment ordering."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from screen2action.data.schema import (
    EdgeRecord,
    NodeRecord,
    NodeType,
    RelationDirection,
    RelationType,
)
from screen2action.ssb.relations import choose_containment_parents


@dataclass(frozen=True, slots=True)
class Hierarchy:
    """Validated rooted hierarchy and its depth-first serialization order."""

    nodes: tuple[NodeRecord, ...]
    root_id: int
    parent_by_child: Mapping[int, int]
    children_by_parent: Mapping[int, tuple[int, ...]]
    depth_by_node: Mapping[int, int]
    containment_edges: tuple[EdgeRecord, ...]
    depth_first_order: tuple[int, ...]


def insert_root(
    nodes: tuple[NodeRecord, ...] | list[NodeRecord],
) -> tuple[tuple[NodeRecord, ...], int]:
    """Return nodes with one full-screen mandatory root node."""

    node_list = list(nodes)
    node_map = {node.node_id: node for node in node_list}
    if len(node_map) != len(node_list):
        raise ValueError("node IDs must be unique")
    full_screen_roots = [
        node
        for node in node_list
        if node.node_type is NodeType.ROOT and node.box_xyxy_norm == (0.0, 0.0, 1.0, 1.0)
    ]
    if len(full_screen_roots) == 1:
        root = full_screen_roots[0]
        if not root.mandatory:
            node_map[root.node_id] = replace(root, mandatory=True)
        return tuple(sorted(node_map.values(), key=lambda node: node.node_id)), root.node_id
    if full_screen_roots:
        raise ValueError("more than one full-screen root node")
    root_id = max(node_map, default=-1) + 1
    root = NodeRecord(
        node_id=root_id,
        node_type=NodeType.ROOT,
        box_xyxy_norm=(0.0, 0.0, 1.0, 1.0),
        detector_confidence=1.0,
        mandatory=True,
        retention_score=1.0,
    )
    node_map[root_id] = root
    return tuple(sorted(node_map.values(), key=lambda node: node.node_id)), root_id


def _make_containment_edge(parent: NodeRecord, child: NodeRecord) -> EdgeRecord:
    return EdgeRecord(
        src=parent.node_id,
        dst=child.node_id,
        relation=RelationType.CONTAINMENT,
        direction=RelationDirection.CONTAINS,
    )


def build_hierarchy(
    nodes: tuple[NodeRecord, ...] | list[NodeRecord],
    *,
    min_child_area_coverage: float = 0.80,
) -> Hierarchy:
    """Insert a root, choose parents, compute depths, and return DFS order."""

    rooted_nodes, root_id = insert_root(nodes)
    node_map = {node.node_id: node for node in rooted_nodes}
    parent_by_child = choose_containment_parents(
        rooted_nodes, min_child_area_coverage=min_child_area_coverage
    )
    for node_id in node_map:
        if node_id != root_id and node_id not in parent_by_child:
            parent_by_child[node_id] = root_id
    parent_by_child.pop(root_id, None)

    children: dict[int, list[int]] = {node_id: [] for node_id in node_map}
    for child_id, parent_id in parent_by_child.items():
        if parent_id not in node_map:
            raise ValueError(f"parent {parent_id} for node {child_id} is missing")
        if child_id == parent_id:
            raise ValueError("hierarchy cannot contain self-parenting")
        children[parent_id].append(child_id)
    children_by_parent = {
        parent_id: tuple(
            sorted(
                child_ids,
                key=lambda child_id: (
                    node_map[child_id].box_xyxy_norm[1],
                    node_map[child_id].box_xyxy_norm[0],
                    child_id,
                ),
            )
        )
        for parent_id, child_ids in children.items()
    }

    depth_by_node: dict[int, int] = {}
    visiting: set[int] = set()

    def visit_depth(node_id: int, depth: int) -> None:
        if node_id in visiting:
            raise ValueError("containment graph contains a cycle")
        previous_depth = depth_by_node.get(node_id)
        if previous_depth is not None and previous_depth != depth:
            raise ValueError("node has inconsistent hierarchy depth")
        if previous_depth is not None:
            return
        visiting.add(node_id)
        depth_by_node[node_id] = depth
        for child_id in children_by_parent[node_id]:
            visit_depth(child_id, depth + 1)
        visiting.remove(node_id)

    visit_depth(root_id, 0)
    if len(depth_by_node) != len(node_map):
        missing = sorted(set(node_map) - set(depth_by_node))
        raise ValueError(f"nodes are disconnected from root: {missing}")

    updated_nodes = tuple(
        replace(
            node_map[node_id],
            hierarchy_depth=depth_by_node[node_id],
            parent_id=parent_by_child.get(node_id),
        )
        for node_id in sorted(node_map)
    )
    updated_map = {node.node_id: node for node in updated_nodes}
    depth_first: list[int] = []

    def visit_order(node_id: int) -> None:
        depth_first.append(node_id)
        for child_id in children_by_parent[node_id]:
            visit_order(child_id)

    visit_order(root_id)
    edges = tuple(
        _make_containment_edge(updated_map[parent_id], updated_map[child_id])
        for child_id, parent_id in sorted(parent_by_child.items())
    )
    return Hierarchy(
        nodes=updated_nodes,
        root_id=root_id,
        parent_by_child=dict(parent_by_child),
        children_by_parent=children_by_parent,
        depth_by_node=depth_by_node,
        containment_edges=edges,
        depth_first_order=tuple(depth_first),
    )
