"""Graph, serialization, and selection invariants."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from screen2action.data.schema import EdgeRecord, NodeRecord, RelationType
from screen2action.ssb.codec import SsbCodec, SsbEncoding


def validate_graph(nodes: Iterable[NodeRecord], edges: Iterable[EdgeRecord]) -> None:
    """Validate IDs, endpoints, containment acyclicity, and proximity bounds."""

    node_list = list(nodes)
    node_ids = {node.node_id for node in node_list}
    if len(node_ids) != len(node_list):
        raise ValueError("graph has duplicate node IDs")
    edge_list = list(edges)
    for edge in edge_list:
        if edge.src not in node_ids or edge.dst not in node_ids:
            raise ValueError(f"edge references missing endpoint: {edge}")
    parent_by_child: dict[int, int] = {}
    proximity_count: defaultdict[int, int] = defaultdict(int)
    for edge in edge_list:
        if edge.relation is RelationType.CONTAINMENT:
            previous = parent_by_child.setdefault(edge.dst, edge.src)
            if previous != edge.src:
                raise ValueError(f"node {edge.dst} has multiple containment parents")
        elif edge.relation is RelationType.PROXIMITY:
            proximity_count[edge.src] += 1
    too_many = {src: count for src, count in proximity_count.items() if count > 4}
    if too_many:
        raise ValueError(f"proximity bound exceeded: {too_many}")
    for child in parent_by_child:
        current = child
        visited: set[int] = set()
        while current in parent_by_child:
            if current in visited:
                raise ValueError("containment graph contains a cycle")
            visited.add(current)
            current = parent_by_child[current]


def remap_selected_edges(
    selected_node_ids: Iterable[int],
    edges: Iterable[EdgeRecord],
) -> tuple[EdgeRecord, ...]:
    """Drop references to omitted nodes so the selected graph has no dangling edge."""

    selected = set(selected_node_ids)
    return tuple(edge for edge in edges if edge.src in selected and edge.dst in selected)


def validate_encoding(encoding: SsbEncoding, codec: SsbCodec | None = None) -> None:
    """Decode an encoding and verify its declared exact cost."""

    decoder = codec or SsbCodec()
    decoded = decoder.decode(encoding.tokens)
    if decoded.tokens_consumed != encoding.total_cost:
        raise ValueError(
            f"decoded token count {decoded.tokens_consumed} differs from {encoding.total_cost}"
        )
    if tuple(encoding.node_indices) != tuple(encoding.ordered_node_ids):
        raise ValueError("node index map keys do not match encoded order")
