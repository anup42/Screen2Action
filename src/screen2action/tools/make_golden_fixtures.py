"""Generate small deterministic graph fixtures without screenshots or datasets."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from screen2action.data.schema import NodeRecord, NodeType
from screen2action.ssb.codec import SsbCodec
from screen2action.ssb.hierarchy import build_hierarchy
from screen2action.ssb.relations import build_ordinal_edges, build_proximity_edges


def make_fixture() -> dict[str, object]:
    """Return a JSON-safe nested-card fixture used for manual inspection."""

    nodes = [
        NodeRecord(1, NodeType.CONTAINER, (0.08, 0.08, 0.92, 0.92), detector_confidence=0.95),
        NodeRecord(2, NodeType.TEXT, (0.16, 0.16, 0.42, 0.24), text_token_ids=(11, 12)),
        NodeRecord(3, NodeType.CONTROL, (0.52, 0.16, 0.84, 0.26), detector_confidence=0.9),
        NodeRecord(4, NodeType.CONTROL, (0.52, 0.34, 0.84, 0.44), detector_confidence=0.8),
    ]
    hierarchy = build_hierarchy(nodes)
    edges = list(hierarchy.containment_edges)
    edges.extend(build_proximity_edges(hierarchy.nodes, parent_by_child=hierarchy.parent_by_child))
    edges.extend(build_ordinal_edges(hierarchy.nodes, parent_by_child=hierarchy.parent_by_child))
    encoding = SsbCodec().encode(
        hierarchy.nodes,
        edges,
        node_order=hierarchy.depth_first_order,
    )
    return {
        "node_order": list(hierarchy.depth_first_order),
        "nodes": [
            {
                "node_id": node.node_id,
                "node_type": node.node_type.value,
                "box_xyxy_norm": list(node.box_xyxy_norm),
                "hierarchy_depth": node.hierarchy_depth,
                "parent_id": node.parent_id,
            }
            for node in hierarchy.nodes
        ],
        "edges": [
            {
                "src": edge.src,
                "dst": edge.dst,
                "relation": edge.relation.value,
                "direction": edge.direction.value,
                "axis": edge.axis.value if edge.axis else None,
            }
            for edge in edges
        ],
        "ssb_tokens": list(encoding.tokens),
        "ssb_cost": encoding.total_cost,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/golden/ssb_fixture.json"),
        help="destination JSON path",
    )
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(make_fixture(), indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
