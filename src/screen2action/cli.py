"""Discoverable command-line interface for Screen2Action workflows."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from screen2action.handoff import repository_root, write_handoff_snapshot


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="screen2action", description=__doc__)
    groups = parser.add_subparsers(dest="group", required=True)
    handoff = groups.add_parser("handoff", help="inspect or snapshot resumable state")
    handoff_commands = handoff.add_subparsers(dest="command", required=True)
    snapshot = handoff_commands.add_parser(
        "snapshot",
        help="generate HANDOFF.md and handoff.json",
    )
    snapshot.add_argument(
        "--output-dir",
        type=Path,
        help="output directory (defaults to repository root)",
    )
    snapshot.add_argument("--json", action="store_true", help="emit snapshot JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the unified CLI and return a process exit code."""

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.group == "handoff" and args.command == "snapshot":
            markdown, machine, snapshot = write_handoff_snapshot(
                repository_root(),
                output_directory=args.output_dir,
            )
            if args.json:
                print(json.dumps(snapshot, indent=2, sort_keys=True))
            else:
                print(f"wrote {markdown}")
                print(f"wrote {machine}")
            return 0
    except (OSError, ValueError) as error:
        payload = {"error": type(error).__name__, "message": str(error)}
        print(json.dumps(payload, sort_keys=True), file=sys.stderr)
        return 2
    parser.error("unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
