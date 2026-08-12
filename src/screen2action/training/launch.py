"""`torchrun -m screen2action.training.launch` stage-run entry point."""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path

from screen2action.config import load_config
from screen2action.handoff import repository_root
from screen2action.model_assets import configured_model_cache_root
from screen2action.training.runner import run_training_stage


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--model-lock", type=Path, default=Path("configs/models/lock.json"))
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/tiny_cpu_e2e.yaml")
    )
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--max-commands", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Launch one rank; only rank zero prints the aggregate result."""

    args = _parser().parse_args(argv)
    root = repository_root()
    config = load_config(args.config, overrides=args.overrides)
    result = run_training_stage(
        config,
        stage=args.stage,
        manifest=args.manifest,
        cache_manifest=args.cache_manifest,
        model_lock=args.model_lock,
        model_cache_root=configured_model_cache_root(root, args.cache_root),
        device=args.device,
        run_directory=args.run_dir,
        resume=args.resume,
        init_checkpoint=args.init_checkpoint,
        max_commands=args.max_commands,
    )
    if result.rank == 0:
        print(json.dumps(dataclasses.asdict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
