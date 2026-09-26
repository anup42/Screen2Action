"""Discoverable command-line interface for Screen2Action workflows."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import yaml

from screen2action.config import load_config, write_resolved_config
from screen2action.doctor import doctor_report
from screen2action.handoff import repository_root, write_handoff_snapshot

REQUIRED_COMMAND_PATHS = (
    "doctor",
    "config validate",
    "models list",
    "models resolve-lock",
    "models fetch",
    "models verify",
    "models probe-mobilevit",
    "data sources",
    "data download",
    "data register-local",
    "data inspect",
    "data normalize",
    "data validate",
    "data split",
    "data dedup",
    "data annotate-references",
    "data build-manifest",
    "data build-shards",
    "data stats",
    "taxonomy icons inspect",
    "taxonomy icons build",
    "taxonomy icons freeze",
    "perception precompute",
    "perception validate",
    "perception stats",
    "train smoke",
    "train probe-batch",
    "train run",
    "evaluate",
    "calibrate",
    "export",
    "handoff snapshot",
)


def _add_json(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="emit structured JSON")


def _add_dry_run(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and print a deterministic plan without writes or downloads",
    )


def _add_config(
    parser: argparse.ArgumentParser,
    *,
    default: str | None = None,
    run_directory: bool = False,
) -> None:
    parser.add_argument("--config", type=Path, default=Path(default) if default else None)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="repeatable dotted configuration override",
    )
    if run_directory:
        parser.add_argument("--run-dir", type=Path)


def _leaf(
    subparsers: Any,
    name: str,
    help_text: str,
    command: str,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(name, help=help_text, description=help_text)
    parser.set_defaults(command_path=command)
    return cast(argparse.ArgumentParser, parser)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="screen2action", description=__doc__)
    groups = parser.add_subparsers(dest="group", required=True)

    doctor = _leaf(groups, "doctor", "diagnose CPU or CUDA readiness", "doctor")
    doctor.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    _add_json(doctor)

    config = groups.add_parser("config", help="validate and resolve configuration")
    config_commands = config.add_subparsers(dest="config_command", required=True)
    validate = _leaf(config_commands, "validate", "validate composed YAML", "config validate")
    _add_config(validate, default="configs/model/tiny_cpu.yaml", run_directory=True)
    _add_json(validate)

    models = groups.add_parser("models", help="manage reproducible public model assets")
    model_commands = models.add_subparsers(dest="models_command", required=True)
    model_list = _leaf(model_commands, "list", "list logical model roles", "models list")
    model_list.add_argument("--registry", type=Path, default=Path("configs/models/registry.yaml"))
    _add_json(model_list)
    resolve_lock = _leaf(
        model_commands,
        "resolve-lock",
        "resolve registry intent to an immutable lock",
        "models resolve-lock",
    )
    resolve_lock.add_argument("--registry", type=Path, default=Path("configs/models/registry.yaml"))
    resolve_lock.add_argument("--lock", type=Path, default=Path("configs/models/lock.json"))
    _add_dry_run(resolve_lock)
    _add_json(resolve_lock)
    fetch = _leaf(model_commands, "fetch", "fetch locked model files", "models fetch")
    fetch.add_argument("--lock", type=Path, default=Path("configs/models/lock.json"))
    fetch.add_argument("--role", action="append", default=[])
    fetch.add_argument("--accept-license", action="append", default=[])
    fetch.add_argument("--cache-root", type=Path)
    _add_dry_run(fetch)
    _add_json(fetch)
    verify = _leaf(model_commands, "verify", "verify locked assets offline", "models verify")
    verify.add_argument("--lock", type=Path, default=Path("configs/models/lock.json"))
    verify.add_argument("--registry", type=Path, default=Path("configs/models/registry.yaml"))
    verify.add_argument("--cache-root", type=Path)
    _add_json(verify)
    probe_mobilevit = _leaf(
        model_commands,
        "probe-mobilevit",
        "probe the locked MobileViT-S crop-token shape contract",
        "models probe-mobilevit",
    )
    probe_mobilevit.add_argument("--weight", type=Path, required=True)
    probe_mobilevit.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    probe_mobilevit.add_argument("--batch-size", type=int, default=1)
    _add_dry_run(probe_mobilevit)
    _add_json(probe_mobilevit)

    data = groups.add_parser("data", help="download, normalize, audit, and freeze data")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    sources = _leaf(data_commands, "sources", "list registered public sources", "data sources")
    sources.add_argument("--registry", type=Path, default=Path("configs/data/sources.yaml"))
    _add_json(sources)
    download = _leaf(data_commands, "download", "download a licensed source", "data download")
    download.add_argument("source")
    download.add_argument("--revision")
    download.add_argument("--sample", type=int)
    download.add_argument("--accept-license", action="store_true")
    download.add_argument("--data-root", type=Path)
    download.add_argument("--registry", type=Path, default=Path("configs/data/sources.yaml"))
    _add_dry_run(download)
    _add_json(download)
    register = _leaf(
        data_commands,
        "register-local",
        "register immutable pre-downloaded source files",
        "data register-local",
    )
    register.add_argument("source")
    register.add_argument("path", type=Path)
    register.add_argument("--revision", required=True)
    register.add_argument("--accept-license", action="store_true")
    register.add_argument("--data-root", type=Path)
    register.add_argument("--registry", type=Path, default=Path("configs/data/sources.yaml"))
    _add_dry_run(register)
    _add_json(register)
    inspect = _leaf(data_commands, "inspect", "inspect source or manifest metadata", "data inspect")
    inspect.add_argument("path", type=Path)
    _add_json(inspect)
    normalize = _leaf(
        data_commands, "normalize", "normalize one registered source", "data normalize"
    )
    normalize.add_argument("source")
    normalize.add_argument("--revision", required=True)
    normalize.add_argument("--dataset-version", required=True)
    normalize.add_argument("--data-root", type=Path)
    normalize.add_argument("--registry", type=Path, default=Path("configs/data/sources.yaml"))
    _add_config(normalize)
    _add_dry_run(normalize)
    _add_json(normalize)
    data_validate = _leaf(
        data_commands,
        "validate",
        "validate canonical data or a frozen manifest",
        "data validate",
    )
    data_validate.add_argument("path", type=Path)
    _add_json(data_validate)
    split = _leaf(data_commands, "split", "create app-disjoint splits", "data split")
    split.add_argument("--dataset", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--seed", type=int, default=7)
    split.add_argument("--alias-map", type=Path)
    split.add_argument("--train-ratio", type=float, default=0.8)
    split.add_argument("--val-ratio", type=float, default=0.1)
    split.add_argument("--test-ratio", type=float, default=0.1)
    _add_dry_run(split)
    _add_json(split)
    dedup = _leaf(data_commands, "dedup", "group and remove cross-split duplicates", "data dedup")
    dedup.add_argument("--dataset", type=Path, required=True)
    dedup.add_argument("--output", type=Path, required=True)
    dedup.add_argument("--split-plan", type=Path, required=True)
    dedup.add_argument("--embeddings", type=Path)
    _add_config(dedup, default="configs/data/dedup.yaml")
    _add_dry_run(dedup)
    _add_json(dedup)
    references = _leaf(
        data_commands,
        "annotate-references",
        "generate masked public weak references",
        "data annotate-references",
    )
    references.add_argument("--dataset", type=Path, required=True)
    references.add_argument("--output", type=Path, required=True)
    references.add_argument("--minimum-confidence", type=float, default=0.8)
    _add_dry_run(references)
    _add_json(references)
    manifest = _leaf(
        data_commands,
        "build-manifest",
        "freeze data provenance and file digests",
        "data build-manifest",
    )
    manifest.add_argument("--dataset", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--split-plan", type=Path, required=True)
    manifest.add_argument("--dedup-audit", type=Path, required=True)
    manifest.add_argument("--taxonomy", type=Path, required=True)
    manifest.add_argument("--preprocessing-policy", default="canonical_rgb_png_v1")
    _add_dry_run(manifest)
    _add_json(manifest)
    shards = _leaf(
        data_commands,
        "build-shards",
        "build optional WebDataset-style shards",
        "data build-shards",
    )
    shards.add_argument("--manifest", type=Path, required=True)
    shards.add_argument("--output", type=Path, required=True)
    shards.add_argument("--max-samples", type=int, default=2048)
    _add_dry_run(shards)
    _add_json(shards)
    stats = _leaf(data_commands, "stats", "summarize canonical data", "data stats")
    stats.add_argument("path", type=Path)
    _add_json(stats)

    taxonomy = groups.add_parser("taxonomy", help="inspect and freeze label taxonomies")
    taxonomy_groups = taxonomy.add_subparsers(dest="taxonomy_group", required=True)
    icons = taxonomy_groups.add_parser("icons", help="manage the 87-class icon taxonomy")
    icon_commands = icons.add_subparsers(dest="icons_command", required=True)
    icon_inspect = _leaf(
        icon_commands,
        "inspect",
        "inspect taxonomy status and review evidence",
        "taxonomy icons inspect",
    )
    icon_inspect.add_argument(
        "--taxonomy", type=Path, default=Path("configs/data/icon_classes.yaml")
    )
    icon_inspect.add_argument(
        "--input",
        type=Path,
        action="append",
        default=[],
        help="repeatable licensed label-count JSON/YAML/CSV input",
    )
    _add_json(icon_inspect)
    icon_build = _leaf(
        icon_commands,
        "build",
        "build a deterministic candidate taxonomy",
        "taxonomy icons build",
    )
    icon_build.add_argument("--input", type=Path, action="append", required=True)
    icon_build.add_argument("--output", type=Path, required=True)
    icon_build.add_argument("--count", type=int, default=87)
    _add_dry_run(icon_build)
    _add_json(icon_build)
    icon_freeze = _leaf(
        icon_commands,
        "freeze",
        "freeze an immutable taxonomy manifest",
        "taxonomy icons freeze",
    )
    icon_freeze.add_argument("--input", type=Path, required=True)
    icon_freeze.add_argument("--output", type=Path, required=True)
    icon_freeze.add_argument("--count", type=int, default=87)
    icon_freeze.add_argument("--accept-reconstruction", action="store_true")
    icon_freeze.add_argument("--reviewer", default="cli-operator")
    _add_dry_run(icon_freeze)
    _add_json(icon_freeze)

    perception = groups.add_parser("perception", help="run command-independent perception")
    perception_commands = perception.add_subparsers(dest="perception_command", required=True)
    precompute = _leaf(
        perception_commands,
        "precompute",
        "precompute a resumable content-addressed cache",
        "perception precompute",
    )
    precompute.add_argument("--manifest", type=Path, required=True)
    precompute.add_argument("--model-lock", type=Path, default=Path("configs/models/lock.json"))
    precompute.add_argument("--cache-root", type=Path)
    precompute.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    precompute.add_argument("--shard-index", type=int)
    precompute.add_argument("--shard-count", type=int)
    precompute.add_argument("--batch-size", type=int, default=4)
    precompute.add_argument("--max-retries", type=int, default=3)
    precompute.add_argument("--max-screens", type=int)
    precompute.add_argument("--visual-checkpoint", type=Path)
    precompute.add_argument(
        "--allow-untrained-heads",
        action="store_true",
        help="allow deterministic random custom heads for smoke testing only",
    )
    _add_config(precompute, default="configs/perception/public_precompute.yaml")
    _add_dry_run(precompute)
    _add_json(precompute)
    perception_validate = _leaf(
        perception_commands,
        "validate",
        "validate cache manifests, shard status, and entries offline",
        "perception validate",
    )
    perception_validate.add_argument("manifest", type=Path)
    _add_json(perception_validate)
    perception_stats = _leaf(
        perception_commands,
        "stats",
        "report bounded cache/run/failure counts",
        "perception stats",
    )
    perception_stats.add_argument("manifest", type=Path)
    _add_json(perception_stats)

    train = groups.add_parser("train", help="probe, smoke-test, and run staged training")
    train_commands = train.add_subparsers(dest="train_command", required=True)
    smoke = _leaf(train_commands, "smoke", "run forward/backward/checkpoint/resume", "train smoke")
    smoke.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    smoke.add_argument("--steps", type=int, default=8)
    smoke.add_argument("--resume", type=Path)
    _add_config(smoke, default="configs/model/tiny_cpu.yaml", run_directory=True)
    _add_dry_run(smoke)
    _add_json(smoke)
    probe = _leaf(train_commands, "probe-batch", "measure one real batch", "train probe-batch")
    probe.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    probe.add_argument("--stage", default="stage2_selector_retrieval")
    probe.add_argument("--manifest", type=Path, required=True)
    probe.add_argument("--cache-manifest", type=Path)
    probe.add_argument("--model-lock", type=Path, default=Path("configs/models/lock.json"))
    probe.add_argument("--cache-root", type=Path)
    probe.add_argument("--max-batch-size", type=int, default=64)
    probe.add_argument("--max-commands", type=int)
    _add_config(probe, default="configs/experiments/tiny_cpu_e2e.yaml")
    _add_dry_run(probe)
    _add_json(probe)
    run = _leaf(train_commands, "run", "run or resume a configured stage", "train run")
    run.add_argument("--stage", required=True)
    run.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--cache-manifest", type=Path)
    run.add_argument("--model-lock", type=Path, default=Path("configs/models/lock.json"))
    run.add_argument("--cache-root", type=Path)
    run.add_argument("--max-commands", type=int)
    run.add_argument("--stop-after-optimizer-steps", type=int, help=argparse.SUPPRESS)
    run.add_argument("--resume", type=Path)
    run.add_argument("--init-checkpoint", type=Path)
    _add_config(run, default="configs/experiments/tiny_cpu_e2e.yaml", run_directory=True)
    _add_dry_run(run)
    _add_json(run)

    evaluate = _leaf(groups, "evaluate", "evaluate a frozen checkpoint", "evaluate")
    evaluate.add_argument("--checkpoint", type=Path)
    evaluate.add_argument("--manifest", type=Path)
    evaluate.add_argument("--cache-manifest", type=Path)
    evaluate.add_argument("--model-lock", type=Path, default=Path("configs/models/lock.json"))
    evaluate.add_argument("--cache-root", type=Path)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    evaluate.add_argument("--split", choices=("train", "val", "test"))
    evaluate.add_argument("--max-commands", type=int)
    evaluate.add_argument("--calibration", type=Path)
    evaluate.add_argument("--sweep", action="store_true")
    _add_config(evaluate, default="configs/eval/default.yaml")
    _add_dry_run(evaluate)
    _add_json(evaluate)

    calibrate = _leaf(groups, "calibrate", "fit validation-only calibration", "calibrate")
    calibrate.add_argument("--predictions", type=Path)
    calibrate.add_argument("--output", type=Path)
    calibrate.add_argument(
        "--method",
        choices=("both", "temperature", "threshold"),
        default="both",
        help="both linked artifacts are emitted; this records the primary consumer",
    )
    calibrate.add_argument("--target-risk", type=float, default=0.10)
    _add_dry_run(calibrate)
    _add_json(calibrate)

    export = _leaf(groups, "export", "export supported fixed-shape partitions", "export")
    export.add_argument("--checkpoint", type=Path)
    export.add_argument("--output", type=Path)
    export.add_argument(
        "--profile", choices=("accurate", "fast_roi", "tiny_cpu"), default="tiny_cpu"
    )
    export.add_argument("--model-lock", type=Path, default=Path("configs/models/lock.json"))
    export.add_argument("--cache-root", type=Path)
    export.add_argument(
        "--visual-checkpoint",
        type=Path,
        help="trained Stage-1 visual-model.pt or compatible stage checkpoint",
    )
    _add_config(export)
    _add_dry_run(export)
    _add_json(export)

    handoff = groups.add_parser("handoff", help="inspect or snapshot resumable state")
    handoff_commands = handoff.add_subparsers(dest="handoff_command", required=True)
    snapshot = _leaf(
        handoff_commands,
        "snapshot",
        "generate HANDOFF.md and handoff.json",
        "handoff snapshot",
    )
    snapshot.add_argument("--output-dir", type=Path)
    _add_json(snapshot)
    return parser


def _emit(payload: Mapping[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    for key, value in payload.items():
        if isinstance(value, dict | list):
            serialized = json.dumps(value, sort_keys=True)
        else:
            serialized = str(value)
        print(f"{key}: {serialized}")


def _portable_argument(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, list):
        return [_portable_argument(item) for item in value]
    return value


def _dry_run_payload(args: argparse.Namespace) -> dict[str, Any]:
    parameters = {
        key: _portable_argument(value)
        for key, value in sorted(vars(args).items())
        if not key.startswith("_")
        and key
        not in {
            "command_path",
            "group",
            "config_command",
            "models_command",
            "data_command",
            "taxonomy_group",
            "icons_command",
            "perception_command",
            "train_command",
            "handoff_command",
            "json",
            "dry_run",
        }
    }
    return {"command": args.command_path, "status": "dry_run", "parameters": parameters}


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"file does not exist: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"top-level YAML must be a mapping: {path}")
    return value


def _precompute_shard(args: argparse.Namespace) -> tuple[int, int]:
    """Resolve explicit sharding or torchrun RANK/WORLD_SIZE without ambiguity."""

    environment_count = int(os.environ.get("WORLD_SIZE", "1"))
    environment_index = int(os.environ.get("RANK", "0"))
    count = args.shard_count if args.shard_count is not None else environment_count
    index = args.shard_index if args.shard_index is not None else environment_index
    if environment_count > 1 and args.shard_count is not None and count != environment_count:
        raise ValueError("--shard-count must equal torchrun WORLD_SIZE")
    if environment_count > 1 and args.shard_index is not None and index != environment_index:
        raise ValueError("--shard-index must equal torchrun RANK")
    if count <= 0 or not 0 <= index < count:
        raise ValueError("perception shard index must be in [0, shard_count)")
    return index, count


def _run_read_only(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.command_path == "models list":
        registry = _load_yaml_mapping(args.registry)
        roles = registry.get("roles")
        if not isinstance(roles, dict):
            raise ValueError("model registry must contain a roles mapping")
        return {
            "schema_version": registry.get("schema_version"),
            "registry_id": registry.get("registry_id"),
            "roles": sorted(roles),
        }
    if args.command_path == "data sources":
        registry = _load_yaml_mapping(args.registry)
        sources = registry.get("sources")
        if not isinstance(sources, dict):
            raise ValueError("source registry must contain a sources mapping")
        return {
            "schema_version": registry.get("schema_version"),
            "registry_id": registry.get("registry_id"),
            "sources": sources,
        }
    if args.command_path == "taxonomy icons inspect":
        if args.input:
            from screen2action.perception.taxonomy import (
                inspect_icon_taxonomy,
                load_label_observations,
            )

            return inspect_icon_taxonomy(load_label_observations(args.input))
        taxonomy = _load_yaml_mapping(args.taxonomy)
        classes = taxonomy.get("classes", [])
        return {
            "schema_version": taxonomy.get("schema_version"),
            "status": taxonomy.get("status"),
            "expected_count": taxonomy.get("expected_count"),
            "observed_count": len(classes) if isinstance(classes, list) else None,
        }
    return None


def _run_command(args: argparse.Namespace) -> dict[str, Any]:
    root = repository_root()
    if args.command_path == "doctor":
        return doctor_report(root, device=args.device)
    if args.command_path == "config validate":
        config = load_config(args.config, overrides=args.overrides)
        snapshot_path = None
        if args.run_dir is not None:
            snapshot_path = write_resolved_config(config, args.run_dir).as_posix()
        return {
            "valid": True,
            "schema_version": config.schema_version,
            "sha256": config.sha256,
            "source_paths": list(config.source_paths),
            "overrides": list(config.overrides),
            "snapshot_path": snapshot_path,
        }
    if args.command_path == "handoff snapshot":
        markdown, machine, snapshot = write_handoff_snapshot(
            root,
            output_directory=args.output_dir,
        )
        if args.json:
            return snapshot
        return {
            "status": "written",
            "markdown": markdown.name,
            "machine": machine.name,
        }
    if args.command_path == "export":
        from screen2action.export.runner import run_partitioned_export
        from screen2action.model_assets import configured_model_cache_root

        default_configs = {
            "accurate": Path("configs/export/accurate.yaml"),
            "fast_roi": Path("configs/export/fast_roi.yaml"),
            "tiny_cpu": Path("configs/export/tiny_cpu.yaml"),
        }
        config = load_config(
            args.config or default_configs[args.profile],
            overrides=args.overrides,
        )
        if args.dry_run:
            payload = _dry_run_payload(args)
            payload["resolved_config_sha256"] = config.sha256
            payload["config_schema_version"] = config.schema_version
            payload["profile_output_subdirectory"] = args.profile
            return payload
        if args.checkpoint is None or args.output is None:
            raise ValueError("export requires --checkpoint and --output")
        return run_partitioned_export(
            config,
            checkpoint=args.checkpoint,
            output_directory=args.output,
            profile=args.profile,
            model_lock=args.model_lock,
            model_cache_root=configured_model_cache_root(root, args.cache_root),
            visual_checkpoint=args.visual_checkpoint,
        ).as_dict()
    if args.command_path == "evaluate" and not args.dry_run:
        from screen2action.eval.runner import run_evaluation
        from screen2action.eval.sweep_runner import run_evaluation_sweep
        from screen2action.model_assets import configured_model_cache_root

        missing = [
            name
            for name in ("checkpoint", "manifest", "cache_manifest", "output")
            if getattr(args, name) is None
        ]
        if missing:
            raise ValueError(
                "evaluate requires " + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            )
        config = load_config(args.config, overrides=args.overrides)
        evaluation_values = config.values.get("evaluation")
        configured_sweep = bool(
            isinstance(evaluation_values, dict) and evaluation_values.get("mode") == "sweep"
        )
        if args.sweep or configured_sweep:
            return run_evaluation_sweep(
                config,
                checkpoint=args.checkpoint,
                manifest=args.manifest,
                cache_manifest=args.cache_manifest,
                output_directory=args.output,
                model_lock=args.model_lock,
                model_cache_root=configured_model_cache_root(root, args.cache_root),
                device=args.device,
                split=args.split,
                max_commands=args.max_commands,
                calibration=args.calibration,
            )
        result = run_evaluation(
            config,
            checkpoint=args.checkpoint,
            manifest=args.manifest,
            cache_manifest=args.cache_manifest,
            output_directory=args.output,
            model_lock=args.model_lock,
            model_cache_root=configured_model_cache_root(root, args.cache_root),
            device=args.device,
            split=args.split,
            max_commands=args.max_commands,
            calibration=args.calibration,
        )
        return result.as_dict()
    if args.command_path == "calibrate" and not args.dry_run:
        from screen2action.eval.reporting import fit_validation_calibration_artifacts

        if args.predictions is None or args.output is None:
            raise ValueError("calibrate requires --predictions and --output")
        calibration_result = fit_validation_calibration_artifacts(
            args.predictions,
            args.output,
            target_risk=args.target_risk,
        )
        return {
            **calibration_result,
            "requested_method": args.method,
            "emitted_methods": ["temperature", "threshold"],
        }
    if args.command_path == "train smoke" and not args.dry_run:
        from screen2action.training.smoke import run_training_smoke

        config = load_config(args.config, overrides=args.overrides)
        if args.run_dir is not None:
            write_resolved_config(config, args.run_dir)
        smoke_result = run_training_smoke(
            device=args.device,
            steps=args.steps,
            run_directory=args.run_dir,
            config=config.values,
        )
        return dataclasses.asdict(smoke_result)
    if args.command_path == "train probe-batch" and not args.dry_run:
        from screen2action.model_assets import configured_model_cache_root
        from screen2action.training.runner import probe_batch_size

        config = load_config(args.config, overrides=args.overrides)
        probe_result = probe_batch_size(
            config,
            stage=args.stage,
            manifest=args.manifest,
            cache_manifest=args.cache_manifest,
            model_lock=args.model_lock,
            model_cache_root=configured_model_cache_root(root, args.cache_root),
            device=args.device,
            maximum_batch_size=args.max_batch_size,
            max_commands=args.max_commands,
        )
        return dataclasses.asdict(probe_result)
    if args.command_path == "train run" and not args.dry_run:
        from screen2action.model_assets import configured_model_cache_root
        from screen2action.training.runner import run_training_stage

        config = load_config(args.config, overrides=args.overrides)
        stage_result = run_training_stage(
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
            stop_after_optimizer_steps=args.stop_after_optimizer_steps,
        )
        return dataclasses.asdict(stage_result)
    if args.command_path == "models probe-mobilevit" and not args.dry_run:
        import torch

        from screen2action.models.mobilevit import MobileVitSCropEncoder

        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA shape probe requested but torch.cuda.is_available() is false")
        model = MobileVitSCropEncoder.from_locked_timm(args.weight).to(torch.device(args.device))
        return model.shape_probe(batch_size=args.batch_size, device=args.device)
    read_only = _run_read_only(args)
    if read_only is not None:
        return read_only
    if args.command_path == "models resolve-lock" and not args.dry_run:
        from screen2action.model_assets import resolve_registry_to_file

        lock = resolve_registry_to_file(args.registry, args.lock)
        return {
            "status": "resolved",
            "lock_path": args.lock.as_posix(),
            "lock_sha256": lock.digest,
            "roles": [model.role for model in lock.models],
            "complete_roles": [model.role for model in lock.models if model.complete],
        }
    if args.command_path == "models fetch" and not args.dry_run:
        from screen2action.model_assets import (
            configured_model_cache_root,
            fetch_locked_models,
        )

        configured_acceptance = [
            value.strip()
            for value in os.environ.get("SCREEN2ACTION_ACCEPTED_LICENSES", "").split(",")
            if value.strip()
        ]
        cache_root = configured_model_cache_root(root, args.cache_root)
        return fetch_locked_models(
            args.lock,
            cache_root,
            accepted_licenses=(*args.accept_license, *configured_acceptance),
            roles=args.role,
        )
    if args.command_path == "models verify":
        from screen2action.model_assets import (
            configured_model_cache_root,
            verify_locked_models,
        )

        cache_root = configured_model_cache_root(root, args.cache_root)
        return verify_locked_models(
            args.lock,
            cache_root,
            registry_path=args.registry,
        )
    if args.command_path == "perception validate":
        from screen2action.perception.precompute import validate_perception_cache

        return dataclasses.asdict(validate_perception_cache(args.manifest))
    if args.command_path == "perception stats":
        from screen2action.perception.precompute import perception_cache_stats

        return perception_cache_stats(args.manifest)
    if args.command_path == "perception precompute" and not args.dry_run:
        from screen2action.model_assets import configured_model_cache_root
        from screen2action.perception.factory import build_locked_perception_service
        from screen2action.perception.precompute import precompute_perception

        config = load_config(args.config, overrides=args.overrides)
        cache_root = configured_model_cache_root(root, args.cache_root)
        factory = build_locked_perception_service(
            model_lock=args.model_lock,
            model_cache_root=cache_root,
            cache_root=cache_root,
            config_values=config.values,
            requested_device=args.device,
            visual_checkpoint=args.visual_checkpoint,
            allow_untrained_heads=args.allow_untrained_heads,
        )
        factory.cache.write_control(
            "resolved-config.json",
            cast(Mapping[str, Any], config.snapshot()),
            immutable=True,
        )
        shard_index, shard_count = _precompute_shard(args)
        precompute_result = precompute_perception(
            args.manifest,
            service=factory.service,
            cache=factory.cache,
            shard_index=shard_index,
            shard_count=shard_count,
            batch_size=args.batch_size,
            max_retries=args.max_retries,
            max_screens=args.max_screens,
        )
        return {
            **dataclasses.asdict(precompute_result),
            "device": factory.device,
            "visual_checkpoint_status": factory.visual_checkpoint_status,
            "resolved_config_sha256": config.sha256,
        }
    if args.command_path == "data download" and not args.dry_run:
        from screen2action.data.assets import download_registered_source
        from screen2action.data.layout import DataLayout
        from screen2action.data.registry import load_source_registry

        registry = load_source_registry(args.registry)
        layout = DataLayout.configured(args.data_root, repository_root=root)
        manifest = download_registered_source(
            registry,
            layout,
            args.source,
            requested_revision=args.revision,
            sample_size=args.sample,
            accept_license=args.accept_license,
        )
        suffix = f"sample-{args.sample}" if args.sample is not None else "full"
        return {
            "status": "registered",
            "source": manifest.source,
            "revision": manifest.revision,
            "sample_size": manifest.sample_size,
            "license_acknowledgement_id": manifest.license_acknowledgement_id,
            "files": len(manifest.files),
            "bytes": sum(item.size for item in manifest.files),
            "manifest_digest": manifest.digest,
            "raw_path": layout.raw_revision(manifest.source, manifest.revision)
            .joinpath(suffix)
            .as_posix(),
        }
    if args.command_path == "data register-local" and not args.dry_run:
        from screen2action.data.assets import register_local_source
        from screen2action.data.layout import DataLayout
        from screen2action.data.registry import load_source_registry

        registry = load_source_registry(args.registry)
        layout = DataLayout.configured(args.data_root, repository_root=root)
        manifest = register_local_source(
            registry,
            layout,
            args.source,
            args.path,
            args.revision,
            accept_license=args.accept_license,
        )
        return {
            "status": "registered",
            "source": manifest.source,
            "revision": manifest.revision,
            "license_acknowledgement_id": manifest.license_acknowledgement_id,
            "files": len(manifest.files),
            "bytes": sum(item.size for item in manifest.files),
            "manifest_digest": manifest.digest,
            "raw_path": layout.raw_revision(manifest.source, manifest.revision).as_posix(),
        }
    if args.command_path == "data normalize" and not args.dry_run:
        from screen2action.data.layout import DataLayout
        from screen2action.data.registry import load_source_registry
        from screen2action.data.storage import normalize_registered_source

        if args.config is not None:
            load_config(args.config, overrides=args.overrides)
        registry = load_source_registry(args.registry)
        layout = DataLayout.configured(args.data_root, repository_root=root)
        return dataclasses.asdict(
            normalize_registered_source(
                registry,
                layout,
                args.source,
                args.revision,
                args.dataset_version,
            )
        )
    if args.command_path == "data validate":
        from screen2action.data.manifest import verify_data_manifest
        from screen2action.data.storage import validate_canonical_dataset

        if args.path.is_file():
            payload = json.loads(args.path.read_text(encoding="utf-8"))
            if "manifest_digest" not in payload:
                raise ValueError("JSON file is not a frozen data manifest")
            return dataclasses.asdict(verify_data_manifest(args.path))
        return validate_canonical_dataset(args.path)
    if args.command_path == "data split" and not args.dry_run:
        from screen2action.data.splits import create_app_disjoint_plan, write_split_plan
        from screen2action.data.storage import read_table_rows

        aliases: Mapping[str, str] = {}
        if args.alias_map is not None:
            loaded_aliases = _load_yaml_mapping(args.alias_map)
            aliases_value = loaded_aliases.get("aliases", loaded_aliases)
            if not isinstance(aliases_value, dict) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in aliases_value.items()
            ):
                raise ValueError("alias map must map raw app IDs to canonical app IDs")
            aliases = cast(dict[str, str], aliases_value)
        plan = create_app_disjoint_plan(
            read_table_rows(args.dataset, "screens"),
            seed=args.seed,
            ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
            alias_map=aliases,
        )
        write_split_plan(plan, args.output)
        return {
            "status": "written",
            "output": args.output.as_posix(),
            "policy_version": plan.policy_version,
            "screens": len(plan.assignments),
            "apps": len(plan.app_assignments),
            "alias_groups": len(plan.aliases),
            "collisions": len(plan.collisions),
        }
    if args.command_path == "data dedup" and not args.dry_run:
        from screen2action.data.dedup import deduplicate_canonical_dataset
        from screen2action.data.splits import load_split_assignments

        config = load_config(args.config, overrides=args.overrides)
        embeddings = None
        if args.embeddings is not None:
            raw_embeddings = json.loads(args.embeddings.read_text(encoding="utf-8"))
            if not isinstance(raw_embeddings, dict):
                raise ValueError("embedding file must map screen IDs to vectors")
            embeddings = {
                str(key): tuple(float(value) for value in values)
                for key, values in raw_embeddings.items()
            }
        values = config.values
        return deduplicate_canonical_dataset(
            args.dataset,
            load_split_assignments(args.split_plan),
            args.output,
            perceptual_hash_distance_threshold=int(
                float(str(values.get("perceptual_hash_distance_threshold", 4)))
            ),
            ocr_jaccard_threshold=float(str(values.get("ocr_jaccard_threshold", 0.90))),
            embedding_similarity_threshold=float(
                str(values.get("image_embedding_similarity_threshold", 0.98))
            ),
            embeddings=embeddings,
        )
    if args.command_path == "data annotate-references" and not args.dry_run:
        from screen2action.data.references import annotate_public_weak_references

        return annotate_public_weak_references(
            args.dataset,
            args.output,
            minimum_confidence=args.minimum_confidence,
        )
    if args.command_path == "data build-manifest" and not args.dry_run:
        from screen2action.data.manifest import build_data_manifest

        return build_data_manifest(
            args.dataset,
            args.output,
            split_plan=args.split_plan,
            dedup_audit=args.dedup_audit,
            taxonomy=args.taxonomy,
            preprocessing_policy=args.preprocessing_policy,
        )
    if args.command_path == "data build-shards" and not args.dry_run:
        from screen2action.data.shards import build_webdataset_shards

        return build_webdataset_shards(
            args.manifest,
            args.output,
            max_samples=args.max_samples,
        )
    if args.command_path == "data stats":
        from screen2action.data.storage import canonical_stats

        return canonical_stats(args.path)
    if args.command_path == "data inspect":
        from screen2action.data.assets import RAW_MANIFEST, verify_raw_source
        from screen2action.data.manifest import verify_data_manifest
        from screen2action.data.storage import DATASET_METADATA, canonical_stats

        if args.path.is_dir() and (args.path / RAW_MANIFEST).is_file():
            manifest = verify_raw_source(args.path)
            return {
                "kind": "raw_source",
                "source": manifest.source,
                "revision": manifest.revision,
                "files": len(manifest.files),
                "bytes": sum(item.size for item in manifest.files),
                "manifest_digest": manifest.digest,
            }
        if args.path.is_dir() and (args.path / DATASET_METADATA).is_file():
            return {"kind": "canonical_dataset", **canonical_stats(args.path)}
        if args.path.is_file():
            payload = json.loads(args.path.read_text(encoding="utf-8"))
            if "manifest_digest" in payload:
                return {
                    "kind": "data_manifest",
                    **dataclasses.asdict(verify_data_manifest(args.path)),
                }
            return {"kind": "json", "keys": sorted(payload) if isinstance(payload, dict) else []}
        raise ValueError(f"unrecognized data path: {args.path}")
    if args.command_path == "taxonomy icons build" and not args.dry_run:
        from screen2action.perception.taxonomy import (
            build_icon_taxonomy,
            inspect_icon_taxonomy,
            load_label_observations,
            write_taxonomy,
        )

        observations = load_label_observations(args.input)
        proposal = build_icon_taxonomy(observations, expected_count=args.count)
        write_taxonomy(proposal, args.output)
        review = inspect_icon_taxonomy(observations)
        review_path = args.output.with_suffix(args.output.suffix + ".review.json")
        review_path.write_text(
            json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return {
            "status": proposal["status"],
            "output": args.output.as_posix(),
            "review_report": review_path.as_posix(),
            "observed_count": proposal["observed_count"],
            "expected_count": proposal["expected_count"],
        }
    if args.command_path == "taxonomy icons freeze" and not args.dry_run:
        from screen2action.perception.taxonomy import freeze_icon_taxonomy, write_taxonomy

        proposal = _load_yaml_mapping(args.input)
        frozen = freeze_icon_taxonomy(
            proposal,
            expected_count=args.count,
            accept_reconstruction=args.accept_reconstruction,
            reviewer=args.reviewer,
        )
        write_taxonomy(frozen, args.output)
        return {
            "status": frozen["status"],
            "output": args.output.as_posix(),
            "count": frozen["count"],
            "digest": frozen["digest"],
        }
    if getattr(args, "dry_run", False):
        if getattr(args, "config", None) is not None:
            config = load_config(args.config, overrides=getattr(args, "overrides", ()))
            payload = _dry_run_payload(args)
            payload["resolved_config_sha256"] = config.sha256
            payload["config_schema_version"] = config.schema_version
            return payload
        return _dry_run_payload(args)
    message = (
        f"{args.command_path} requires its declared external inputs; "
        "inspect --help or use --dry-run"
    )
    raise RuntimeError(message)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the unified CLI and return a process exit code."""

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        payload = _run_command(args)
        _emit(payload, as_json=bool(args.json))
        if (
            args.command_path in {"doctor", "models verify", "perception validate", "export"}
            and payload.get("ok") is False
        ):
            return 1
        return 0
    except (FileNotFoundError, OSError, PermissionError, RuntimeError, ValueError) as error:
        payload = {"error": type(error).__name__, "message": str(error)}
        if bool(getattr(args, "json", False)):
            print(json.dumps(payload, sort_keys=True), file=sys.stderr)
        else:
            print(f"error[{payload['error']}]: {payload['message']}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
