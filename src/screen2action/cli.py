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

import yaml  # type: ignore[import-untyped]

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
    _add_dry_run(split)
    _add_json(split)
    dedup = _leaf(data_commands, "dedup", "group and remove cross-split duplicates", "data dedup")
    dedup.add_argument("--dataset", type=Path, required=True)
    dedup.add_argument("--output", type=Path, required=True)
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
    _add_json(icon_inspect)
    icon_build = _leaf(
        icon_commands,
        "build",
        "build a deterministic candidate taxonomy",
        "taxonomy icons build",
    )
    icon_build.add_argument("--input", type=Path, required=True)
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
    precompute.add_argument("--shard-index", type=int, default=0)
    precompute.add_argument("--shard-count", type=int, default=1)
    _add_config(precompute)
    _add_dry_run(precompute)
    _add_json(precompute)

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
    probe.add_argument("--manifest", type=Path)
    probe.add_argument("--cache-manifest", type=Path)
    _add_config(probe, default="configs/model/tiny_cpu.yaml", run_directory=True)
    _add_dry_run(probe)
    _add_json(probe)
    run = _leaf(train_commands, "run", "run or resume a configured stage", "train run")
    run.add_argument("--stage", required=True)
    run.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--cache-manifest", type=Path)
    run.add_argument("--resume", type=Path)
    _add_config(run, run_directory=True)
    _add_dry_run(run)
    _add_json(run)

    evaluate = _leaf(groups, "evaluate", "evaluate a frozen checkpoint", "evaluate")
    evaluate.add_argument("--checkpoint", type=Path)
    evaluate.add_argument("--manifest", type=Path)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    _add_config(evaluate, default="configs/eval/default.yaml")
    _add_dry_run(evaluate)
    _add_json(evaluate)

    calibrate = _leaf(groups, "calibrate", "fit validation-only calibration", "calibrate")
    calibrate.add_argument("--predictions", type=Path)
    calibrate.add_argument("--output", type=Path)
    calibrate.add_argument("--method", choices=("temperature", "threshold"), default="temperature")
    _add_dry_run(calibrate)
    _add_json(calibrate)

    export = _leaf(groups, "export", "export supported fixed-shape partitions", "export")
    export.add_argument("--checkpoint", type=Path)
    export.add_argument("--output", type=Path)
    export.add_argument(
        "--profile", choices=("accurate", "fast_roi", "tiny_cpu"), default="tiny_cpu"
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
    if args.command_path == "train smoke" and not args.dry_run:
        from screen2action.training.smoke import run_training_smoke

        config = load_config(args.config, overrides=args.overrides)
        if args.run_dir is not None:
            write_resolved_config(config, args.run_dir)
        result = run_training_smoke(
            device=args.device,
            steps=args.steps,
            run_directory=args.run_dir,
            config=config.values,
        )
        return dataclasses.asdict(result)
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
        if args.command_path in {"doctor", "models verify"} and not payload["ok"]:
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
