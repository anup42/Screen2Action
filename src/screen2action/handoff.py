"""Generate secret-safe, machine-readable repository handoff snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from screen2action import (
    CONFIG_SCHEMA_VERSION,
    HANDOFF_SCHEMA_VERSION,
    PACKAGE_SCHEMA_VERSION,
    __version__,
)

ACTIVE_PLAN = Path("docs/plans/complete_pipeline.md")
ROOT_VARIABLES = {
    "data": "SCREEN2ACTION_DATA_ROOT",
    "cache": "SCREEN2ACTION_CACHE_ROOT",
    "runs": "SCREEN2ACTION_RUN_ROOT",
}
DEFAULT_EVALUATE_COMMAND = " ".join(
    (
        "screen2action evaluate",
        "--config configs/eval/default.yaml",
        "--device cpu --dry-run",
    )
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )


def _git_state(root: Path) -> dict[str, Any]:
    revision = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain", "--untracked-files=normal")
    branch = _git(root, "branch", "--show-current")
    ignored_generated = {"HANDOFF.md", "handoff.json"}
    dirty_entries = []
    if status.returncode == 0:
        for line in status.stdout.splitlines():
            candidate = line[3:].replace("\\", "/") if len(line) > 3 else line
            if candidate not in ignored_generated:
                dirty_entries.append(line)
    return {
        "commit": revision.stdout.strip() if revision.returncode == 0 else "unborn",
        "branch": branch.stdout.strip() if branch.returncode == 0 else "unknown",
        "dirty": bool(dirty_entries),
        "dirty_entry_count": len(dirty_entries),
        "generated_files_excluded_from_dirty_check": sorted(ignored_generated),
    }


def _configured_root(
    repository_root: Path,
    name: str,
    environment: Mapping[str, str],
) -> tuple[Path, str]:
    variable = ROOT_VARIABLES[name]
    raw = environment.get(variable)
    resolved = Path(raw).expanduser().resolve() if raw else (repository_root / name).resolve()
    return resolved, f"${{{variable}}}"


def _portable_path(path: Path, root: Path, label: str) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return label
    suffix = relative.as_posix()
    return label if suffix == "." else f"{label}/{suffix}"


def _discover_digest_files(
    root: Path,
    label: str,
    patterns: Sequence[str],
    *,
    limit: int = 100,
) -> list[dict[str, str]]:
    if not root.exists():
        return []
    found: dict[Path, None] = {}
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file():
                found[path.resolve()] = None
    rows = []
    for path in sorted(found, key=lambda item: item.as_posix())[:limit]:
        rows.append({"path": _portable_path(path, root, label), "sha256": _sha256(path)})
    return rows


def _plan_state(root: Path) -> tuple[list[str], str, list[str]]:
    path = root / ACTIVE_PLAN
    if not path.exists():
        return [], "unknown", []
    text = path.read_text(encoding="utf-8")
    active_match = re.search(r"\*\*Active milestone:\*\*\s*(.+)", text)
    active = active_match.group(1).strip() if active_match else "unknown"
    completed = [
        match.group(1).strip()
        for match in re.finditer(r"^###\s+(.+?)\s+\(COMPLETE\)\s*$", text, re.MULTILINE)
    ]
    successful_commands: list[str] = []
    for line in text.splitlines():
        if "exit 0" not in line.lower():
            continue
        successful_commands.extend(re.findall(r"`([^`]+)`", line))
    return completed, active, successful_commands[-10:]


def _open_questions(root: Path) -> list[str]:
    path = root / "docs" / "OPEN_QUESTIONS.md"
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    blocks = re.findall(
        r"^\d+\.\s+(.+?)(?=^\d+\.\s+|\n\n|\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    return [" ".join(block.split()) for block in blocks]


def _checkpoint_inventory(run_root: Path, label: str) -> dict[str, dict[str, str | None]]:
    if not run_root.exists():
        return {}
    grouped: dict[str, list[Path]] = {}
    for path in run_root.glob("**/*.pt"):
        lowered = path.as_posix().lower()
        stage_match = re.search(r"stage[_-]?[1-4][a-z0-9_-]*", lowered)
        stage = stage_match.group(0).replace("-", "_") if stage_match else "unclassified"
        grouped.setdefault(stage, []).append(path)
    result: dict[str, dict[str, str | None]] = {}
    for stage, paths in sorted(grouped.items()):
        latest = max(paths, key=lambda item: item.stat().st_mtime_ns)
        best_candidates = [path for path in paths if "best" in path.name.lower()]
        best = max(best_candidates, key=lambda item: item.stat().st_mtime_ns, default=None)
        result[stage] = {
            "latest": _portable_path(latest, run_root, label),
            "best": _portable_path(best, run_root, label) if best is not None else None,
        }
    return result


def _host_state(repository_root: Path, roots: Mapping[str, Path]) -> dict[str, Any]:
    disk = shutil.disk_usage(repository_root)
    writable: dict[str, bool] = {}
    for name, path in roots.items():
        probe = path if path.exists() else path.parent
        writable[name] = os.access(probe, os.W_OK)
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "repository_volume_free_bytes": disk.free,
        "configured_roots_writable": writable,
    }


def build_handoff_snapshot(
    repository_root: str | Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Inspect repository-local evidence and return a portable handoff document."""

    root = Path(repository_root).resolve()
    env = os.environ if environment is None else environment
    completed, active, successful_commands = _plan_state(root)
    data_root, data_label = _configured_root(root, "data", env)
    cache_root, cache_label = _configured_root(root, "cache", env)
    run_root, run_label = _configured_root(root, "runs", env)
    model_registry = root / "configs" / "models" / "registry.yaml"
    model_lock = root / "configs" / "models" / "lock.json"
    return {
        "handoff_schema_version": HANDOFF_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "git": _git_state(root),
        "versions": {
            "package": __version__,
            "package_schema": PACKAGE_SCHEMA_VERSION,
            "config_schema": CONFIG_SCHEMA_VERSION,
        },
        "plan": {
            "active_path": ACTIVE_PLAN.as_posix(),
            "active_milestone": active,
            "completed_milestones": completed,
        },
        "models": {
            "registry_path": "configs/models/registry.yaml",
            "registry_sha256": _sha256(model_registry) if model_registry.exists() else None,
            "lock_path": "configs/models/lock.json",
            "lock_sha256": _sha256(model_lock) if model_lock.exists() else None,
            "status": "resolved" if model_lock.exists() else "unresolved",
        },
        "data_manifests": _discover_digest_files(
            data_root,
            data_label,
            ("normalized/*/manifests/*.json", "normalized/*/manifest.json"),
        ),
        "perception_cache_manifests": _discover_digest_files(
            cache_root,
            cache_label,
            ("perception/**/manifest.json",),
        ),
        "checkpoints": _checkpoint_inventory(run_root, run_label),
        "commands": {
            "resume": "screen2action train smoke --device cpu",
            "evaluate": DEFAULT_EVALUATE_COMMAND,
            "refresh_handoff": "screen2action handoff snapshot",
            "last_successful": successful_commands,
        },
        "unresolved_decisions": _open_questions(root),
        "host": _host_state(
            root,
            {"data": data_root, "cache": cache_root, "runs": run_root},
        ),
    }


def render_handoff_markdown(snapshot: Mapping[str, Any]) -> str:
    """Render the compact human entry point from a machine snapshot."""

    git = snapshot["git"]
    plan = snapshot["plan"]
    models = snapshot["models"]
    commands = snapshot["commands"]
    completed = plan["completed_milestones"] or ["None yet"]
    questions = snapshot["unresolved_decisions"] or ["None recorded"]
    successful = commands["last_successful"] or ["None recorded in the active ExecPlan"]
    lines = [
        "# Screen2Action handoff",
        "",
        "> Generated by `screen2action handoff snapshot`. Do not edit manually.",
        "",
        "## Resume here",
        "",
        f"- Active plan: `{plan['active_path']}`",
        f"- Active milestone: {plan['active_milestone']}",
        f"- Git branch/commit: `{git['branch']}` / `{git['commit']}`",
        f"- Source dirty at snapshot: `{str(git['dirty']).lower()}` ",
        "  (`HANDOFF.md` and `handoff.json` are excluded from this check).",
        f"- Model lock: {models['status']} (`{models['lock_path']}`).",
        "",
        "Read `AGENTS.md` first, then this file's active plan and the ordered",
        "architecture/runbook documents listed in `AGENTS.md`.",
        "",
        "## Completed milestones",
        "",
        *[f"- {item}" for item in completed],
        "",
        "## Exact continuation commands",
        "",
        f"- Resume: `{commands['resume']}`",
        f"- Evaluate: `{commands['evaluate']}`",
        f"- Refresh this snapshot: `{commands['refresh_handoff']}`",
        "",
        "## Last successful commands",
        "",
        *[f"- `{item}`" for item in successful],
        "",
        "## External state",
        "",
        f"- Data manifests: {len(snapshot['data_manifests'])}",
        f"- Perception-cache manifests: {len(snapshot['perception_cache_manifests'])}",
        f"- Checkpoint stage groups: {len(snapshot['checkpoints'])}",
        "- Paths in `handoff.json` are repository-relative or use",
        "  `${SCREEN2ACTION_DATA_ROOT}`, `${SCREEN2ACTION_CACHE_ROOT}`, and",
        "  `${SCREEN2ACTION_RUN_ROOT}`. No credentials or user paths are emitted.",
        "",
        "## Known unresolved paper details",
        "",
        *[f"- {item}" for item in questions],
        "",
        "## Machine-readable state",
        "",
        "See `handoff.json` for digests, checkpoint inventory, schema versions,",
        "and secret-safe host diagnostics.",
        "",
    ]
    return "\n".join(lines)


def write_handoff_snapshot(
    repository_root: str | Path,
    *,
    output_directory: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Write `HANDOFF.md` and `handoff.json` for local handoff."""

    root = Path(repository_root).resolve()
    output = root if output_directory is None else Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshot = build_handoff_snapshot(root, environment=environment)
    markdown_path = output / "HANDOFF.md"
    json_path = output / "handoff.json"
    markdown_path.write_text(render_handoff_markdown(snapshot), encoding="utf-8")
    json_path.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return markdown_path, json_path, snapshot


def repository_root() -> Path:
    """Resolve the installed source tree's repository root."""

    candidate = Path(__file__).resolve().parents[2]
    return candidate.parent if candidate.name == "src" else candidate


def print_json(value: Mapping[str, Any]) -> None:
    """Write stable JSON to stdout."""

    json.dump(value, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
