"""Deterministic optional WebDataset-style screen shards."""

from __future__ import annotations

import io
import json
import os
import tarfile
import uuid
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from screen2action.data.assets import sha256_file
from screen2action.data.manifest import verify_data_manifest
from screen2action.data.storage import read_table_rows


def _tar_add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mtime = 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    archive.addfile(info, io.BytesIO(payload))


def _chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def build_webdataset_shards(
    manifest_path: Path,
    output: Path,
    *,
    max_samples: int = 2048,
) -> dict[str, object]:
    """Write each unique screen image once with its elements and commands JSON."""

    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    verification = verify_data_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_root = Path(verification.dataset_root)
    membership_path = manifest_path.parent / manifest["split"]["membership_path"]
    membership = {
        row["screen_id"]: row
        for row in (
            json.loads(line)
            for line in membership_path.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    screens = {
        str(row["screen_id"]): row
        for row in read_table_rows(dataset_root, "screens")
        if str(row["screen_id"]) in membership
    }
    commands: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in read_table_rows(dataset_root, "commands"):
        if str(row["screen_id"]) in membership:
            commands[str(row["screen_id"])].append(row)
    elements: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in read_table_rows(dataset_root, "elements"):
        if str(row["screen_id"]) in membership:
            elements[str(row["screen_id"])].append(row)
    output.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, object]] = []
    for shard_index, screen_ids in enumerate(_chunks(sorted(screens), max_samples)):
        destination = output / f"shard-{shard_index:05d}.tar"
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial")
        with tarfile.open(temporary, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for screen_id in screen_ids:
                screen = screens[screen_id]
                key = str(screen["screen_sha256"])
                image_path = dataset_root / str(screen["image_path"])
                _tar_add_bytes(archive, f"{key}.png", image_path.read_bytes())
                sample = {
                    "screen": screen,
                    "split": membership[screen_id]["split"],
                    "elements": sorted(
                        elements.get(screen_id, ()), key=lambda row: str(row["element_id"])
                    ),
                    "commands": sorted(
                        commands.get(screen_id, ()), key=lambda row: str(row["command_id"])
                    ),
                }
                _tar_add_bytes(
                    archive,
                    f"{key}.json",
                    (json.dumps(sample, sort_keys=True) + "\n").encode("utf-8"),
                )
        os.replace(temporary, destination)
        shards.append(
            {
                "path": destination.name,
                "screens": len(screen_ids),
                "size": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        )
    payload = {
        "schema_version": 1,
        "format": "webdataset_screen_v1",
        "source_manifest_digest": manifest["manifest_digest"],
        "screen_count": len(screens),
        "max_samples": max_samples,
        "shards": shards,
    }
    index_path = output / "shards.json"
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if index_path.is_file() and index_path.read_text(encoding="utf-8") != serialized:
        raise FileExistsError(f"refusing to overwrite a different shard index: {index_path}")
    index_path.write_text(serialized, encoding="utf-8")
    return {**payload, "index_path": index_path.as_posix()}
