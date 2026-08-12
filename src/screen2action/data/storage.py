"""Content-addressed images and versioned canonical Parquet tables."""

from __future__ import annotations

import hashlib
import io
import json
import os
import uuid
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from screen2action.data.adapters.base import AdapterReject, CanonicalExample
from screen2action.data.adapters.common import with_screen_digest
from screen2action.data.adapters.factory import iter_adapters, locate_raw_source
from screen2action.data.assets import sha256_file
from screen2action.data.layout import DataLayout
from screen2action.data.registry import SourceRegistry
from screen2action.data.schema import CANONICAL_SCHEMA_VERSION, RecordProvenance

DATASET_METADATA = "dataset.json"


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    source: str
    revision: str
    dataset_version: str
    screens: int
    elements: int
    commands: int
    rejected: int
    reused_images: int
    written_images: int
    partition_paths: Mapping[str, str]
    audit_path: str


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _provenance_columns(provenance: RecordProvenance) -> dict[str, object]:
    return {
        "schema_version": provenance.schema_version,
        "source_dataset": provenance.source_dataset,
        "source_item_id": provenance.source_item_id,
        "source_revision": provenance.source_revision,
        "source_license_ack_id": provenance.source_license_ack_id,
        "screen_sha256": provenance.screen_sha256,
        "image_format": provenance.image_format,
        "app_id_canonical": provenance.app_id_canonical,
        "app_id_raw": provenance.app_id_raw,
        "domain_id_canonical": provenance.domain_id_canonical,
        "domain_id_raw": provenance.domain_id_raw,
        "platform": provenance.platform,
        "split_origin": provenance.split_origin,
        "annotation_source": provenance.annotation_source,
        "annotation_confidence": provenance.annotation_confidence,
        "label_masks_json": _json(provenance.label_masks),
        "target_match_method": provenance.target_match_method,
        "reference_match_method": provenance.reference_match_method,
        "action_trace_id": provenance.action_trace_id,
        "action_step_id": provenance.action_step_id,
        "synthetic_parent_id": provenance.synthetic_parent_id,
        "transformation_provenance_json": _json(provenance.transformation_provenance),
    }


def screen_row(example: CanonicalExample) -> dict[str, object]:
    screen = example.screen
    return {
        **_provenance_columns(screen.provenance),
        "screen_id": screen.screen_id,
        "app_id": screen.app_id,
        "image_path": screen.image_path,
        "width": screen.width,
        "height": screen.height,
        "split": screen.split,
        "source": screen.source,
    }


def element_rows(example: CanonicalExample) -> Iterator[dict[str, object]]:
    for element in example.screen.elements:
        yield {
            **_provenance_columns(element.provenance),
            "screen_id": example.screen.screen_id,
            "element_id": element.element_id,
            "node_type": element.node_type.value,
            "x1": element.box_xyxy_norm[0],
            "y1": element.box_xyxy_norm[1],
            "x2": element.box_xyxy_norm[2],
            "y2": element.box_xyxy_norm[3],
            "text": element.text or "",
            "has_text": element.text is not None,
            "actionability_labels_json": _json(element.actionability_labels),
            "icon_class_id": element.icon_class_id if element.icon_class_id is not None else -1,
            "parent_element_id": element.parent_element_id or "",
            "metadata_json": _json(element.metadata),
            "element_annotation_source": element.annotation_source,
            "element_annotation_confidence": element.annotation_confidence,
            "element_label_masks_json": _json(element.label_masks),
        }


def command_rows(example: CanonicalExample) -> Iterator[dict[str, object]]:
    for command in example.commands:
        point = command.target_point_xy_norm
        yield {
            **_provenance_columns(command.provenance),
            "command_id": command.command_id,
            "screen_id": command.screen_id,
            "text": command.text,
            "action_type": command.action_type.value,
            "target_x1": command.target_box_xyxy_norm[0],
            "target_y1": command.target_box_xyxy_norm[1],
            "target_x2": command.target_box_xyxy_norm[2],
            "target_y2": command.target_box_xyxy_norm[3],
            "target_point_x": point[0] if point is not None else None,
            "target_point_y": point[1] if point is not None else None,
            "has_target_point": point is not None,
            "target_point_source": command.target_point_source.value,
            "action_parameters_json": _json(command.action_parameters),
            "relation_type": command.relation_type,
            "reference_element_ids_json": _json(command.reference_element_ids),
            "command_label_masks_json": _json(command.label_masks),
        }


def _schema(name: str) -> Any:
    try:
        import pyarrow as pa
    except ImportError as error:
        raise RuntimeError("canonical Parquet storage requires: pip install -e .[data]") from error
    provenance = [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("source_dataset", pa.string(), nullable=False),
        pa.field("source_item_id", pa.string(), nullable=False),
        pa.field("source_revision", pa.string(), nullable=False),
        pa.field("source_license_ack_id", pa.string(), nullable=False),
        pa.field("screen_sha256", pa.string(), nullable=False),
        pa.field("image_format", pa.string(), nullable=False),
        pa.field("app_id_canonical", pa.string(), nullable=False),
        pa.field("app_id_raw", pa.string(), nullable=False),
        pa.field("domain_id_canonical", pa.string(), nullable=False),
        pa.field("domain_id_raw", pa.string(), nullable=False),
        pa.field("platform", pa.string(), nullable=False),
        pa.field("split_origin", pa.string(), nullable=False),
        pa.field("annotation_source", pa.string(), nullable=False),
        pa.field("annotation_confidence", pa.float64()),
        pa.field("label_masks_json", pa.string(), nullable=False),
        pa.field("target_match_method", pa.string(), nullable=False),
        pa.field("reference_match_method", pa.string(), nullable=False),
        pa.field("action_trace_id", pa.string(), nullable=False),
        pa.field("action_step_id", pa.string(), nullable=False),
        pa.field("synthetic_parent_id", pa.string(), nullable=False),
        pa.field("transformation_provenance_json", pa.string(), nullable=False),
    ]
    if name == "screens":
        fields = provenance + [
            pa.field("screen_id", pa.string(), nullable=False),
            pa.field("app_id", pa.string(), nullable=False),
            pa.field("image_path", pa.string(), nullable=False),
            pa.field("width", pa.int64(), nullable=False),
            pa.field("height", pa.int64(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("source", pa.string(), nullable=False),
        ]
    elif name == "elements":
        fields = provenance + [
            pa.field("screen_id", pa.string(), nullable=False),
            pa.field("element_id", pa.string(), nullable=False),
            pa.field("node_type", pa.string(), nullable=False),
            *[pa.field(key, pa.float64(), nullable=False) for key in ("x1", "y1", "x2", "y2")],
            pa.field("text", pa.string(), nullable=False),
            pa.field("has_text", pa.bool_(), nullable=False),
            pa.field("actionability_labels_json", pa.string(), nullable=False),
            pa.field("icon_class_id", pa.int64(), nullable=False),
            pa.field("parent_element_id", pa.string(), nullable=False),
            pa.field("metadata_json", pa.string(), nullable=False),
            pa.field("element_annotation_source", pa.string(), nullable=False),
            pa.field("element_annotation_confidence", pa.float64()),
            pa.field("element_label_masks_json", pa.string(), nullable=False),
        ]
    elif name == "commands":
        fields = provenance + [
            pa.field("command_id", pa.string(), nullable=False),
            pa.field("screen_id", pa.string(), nullable=False),
            pa.field("text", pa.string(), nullable=False),
            pa.field("action_type", pa.string(), nullable=False),
            *[
                pa.field(key, pa.float64(), nullable=False)
                for key in ("target_x1", "target_y1", "target_x2", "target_y2")
            ],
            pa.field("target_point_x", pa.float64()),
            pa.field("target_point_y", pa.float64()),
            pa.field("has_target_point", pa.bool_(), nullable=False),
            pa.field("target_point_source", pa.string(), nullable=False),
            pa.field("action_parameters_json", pa.string(), nullable=False),
            pa.field("relation_type", pa.string(), nullable=False),
            pa.field("reference_element_ids_json", pa.string(), nullable=False),
            pa.field("command_label_masks_json", pa.string(), nullable=False),
        ]
    else:
        raise ValueError(f"unknown canonical table {name!r}")
    metadata = {
        b"screen2action_schema_version": CANONICAL_SCHEMA_VERSION.encode(),
        b"screen2action_table": name.encode(),
    }
    return pa.schema(fields, metadata=metadata)


class _PartitionWriter:
    def __init__(self, path: Path, name: str, *, batch_size: int = 512) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise RuntimeError(
                "canonical Parquet storage requires: pip install -e .[data]"
            ) from error
        self._pa = pa
        self._path = path
        self._temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
        self._temporary.parent.mkdir(parents=True, exist_ok=True)
        self._schema = _schema(name)
        self._writer = parquet.ParquetWriter(
            self._temporary,
            self._schema,
            compression="zstd",
            use_dictionary=True,
        )
        self._rows: list[Mapping[str, object]] = []
        self._batch_size = batch_size
        self._closed = False
        self.count = 0

    def append(self, row: Mapping[str, object]) -> None:
        self._rows.append(row)
        if len(self._rows) >= self._batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return
        table = self._pa.Table.from_pylist(self._rows, schema=self._schema)
        self._writer.write_table(table)
        self.count += len(self._rows)
        self._rows.clear()

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self._writer.close()
        self._closed = True
        os.replace(self._temporary, self._path)

    def abort(self) -> None:
        if not self._closed:
            with suppress(Exception):
                self._writer.close()
            self._closed = True
        self._temporary.unlink(missing_ok=True)


def _canonicalize_image(
    example: CanonicalExample,
    *,
    raw_root: Path,
    layout: DataLayout,
    dataset_version: str,
) -> tuple[CanonicalExample, bool]:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("image normalization requires: pip install -e .[data]") from error
    payload = example.image_bytes
    if payload is None:
        source_path = (raw_root / example.screen.image_path).resolve()
        if not source_path.is_relative_to(raw_root.resolve()):
            raise ValueError("source image path escapes the immutable raw root")
        if not source_path.is_file():
            raise FileNotFoundError(f"source image is missing: {example.screen.image_path}")
        payload = source_path.read_bytes()
    with Image.open(io.BytesIO(payload)) as image:
        if image.size != (example.screen.width, example.screen.height):
            raise ValueError(
                f"declared dimensions {(example.screen.width, example.screen.height)} "
                f"do not match image dimensions {image.size}"
            )
        normalized = image.convert("RGB")
        stream = io.BytesIO()
        normalized.save(stream, format="PNG", optimize=False, compress_level=9)
        canonical_bytes = stream.getvalue()
    digest = hashlib.sha256(canonical_bytes).hexdigest()
    destination = layout.image(dataset_version, digest)
    reused = destination.is_file()
    if reused:
        if sha256_file(destination) != digest:
            raise ValueError(f"content-addressed image digest mismatch: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial")
        temporary.write_bytes(canonical_bytes)
        os.replace(temporary, destination)
    relative = destination.relative_to(layout.normalized(dataset_version)).as_posix()
    return with_screen_digest(example, digest, "png", relative), reused


def _partition_path(root: Path, table: str, source: str, revision: str) -> Path:
    return root / table / f"part-{source}-{revision[:12]}.parquet"


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def normalize_registered_source(
    registry: SourceRegistry,
    layout: DataLayout,
    source_name: str,
    revision: str,
    dataset_version: str,
) -> NormalizationResult:
    """Stream one immutable raw source into content-addressed canonical tables."""

    source = registry.source(source_name)
    if not dataset_version.strip() or any(character in dataset_version for character in "\\/:"):
        raise ValueError("dataset_version must be a portable non-empty name")
    raw_root, raw_manifest = locate_raw_source(layout.raw_revision(source_name, revision))
    if raw_manifest.revision != revision:
        raise ValueError("registered raw source revision does not match requested revision")
    normalized_root = layout.normalized(dataset_version)
    partition_paths = {
        table: _partition_path(normalized_root, table, source_name, revision)
        for table in ("screens", "elements", "commands")
    }
    audit_path = layout.audits(dataset_version) / f"normalize-{source_name}-{revision[:12]}.json"
    if any(path.exists() for path in partition_paths.values()) or audit_path.exists():
        if not all(path.is_file() for path in partition_paths.values()) or not audit_path.is_file():
            raise ValueError("partial canonical partition exists; move it aside before retrying")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        for table, path in partition_paths.items():
            expected = audit["partition_digests"][table]
            if sha256_file(path) != expected:
                raise ValueError(f"existing canonical {table} partition digest mismatch")
        return NormalizationResult(**audit["result"])

    writers = {table: _PartitionWriter(path, table) for table, path in partition_paths.items()}
    rejection_counts: Counter[str] = Counter()
    rejects: list[dict[str, str]] = []
    action_counts: Counter[str] = Counter()
    platform_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    reused_images = 0
    written_images = 0
    try:
        for adapter in iter_adapters(
            source_name,
            raw_root,
            revision=revision,
            license_acknowledgement_id=raw_manifest.license_acknowledgement_id,
        ):
            for reject in adapter.rejects():
                rejection_counts[reject.reason_code] += 1
                rejects.append(asdict(reject))
            for example in adapter.examples():
                if source.evaluation_only and example.screen.split != "test":
                    raise ValueError("evaluation-only source emitted a non-test split")
                try:
                    example, reused = _canonicalize_image(
                        example,
                        raw_root=raw_root,
                        layout=layout,
                        dataset_version=dataset_version,
                    )
                except (FileNotFoundError, OSError, ValueError) as error:
                    rejection_counts["image_normalization_failed"] += 1
                    rejects.append(
                        asdict(
                            AdapterReject(
                                source_name,
                                example.screen.provenance.source_item_id,
                                "image_normalization_failed",
                                str(error),
                            )
                        )
                    )
                    continue
                reused_images += int(reused)
                written_images += int(not reused)
                writers["screens"].append(screen_row(example))
                for row in element_rows(example):
                    writers["elements"].append(row)
                for command, row in zip(example.commands, command_rows(example), strict=True):
                    writers["commands"].append(row)
                    action_counts[command.action_type.value] += 1
                platform_counts[example.screen.provenance.platform] += 1
                split_counts[example.screen.split] += 1
        for writer in writers.values():
            writer.close()
    except Exception:
        for writer in writers.values():
            writer.abort()
        raise
    if writers["screens"].count == 0:
        for path in partition_paths.values():
            path.unlink(missing_ok=True)
        raise ValueError(f"source {source_name!r} produced no canonical screens")
    result = NormalizationResult(
        source=source_name,
        revision=revision,
        dataset_version=dataset_version,
        screens=writers["screens"].count,
        elements=writers["elements"].count,
        commands=writers["commands"].count,
        rejected=sum(rejection_counts.values()),
        reused_images=reused_images,
        written_images=written_images,
        partition_paths={
            table: path.relative_to(normalized_root).as_posix()
            for table, path in partition_paths.items()
        },
        audit_path=audit_path.relative_to(normalized_root).as_posix(),
    )
    audit = {
        "schema_version": 1,
        "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
        "result": asdict(result),
        "raw_manifest_digest": raw_manifest.digest,
        "partition_digests": {table: sha256_file(path) for table, path in partition_paths.items()},
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "rejects": rejects,
        "action_counts": dict(sorted(action_counts.items())),
        "platform_counts": dict(sorted(platform_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "evaluation_only": source.evaluation_only,
    }
    _atomic_json(audit_path, audit)
    _update_dataset_metadata(normalized_root, dataset_version, audit_path, result)
    return result


def _update_dataset_metadata(
    normalized_root: Path,
    dataset_version: str,
    audit_path: Path,
    result: NormalizationResult,
) -> None:
    metadata_path = normalized_root / DATASET_METADATA
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        metadata = {
            "schema_version": 1,
            "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
            "dataset_version": dataset_version,
            "partitions": [],
        }
    key = (result.source, result.revision)
    partitions = [
        partition
        for partition in metadata["partitions"]
        if (partition["source"], partition["revision"]) != key
    ]
    partitions.append(
        {
            "source": result.source,
            "revision": result.revision,
            "paths": dict(result.partition_paths),
            "audit_path": audit_path.relative_to(normalized_root).as_posix(),
        }
    )
    metadata["partitions"] = sorted(partitions, key=lambda item: (item["source"], item["revision"]))
    _atomic_json(metadata_path, metadata)


def dataset_table_paths(dataset_root: Path, table: str) -> tuple[Path, ...]:
    if table not in {"screens", "elements", "commands", "references"}:
        raise ValueError(f"unknown canonical table {table!r}")
    return tuple(sorted((dataset_root / table).glob("*.parquet")))


def read_table_rows(dataset_root: Path, table: str) -> Iterator[dict[str, object]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("canonical Parquet reads require: pip install -e .[data]") from error
    for path in dataset_table_paths(dataset_root, table):
        file = parquet.ParquetFile(path)
        for batch in file.iter_batches(batch_size=1024):
            yield from batch.to_pylist()


def validate_canonical_dataset(dataset_root: Path) -> dict[str, object]:
    """Validate joins, geometry, image hashes, and evaluation leakage."""

    metadata_path = dataset_root / DATASET_METADATA
    if not metadata_path.is_file():
        raise FileNotFoundError(f"canonical dataset metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("canonical_schema_version") != CANONICAL_SCHEMA_VERSION:
        raise ValueError("canonical schema version mismatch")
    screens: dict[str, dict[str, object]] = {}
    apps: dict[str, set[str]] = {}
    for row in read_table_rows(dataset_root, "screens"):
        screen_id = str(row["screen_id"])
        if screen_id in screens:
            raise ValueError(f"duplicate canonical screen_id: {screen_id}")
        screens[screen_id] = row
        split = str(row["split"])
        apps.setdefault(str(row["app_id_canonical"]), set()).add(split)
        if row["source_dataset"] == "screenspot" and split != "test":
            raise ValueError("ScreenSpot leakage guard: evaluation image is outside test split")
        digest = str(row["screen_sha256"])
        image_path = dataset_root / str(row["image_path"])
        if not image_path.is_file() or sha256_file(image_path) != digest:
            raise ValueError(f"canonical image digest mismatch for screen {screen_id}")
    command_count = 0
    for row in read_table_rows(dataset_root, "commands"):
        command_count += 1
        if str(row["screen_id"]) not in screens:
            raise ValueError(f"command references unknown screen: {row['screen_id']}")
        box = tuple(
            float(str(row[key])) for key in ("target_x1", "target_y1", "target_x2", "target_y2")
        )
        if any(not 0.0 <= value <= 1.0 for value in box) or box[0] > box[2] or box[1] > box[3]:
            raise ValueError(f"command target is not normalized xyxy: {row['command_id']}")
    element_count = 0
    for row in read_table_rows(dataset_root, "elements"):
        element_count += 1
        if str(row["screen_id"]) not in screens:
            raise ValueError(f"element references unknown screen: {row['screen_id']}")
        box = tuple(float(str(row[key])) for key in ("x1", "y1", "x2", "y2"))
        if any(not 0.0 <= value <= 1.0 for value in box) or box[0] > box[2] or box[1] > box[3]:
            raise ValueError(f"element box is not normalized xyxy: {row['element_id']}")
    conflicts = {app: sorted(splits) for app, splits in apps.items() if len(splits) > 1}
    return {
        "ok": True,
        "dataset_version": metadata["dataset_version"],
        "screens": len(screens),
        "elements": element_count,
        "commands": command_count,
        "app_split_conflicts": conflicts,
    }


def canonical_stats(dataset_root: Path) -> dict[str, object]:
    screens = list(read_table_rows(dataset_root, "screens"))
    commands = list(read_table_rows(dataset_root, "commands"))
    return {
        "screens": len(screens),
        "commands": len(commands),
        "sources": dict(sorted(Counter(str(row["source_dataset"]) for row in screens).items())),
        "splits": dict(sorted(Counter(str(row["split"]) for row in screens).items())),
        "platforms": dict(sorted(Counter(str(row["platform"]) for row in screens).items())),
        "actions": dict(sorted(Counter(str(row["action_type"]) for row in commands).items())),
        "apps": len({str(row["app_id_canonical"]) for row in screens}),
    }
