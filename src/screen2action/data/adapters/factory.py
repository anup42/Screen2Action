"""Adapter registry and streaming loaders for registered raw source layouts."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path

from screen2action.data.adapters.amex import AmexAdapter
from screen2action.data.adapters.android_control import (
    AndroidControlAdapter,
    iter_android_control_tfrecords,
)
from screen2action.data.adapters.common import MaterializedPublicAdapter, load_json_records
from screen2action.data.adapters.grounding import (
    RicoSemanticsAdapter,
    ScreenSpotAdapter,
    WaveUiAdapter,
)
from screen2action.data.adapters.guicourse import GuiActAdapter, GuiEnvAdapter
from screen2action.data.assets import RAW_MANIFEST, RawSourceManifest, verify_raw_source

ADAPTERS: Mapping[str, type[MaterializedPublicAdapter]] = {
    "guicourse_guiact": GuiActAdapter,
    "guicourse_guienv": GuiEnvAdapter,
    "amex": AmexAdapter,
    "android_control": AndroidControlAdapter,
    "wave_ui": WaveUiAdapter,
    "rico_semantics": RicoSemanticsAdapter,
    "screenspot": ScreenSpotAdapter,
}


def adapter_from_json(
    source: str,
    path: Path,
    *,
    revision: str,
    license_acknowledgement_id: str,
) -> MaterializedPublicAdapter:
    try:
        adapter_type = ADAPTERS[source]
    except KeyError as error:
        raise ValueError(f"no adapter registered for source {source!r}") from error
    return adapter_type.from_json(
        path,
        revision=revision,
        license_acknowledgement_id=license_acknowledgement_id,
    )


def locate_raw_source(raw_revision: Path) -> tuple[Path, RawSourceManifest]:
    """Locate one immutable full/sample registration under a revision root."""

    if (raw_revision / RAW_MANIFEST).is_file():
        return raw_revision, verify_raw_source(raw_revision)
    candidates = sorted(
        path.parent for path in raw_revision.glob(f"*/{RAW_MANIFEST}") if path.is_file()
    )
    full = raw_revision / "full"
    if full in candidates:
        return full, verify_raw_source(full)
    if len(candidates) != 1:
        raise ValueError(
            f"expected one registered raw source under {raw_revision}; found {len(candidates)}"
        )
    return candidates[0], verify_raw_source(candidates[0])


def _parquet_rows(path: Path, *, batch_size: int = 256) -> Iterator[dict[str, object]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("Parquet sources require: pip install -e .[data]") from error
    file = parquet.ParquetFile(path)
    for batch in file.iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def _json_rows(path: Path) -> Iterator[dict[str, object]]:
    for value in load_json_records(path):
        yield dict(value)


def _sample_rows(root: Path) -> Iterator[dict[str, object]]:
    sample = root / "sample.jsonl"
    if sample.is_file():
        yield from _json_rows(sample)


def _guicourse_rows(root: Path, *, env: bool) -> Iterator[dict[str, object]]:
    sample_rows = iter(_sample_rows(root))
    try:
        first_sample = next(sample_rows)
    except StopIteration:
        first_sample = None
    if first_sample is not None:
        yield first_sample
        yield from sample_rows
        return
    patterns = (
        ("ocr_grounding_*_data.json",)
        if env
        else (
            "web-single_*_data.json",
            "smartphone_*_data.json",
        )
    )
    for pattern in patterns:
        for data_path in sorted(root.rglob(pattern)):
            stem = data_path.name.removesuffix("_data.json")
            image_path = data_path.with_name(f"{stem}_images.parquet")
            if not image_path.is_file():
                raise FileNotFoundError(f"paired GUICourse image Parquet is missing: {image_path}")
            metadata = {
                str(row.get("image_id") or row.get("uid")): row for row in _json_rows(data_path)
            }
            subset = "guienv" if env else stem.rsplit("_", 1)[0].replace("-", "_")
            split = stem.rsplit("_", 1)[-1]
            seen: set[str] = set()
            for image in _parquet_rows(image_path, batch_size=32):
                image_id = str(
                    image.get("__index_level_0__") or image.get("image_id") or image.get("uid")
                )
                if image_id not in metadata:
                    continue
                row = dict(metadata[image_id])
                row.update(image)
                row["subset"] = subset
                row["split"] = split
                seen.add(image_id)
                yield row
            missing = sorted(set(metadata) - seen)
            if missing:
                raise ValueError(
                    f"{len(missing)} GUICourse annotations have no matching image; "
                    f"first={missing[0]}"
                )


def _wave_ui_rows(root: Path) -> Iterator[dict[str, object]]:
    sample_rows = iter(_sample_rows(root))
    try:
        first_sample = next(sample_rows)
    except StopIteration:
        first_sample = None
    if first_sample is not None:
        yield first_sample
        yield from sample_rows
        return
    for path in sorted(root.rglob("*.parquet")):
        split = path.name.split("-", 1)[0]
        for index, row in enumerate(_parquet_rows(path, batch_size=64)):
            row.setdefault("id", f"{path.stem}:{index}")
            row["split"] = split
            yield row


def _amex_rows(root: Path) -> Iterator[dict[str, object]]:
    instruction_paths = sorted(root.rglob("instruction_anno*.json"))
    if not instruction_paths:
        raise FileNotFoundError(
            "AMEX instruction JSON is unavailable; repair/extract the official multipart ZIP first"
        )
    element_paths = {
        path.stem.removeprefix("element_anno_"): path for path in root.rglob("element_anno*.json")
    }
    image_paths = {path.name: path for path in root.rglob("*.png")}
    for instruction_path in instruction_paths:
        for episode in _json_rows(instruction_path):
            enriched_steps: list[dict[str, object]] = []
            raw_steps = episode.get("steps", [])
            if not isinstance(raw_steps, list):
                raise ValueError("AMEX episode steps must be a list")
            for raw_step in raw_steps:
                if not isinstance(raw_step, Mapping):
                    enriched_steps.append({"invalid_step": str(raw_step)})
                    continue
                step = dict(raw_step)
                image_name = str(step.get("image_path") or "")
                element_path = element_paths.get(Path(image_name).stem)
                if element_path is not None:
                    annotation = json.loads(element_path.read_text(encoding="utf-8"))
                    step["clickable_elements"] = annotation.get("clickable_elements", [])
                    step["scrollable_elements"] = annotation.get("scrollable_elements", [])
                image_path = image_paths.get(Path(image_name).name)
                if image_path is not None:
                    step["image"] = image_path.read_bytes()
                enriched_steps.append(step)
            episode["steps"] = enriched_steps
            yield episode


def _load_split_mapping(root: Path) -> dict[str, str]:
    split_files = sorted(path for path in root.rglob("*.json") if "split" in path.name.casefold())
    if not split_files:
        raise FileNotFoundError("AndroidControl official split JSON is missing")
    payload = json.loads(split_files[0].read_text(encoding="utf-8"))
    result: dict[str, str] = {}
    if isinstance(payload, Mapping):
        if all(isinstance(value, str) for value in payload.values()):
            result = {str(key): str(value) for key, value in payload.items()}
        else:
            for split, ids in payload.items():
                if isinstance(ids, list):
                    result.update({str(identifier): str(split) for identifier in ids})
    if not result:
        raise ValueError("unsupported AndroidControl official split-file structure")
    return result


def _android_control_rows(root: Path) -> Iterator[Mapping[str, object]]:
    fixture_json = sorted(
        path for path in root.rglob("*.json") if "split" not in path.name.casefold()
    )
    if fixture_json:
        for path in fixture_json:
            yield from _json_rows(path)
        return
    tfrecords = sorted(root.rglob("*.tfrecord*"))
    if not tfrecords:
        raise FileNotFoundError("AndroidControl GZIP TFRecords are missing")
    yield from iter_android_control_tfrecords(
        tfrecords,
        official_splits=_load_split_mapping(root),
    )


def _generic_json_rows(root: Path) -> Iterator[dict[str, object]]:
    for path in sorted(root.rglob("*.json")):
        if path.name == RAW_MANIFEST:
            continue
        yield from _json_rows(path)


def _rico_rows(root: Path) -> Iterator[dict[str, object]]:
    app_metadata: dict[str, str] = {}
    image_paths = {path.stem: path.relative_to(root).as_posix() for path in root.rglob("*.png")}
    metadata_path = next(iter(sorted(root.rglob("ui-app-map.json"))), None)
    if metadata_path is not None:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("RICO ui-app-map.json must map screen IDs to package IDs")
        app_metadata = {str(key): str(value) for key, value in payload.items()}
    for path in sorted(root.rglob("*.json")):
        if path.name in {RAW_MANIFEST, "ui-app-map.json"}:
            continue
        split = path.stem if path.stem in {"train", "val", "test"} else "train"
        for row in _json_rows(path):
            row["split"] = split
            row["annotation_task"] = path.parent.name
            screen_id = str(row.get("screen_id") or "")
            if screen_id in app_metadata:
                row["app_id"] = app_metadata[screen_id]
            if screen_id in image_paths:
                row["image_path"] = image_paths[screen_id]
            yield row


def iter_source_records(source: str, root: Path) -> Iterator[Mapping[str, object]]:
    """Stream source rows from the official registered file layout."""

    if source == "guicourse_guiact":
        yield from _guicourse_rows(root, env=False)
    elif source == "guicourse_guienv":
        yield from _guicourse_rows(root, env=True)
    elif source == "amex":
        yield from _amex_rows(root)
    elif source == "android_control":
        yield from _android_control_rows(root)
    elif source == "wave_ui":
        yield from _wave_ui_rows(root)
    elif source == "rico_semantics":
        yield from _rico_rows(root)
    elif source == "screenspot":
        yield from _generic_json_rows(root)
    else:
        raise ValueError(f"no source loader registered for {source!r}")


def iter_adapters(
    source: str,
    root: Path,
    *,
    revision: str,
    license_acknowledgement_id: str,
) -> Iterator[MaterializedPublicAdapter]:
    """Adapt one raw record at a time so full datasets stay streaming."""

    try:
        adapter_type = ADAPTERS[source]
    except KeyError as error:
        raise ValueError(f"no adapter registered for source {source!r}") from error
    for record in iter_source_records(source, root):
        yield adapter_type(
            (record,),
            revision=revision,
            license_acknowledgement_id=license_acknowledgement_id,
        )
