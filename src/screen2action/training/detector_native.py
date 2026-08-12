"""Optional Ultralytics-native ScreenParser detector fine-tuning boundary."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import yaml  # type: ignore[import-untyped]

from screen2action.config import ResolvedConfig, write_resolved_config
from screen2action.handoff import repository_root
from screen2action.model_assets import load_model_lock
from screen2action.training.artifacts import write_run_manifest
from screen2action.training.data import CanonicalTrainingCorpus, load_training_corpus
from screen2action.training.model_factory import _locked_model, _verified_paths, _weight_file


@dataclass(frozen=True, slots=True)
class YoloDatasetResult:
    data_yaml: Path
    train_images: int
    validation_images: int
    labeled_boxes: int
    skipped_unmapped_elements: int


@dataclass(frozen=True, slots=True)
class NativeStageResult:
    status: str
    stage: str
    run_directory: str
    rank: int
    world_size: int
    device: str
    epochs_completed: int
    latest_checkpoint: str | None
    best_checkpoint: str | None
    data_manifest_digest: str
    model_lock_digest: str
    run_manifest_sha256: str
    native_backend: str
    details: Mapping[str, object]


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _metadata(row: Mapping[str, object]) -> Mapping[str, object]:
    try:
        value = json.loads(str(row.get("metadata_json", "{}")))
    except json.JSONDecodeError as error:
        raise ValueError("element metadata_json is malformed") from error
    return cast(dict[str, object], value) if isinstance(value, dict) else {}


def _class_id(row: Mapping[str, object]) -> int | None:
    values: tuple[object | None, ...] = (
        row.get("original_class_id"),
        _metadata(row).get("original_class_id"),
        _metadata(row).get("detector_class_id"),
    )
    for value in values:
        if value is None or str(value) == "":
            continue
        try:
            parsed = int(str(value))
        except ValueError as error:
            raise ValueError("ScreenParser original class ID must be an integer") from error
        if not 0 <= parsed < 55:
            raise ValueError("ScreenParser original class ID must be in [0, 55)")
        return parsed
    return None


def generate_yolo_dataset(
    corpus: CanonicalTrainingCorpus,
    output_directory: Path,
) -> YoloDatasetResult:
    """Materialize only source labels carrying original ScreenParser-55 IDs."""

    split_by_screen = {command.screen_id: command.split for command in corpus.commands}
    counts = {"train": 0, "val": 0}
    boxes = 0
    skipped = 0
    for split in ("train", "val"):
        (output_directory / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_directory / "labels" / split).mkdir(parents=True, exist_ok=True)
    for screen_id in sorted(corpus.screens):
        assigned_split = split_by_screen.get(screen_id)
        if assigned_split not in {"train", "val"}:
            continue
        labels: list[str] = []
        for element_id in sorted(corpus.elements.get(screen_id, {})):
            row = corpus.elements[screen_id][element_id]
            class_id = _class_id(row)
            if class_id is None:
                skipped += 1
                continue
            x1, y1, x2, y2 = (float(str(row[key])) for key in ("x1", "y1", "x2", "y2"))
            width = x2 - x1
            height = y2 - y1
            if width <= 0.0 or height <= 0.0:
                continue
            labels.append(
                f"{class_id} {(x1 + x2) / 2:.9f} {(y1 + y2) / 2:.9f} {width:.9f} {height:.9f}"
            )
        if not labels:
            continue
        source = corpus.dataset_root / str(corpus.screens[screen_id]["image_path"])
        suffix = source.suffix.casefold() or ".png"
        stem = str(corpus.screens[screen_id]["screen_sha256"])
        image = output_directory / "images" / assigned_split / f"{stem}{suffix}"
        label = output_directory / "labels" / assigned_split / f"{stem}.txt"
        if not image.is_file():
            try:
                os.link(source, image)
            except OSError:
                shutil.copy2(source, image)
        _atomic_text(label, "\n".join(labels) + "\n")
        counts[assigned_split] += 1
        boxes += len(labels)
    if counts["train"] == 0 or counts["val"] == 0 or boxes == 0:
        raise ValueError(
            "stage1_detector requires original ScreenParser-55 IDs in both train and val splits"
        )
    data = {
        "path": output_directory.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": {index: f"screenparser_{index:02d}" for index in range(55)},
    }
    data_yaml = output_directory / "dataset.yaml"
    _atomic_text(data_yaml, yaml.safe_dump(data, sort_keys=False))
    return YoloDatasetResult(data_yaml, counts["train"], counts["val"], boxes, skipped)


def run_detector_stage(
    config: ResolvedConfig,
    *,
    manifest: Path,
    model_lock: Path | None,
    model_cache_root: Path,
    device: str,
    run_directory: Path,
    resume: Path | None = None,
) -> NativeStageResult:
    """Run Ultralytics' native objective; NMS remains outside downstream autograd."""

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA detector training requested but CUDA is unavailable")
    if model_lock is None:
        raise ValueError("stage1_detector requires --model-lock")
    try:
        ultralytics = importlib.import_module("ultralytics")
    except ImportError as error:
        raise RuntimeError(
            "stage1_detector requires the perception extra: pip install -e .[perception]"
        ) from error
    yolo_factory = getattr(ultralytics, "YOLO", None)
    if not callable(yolo_factory):
        raise RuntimeError("installed Ultralytics does not expose YOLO")
    corpus = load_training_corpus(manifest)
    lock = load_model_lock(model_lock)
    paths = _verified_paths(_locked_model(lock, "ui_detector_screenparser"), model_cache_root)
    weight = _weight_file(paths, role="ui_detector_screenparser")
    settings_raw = config.values.get("detector_native")
    settings = cast(dict[str, object], settings_raw) if isinstance(settings_raw, dict) else {}
    training_raw = config.values.get("training")
    training = cast(dict[str, object], training_raw) if isinstance(training_raw, dict) else {}
    epochs = int(str(training.get("epochs", 12)))
    batch = int(str(training.get("per_device_batch_size", 1)))
    image_size = int(str(settings.get("image_size", 1280)))
    workers = int(str(settings.get("workers", 4)))
    generated = generate_yolo_dataset(corpus, run_directory / "detector-data")
    write_resolved_config(config, run_directory)
    _, run_digest, _ = write_run_manifest(
        run_directory,
        root=repository_root(),
        stage="stage1_detector",
        config_sha256=config.sha256,
        data_manifest_digest=corpus.manifest_digest,
        model_lock_digest=lock.digest,
        cache_manifest_sha256="not-applicable-detector-native",
        world_size=1,
        device_type=device,
    )
    model = yolo_factory(str(resume or weight))
    extra_raw = settings.get("extra_arguments", {})
    if not isinstance(extra_raw, dict):
        raise ValueError("detector_native.extra_arguments must be a mapping")
    reserved = {"data", "epochs", "imgsz", "batch", "device", "project", "name", "resume"}
    overlap = reserved & set(extra_raw)
    if overlap:
        raise ValueError(
            f"detector_native.extra_arguments overrides reserved keys: {sorted(overlap)}"
        )
    target_device: str | int = "cpu" if device == "cpu" else 0
    result = cast(Any, model).train(
        data=str(generated.data_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        workers=workers,
        device=target_device,
        project=str(run_directory),
        name="ultralytics",
        exist_ok=True,
        resume=bool(resume),
        **cast(dict[str, Any], extra_raw),
    )
    save_dir = Path(str(getattr(result, "save_dir", run_directory / "ultralytics")))
    latest = save_dir / "weights" / "last.pt"
    best = save_dir / "weights" / "best.pt"
    return NativeStageResult(
        "complete",
        "stage1_detector",
        run_directory.as_posix(),
        0,
        1,
        device,
        epochs,
        latest.as_posix() if latest.is_file() else None,
        best.as_posix() if best.is_file() else None,
        corpus.manifest_digest,
        lock.digest,
        run_digest,
        "ultralytics",
        {
            "train_images": generated.train_images,
            "validation_images": generated.validation_images,
            "labeled_boxes": generated.labeled_boxes,
            "skipped_unmapped_elements": generated.skipped_unmapped_elements,
            "nms_downstream_gradient": False,
        },
    )
