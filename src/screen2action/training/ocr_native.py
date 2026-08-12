"""Optional docTR-native CRNN/CTC Stage-1 branch over tight UI text crops."""

from __future__ import annotations

import importlib
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional as F

from screen2action.data.schema import Box
from screen2action.model_assets import load_model_lock
from screen2action.perception.base import normalize_image
from screen2action.training.data import CanonicalTrainingCorpus
from screen2action.training.model_factory import _locked_model, _verified_paths, _weight_file

_DOCTR_MEAN = (0.694, 0.695, 0.693)
_DOCTR_STD = (0.299, 0.296, 0.301)


@dataclass(frozen=True, slots=True)
class OcrExample:
    screen_id: str
    image_path: Path
    box: Box
    text: str


@dataclass(frozen=True, slots=True)
class OcrTrainingCorpus:
    by_split: Mapping[str, tuple[OcrExample, ...]]

    def batches(
        self,
        split: str,
        *,
        epoch: int,
        seed: int,
        rank: int,
        world_size: int,
        per_device_batch_size: int,
        training: bool,
        max_samples: int | None,
    ) -> tuple[tuple[OcrExample, ...], ...]:
        values = list(self.by_split.get(split, ()))
        if max_samples is not None:
            values = values[:max_samples]
        if not values:
            raise ValueError(f"stage1_ocr has no tight text crops in split {split!r}")
        if training:
            random.Random(seed + epoch).shuffle(values)
        global_batch = per_device_batch_size * world_size
        padded_count = math.ceil(len(values) / global_batch) * global_batch
        values.extend(values[index % len(values)] for index in range(padded_count - len(values)))
        local: list[OcrExample] = []
        for offset in range(0, len(values), global_batch):
            rank_offset = offset + rank * per_device_batch_size
            local.extend(values[rank_offset : rank_offset + per_device_batch_size])
        return tuple(
            tuple(local[offset : offset + per_device_batch_size])
            for offset in range(0, len(local), per_device_batch_size)
        )


def build_ocr_corpus(corpus: CanonicalTrainingCorpus) -> OcrTrainingCorpus:
    """Select only source-visible, non-empty text labels from frozen splits."""

    split_by_screen = {command.screen_id: command.split for command in corpus.commands}
    grouped: dict[str, list[OcrExample]] = {"train": [], "val": [], "test": []}
    for screen_id in sorted(corpus.screens):
        split = split_by_screen.get(screen_id)
        if split is None:
            continue
        image_path = corpus.dataset_root / str(corpus.screens[screen_id]["image_path"])
        for element_id in sorted(corpus.elements.get(screen_id, {})):
            row = corpus.elements[screen_id][element_id]
            text = str(row["text"]).strip() if bool(row["has_text"]) else ""
            if not text:
                continue
            box = cast(
                Box,
                tuple(float(str(row[key])) for key in ("x1", "y1", "x2", "y2")),
            )
            grouped[split].append(OcrExample(screen_id, image_path, box, text))
    if not grouped["train"]:
        raise ValueError("stage1_ocr requires source text labels in the frozen train split")
    return OcrTrainingCorpus({name: tuple(values) for name, values in grouped.items()})


@dataclass(frozen=True, slots=True)
class OcrBatch:
    crops: torch.Tensor
    targets: tuple[str, ...]


class OcrBatchBuilder:
    def __init__(self, device: torch.device, *, height: int = 32, width: int = 128) -> None:
        if height <= 0 or width <= 0:
            raise ValueError("OCR crop dimensions must be positive")
        self.device = device
        self.height = height
        self.width = width

    def _resize(self, crop: torch.Tensor) -> torch.Tensor:
        height, width = crop.shape[-2:]
        scaled_width = max(1, min(self.width, round(width * self.height / max(height, 1))))
        resized = F.interpolate(
            crop.unsqueeze(0).float() / 255.0,
            size=(self.height, scaled_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        padded = torch.ones((3, self.height, self.width), dtype=resized.dtype)
        padded[:, :, :scaled_width] = resized
        mean = torch.tensor(_DOCTR_MEAN, dtype=padded.dtype).view(3, 1, 1)
        std = torch.tensor(_DOCTR_STD, dtype=padded.dtype).view(3, 1, 1)
        return (padded - mean) / std

    def build(self, examples: Sequence[OcrExample]) -> OcrBatch:
        if not examples:
            raise ValueError("OCR batch cannot be empty")
        crops = []
        for example in examples:
            image = normalize_image(example.image_path)
            x1 = max(0, min(image.width - 1, math.floor(example.box[0] * image.width)))
            y1 = max(0, min(image.height - 1, math.floor(example.box[1] * image.height)))
            x2 = max(x1 + 1, min(image.width, math.ceil(example.box[2] * image.width)))
            y2 = max(y1 + 1, min(image.height, math.ceil(example.box[3] * image.height)))
            tight = image.pixels[:, y1:y2, x1:x2]
            crops.append(self._resize(tight))
        return OcrBatch(
            torch.stack(crops).to(self.device),
            tuple(example.text for example in examples),
        )


class DoctrCrnnTrainingModel(nn.Module):
    """Thin module retaining docTR's native target encoding and CTC loss."""

    def __init__(self, recognizer: nn.Module) -> None:
        super().__init__()
        self.recognizer = recognizer

    def forward(self, batch: OcrBatch) -> torch.Tensor:
        output = self.recognizer(batch.crops, list(batch.targets))
        if not isinstance(output, dict) or not isinstance(output.get("loss"), torch.Tensor):
            raise RuntimeError("docTR CRNN did not return its native loss tensor")
        return cast(torch.Tensor, output["loss"])


@dataclass(frozen=True, slots=True)
class OcrModelBundle:
    model: DoctrCrnnTrainingModel
    model_lock_digest: str
    pretrained_prefixes: tuple[str, ...]
    crop_height: int
    crop_width: int


def build_locked_ocr_model(
    config_values: Mapping[str, object],
    *,
    model_lock_path: Path | None,
    cache_root: Path,
) -> OcrModelBundle:
    """Construct docTR without downloads and load only the verified lock artifact."""

    if model_lock_path is None:
        raise ValueError("stage1_ocr requires --model-lock")
    try:
        models = importlib.import_module("doctr.models")
    except ImportError as error:
        raise RuntimeError(
            "stage1_ocr requires the perception extra: pip install -e .[perception]"
        ) from error
    factory = getattr(models, "crnn_vgg16_bn", None)
    if not callable(factory):
        raise RuntimeError("installed docTR does not expose crnn_vgg16_bn")
    lock = load_model_lock(model_lock_path)
    paths = _verified_paths(_locked_model(lock, "ocr_crnn_vgg16_bn"), cache_root)
    weight = _weight_file(paths, role="ocr_crnn_vgg16_bn")
    recognizer = factory(pretrained=False, pretrained_backbone=False)
    if not isinstance(recognizer, nn.Module) or not callable(
        getattr(recognizer, "from_pretrained", None)
    ):
        raise RuntimeError("installed docTR CRNN lacks the locked-weight loading interface")
    cast(Any, recognizer).from_pretrained(str(weight))
    native = config_values.get("ocr_native")
    settings = cast(dict[str, object], native) if isinstance(native, dict) else {}
    height = int(str(settings.get("crop_size", 32)))
    width = int(str(settings.get("max_width", 128)))
    return OcrModelBundle(
        DoctrCrnnTrainingModel(recognizer),
        lock.digest,
        ("recognizer.",),
        height,
        width,
    )
