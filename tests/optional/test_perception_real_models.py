"""Opt-in real locked-model perception smoke tests; excluded by default."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from screen2action.data.schema import NodeType
from screen2action.model_assets import load_model_lock, verify_locked_models
from screen2action.models.mobilevit import MobileVitSCropEncoder
from screen2action.perception.base import PreprocessingMetadata, UiDetection, normalize_image
from screen2action.perception.doctr_crnn import DoctrCrnnBackend, DoctrCrnnVgg16Recognizer
from screen2action.perception.icon_actionability import MobileNetV3IconActionability
from screen2action.perception.screenparser import (
    ScreenParserConfig,
    ScreenParserUiDetector,
    UltralyticsScreenParserBackend,
)

pytestmark = pytest.mark.network


def _locked_paths() -> dict[str, Path]:
    cache_value = os.environ.get("SCREEN2ACTION_CACHE_ROOT")
    lock_path = Path(os.environ.get("SCREEN2ACTION_MODEL_LOCK", "configs/models/lock.json"))
    if cache_value is None or not lock_path.is_file():
        pytest.skip("set SCREEN2ACTION_CACHE_ROOT and provide a completed model lock")
    cache_root = Path(cache_value)
    report = verify_locked_models(lock_path, cache_root)
    if not report["ok"]:
        pytest.skip("the configured public model bundle is not fully verified")
    lock = load_model_lock(lock_path)
    paths: dict[str, Path] = {}
    for model in lock.models:
        if not model.files:
            raise AssertionError(f"locked role has no file: {model.role}")
        paths[model.role] = cache_root / model.files[0].path
    return paths


def _run_real_smoke(device: str) -> None:
    paths = _locked_paths()
    pixels = torch.zeros((3, 96, 192), dtype=torch.uint8)
    pixels[:, 20:60, 20:170] = 180
    detector = ScreenParserUiDetector(
        UltralyticsScreenParserBackend(paths["ui_detector_screenparser"]),
        config=ScreenParserConfig(input_size=1280, device=device),
    )
    detected = detector.detect((pixels,))[0]
    assert detected.detections[0].node_type is NodeType.ROOT

    image = normalize_image(pixels)
    preprocessing = PreprocessingMetadata(192, 96, 1280, 1.0, 0.0, 0.0, 192, 96, "smoke")
    text_detection = UiDetection(
        1,
        49,
        "Text",
        1.0,
        (0.1, 0.2, 0.9, 0.7),
        NodeType.TEXT,
        ("text",),
        (),
        "smoke_fixture",
        "smoke_v1",
        1.0,
        preprocessing,
        {},
    )
    recognizer = DoctrCrnnVgg16Recognizer(
        DoctrCrnnBackend(paths["ocr_crnn_vgg16_bn"], device=device)
    )
    recognized = recognizer.recognize(image, (text_detection,))
    assert len(recognized) == 1
    assert 0.0 <= recognized[0].confidence <= 1.0

    visual = MobileNetV3IconActionability.from_locked_torchvision(
        paths["icon_backbone_mobilenet_v3_small"]
    ).to(torch.device(device))
    output = visual(torch.rand((2, 3, 224, 224), device=torch.device(device)))
    assert output.icon_logits.shape == (2, 87)
    assert output.actionability_logits.shape == (2, 4)
    assert output.visual_features.shape == (2, 256)

    crop_encoder = MobileVitSCropEncoder.from_locked_timm(paths["crop_encoder_mobilevit_s"]).to(
        torch.device(device)
    )
    crop_tokens = crop_encoder(torch.rand((1, 3, 192, 192), device=torch.device(device)))
    assert crop_tokens.shape == (1, 144, 256)


def test_real_locked_perception_models_on_cpu() -> None:
    _run_real_smoke("cpu")


@pytest.mark.gpu
def test_real_locked_perception_models_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    _run_real_smoke("cuda")
