"""Construct the real perception service exclusively from verified local assets."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from screen2action.model_assets import ModelLock, load_model_lock, sha256_file
from screen2action.perception.cache import ContentAddressedPerceptionCache
from screen2action.perception.doctr_crnn import DoctrCrnnBackend, DoctrCrnnVgg16Recognizer
from screen2action.perception.icon_actionability import MobileNetV3IconActionability
from screen2action.perception.pipeline import FullScreenPerception, FullScreenPerceptionConfig
from screen2action.perception.precompute import PerceptionBundleSpec, bundle_cache
from screen2action.perception.screenparser import (
    ContainerAugmentationPolicy,
    ScreenParserConfig,
    ScreenParserUiDetector,
    UltralyticsScreenParserBackend,
)

PERCEPTION_MODEL_ROLES = (
    "ui_detector_screenparser",
    "ocr_crnn_vgg16_bn",
    "icon_backbone_mobilenet_v3_small",
)


@dataclass(frozen=True, slots=True)
class PerceptionFactoryResult:
    """Real service plus exact local compatibility identity."""

    service: FullScreenPerception
    cache: ContentAddressedPerceptionCache
    bundle: PerceptionBundleSpec
    device: str
    visual_checkpoint_status: str


def _mapping(value: object, *, context: str) -> Mapping[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    return cast(dict[str, object], value)


def _integer(mapping: Mapping[str, object], key: str, default: int) -> int:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _number(mapping: Mapping[str, object], key: str, default: float) -> float:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _boolean(mapping: Mapping[str, object], key: str, default: bool) -> bool:
    value = mapping.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be boolean")
    return value


def resolve_precompute_device(requested: str) -> torch.device:
    """Resolve CPU or this torchrun process's validated CUDA device."""

    if requested == "cpu":
        return torch.device("cpu")
    if requested != "cuda":
        raise ValueError("perception device must be cpu or cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA perception requested but torch.cuda.is_available() is false")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not 0 <= local_rank < torch.cuda.device_count():
        raise RuntimeError("LOCAL_RANK is outside the visible CUDA device range")
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


def _verified_role_files(lock: ModelLock, cache_root: Path, role: str) -> tuple[Path, ...]:
    try:
        model = next(model for model in lock.models if model.role == role)
    except StopIteration as error:
        raise ValueError(f"model lock is missing perception role {role}") from error
    if not model.complete or model.license_acknowledgement_id is None:
        raise RuntimeError(f"perception role is not fetched/license-acknowledged: {role}")
    paths: list[Path] = []
    for declaration in model.files:
        path = (cache_root / declaration.path).resolve()
        if not path.is_relative_to(cache_root.resolve()):
            raise ValueError(f"locked model path escapes cache root: {declaration.path}")
        if declaration.sha256 is None or not path.is_file():
            raise FileNotFoundError(f"locked perception asset is unavailable: {declaration.path}")
        if sha256_file(path) != declaration.sha256:
            raise ValueError(f"locked perception asset digest mismatch: {declaration.path}")
        paths.append(path)
    return tuple(paths)


def _single_weight(paths: tuple[Path, ...], *, role: str) -> Path:
    candidates = tuple(path for path in paths if path.suffix.casefold() in {".pt", ".pth"})
    if len(candidates) != 1:
        raise ValueError(f"{role} must resolve to exactly one PT/PTH weight file")
    return candidates[0]


def _checkpoint_state(payload: object) -> Mapping[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("visual checkpoint must be a state-dict mapping")
    for key in ("visual_model_state", "state_dict", "model_state_dict"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            payload = nested
            break
    if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
        raise ValueError("visual checkpoint state keys must be strings")
    state = cast(dict[str, Any], payload)
    prefixes = (
        "visual_model.",
        "visual.",
        "perception.visual_model.",
        "module.visual_model.",
        "module.visual.",
        "semantics.visual.",
        "module.semantics.visual.",
    )
    for prefix in prefixes:
        selected = {
            key.removeprefix(prefix): value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if selected:
            return selected
    return state


def _load_visual_model(
    backbone_path: Path,
    *,
    checkpoint: Path | None,
    allow_untrained_heads: bool,
    untrained_head_seed: int,
) -> tuple[MobileNetV3IconActionability, str, str]:
    checkpoint_digest = "0" * 64
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(untrained_head_seed)
        model = MobileNetV3IconActionability.from_locked_torchvision(backbone_path)
    if checkpoint is None:
        if not allow_untrained_heads:
            raise RuntimeError(
                "precompute requires --visual-checkpoint from stage1_semantics_graph; "
                "use --allow-untrained-heads only for a deterministic smoke run"
            )
        return model, checkpoint_digest, "deterministic_untrained_heads"
    resolved = checkpoint.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"visual checkpoint does not exist: {checkpoint}")
    checkpoint_digest = sha256_file(resolved)
    state = _checkpoint_state(torch.load(resolved, map_location="cpu", weights_only=True))
    model.load_state_dict(state, strict=True)
    return model, checkpoint_digest, "trained_checkpoint"


def build_locked_perception_service(
    *,
    model_lock: Path,
    model_cache_root: Path,
    cache_root: Path,
    config_values: Mapping[str, object],
    requested_device: str,
    visual_checkpoint: Path | None,
    allow_untrained_heads: bool,
) -> PerceptionFactoryResult:
    """Build real ScreenParser/docTR/MobileNet adapters from an offline lock."""

    lock = load_model_lock(model_lock)
    device = resolve_precompute_device(requested_device)
    detector_values = _mapping(config_values.get("detector"), context="detector")
    perception_values = _mapping(config_values.get("perception"), context="perception")
    container_values = _mapping(
        detector_values.get("container_augmentation"),
        context="detector.container_augmentation",
    )
    untrained_seed = _integer(perception_values, "untrained_head_seed", 17)
    detector_weight = _single_weight(
        _verified_role_files(lock, model_cache_root, "ui_detector_screenparser"),
        role="ui_detector_screenparser",
    )
    ocr_weight = _single_weight(
        _verified_role_files(lock, model_cache_root, "ocr_crnn_vgg16_bn"),
        role="ocr_crnn_vgg16_bn",
    )
    backbone_weight = _single_weight(
        _verified_role_files(lock, model_cache_root, "icon_backbone_mobilenet_v3_small"),
        role="icon_backbone_mobilenet_v3_small",
    )
    visual_model, checkpoint_digest, checkpoint_status = _load_visual_model(
        backbone_weight,
        checkpoint=visual_checkpoint,
        allow_untrained_heads=allow_untrained_heads,
        untrained_head_seed=untrained_seed,
    )
    visual_model.to(device)
    visual_model.eval()
    pipeline_config = FullScreenPerceptionConfig(
        node_crop_size=_integer(perception_values, "node_crop_size", 224),
        node_crop_margin=_number(perception_values, "node_crop_margin", 0.0),
        include_proximity=_boolean(perception_values, "include_proximity", True),
        include_ordinal=_boolean(perception_values, "include_ordinal", True),
        ocr_control_propagation_coverage=_number(
            perception_values,
            "ocr_control_propagation_coverage",
            0.95,
        ),
        include_detector_roi_features=_boolean(
            perception_values,
            "include_detector_roi_features",
            False,
        ),
        visual_checkpoint_sha256=checkpoint_digest,
        untrained_head_seed=untrained_seed,
    )
    detector = ScreenParserUiDetector(
        UltralyticsScreenParserBackend(detector_weight),
        config=ScreenParserConfig(
            input_size=_integer(detector_values, "input_size", 1280),
            confidence_threshold=_number(detector_values, "confidence_threshold", 0.10),
            nms_iou_threshold=_number(detector_values, "nms_iou_threshold", 0.10),
            device=str(device),
        ),
        container_policy=ContainerAugmentationPolicy(
            enabled=_boolean(container_values, "enabled", False),
            minimum_children=_integer(container_values, "minimum_children", 3),
            alignment_tolerance=_number(container_values, "alignment_tolerance", 0.025),
            margin=_number(container_values, "margin", 0.01),
            confidence=_number(container_values, "confidence", 0.65),
        ),
    )
    recognizer = DoctrCrnnVgg16Recognizer(
        DoctrCrnnBackend(ocr_weight, device=str(device)),
        aspect_batch_size=_integer(perception_values, "ocr_aspect_batch_size", 64),
    )
    provisional = FullScreenPerception(
        detector=detector,
        recognizer=recognizer,
        visual_model=visual_model,
        model_bundle_lock_digest=lock.digest,
        config=pipeline_config,
    )
    bundle = PerceptionBundleSpec.from_service(provisional)
    cache = bundle_cache(cache_root, bundle)
    service = FullScreenPerception(
        detector=detector,
        recognizer=recognizer,
        visual_model=visual_model,
        cache=cache,
        model_bundle_lock_digest=lock.digest,
        config=pipeline_config,
    )
    return PerceptionFactoryResult(
        service=service,
        cache=cache,
        bundle=bundle,
        device=str(device),
        visual_checkpoint_status=checkpoint_status,
    )
