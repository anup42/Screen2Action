"""Build tiny or locked paper-profile trainable models without network access."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from screen2action.model_assets import (
    LockedModel,
    ModelLock,
    load_model_lock,
    sha256_file,
)
from screen2action.models.bert_adapter import CompactBertAdapter
from screen2action.models.mobilevit import MobileVitSCropEncoder
from screen2action.models.screen2action_model import Screen2ActionModel, Screen2ActionModelConfig


@dataclass(frozen=True, slots=True)
class TrainModelBundle:
    """Constructed model plus optimizer grouping and provenance identity."""

    model: Screen2ActionModel
    profile: str
    model_lock_digest: str
    pretrained_prefixes: tuple[str, ...]


def _mapping(value: object, *, context: str) -> Mapping[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    return cast(dict[str, object], value)


def _integer(values: Mapping[str, object], key: str, default: int) -> int:
    value = values.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"model.{key} must be an integer")
    return value


def _number(values: Mapping[str, object], key: str, default: float) -> float:
    value = values.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"model.{key} must be numeric")
    return float(value)


def _string(values: Mapping[str, object], key: str, default: str) -> str:
    value = values.get(key, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"model.{key} must be a non-empty string")
    return value


def _config(
    values: Mapping[str, object], *, profile: str, vocab_size: int
) -> Screen2ActionModelConfig:
    base = (
        Screen2ActionModelConfig.paper_reference()
        if profile == "paper_reference"
        else Screen2ActionModelConfig.tiny_cpu()
    )
    return replace(
        base,
        embedding_dim=_integer(values, "embedding_dim", base.embedding_dim),
        vocab_size=max(vocab_size, _integer(values, "vocab_size", base.vocab_size)),
        node_text_length=_integer(values, "node_text_length", base.node_text_length),
        visual_feature_dim=_integer(
            values,
            "visual_feature_dim",
            base.visual_feature_dim,
        ),
        graph_layers=_integer(values, "graph_layers", base.graph_layers),
        graph_heads=_integer(values, "graph_heads", base.graph_heads),
        graph_variant=_string(values, "graph_variant", base.graph_variant),
        command_layers=_integer(values, "command_layers", base.command_layers),
        command_heads=_integer(values, "command_heads", base.command_heads),
        command_ffn=_integer(values, "command_ffn", base.command_ffn),
        command_max_length=_integer(
            values,
            "command_max_length",
            base.command_max_length,
        ),
        node_text_strategy=_string(
            values,
            "node_text_strategy",
            base.node_text_strategy,
        ),
        top_k=_integer(values, "top_k", base.top_k),
        ssb_budget=_integer(values, "ssb_budget", base.ssb_budget),
        crop_size=_integer(values, "crop_size", base.crop_size),
        crop_tokens=_integer(values, "crop_tokens", base.crop_tokens),
        crop_margin_fraction=_number(
            values,
            "crop_margin_fraction",
            base.crop_margin_fraction,
        ),
        actionability_threshold=_number(
            values,
            "actionability_threshold",
            base.actionability_threshold,
        ),
        relation_lambda=_number(values, "relation_lambda", base.relation_lambda),
    )


def _locked_model(lock: ModelLock, role: str) -> LockedModel:
    try:
        model = next(model for model in lock.models if model.role == role)
    except StopIteration as error:
        raise ValueError(f"model lock is missing role {role}") from error
    if not model.complete or model.license_acknowledgement_id is None:
        raise RuntimeError(f"model role is not completely fetched: {role}")
    return model


def _verified_paths(model: LockedModel, cache_root: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    resolved_root = cache_root.resolve()
    for declaration in model.files:
        path = (resolved_root / declaration.path).resolve()
        if not path.is_relative_to(resolved_root):
            raise ValueError(f"locked model path escapes cache root: {declaration.path}")
        if declaration.sha256 is None or not path.is_file():
            raise FileNotFoundError(f"locked model file is unavailable: {declaration.path}")
        if sha256_file(path) != declaration.sha256:
            raise ValueError(f"locked model digest mismatch: {declaration.path}")
        paths.append(path)
    return tuple(paths)


def _model_directory(paths: tuple[Path, ...], *, role: str) -> Path:
    if not paths:
        raise ValueError(f"locked role {role} has no files")
    common = Path(os.path.commonpath(paths))
    directory = common if common.is_dir() else common.parent
    if not (directory / "config.json").is_file():
        raise ValueError(f"locked role {role} does not contain config.json")
    return directory


def _weight_file(paths: tuple[Path, ...], *, role: str) -> Path:
    preferred = tuple(
        path
        for path in paths
        if path.name.casefold()
        in {
            "model.safetensors",
            "pytorch_model.bin",
            "weights.pt",
        }
    )
    candidates = preferred or tuple(
        path for path in paths if path.suffix.casefold() in {".safetensors", ".bin", ".pt"}
    )
    if len(candidates) != 1:
        raise ValueError(f"locked role {role} must resolve to one model weight file")
    return candidates[0]


def build_screen2action_model(
    config_values: Mapping[str, object],
    *,
    tokenizer_vocab_size: int,
    model_lock_path: Path | None,
    cache_root: Path,
) -> TrainModelBundle:
    """Build a CPU-safe tiny model or locked compact-BERT/MobileViT paper model."""

    values = _mapping(config_values.get("model"), context="model")
    profile = _string(values, "profile", "tiny_cpu")
    if profile not in {"tiny_cpu", "paper_reference"}:
        raise ValueError("model.profile must be tiny_cpu or paper_reference")
    model_config = _config(values, profile=profile, vocab_size=tokenizer_vocab_size)
    if profile == "tiny_cpu":
        return TrainModelBundle(
            Screen2ActionModel(config=model_config),
            profile,
            "fixture-no-model-lock",
            (),
        )
    if model_lock_path is None:
        raise ValueError("paper_reference training requires --model-lock")
    lock = load_model_lock(model_lock_path)
    bert_paths = _verified_paths(
        _locked_model(lock, "command_encoder_bert_l6_h256_a4"),
        cache_root,
    )
    crop_paths = _verified_paths(
        _locked_model(lock, "crop_encoder_mobilevit_s"),
        cache_root,
    )
    command_encoder = CompactBertAdapter.from_locked_pretrained(
        _model_directory(bert_paths, role="command_encoder_bert_l6_h256_a4"),
        max_length=model_config.command_max_length,
    )
    crop_encoder = MobileVitSCropEncoder.from_locked_timm(
        _weight_file(crop_paths, role="crop_encoder_mobilevit_s")
    )
    return TrainModelBundle(
        Screen2ActionModel(
            config=model_config,
            command_encoder=command_encoder,
            crop_encoder=crop_encoder,
        ),
        profile,
        lock.digest,
        ("command_encoder.model.", "crop_encoder.backbone."),
    )
