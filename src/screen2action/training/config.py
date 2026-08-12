"""Typed stage-run settings resolved from composed experiment configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from screen2action.training.stages import stage_spec


def _mapping(value: object, *, context: str) -> Mapping[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    return cast(dict[str, object], value)


def _value(
    nested: Mapping[str, object],
    root: Mapping[str, object],
    *names: str,
    default: object,
) -> object:
    for name in names:
        if name in nested:
            return nested[name]
        if name in root:
            return root[name]
    return default


def _integer(value: object, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"training.{name} must be an integer >= {minimum}")
    return value


def _optional_integer(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    return _integer(value, name=name)


def _number(value: object, *, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"training.{name} must be numeric")
    result = float(value)
    if result < minimum:
        raise ValueError(f"training.{name} must be >= {minimum}")
    return result


def _fraction(value: object, *, name: str, upper_inclusive: bool = True) -> float:
    result = _number(value, name=name)
    valid = result <= 1.0 if upper_inclusive else result < 1.0
    if not valid:
        boundary = "[0, 1]" if upper_inclusive else "[0, 1)"
        raise ValueError(f"training.{name} must be in {boundary}")
    return result


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"training.{name} must be boolean")
    return value


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"training.{name} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class StageRunSettings:
    """Validated knobs that affect training state, scheduling, or artifacts."""

    stage: str
    epochs: int
    global_batch_size: int
    per_device_batch_size: int
    seed: int
    precision: str
    new_head_lr: float
    pretrained_lr: float
    weight_decay: float
    warmup_fraction: float
    scheduler: str
    gradient_clip_norm: float
    log_interval: int
    validation_interval_epochs: int
    checkpoint_interval_optimizer_steps: int
    validation_required: bool
    frozen_crop_warmup_epochs: int
    max_commands: int | None
    tensorboard: bool
    gumbel_temperature_start: float
    gumbel_temperature_end: float
    gumbel_anneal_fraction: float
    forced_positive_probability: float

    @classmethod
    def from_values(
        cls,
        values: Mapping[str, object],
        *,
        stage_name: str | None = None,
        max_commands: int | None = None,
    ) -> StageRunSettings:
        """Resolve legacy top-level and current nested training keys."""

        nested = _mapping(values.get("training"), context="training")
        selected = stage_name or _string(
            _value(nested, values, "stage", "name", default="stage2_selector_retrieval"),
            name="stage",
        )
        spec = stage_spec(selected)
        precision = _string(
            _value(nested, values, "precision", default="auto"),
            name="precision",
        )
        if precision not in {"auto", "fp32", "fp16", "bf16"}:
            raise ValueError("training.precision must be auto, fp32, fp16, or bf16")
        scheduler = _string(
            _value(nested, values, "scheduler", default="cosine"),
            name="scheduler",
        )
        if scheduler not in {"cosine", "linear", "constant"}:
            raise ValueError("training.scheduler must be cosine, linear, or constant")
        configured_max = _optional_integer(
            _value(nested, values, "max_commands", default=None),
            name="max_commands",
        )
        resolved_max = max_commands if max_commands is not None else configured_max
        if resolved_max is not None and resolved_max <= 0:
            raise ValueError("training.max_commands must be positive")
        start = _number(
            _value(
                nested,
                values,
                "gumbel_temperature_start",
                default=spec.gumbel_temperature_start,
            ),
            name="gumbel_temperature_start",
        )
        end = _number(
            _value(
                nested,
                values,
                "gumbel_temperature_end",
                default=spec.gumbel_temperature_end,
            ),
            name="gumbel_temperature_end",
        )
        if start <= 0.0 or end <= 0.0:
            raise ValueError("Gumbel temperatures must be positive")
        return cls(
            stage=spec.name,
            epochs=_integer(
                _value(nested, values, "epochs", default=spec.epochs),
                name="epochs",
            ),
            global_batch_size=_integer(
                _value(
                    nested,
                    values,
                    "global_batch_size",
                    "global_command_batch_size",
                    default=spec.global_batch_size,
                ),
                name="global_batch_size",
            ),
            per_device_batch_size=_integer(
                _value(nested, values, "per_device_batch_size", default=1),
                name="per_device_batch_size",
            ),
            seed=_integer(
                _value(nested, values, "seed", default=17),
                name="seed",
                minimum=0,
            ),
            precision=precision,
            new_head_lr=_number(
                _value(
                    nested,
                    values,
                    "new_head_lr",
                    "learning_rate_new_heads",
                    default=spec.new_head_lr,
                ),
                name="new_head_lr",
            ),
            pretrained_lr=_number(
                _value(
                    nested,
                    values,
                    "pretrained_lr",
                    "pretrained_encoder_lr",
                    "learning_rate_pretrained",
                    default=spec.pretrained_lr,
                ),
                name="pretrained_lr",
            ),
            weight_decay=_number(
                _value(nested, values, "weight_decay", default=spec.weight_decay),
                name="weight_decay",
            ),
            warmup_fraction=_fraction(
                _value(nested, values, "warmup_fraction", default=spec.warmup_fraction),
                name="warmup_fraction",
                upper_inclusive=False,
            ),
            scheduler=scheduler,
            gradient_clip_norm=_number(
                _value(
                    nested,
                    values,
                    "gradient_clip_norm",
                    default=spec.gradient_clip_norm,
                ),
                name="gradient_clip_norm",
                minimum=1e-12,
            ),
            log_interval=_integer(
                _value(nested, values, "log_interval", default=10),
                name="log_interval",
            ),
            validation_interval_epochs=_integer(
                _value(nested, values, "validation_interval_epochs", default=1),
                name="validation_interval_epochs",
            ),
            checkpoint_interval_optimizer_steps=_integer(
                _value(
                    nested,
                    values,
                    "checkpoint_interval_optimizer_steps",
                    default=100,
                ),
                name="checkpoint_interval_optimizer_steps",
            ),
            validation_required=_boolean(
                _value(nested, values, "validation_required", default=True),
                name="validation_required",
            ),
            frozen_crop_warmup_epochs=_integer(
                _value(nested, values, "frozen_crop_warmup_epochs", default=0),
                name="frozen_crop_warmup_epochs",
                minimum=0,
            ),
            max_commands=resolved_max,
            tensorboard=_boolean(
                _value(nested, values, "tensorboard", default=True),
                name="tensorboard",
            ),
            gumbel_temperature_start=start,
            gumbel_temperature_end=end,
            gumbel_anneal_fraction=_fraction(
                _value(
                    nested,
                    values,
                    "gumbel_anneal_fraction",
                    default=spec.gumbel_anneal_fraction,
                ),
                name="gumbel_anneal_fraction",
            ),
            forced_positive_probability=_fraction(
                _value(
                    nested,
                    values,
                    "forced_positive_probability_first_two_epochs",
                    "forced_positive_probability",
                    default=spec.forced_positive_probability,
                ),
                name="forced_positive_probability",
            ),
        )
