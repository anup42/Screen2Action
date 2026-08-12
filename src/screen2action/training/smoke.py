"""Small device-safe forward/backward/checkpoint/resume smoke workflow."""

from __future__ import annotations

import math
import tempfile
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from screen2action.data.tokenizer import VocabularyTokenizer
from screen2action.training.checkpoints import load_checkpoint, save_checkpoint
from screen2action.training.seed import seed_everything
from screen2action.training.tiny import TinyDataset, TinyGroundingModel, build_tiny_dataset


@dataclass(frozen=True, slots=True)
class SmokeResult:
    """Observed evidence from a tiny training and exact-resume cycle."""

    device: str
    steps: int
    initial_loss: float
    final_loss: float
    checkpoint_resume_parity: bool
    resumed_step: int
    checkpoint_path: str | None


def _move_dataset(dataset: TinyDataset, device: torch.device) -> TinyDataset:
    return TinyDataset(
        node_features=dataset.node_features.to(device),
        command_ids=dataset.command_ids.to(device),
        command_mask=dataset.command_mask.to(device),
        target_index=dataset.target_index.to(device),
        target_point=dataset.target_point.to(device),
    )


def _loss(model: TinyGroundingModel, dataset: TinyDataset) -> torch.Tensor:
    logits, points = model(dataset.node_features, dataset.command_ids, dataset.command_mask)
    rows = torch.arange(len(dataset.target_index), device=dataset.target_index.device)
    selected_points = points[rows, dataset.target_index]
    return F.cross_entropy(logits, dataset.target_index) + F.smooth_l1_loss(
        selected_points,
        dataset.target_point,
    )


def _run(
    checkpoint: Path,
    *,
    device: torch.device,
    steps: int,
    config: Mapping[str, Any],
) -> tuple[float, float, bool, int]:
    seed_everything(17)
    corpus = ("tap red", "tap blue", "tap green", "tap yellow")
    tokenizer = VocabularyTokenizer.from_corpus(corpus)
    dataset = _move_dataset(build_tiny_dataset(tokenizer, count=8), device)
    model = TinyGroundingModel(vocab_size=tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    model.train()
    with torch.no_grad():
        initial = float(_loss(model, dataset).detach().cpu())
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model, dataset)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError("smoke loss must be a finite scalar")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        expected_logits, expected_points = model(
            dataset.node_features,
            dataset.command_ids,
            dataset.command_mask,
        )
        final = float(_loss(model, dataset).detach().cpu())
    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        epoch=0,
        step=steps,
        config=config,
        data_manifest_hash="fixture-smoke-v1",
        git_revision="smoke",
    )
    resumed = TinyGroundingModel(vocab_size=tokenizer.vocab_size).to(device)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.01, weight_decay=0.0)
    state = load_checkpoint(
        checkpoint,
        resumed,
        resumed_optimizer,
        device=device,
    )
    resumed.eval()
    with torch.no_grad():
        actual_logits, actual_points = resumed(
            dataset.node_features,
            dataset.command_ids,
            dataset.command_mask,
        )
    parity = bool(
        torch.equal(expected_logits, actual_logits) and torch.equal(expected_points, actual_points)
    )
    if not parity:
        raise RuntimeError("checkpoint resume changed smoke model outputs")
    resumed.train()
    resumed_optimizer.zero_grad(set_to_none=True)
    resumed_loss = _loss(resumed, dataset)
    resumed_loss.backward()
    resumed_optimizer.step()
    if not all(
        math.isfinite(value) for value in (initial, final, float(resumed_loss.detach().cpu()))
    ):
        raise ValueError("smoke workflow produced a non-finite measurement")
    return initial, final, parity, state.step


def run_training_smoke(
    *,
    device: str = "cpu",
    steps: int = 8,
    run_directory: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
) -> SmokeResult:
    """Exercise training and exact checkpoint resume on CPU or available CUDA."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requested but torch.cuda.is_available() is false")
    context = (
        tempfile.TemporaryDirectory(prefix="screen2action-smoke-")
        if run_directory is None
        else nullcontext(None)
    )
    with context as temporary:
        persistent = Path(run_directory) if run_directory is not None else None
        directory = persistent if persistent is not None else Path(str(temporary))
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint = directory / "smoke-checkpoint.pt"
        initial, final, parity, resumed_step = _run(
            checkpoint,
            device=target,
            steps=steps,
            config=config or {},
        )
        return SmokeResult(
            device=str(target),
            steps=steps,
            initial_loss=initial,
            final_loss=final,
            checkpoint_resume_parity=parity,
            resumed_step=resumed_step,
            checkpoint_path=(checkpoint.name if persistent is not None else None),
        )
