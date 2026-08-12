"""Offline tiny training command for the CPU milestone."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from screen2action.data.tokenizer import VocabularyTokenizer
from screen2action.training.stages import stage_spec
from screen2action.training.tiny import build_tiny_dataset, train_tiny_model


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "tiny_cpu",
            "stage1_perception",
            "stage2_selector_retrieval",
            "stage3_joint",
            "stage4_qat",
        ),
        default="tiny_cpu",
    )
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--steps", type=int, default=60)
    args = parser.parse_args(argv)
    corpus = ("tap red", "tap blue", "tap green", "tap yellow")
    tokenizer = VocabularyTokenizer.from_corpus(corpus)
    dataset = build_tiny_dataset(tokenizer, count=32)
    result = train_tiny_model(dataset, vocab_size=tokenizer.vocab_size, steps=args.steps)
    stage = stage_spec(args.stage) if args.stage != "tiny_cpu" else None
    output = {
        "stage": args.stage,
        "device": args.device,
        "steps": result.steps,
        "initial_loss": result.initial_loss,
        "final_loss": result.final_loss,
        "final_accuracy": result.final_accuracy,
    }
    if stage is not None:
        output["reference_epochs"] = stage.epochs
        output["ocr_frozen"] = stage.freeze_ocr
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
