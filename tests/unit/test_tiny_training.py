from __future__ import annotations

from screen2action.data.tokenizer import VocabularyTokenizer
from screen2action.training.tiny import build_tiny_dataset, train_tiny_model


def test_tiny_cpu_model_overfits_32_examples() -> None:
    tokenizer = VocabularyTokenizer.from_corpus(("tap red", "tap blue", "tap green", "tap yellow"))
    dataset = build_tiny_dataset(tokenizer, count=32)
    result = train_tiny_model(
        dataset, vocab_size=tokenizer.vocab_size, steps=60, learning_rate=0.03
    )
    assert result.final_loss < result.initial_loss
    assert result.final_accuracy >= 0.95
