from __future__ import annotations

import pytest
import torch

from screen2action.data.tokenizer import VocabularyTokenizer
from screen2action.export.onnx import export_onnx
from screen2action.export.parity import onnx_parity
from screen2action.training.tiny import TinyGroundingModel


def test_tiny_model_cpu_onnx_export_and_parity(tmp_path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    tokenizer = VocabularyTokenizer.from_corpus(("tap red", "tap blue"))
    model = TinyGroundingModel(vocab_size=tokenizer.vocab_size)
    inputs = (
        torch.eye(4).unsqueeze(0),
        torch.tensor([[1, 2, 3] for _ in range(1)]),
        torch.ones((1, 3), dtype=torch.bool),
    )
    path = export_onnx(
        model,
        inputs,
        tmp_path / "tiny.onnx",
        input_names=("node_features", "command_ids", "command_mask"),
        output_names=("candidate_logits", "point_local"),
    )
    errors = onnx_parity(model, path, inputs)
    assert max(errors) < 1e-3
