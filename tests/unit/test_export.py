from __future__ import annotations

import pytest
import torch

from screen2action.data.tokenizer import VocabularyTokenizer
from screen2action.export.host_contract import validate_fixed_graph_inputs
from screen2action.export.onnx import export_onnx
from screen2action.export.parity import onnx_parity
from screen2action.export.static_shapes import ExportShapeContract
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


def test_host_graph_contract_counts_overlapping_relations_and_rejects_overflow() -> None:
    contract = ExportShapeContract(4, 2, 8, 4, 4, 16)
    relation_mask = torch.zeros((1, 3, 4, 4), dtype=torch.bool)
    relation_mask[0, 0, 0, 1] = True
    relation_mask[0, 2, 0, 1] = True
    geometry = torch.zeros((1, 3, 4, 4, 12))
    valid = torch.ones((1, 4), dtype=torch.bool)

    assert validate_fixed_graph_inputs(contract, relation_mask, geometry, valid) == 2

    relation_mask[0, 1, 1, 2] = True
    with pytest.raises(ValueError, match="max_edges"):
        validate_fixed_graph_inputs(contract, relation_mask, geometry, valid)

    relation_mask[0, 1, 1, 2] = False
    valid[0, 1] = False
    with pytest.raises(ValueError, match="padded node"):
        validate_fixed_graph_inputs(contract, relation_mask, geometry, valid)
