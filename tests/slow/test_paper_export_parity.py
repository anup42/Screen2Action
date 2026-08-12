"""Opt-in CPU parity for locally constructible paper-shape neural partitions."""

from __future__ import annotations

import pytest
import torch

from screen2action.export.onnx import export_onnx
from screen2action.export.parity import onnx_parity
from screen2action.export.partitions import (
    COMMAND_INPUT_NAMES,
    COMMAND_OUTPUT_NAMES,
    GRAPH_INPUT_NAMES,
    GRAPH_OUTPUT_NAMES,
    CommandRetrievalPartition,
    GraphRetentionPartition,
)
from screen2action.export.runner import _command_inputs, _graph_inputs
from screen2action.export.static_shapes import ExportShapeContract
from screen2action.models.screen2action_model import Screen2ActionModel, Screen2ActionModelConfig


@pytest.mark.slow
@pytest.mark.integration_model
def test_paper_shape_graph_and_command_partitions_have_cpu_onnx_parity(tmp_path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    torch.manual_seed(31)
    config = Screen2ActionModelConfig.paper_reference()
    model = Screen2ActionModel(config=config).eval()
    contract = ExportShapeContract(
        max_nodes=8,
        max_edges=32,
        max_command_tokens=64,
        top_k=8,
        crop_tokens=144,
        embedding_dim=256,
        node_text_tokens=16,
        icon_class_count=87,
        visual_feature_dim=256,
        crop_size=192,
        semantic_crop_size=224,
    )
    generator = torch.Generator().manual_seed(37)
    definitions = (
        (
            GraphRetentionPartition(
                model.node_encoder,
                model.graph_encoder,
                model.retention_scorer,
            ),
            _graph_inputs(contract, vocab_size=config.vocab_size, generator=generator),
            GRAPH_INPUT_NAMES,
            GRAPH_OUTPUT_NAMES,
            tmp_path / "paper-graph.onnx",
        ),
        (
            CommandRetrievalPartition(model.command_encoder, model.retriever, model.reranker),
            _command_inputs(contract, vocab_size=config.vocab_size, generator=generator),
            COMMAND_INPUT_NAMES,
            COMMAND_OUTPUT_NAMES,
            tmp_path / "paper-command.onnx",
        ),
    )

    for module, inputs, input_names, output_names, destination in definitions:
        export_onnx(
            module,
            inputs,
            destination,
            input_names=input_names,
            output_names=output_names,
        )
        errors = onnx_parity(module, destination, inputs, atol=1e-4, rtol=1e-3)
        assert max(errors) < 1e-3
