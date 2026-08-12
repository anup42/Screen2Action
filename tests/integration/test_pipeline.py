from __future__ import annotations

from screen2action.data.synthetic import make_synthetic_examples
from screen2action.data.tokenizer import VocabularyTokenizer
from screen2action.perception.oracle import OraclePerception
from screen2action.runtime.pipeline import PipelineConfig, Screen2ActionPipeline
from screen2action.ssb.codec import SsbCodec


def test_synthetic_oracle_pipeline_reaches_screen_point() -> None:
    examples = make_synthetic_examples()
    tokenizer = VocabularyTokenizer.from_corpus(example.command.text for example in examples)
    oracle = OraclePerception(tokenizer=tokenizer)
    frame = oracle.perceive(examples[0].screenshot, examples[0].screen)
    pipeline = Screen2ActionPipeline(tokenizer=tokenizer, config=PipelineConfig.tiny_cpu())
    state = pipeline.prepare_oracle_frame(frame)
    result = pipeline.ground(state, examples[0].command.text)
    assert state.ssb.total_cost <= pipeline.config.ssb_budget
    assert state.ssb.total_cost == len(state.ssb.tokens)
    assert result.node_id in state.ssb.ordered_node_ids
    assert result.point_xy_norm is not None
    assert all(0.0 <= value <= 1.0 for value in result.point_xy_norm)
    assert SsbCodec().decode(state.ssb.tokens).tokens_consumed == state.ssb.total_cost


def test_selector_and_ssb_are_command_independent() -> None:
    examples = make_synthetic_examples()
    tokenizer = VocabularyTokenizer.from_corpus(example.command.text for example in examples)
    frame = OraclePerception(tokenizer=tokenizer).perceive(
        examples[0].screenshot, examples[0].screen
    )
    pipeline = Screen2ActionPipeline(tokenizer=tokenizer, config=PipelineConfig.tiny_cpu())
    first = pipeline.prepare_oracle_frame(frame)
    selected_before = first.selection.selected_node_ids
    ssb_before = first.ssb.tokens
    cost_before = first.selection.total_cost
    pipeline.ground(first, examples[0].command.text)
    pipeline.ground(first, examples[1].command.text)
    assert first.selection.selected_node_ids == selected_before
    assert first.ssb.tokens == ssb_before
    assert first.selection.total_cost == cost_before
