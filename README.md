# Screen2Action

CPU-first reference implementation of the structured semantic bottleneck
architecture described in *Screen2Action: Structured Semantic Bottleneck for
On-Device GUI Grounding*.

The repository is built in explicit fidelity tiers. The verified CPU
architecture milestone includes oracle perception, deterministic SSB, budget
selection, retrieval, crop transforms, sparse grounding, losses, tiny
training, checkpoint resume, evaluation, calibration, and tiny ONNX parity.
Production public-model integration, paper-reference training, and deployment
are tracked separately in `docs/TRACEABILITY.md`.

## Quick start

The core has no network or dataset requirement. With Python 3.12:

```text
python -m pip install -e .[cpu,dev,export]
python -m pytest
python -m screen2action.tools.verify all
python -m screen2action handoff snapshot
```

The installed console and module entry points are identical. Discover the
workflow with `screen2action --help`; useful first checks are:

```text
screen2action doctor --device cpu --json
screen2action config validate --config configs/model/tiny_cpu.yaml --json
screen2action train smoke --device cpu --json
```

Manifest-bound stage training is available through `screen2action train run`
and `torchrun -m screen2action.training.launch`. The checked-in experiment
configs cover Stage 1 semantic/native perception, Stage 2 cached selection and
grounding, Stage 3 joint reconstruction, and optional Stage 4 QAT. See
`docs/GPU_RUNBOOK.md` for the immutable-input and exact-resume workflow.

Checkpoint evaluation uses the same frozen data and perception manifests and
emits JSON metrics, CSV/Parquet predictions, Markdown, resolved config, and
provenance. Confidence calibration is a separate validation-only command;
ScreenSpot is rejected from calibration. See `docs/EVALUATION_RUNBOOK.md`.

Configuration supports recursive `extends`, `${ENV_VAR}` or
`${ENV_VAR:-default}` expansion, repeatable `--set dotted.key=value`
overrides, schema validation, and resolved snapshots in run directories.

The `export` extra supplies ONNX and ONNXRuntime for the CPU numerical parity
test; the core runtime itself does not require either package.

If GNU Make is available, the same checks are exposed as `make test`,
`make lint`, `make typecheck`, and `make verify-cpu`.

## Repository contract

- `docs/PAPER_SPEC.md` separates paper facts from reconstruction defaults.
- `docs/TRACEABILITY.md` connects the plan to implementation and tests.
- `docs/DECISIONS.md` records choices that the paper does not specify.
- `HANDOFF.md` links the active plan and exact continuation commands.
- `configs/` contains model, data, training, and evaluation defaults.
- `src/screen2action/ssb/` contains the deterministic CPU core.
- `tests/` contains golden and invariant tests; no external data is needed.

Raw screenshots, datasets, checkpoints, generated exports, and experiment
outputs must remain outside Git.

## Current status

The CPU architecture milestone is implemented and tested. Tiny ONNX parity is
not paper-reference/mobile export. Public ScreenSpot and mobile-latency numbers
in the paper are diagnostic targets, not claims made by this repository.
