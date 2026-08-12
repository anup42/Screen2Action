# Repository map

| Path | Responsibility | Generated or external state |
| --- | --- | --- |
| `src/screen2action/data/` | schema/layout/storage, immutable assets, seven source adapters, app splits, dedup, weak references, manifests, shards | data lake external |
| `src/screen2action/perception/` | detector/OCR/icon protocols, orchestration, model factory, hash-sharded precompute, cache-only loader | weights/cache external |
| `src/screen2action/ssb/` | geometry, typed relations, codec, BudgetSelect, validation | none |
| `src/screen2action/models/` | node/graph/command/crop/retrieval/grounding modules | weights external |
| `src/screen2action/training/` | staged trainer, distributed launch, checkpoints | runs/checkpoints external |
| `src/screen2action/eval/` | metrics, calibration, sweeps, reports | reports generated |
| `src/screen2action/export/` | static contracts, ONNX/quantization/parity | artifacts external |
| `src/screen2action/cli.py` | shared console and `python -m` command tree | none |
| `configs/` | versioned reconstruction intent and experiment profiles | model lock is generated |
| `docs/` | paper boundary, ADRs, traceability, runbooks, active plan | handoff is generated |
| `tests/` | offline unit/integration fixtures and opt-in markers | no public data |
| `scripts/` | idempotent operator bootstrap/smoke helpers | none |
| `HANDOFF.md`, `handoff.json` | human/machine continuation snapshot | generated and versioned |

## Environment roots

- `${SCREEN2ACTION_DATA_ROOT}`: immutable raw sources, normalized datasets,
  shards, data manifests, and source license acknowledgements.
- `${SCREEN2ACTION_CACHE_ROOT}`: resolved model weights and perception caches.
- `${SCREEN2ACTION_RUN_ROOT}`: resolved configs, logs, checkpoints, calibration,
  evaluation reports, and export artifacts.

Defaults are repository-local ignored directories for fixture/smoke use. A
production operator should set all three roots explicitly.

## First files to read

Follow the order in `AGENTS.md`. The active continuation command and observed
state are always in `HANDOFF.md`; detailed milestone checkboxes and validation
evidence are in `docs/plans/complete_pipeline.md`.
