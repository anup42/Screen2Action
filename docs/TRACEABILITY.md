# Traceability

Status is dimensional. `complete` in the CPU column means an offline
architecture/fixture contract exists; it does not imply production public
models, paper-reference training, or mobile deployment. `partial` identifies
an implemented boundary with remaining required behavior. `not-started` means
no qualifying implementation evidence exists.

| Issue | Requirement and evidence | CPU architecture | Production integration | Paper-reference training | Deployment/export |
| --- | --- | --- | --- | --- | --- |
| S2A-000 | scaffold/config/verification: root, `configs/`, `tools/verify.py` | complete | partial | partial | not-started |
| S2A-001 | provenance-rich canonical schema 2.0, normalized `xyxy`, content-addressed PNG/Parquet: `data/schema.py`, `data/storage.py` | complete | partial | partial | host-only |
| S2A-002 | geometry/matching/point-in-target: `ssb/geometry.py`, `matching.py` | complete | complete | complete | host-only |
| S2A-003 | containment/proximity/ordinal graph: `relations.py`, `hierarchy.py` | complete | partial | partial | host-only |
| S2A-004 | versioned 10-bit SSB reconstruction codec: `ssb/codec.py` | complete | partial | partial | host-only |
| S2A-005 | BudgetSelect and closure repair: `ssb/selector.py`, `validation.py` | complete | partial | partial | host-only |
| S2A-006 | oracle/cached perception and synthetic fixtures | complete | partial | not-started | not-started |
| S2A-007 | node encoder/relation GAT with shape/gradient tests | complete | partial | partial | not-started |
| S2A-008 | tiny command encoder/tokenizer | complete | not-started | not-started | not-started |
| S2A-009 | cosine retrieval/reconstruction reranker/hard negatives | complete | partial | partial | not-started |
| S2A-010 | crop extraction/coordinate transforms | complete | complete | partial | host-only |
| S2A-011 | tiny sparse candidate grounder/crop encoder | complete | not-started | not-started | not-started |
| S2A-012 | objective loss functions and mask tests | complete | partial | not-started | n/a |
| S2A-013 | tiny CPU overfit workflow | complete | n/a | not-started | n/a |
| S2A-014 | seven fixture-backed public adapters plus retained synthetic framework and exact reject audits | complete | partial | partial | n/a |
| S2A-015 | oracle perception interface boundary | complete | not-started | not-started | not-started |
| S2A-016 | basic trainer/checkpoint/stage helpers | complete | partial | not-started | n/a |
| S2A-017 | metric/sweep/timing primitives | complete | partial | not-started | n/a |
| S2A-018 | confidence calibration primitive | complete | partial | not-started | n/a |
| S2A-019 | fixed-shape tiny ONNX export and ONNXRuntime parity | complete | not-started | not-started | partial |
| S2A-020 | PTQ/QAT utility interfaces | partial | not-started | not-started | not-started |
| S2A-021 | structured runtime logging/toy duplicate diagnostics | complete | partial | partial | partial |
| S2A-022 | resumable handoff/provenance/runbooks | complete | partial | partial | partial |
| S2A-023 | unified CLI/config/doctor | complete | partial | partial | partial |
| S2A-024 | model registry/lock/fetch/verify | complete | partial | partial | partial |
| S2A-025 | public production perception/taxonomy/cache: typed adapters, fake-backed integration tests | complete | partial | partial | host-only |
| S2A-026 | unified trainable model, batched masks, paper graph/reranker, retention, BERT/MobileViT adapters, typed heads | complete | partial | partial | partial |
| S2A-027 | immutable source acquisition, seven adapters, app split, scalable dedup, weak references, manifest enforcement, deterministic shards | complete | partial | partial | n/a |
| S2A-028 | resumable production perception precompute | not-started | not-started | not-started | n/a |
| S2A-029 | complete torchrun/DDP stage system | not-started | partial | not-started | n/a |
| S2A-030 | production evaluation/calibration/reporting | not-started | not-started | not-started | n/a |
| S2A-031 | four-partition paper-reference export/runtime | not-started | not-started | not-started | not-started |

`host-only` means the operation is deliberately outside a neural export graph.
`n/a` means the dimension is not applicable. Production and training columns
remain incomplete until the corresponding public assets/data and real stage
paths are exercised; no status is inferred from architecture-only tests.

## Verification mapping

- `python -m screen2action.tools.verify format` checks syntax, compilation, and
  repository formatting invariants.
- `python -m screen2action.tools.verify lint` runs Ruff when installed and
  enforces device-safety policy.
- `python -m screen2action.tools.verify typecheck` runs Mypy when installed.
- `python -m screen2action.tools.verify test` runs the default offline suite.

## Principal configuration traceability

| Configuration | Truth class | Evidence |
| --- | --- | --- |
| `B=512`, `d=256`, `K=8`, `P=144` | paper | `PAPER_SPEC.md`, `configs/model/paper_reference.yaml` |
| tiny `B=256`, `d=64`, `K=4`, `P=16` | CPU reconstruction | `configs/model/tiny_cpu.yaml`, integration tests |
| relation rerank `lambda=0.30` | paper | `PAPER_SPEC.md`, retrieval tests |
| 12/8/6 epochs, optimizer/LR/objective weights | paper/supplied contract | train configs, `training/stages.py`, losses |
| 12% per-side crop margin | reconstruction ADR-0008 | `runtime/cropper.py`, crop test |
| Stage 2 positive insertion 0.5 for first two epochs | supplied contract ADR-0005 | `training/stages.py`, adapter/sweep tests |
| public model IDs and initial detector thresholds | supplied reconstruction | `configs/models/registry.yaml`, `MODEL_PROVENANCE.md` |
| public source revisions, license notes, and conversion boundaries | upstream evidence plus reconstruction | `configs/data/sources.yaml`, `DATASET_RUNBOOK.md`, ADR-0020 |
