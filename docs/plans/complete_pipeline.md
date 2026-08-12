# Complete production, data, and GPU-training pipeline

**Status:** ACTIVE
**Active milestone:** Milestone 2 - model registry and reproducible assets
**Started:** 2026-08-12
**Specification:** `Screen2Action_Remaining_Gaps_Codex_Prompt.md` supplied by
the user (external input; not copied into the repository)

## Purpose and observable outcome

Turn the verified CPU architecture scaffold into a production-oriented,
configuration-driven Screen2Action repository that can be transferred to a
GPU machine and resumed from repository documentation alone. A user must be
able to inspect the environment, resolve optional public model assets,
register or download licensed data, normalize/audit it, precompute
command-independent perception, run stage-aware training, evaluate and
calibrate, export supported subgraphs, and produce a reproducible handoff.

The implementation must not claim the paper's 80.2% ScreenSpot accuracy, 185M
parameter count, 190 MB disk size, or Galaxy latency without direct
measurement under the stated protocol.

## Truth hierarchy

1. `docs/PAPER_SPEC.md` for facts explicitly stated by the paper.
2. The architecture/equations and locked public reconstruction choices in the
   user-supplied remaining-gaps specification.
3. Existing deterministic invariants and public APIs where compatible.
4. Accepted decisions in `docs/DECISIONS.md`.
5. New, visibly labeled reconstruction decisions recorded during this plan.

## Non-negotiable constraints

- CPU is supported by every public API and default smoke test.
- No direct model `.cuda()` calls and no unconditional CUDA imports.
- Default tests require no network, model weights, datasets, credentials, or
  GPU.
- Heavy dependencies are lazy and optional.
- Internal boxes remain normalized `xyxy`.
- ScreenSpot is evaluation-only and protected by a hard leakage guard.
- Raw data, screenshots, weights, caches, checkpoints, secrets, and
  machine-specific paths are excluded from Git.
- Downloads are explicit, license-gated, resumable, revision/checksum aware.
- Missing supervision uses masks.
- The deterministic graph/SSB/knapsack/closure implementation remains intact.
- Commands provide help, structured errors, optional JSON, and dry-run where
  mutation/download is meaningful.

## Baseline observed before this plan

- Branch `main` exists but has no commits; the implementation is untracked.
- The CPU baseline was rerun on 2026-08-12: 41 tests passed with full offline
  formatting, lint, and type-check verification.
- The repository contains canonical schemas, deterministic SSB and selection,
  tiny neural modules, oracle perception, losses/metrics/calibration,
  checkpointing, and tiny ONNX parity.
- It lacks the production model registry, real perception, unified trainable
  model, canonical data lake/source adapters, scalable cache, full DDP stage
  runner, production evaluation reports, and complete handoff.
- Several ignored/local presentation and test-output directories are present
  outside the source scope; they must not be staged or modified by this plan.

## Milestones and acceptance evidence

### Milestone 0 - repository truth and resumable handoff (COMPLETE)

- [x] Add `.agent/PLANS.md` and this living ExecPlan.
- [x] Require the read order and ExecPlan discipline in `AGENTS.md`.
- [x] Audit `docs/TRACEABILITY.md` into CPU, production, training, and
      deployment dimensions.
- [x] Reconcile tiny ONNX completion with paper-reference export status.
- [x] Add architecture/repository/provenance/data/training/GPU/handoff docs.
- [x] Add `.env.example`, model registry, and source registry.
- [x] Implement `screen2action handoff snapshot` and generate `HANDOFF.md` plus
      `handoff.json` using relative/environment-root paths.
- [x] Focused tests and full offline verification pass.
- [x] Prepare a Git checkpoint containing only repository source/config/docs.

### Milestone 1 - typed configuration, CLI, and environment diagnostics (COMPLETE)

- [x] Stable schema-versioned YAML composition with environment expansion,
      repeatable dotted overrides, validation, and resolved snapshots.
- [x] One `screen2action` CLI and matching `python -m screen2action` entrypoint
      exposing every required command group.
- [x] `doctor --device cpu|cuda --json` reports environment, roots, optional
      features, assets/manifests/caches/checkpoints, warnings, and next command.
- [x] Package extras do not force a CUDA-specific PyTorch wheel.
- [x] Idempotent CPU/GPU bootstrap and smoke shell scripts.
- [x] CLI/config/doctor tests and full offline verification pass.

### Milestone 2 - model registry and reproducible assets (IN PROGRESS)

- [ ] Registry covers ScreenParser, docTR CRNN, MobileNetV3-small, compact
      BERT, and MobileViT-S with licenses and I/O contracts.
- [ ] Immutable lock entries record resolved revisions/files/SHA256 digests.
- [ ] Resumable cache-root fetch and fully offline verify paths.
- [ ] Mocked unit tests; opt-in `network`/`gpu` smoke tests.
- [ ] Full offline verification and checkpoint.

### Milestone 3 - production perception and taxonomy

- [ ] Typed detector/OCR/text-detector protocols and normalized image boundary.
- [ ] ScreenParser conversion/class mapping/diagnostics/root/container policy.
- [ ] docTR CRNN crop normalization, aspect batching, ordering, diagnostics,
      and deterministic OCR association.
- [ ] Shared MobileNetV3 icon/actionability/visual module with masked losses.
- [ ] Inspect/build/freeze icon taxonomy workflow with immutable 87-class
      reconstruction manifest and review report.
- [ ] Command-independent real/oracle/cached `FullScreenPerception`.
- [ ] Atomic, sharded, content-addressed perception cache with retries.
- [ ] Offline fake-backbone/cache tests and opt-in real-model tests.

### Milestone 4 - unified trainable paper-reference model

- [ ] `Screen2ActionModel` separates perceive/encode/ground/forward and permits
      caller-controlled train/eval/freezing.
- [ ] Batched multi-screen/multi-command representation and grouped sampler.
- [ ] 256-D multimodal node fusion with provenance/missing-modality masks.
- [ ] Explicit paper-equation and CPU-reconstruction graph/reranker variants;
      edge-list/dense/batched parity.
- [ ] Learned command-independent retention connected to Gumbel training and
      deterministic inference selection.
- [ ] Lazy compact BERT and MobileViT adapters with fake-backbone tests.
- [ ] MobileViT contract `[B*K, 144, 256]` and shape probe.
- [ ] 2-D box encoding, candidate/crop masks, grouped sparse isolation.
- [ ] Typed click/long-press/scroll/drag/confidence heads and masked losses.
- [ ] Tiny cached forward/backward and model command-independence tests.

### Milestone 5 - canonical data lake and audits

- [ ] Backward-compatible provenance-rich schemas and environment-root layout.
- [ ] Source registry and fixture adapters for GUIAct, GUIEnv, AMEX,
      AndroidControl, WaveUI, RICO Semantics, and ScreenSpot.
- [ ] Explicit license-gated resumable download/register-local boundaries.
- [ ] App-disjoint split creation with alias/collision audit.
- [ ] Scalable staged dedup grouping/survivor/removal audit.
- [ ] Hard ScreenSpot train/calibration/model-selection leakage guard.
- [ ] `public_weak_reference_v1` annotation with confidence/provenance.
- [ ] Manifest/shard creation and digest enforcement.
- [ ] CLI fixture workflow exercises register -> normalize -> validate -> split
      -> dedup -> manifest without copyrighted assets.

### Milestone 6 - resumable perception precompute

- [ ] Unique-screen deterministic sharding, atomic writes, resume/retry,
      failure logs, compatibility digests, cache validation/stats.
- [ ] CPU sample path and CUDA/multiprocess-ready path.
- [ ] Stage-2 cached loader does not instantiate ScreenParser/docTR.
- [ ] Resume/atomicity/command-independence tests.

### Milestone 7 - GPU/DDP stage training

- [ ] Torchrun local-rank setup, Gloo/NCCL selection, DDP, samplers, cleanup,
      no-sync accumulation, all-reduced metrics, deterministic seeds.
- [ ] Stage-aware freezing and Stage 1 submodes, Stage 2 schedule, Stage 3
      non-OCR joint path, optional Stage 4 quantization.
- [ ] Correct global-batch accumulation including final partial flush.
- [ ] Conditional BF16/FP16/scaler/clip, validation, latest/best rank-zero
      checkpoints, exact resume including sampler state.
- [ ] JSONL/TensorBoard/run-manifest timing/throughput/VRAM/ETA reporting.
- [ ] CPU smoke, batch probe, one-process and supported two-process tests.

### Milestone 8 - evaluation, calibration, sweeps, and reports

- [ ] ScreenSpot evaluation-only adapter and official point-in-target path.
- [ ] Direct cascade/subset/action/confidence/latency/memory/failure metrics.
- [ ] Per-example CSV/Parquet, JSON metrics, Markdown report, and provenance.
- [ ] Validation-only temperature/threshold artifacts tied to digests.
- [ ] B/K, selector/loss/relation/model/quantization ablations from config.
- [ ] Offline report/calibration/leakage tests.

### Milestone 9 - partitioned export/runtime

- [ ] Explicit host operations and four logical export partitions.
- [ ] Fixed-shape contracts and CPU PyTorch/export parity for supported
      paper-reference subgraphs.
- [ ] Clear partition report for unsupported operations.
- [ ] Accurate/Fast artifacts and metrics remain separate.
- [ ] Export tests and complete handoff.

## Progress log

- **2026-08-12:** Read the complete 1,006-line task specification, inspected
  the uncommitted repository baseline, and created the required plan rules and
  active ExecPlan. No production-integration gate has yet been claimed.
- **2026-08-12:** Reverified the preserved CPU baseline (41 tests), split
  traceability by fidelity tier, corrected the ONNX contradiction, added the
  operational/provenance documents and registries, and implemented the
  portable human/machine handoff generator.
- **2026-08-12:** Milestone 0 focused handoff tests passed (2 tests); the full
  verifier passed with 43 tests. GitHub connector access confirmed admin/push
  permission to the user-supplied private repository. Milestone 1 is active.
- **2026-08-12:** Milestone 0 was pushed to `anup42/Screen2Action`; local and
  remote `main` matched at `e98440e59f1e2bc2a6554b198ff453ed1e272b5a`.
- **2026-08-12:** Added schema-versioned YAML composition/overrides/snapshots,
  all required CLI paths and help, a secret-safe CPU/CUDA doctor, feature
  extras, bootstrap/smoke scripts, and a CPU training/checkpoint/resume smoke.
  Milestone 1 focused tests (8) and the full suite (51) passed. Milestone 2 is
  active.

## Discoveries and surprises

- Git is initialized but `main` has no commit. Milestone checkpoints therefore
  need a carefully scoped first source commit rather than a diff against an
  existing baseline.
- The workspace contains local presentation/test-output directories with
  malformed flattened names and at least one inaccessible path. They are
  outside repository source scope and will be ignored, not cleaned up.
- The task prompt's baseline test count was stale: current baseline collection
  is 41 tests, all passing before Milestone 0 changes.

## Decision log

- **D-001 (process reconstruction):** Follow milestone dependencies
  sequentially, while implementing independent files/tests in parallel where
  safe. One writer owns each file and the full verifier is the merge gate.
- **D-002 (Git checkpoint):** Because no initial commit exists, the first
  checkpoint will include the preserved CPU scaffold plus Milestone 0 source
  artifacts, excluding ignored/local outputs. Later checkpoints will be
  milestone-scoped.
- **D-003 (external gates):** Real model, public data, CUDA/DDP, and device
  checks are opt-in. Offline fake/fixture boundaries can be complete locally;
  real external gates remain explicitly unverified until run.
- **D-004 (publication):** Push scoped milestone checkpoints to the private
  `anup42/Screen2Action` repository. Because the remote is empty, establish
  `main` directly; no pull request has a meaningful base for the initial
  checkpoint. Exclude all local presentation/output directories.
- **D-005 (configuration):** Keep YAML as the only core runtime dependency.
  All heavy model/data/training/export imports remain lazy and feature-gated;
  no package metadata selects a CUDA wheel.

## Validation ledger

- **2026-08-12:** `python -m screen2action.tools.verify all` - exit 0; Ruff,
  Mypy, format/compile checks, and 41 offline tests passed in 8.9 seconds.
- **2026-08-12:** `python -m pytest tests/unit/test_handoff.py -q` - exit 0;
  2 focused tests passed.
- **2026-08-12:** `python -m screen2action.tools.verify all` - exit 0; Ruff,
  Mypy, format/compile checks, and 43 offline tests passed in 13.2 seconds.
- **2026-08-12:** `python -m pytest tests/unit/test_config.py
  tests/unit/test_cli.py tests/unit/test_doctor.py
  tests/unit/test_training_smoke.py -q` - exit 0; 8 focused tests passed.
- **2026-08-12:** `screen2action doctor --device cpu --json` - exit 0; CPU
  PyTorch, root writability, RAM/disk, optional features, and unresolved model
  lock reported without absolute configured paths or credential values.
- **2026-08-12:** `screen2action config validate --config
  configs/model/tiny_cpu.yaml --set retrieval_top_k=5 --json` - exit 0;
  schema 1 validated with deterministic digest.
- **2026-08-12:** `screen2action train smoke --device cpu --steps 2 --json` -
  exit 0; finite forward/backward loss and exact checkpoint/resume parity.
- **2026-08-12:** `python -m screen2action.tools.verify all` - exit 0; Ruff,
  Mypy, format/compile checks, and 51 offline tests passed in 9.5 seconds.
- **2026-08-12:** `python -m screen2action.tools.verify all` after final doctor
  package-version reporting - exit 0; 51 tests and all static checks passed in
  9.9 seconds.

## Remaining blockers and continuation

Current host is CPU-oriented and no public data/model license acceptance was
provided. Full downloads and GPU runs are intentionally excluded. The exact
first GPU-machine command will be generated after the doctor, asset, data,
cache, and stage-runner interfaces exist.

## Outcome summary

Active; no outcome claimed.
