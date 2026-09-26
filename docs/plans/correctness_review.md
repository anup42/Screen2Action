# Implementation correctness review

**Status:** IN PROGRESS
**Active milestone:** Publication
**Started:** 2026-09-26

## Purpose and observable outcome

Review the completed implementation against the paper and repository contracts,
fix demonstrated defects, verify the changes offline, and publish a scoped
commit to the existing GitHub repository.

## Truth hierarchy and constraints

Follow `AGENTS.md`, `PAPER_SPEC.md`, accepted reconstruction decisions, and
traceability. Preserve normalized geometry, CPU support, offline tests, explicit
errors, command-independent frame state, and unrelated local artifacts. Public
model, dataset, GPU, and mobile results require direct evidence.

## Baseline

The checkout is `main` at `0866d62` with no tracked changes. The prior plan is
complete with 137 offline tests recorded; current verification is pending.
Several pre-existing inaccessible local output directories remain out of scope.

## Milestones

- [x] Review model/training, data/evaluation, and export contracts;
      reproduce actionable defects with focused regression tests.
- [x] Fix verified defects and update traceability and operational documentation.
- [x] Run full verification and relevant slow tests.
- [ ] IN PROGRESS: Refresh handoff, commit, push, and verify remote commit identity.

## Progress

- 2026-09-26: Read the required project documents and runbooks, confirmed the
  clean tracked baseline and existing remote, and started the correctness audit.
- 2026-09-26: Regression tests reproduced missing retrieval gradients, action-label
  leakage into candidates, incorrect long-press/drag coordinate targets, unmasked
  drag slots, and K=1/4 evaluation silently using K=8. Added fixes and checks for
  frozen metadata additions, linked calibration artifacts, and partial labels.
- 2026-09-26: Found FP16 mask overflow with a direct CPU half-tensor reproduction;
  made masking and empty-neighborhood normalization representable in FP16.
- 2026-09-26: Expanded distributed coverage to the actual unified model and uneven
  validation shards with BatchNorm buffers. Restore checkpoint RNG on CPU before
  moving optimizer state to the requested device.
- 2026-09-26: The expanded DDP test reproduced a reducer failure on the second
  optimizer iteration. An optimization-output adapter now exposes only the
  total-loss graph to DDP while preserving public model diagnostic gradients.
  Two-rank training with different label masks and uneven validation now passes.
- 2026-09-26: Corrected frozen crop BatchNorm behavior and preserved the original
  initialization digest on exact resume. Added ADR-0027, exposed loss weights in
  experiment YAML, and updated traceability and training guidance.
- 2026-09-26: Final offline verifier and separate paper-dimension ONNX parity
  passed. Implementation, regression tests, and documentation are ready to publish.

## Discoveries and surprises

- Baseline format and Ruff passed, but Mypy rejected nine stale import ignores
  with the currently installed stubs. Removed those suppressions.
- Initial pytest temporary roots were inaccessible; isolated ignored workspace
  roots avoid those pre-existing paths. Data registration then hit Windows path
  length limits because the revision and two UUIDs were repeated in temporary
  names. Local registration now uses a UUID-only staging directory.

## Decision log

- This is a correctness audit of implemented contracts. No new paper claims,
  dependencies, or external model/data acquisition are required.
- ADR-0027 records the supervised retrieval reconstruction and inference,
  supervision, distributed execution, resume, and artifact-integrity boundaries.

## Validation ledger

- `python -m screen2action.tools.verify all`: exit 1 at Mypy (nine stale ignores).
- Initial focused regression run: six expected assertion failures, 14 passes,
  and five temporary-directory permission errors.
- Focused run after fixes: 23 passed, two Windows path-length failures in local
  data registration; the staging path has subsequently been shortened.
- Expanded focused run: 26 passed; unified-model DDP reducer failure reproduced.
- After DDP correction: 21 passed; one new test used the wrong sweep-artifact
  path. Corrected the test to inspect `sweep/jobs/budget-512-k-1/predictions.csv`.
- First full post-fix verifier: format/Ruff/Mypy passed; 149 passed, ten opt-in
  tests deselected, one configuration test assumed all temporary files live
  outside the repository. Its assertion now checks portable relative paths and
  ordered source names regardless of the test root location.
- Final focused training/semantics run: 20 passed and one missing test import;
  added the import and reran the full verifier below.
- Final `python -m screen2action.tools.verify all`: exit 0; 177 Python files
  formatted, Ruff passed, Mypy passed on 123 source files, **152 tests passed**,
  ten opt-in tests deselected, 67 framework/export warnings; 51.17 seconds.
  Environment: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `PYTEST_ADDOPTS="-p no:cacheprovider --basetemp=outputs/r7"`.
- `python -m pytest tests/slow/test_paper_export_parity.py -m slow -q --tb=short
  -p no:cacheprovider --basetemp=outputs/rs1`: exit 0; **one passed**, 20
  framework/export warnings; 8.09 seconds. Covers paper-dimension graph and
  command partitions with locally initialized weights on CPU.
- `git -c core.safecrlf=false diff --check`: exit 0.

## Risks and continuation

Real pretrained assets, public data, CUDA training, and mobile runtime validation
remain external gates. On a CUDA host run `screen2action doctor --device cuda
--json` and `python -m pytest tests/unit/test_checkpoint.py -m gpu -q`, then
follow `docs/GPU_RUNBOOK.md`. CUDA RNG restoration is implemented but not
executed on this CPU host. Fast ROI remains explicitly unsupported.

The corrected retrieval objective requires a new training run; initialize from
old weights if desired, but do not claim an exact resume across changed code or
configuration. Regenerate prior small-K or action-label-conditioned evaluation
reports and their dependent calibration artifacts.

Publication continuation: review the staged diff, commit, `git push origin main`,
and compare `git rev-parse HEAD` with `git ls-remote origin refs/heads/main`.

## Outcome

Correctness fixes are implemented and locally verified. Publication is pending.
