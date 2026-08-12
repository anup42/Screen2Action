# AGENTS.md - Screen2Action

## Project objective

Implement the Screen2Action paper as a CPU-first, configuration-driven PyTorch
project, with later GPU training and mobile export.

## Sources of truth

- `docs/PAPER_SPEC.md` contains paper-specified behavior.
- `docs/TRACEABILITY.md` maps paper sections to code and tests.
- `docs/DECISIONS.md` records reconstruction choices where the paper is silent.
- Never present a reconstruction choice as paper-specified.

## Required read order

Before cross-cutting work, read in order: `AGENTS.md`, `HANDOFF.md`, the active
ExecPlan linked there, `docs/PAPER_SPEC.md`, `docs/ARCHITECTURE.md`,
`docs/REPO_MAP.md`, `docs/TRACEABILITY.md`, then `docs/DECISIONS.md`,
`docs/OPEN_QUESTIONS.md`, and the relevant data/model/training runbooks.

Cross-cutting work requires a living ExecPlan under `docs/plans/` following
`.agent/PLANS.md`. Keep progress, discoveries, decisions, validation evidence,
and outcomes current throughout implementation.

## Hard constraints

- Every supported workflow must run on CPU; CUDA is optional.
- Never use direct `.cuda()` calls or unconditional CUDA imports.
- Internal boxes use normalized `xyxy`; conversions occur only at boundaries.
- Raw screenshots, datasets, checkpoints and secrets are never committed.
- Tests must not require network access.
- Do not add a production dependency without documenting why.
- Do not modify unrelated code.

## Engineering standards

- Use typed public APIs and dataclasses or validated models for schemas.
- Keep model and training constants in configuration, not source files.
- Add tests for every behavior change.
- Geometry and selector code require deterministic golden tests.
- Neural modules require shape, mask, finite-loss and gradient tests.
- Exports require numerical parity tests against PyTorch.
- Use explicit errors rather than silent fallback for invalid graphs or budgets.

## Required verification

Run the narrow tests while working, then before completion run:

```text
make format-check
make lint
make typecheck
make test
```

On Windows without GNU Make, the equivalent is:

```text
python -m screen2action.tools.verify all
```

Run slow tests only when relevant:

```text
make test-slow
```

## Definition of done

A task is done only when implementation, tests and concise documentation are
complete; verification passes; traceability is updated; and remaining
assumptions are listed.

## Task completion report

Always end with:

1. changed files;
2. commands run;
3. test results;
4. behavior decisions or ADRs added;
5. unresolved risks.
