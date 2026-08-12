# Screen2Action ExecPlan rules

An ExecPlan is required for work that crosses two or more of the data,
perception, model, training, evaluation, export, CLI, or documentation
boundaries. The active plan is linked from `HANDOFF.md` and must be usable by a
new contributor without access to prior chat.

## Required sections

Every ExecPlan must contain:

1. purpose and externally observable outcome;
2. truth hierarchy and non-negotiable constraints;
3. current repository baseline;
4. milestone checklist with acceptance evidence;
5. progress log with timestamps;
6. discoveries and surprises;
7. decision log separating paper facts from reconstructions;
8. validation commands and observed results;
9. remaining risks, blockers, and exact continuation command;
10. outcome summary when work finishes.

## Operating rules

- Keep the plan in the repository under `docs/plans/`.
- Update it before beginning a milestone and after every focused/full test run.
- Record observed output, not intended output. A command that was not run is
  never listed as passing.
- Use repository-relative paths and environment-variable roots. Do not record
  secrets, access tokens, user-specific cache paths, or raw data paths.
- Mark public reconstruction decisions explicitly and mirror durable decisions
  in `docs/DECISIONS.md`.
- Keep paper claims in `docs/PAPER_SPEC.md`; unresolved details belong in
  `docs/OPEN_QUESTIONS.md`.
- A milestone can be marked complete only when its implementation, tests,
  documentation, traceability, and handoff state agree.
- Preserve the deterministic SSB core unless a demonstrated defect requires a
  narrow correction.
- Default tests remain offline. Network, GPU, and integration-model tests must
  be opt-in and marked.
- Before a milestone checkpoint, run focused tests and
  `python -m screen2action.tools.verify all`, review the staged scope, and
  update the handoff snapshot.
- If a required real-data/model/GPU gate cannot run on the current host,
  implement the offline boundary and fixtures, record the unverified gate, and
  provide the exact operator command. Do not report the gate as complete.

## Plan maintenance

Use checkboxes for deliverables and keep exactly one milestone marked
`IN PROGRESS`. Append progress entries instead of rewriting history. When a
decision changes, retain the old entry with a superseded note and add the new
decision. The outcome section must distinguish:

- implemented and locally verified;
- implemented but requiring opt-in network/model/GPU verification;
- intentionally deferred because an external license, credential, dataset, or
  device is required.
