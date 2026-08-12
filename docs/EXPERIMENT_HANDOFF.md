# Experiment handoff protocol

Run `screen2action handoff snapshot` after verification and before transferring
the checkout. Commit source/config/docs plus the generated `HANDOFF.md` and
`handoff.json`; do not commit datasets, weights, caches, checkpoints, reports,
or credentials.

The receiving operator should:

1. read `AGENTS.md`, `HANDOFF.md`, and the active ExecPlan;
2. configure the three environment roots and install extras for the target;
3. run `screen2action doctor --device cpu|cuda --json`;
4. verify model lock, data manifest, perception cache, and checkpoint digests;
5. run the exact resume or evaluate command from `handoff.json`;
6. refresh the snapshot after new evidence exists.

Snapshot statuses distinguish locally verified work, implemented opt-in gates
not run on this host, and blockers requiring a license, credential, user-owned
data, model asset, or device. Test fixtures and dry runs never count as real
model/data/GPU evidence. Reported accuracy, latency, memory, disk, and
parameter measurements must cite the profile, artifact digests, hardware,
dataset split, and command that produced them.
