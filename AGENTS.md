# Repository guidance for Codex

## Project purpose

This repository studies learning from demonstration for bimanual manipulation with two Franka
FR3 arms in DuoBench/RCS/MuJoCo. The active data-collection work is the scripted
`duobench/pour_marbles` controller and its two-arm joint-state demonstrations. The repository
also contains the earlier custom `bimanual_rope/sling_hook` environment.

Before changing the pour-marbles controller, generators, recording format, or dataset tooling,
read [`docs/CODEX_HANDOFF.md`](docs/CODEX_HANDOFF.md). It records the decisions and empirical
failure modes that produced the current successful controller.

## Environment and commands

- Target platform: Linux x86-64.
- Conda environment: `bimanual-rope` from `environment.yml`.
- Prefer `conda run -n bimanual-rope <command>` for reproducible non-interactive checks.
- Headless controller runs set `MUJOCO_GL=egl` themselves. Interactive `--render` runs require a
  local display and must not be forced through EGL.
- DuoBench/RCS assets are cached outside the repository under `~/.duobench` and `~/.rcs` unless
  the documented prefix variables are set.

Useful checks:

```bash
conda run -n bimanual-rope python -m py_compile \
  scripts/pour_marbles_controller.py \
  scripts/generate_pour_marbles_demos.py \
  scripts/generate_pour_marbles_variants.py \
  scripts/audit_pour_marbles_demos.py \
  scripts/migrate_pour_marbles_v2.py

conda run -n bimanual-rope python scripts/audit_pour_marbles_demos.py
```

Run the targeted checks above after pour-marbles changes. Run the smoke tests documented in
`README.md` when changing environment setup, dependencies, or the custom sling task.

## Pour-marbles behavioral invariants

- A recorded demonstration is acceptable only if it reaches stage 6, transfers all 20 marbles
  into the target cup with zero left in the source cup, and leaves both cups in their target
  squares and upright.
- `--record` must never publish a partial or merely task-latched success. Keep the strict
  acceptance gate in `strict_demonstration_success` and independently validate generated files.
- Preserve both pour directions. Dataset generation targets an exact 50/50 left-source and
  right-source balance for a 100-file corpus.
- Preserve the approximately synchronized setup motion: the source cup rises while the receiver
  moves inward, then both cups move toward the center together. A short final source-only drop is
  intentional. Do not regress to two visibly sequential long moves.
- The post-release retreat is diagonally outward and upward. A vertical retreat can catch a cup
  with the gripper or wrist and tip it after an otherwise successful placement.
- Abort after a failed grasp/lift instead of attempting the rest of the choreography.

## State and trajectory semantics

- This project saves an initial object state, not a mid-episode simulator checkpoint.
- Canonical recordings are pickle-free NPZ format version 3.
- Restore cup and marble positions **and velocities**. Fresh DuoBench reset scenes are not fully
  settled, so replacing captured velocity with zero can change the outcome. Legacy v2 pose-only
  states intentionally fall back to zero velocity for backward compatibility.
- Do not restore robot dynamics. The arms start from the environment's standard reset. Initial
  robot joint positions are recorded for inspection, but object-state load/replay does not apply
  them.
- Keep environment reset `seed` separate from unique dataset `record_id`.
- For learning, `left_joint_qpos` and `right_joint_qpos` are the measured post-step trajectories.
  `left_joint_targets`, `right_joint_targets`, gripper commands, combined `actions`, and strictly
  increasing `timestamps_s` are recorded separately.

## Dataset integrity and generation

- Default dataset directory: `data/duo/pour_marbles`.
- `*.npz` is ignored by Git. Never assume cloning the repository transfers demonstrations.
- Candidate recordings must remain temporary and be atomically promoted only after validation.
- Generation is resumable and must stop at the exact requested count; do not silently add files
  beyond it.
- Keep generation at no more than four concurrent MuJoCo workers on a roughly 14 GiB machine
  unless memory use has first been measured. Four simulators already use most available RAM.
- Preserve validated data. Do not delete, replace, or bulk-migrate recordings without resolving
  exact targets and validating replacements.
- `generate_pour_marbles_variants.py` depends on proven base recordings with IDs `000000`
  (right source) and `000030` (left source).
- The current `--motion-variant` value is dataset metadata; task-critical motion parameters remain
  on the proven nominal path because endpoint noise caused placement failures.

## Research-quality cautions

- The audited 100-file corpus contains 20 unique measured joint trajectories across 9 initial
  scenes, not 100 statistically independent motions. Do not make a random sample-level train/test
  split that places replay-equivalent trajectories on both sides.
- Group evaluation splits by initial-scene and/or joint-trajectory fingerprint and stratify by
  `source_cup`.
- If more diversity is needed, generate and validate genuinely different initial scenes or
  choreography parameters. Never manufacture diversity by adding noise to recorded joint states
  while retaining success labels.
- Random DuoBench seeds are not uniformly controllable by the current grasp. Small geometric or
  motion perturbations require executed validation; many apparently reasonable variants fail.

## Working style

- Keep edits scoped and preserve unrelated user changes.
- Prefer measured evidence from an executed simulation or the audit tool over assumptions about
  replay fidelity or success.
- Update `docs/CODEX_HANDOFF.md` when a decision, schema, validated base state, known failure mode,
  or dataset result materially changes.
