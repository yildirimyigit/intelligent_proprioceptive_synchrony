# Codex project handoff

Status snapshot: 2026-07-24  
Repository: `yildirimyigit/intelligent_proprioceptive_synchrony`  
Snapshot commit: `a217a47f9d3ccfd4cb15eab3e95ebba175b1f382` (`main`)

This is durable context for future Codex sessions and human collaborators. Read it before
modifying the pour-marbles controller, demonstration schema, generators, or training pipeline.
When the implementation or dataset changes, update the dated status and the affected sections.

## Research objective

The immediate objective is to collect successful bimanual `duobench/pour_marbles`
demonstrations for training a neural movement primitive model. Each demonstration must contain
the measured joint-state trajectory for both Franka FR3 manipulators, enough initial object state
to reproduce the rollout, and an unambiguous strict-success result. Both pour directions and real
trajectory variance are required.

The user explicitly chose initial-state restoration rather than resuming a mid-episode physics
checkpoint. Robot dynamics are not restored; both arms begin at the normal environment reset.

## Current implementation

### Controller

The main controller is `scripts/pour_marbles_controller.py`. It uses privileged simulator state
and RCS inverse kinematics to:

1. approach both cups from their outer sides;
2. close both grippers and measure the cup-to-TCP grasp transforms;
3. lift both cups smoothly and abort early if either grasp/lift failed;
4. bring both cups toward the pour setup with overlapping bimanual motion;
5. tilt the source cup while tracking the predicted outlet path with the receiving cup;
6. retreat the source diagonally to clear the receiver, return it upright, and carry both cups
   home;
7. place both cups upright, release, and retreat outward/upward without tipping them.

The previously sequential setup was changed deliberately. The source now rises while the target
moves inward, followed by a simultaneous move toward the center. Only a short final vertical
source adjustment remains after the target arrives. The result is visually bimanual without
requiring hard real-time synchronization.

The pour geometry uses the physical MuJoCo rim height (`0.0725 m`), not the task's larger logical
containment height. The receiver follows keyframes derived from the predicted lowest rim/outlet
path. Left-source and right-source branches use slightly different clearance schedules because
their outlet paths and final marble release differ.

The post-release retreat is also task-critical: move outward and upward. Earlier vertical
retreats could touch the cup and turn a successful placement into a failure.

### Initial state and replay

The controller supports:

```bash
# Save a reset's initial object state and exit.
conda run -n bimanual-rope python scripts/pour_marbles_controller.py \
  --seed 0 --save-state /tmp/pour_initial.npz

# Load that object state and execute the oracle.
conda run -n bimanual-rope python scripts/pour_marbles_controller.py \
  --seed 0 --load-state /tmp/pour_initial.npz --record /tmp/pour_demo.npz

# Restore a recording's object state and replay its recorded actions.
conda run -n bimanual-rope python scripts/pour_marbles_controller.py \
  --replay /tmp/pour_demo.npz
```

Initial-state restore applies cup/marble `qpos` and `qvel`, calls `mj_forward`, and updates the
DuoBench stage tracker. It intentionally leaves robot state at the environment's normal reset.

An important empirical finding was that object velocity is relevant even at the nominal initial
state. DuoBench exposes the reset scene before all objects have settled, and the captured marble
velocity norms were substantial for the proven raw reset seeds. Restoring the same positions
with zero velocity could change whether the scripted rollout succeeded. This is why format v3
stores velocities. Old pose-only v2 files still load with explicit zero velocity because that was
their original execution behavior; the collected corpus was normalized to v3.

### Recording format v3

Recordings use `np.savez` and `allow_pickle=False`. Important fields are:

| Field | Shape | Meaning |
| --- | ---: | --- |
| `format_version` | scalar | `3` |
| `file_type` | scalar string | `pour_marbles_recording` |
| `seed` | scalar | environment reset seed |
| `record_id` | scalar | unique demonstration identifier |
| `source_cup` | scalar string | `left` or `right` |
| `cup_qpos` | `(2, 7)` | initial free-joint cup positions/orientations |
| `marble_qpos` | `(20, 7)` | initial free-joint marble positions/orientations |
| `cup_qvel` | `(2, 6)` | initial cup linear/angular velocities |
| `marble_qvel` | `(20, 6)` | initial marble linear/angular velocities |
| `initial_left_joint_qpos` | `(7,)` | observed arm reset state; not restored |
| `initial_right_joint_qpos` | `(7,)` | observed arm reset state; not restored |
| `actions` | `(T, 16)` | left 7 joints, left grip, right 7 joints, right grip |
| `left_joint_qpos` | `(T, 7)` | measured post-step left-arm joint state |
| `right_joint_qpos` | `(T, 7)` | measured post-step right-arm joint state |
| `left_joint_targets` | `(T, 7)` | commanded left-arm joint targets |
| `right_joint_targets` | `(T, 7)` | commanded right-arm joint targets |
| `left_gripper_commands` | `(T,)` | commanded left gripper values |
| `right_gripper_commands` | `(T,)` | commanded right gripper values |
| `timestamps_s` | `(T,)` | strictly increasing simulator timestamps |
| `joint_state_sample` | scalar string | `post_step` |
| `final_*`, `strict_success` | scalars | terminal outcome and validation metadata |
| `motion_variant` | scalar | dataset variant identifier |
| `motion_parameters` | `(5,)` | controller parameters used for the rollout |

Use measured `left_joint_qpos` and `right_joint_qpos` as the demonstrated robot motion unless a
training method explicitly needs commanded targets instead.

### Strict success definition

`--record` writes an NPZ only when all of the following hold at the terminal evaluation:

- stage and maximum stage are 6;
- the environment reports success;
- all 20 marbles are in the target cup;
- zero marbles remain in the source cup;
- both cups are inside their target regions;
- both cups are upright.

The generators write a hidden candidate first, validate it, and atomically rename it into the
dataset only on success. A latched intermediate stage is not sufficient.

## Dataset status

Local path:

```text
data/duo/pour_marbles
```

The 2026-07-24 independent audit reported:

- 100 NPZ recordings;
- 100/100 format v3 with explicit initial object velocities;
- 50 right-source and 50 left-source demonstrations;
- 100/100 strict successes;
- 20/20 marbles transferred in every file;
- both cups placed and upright in every file;
- 51,705 total time steps;
- 508 to 527 steps per demonstration, with 7 distinct lengths;
- 168.190 to 207.390 simulator seconds per demonstration;
- 9 unique initial-scene fingerprints;
- 20 unique measured two-arm joint-trajectory fingerprints;
- approximately 10.80 MiB total.

Reproduce the audit with:

```bash
conda run -n bimanual-rope python scripts/audit_pour_marbles_demos.py
```

The audit checks exact count, source balance, strict terminal metadata, marble counts, placement
and upright flags, array shapes, finite values, timestamp monotonicity, unique record IDs,
initial-state fields, and diversity summaries.

### Dataset portability

The repository's `.gitignore` contains `*.npz`. The Git repository therefore does **not** carry
the local demonstrations even when `main` is fully pushed. Transfer or restore
`data/duo/pour_marbles` separately before auditing or training on another computer.

### Diversity caveat

The corpus has genuine variance, but it does not contain 100 independent motion shapes. It has
20 exact measured-trajectory fingerprints and 9 initial-scene fingerprints; the reliable bulk
fill reused proven initial states. This was an intentional quality tradeoff after many small
geometric and motion perturbations failed strict success.

For model evaluation:

- stratify by `source_cup`;
- group train/validation/test splits by initial-scene and/or trajectory fingerprint;
- do not allow exact or replay-equivalent trajectories into multiple splits;
- report results separately for left-source and right-source motions;
- resample using `timestamps_s` if the movement primitive requires a common phase grid.

If substantially more diversity is needed, collect additional executed and validated scenes or
develop controller parameter ranges with a high empirical success rate. Do not inject artificial
noise into a successful trajectory and preserve its success label without running the simulator.

## Generation tools

### Raw reset-seed generator

`scripts/generate_pour_marbles_demos.py` schedules controller processes, validates recordings,
limits numerical-library thread counts, resumes from existing valid files, and publishes only
strict successes.

```bash
conda run -n bimanual-rope python scripts/generate_pour_marbles_demos.py \
  --count 100 --workers 4
```

This is not currently a high-yield way to obtain diversity. Broad reset-seed sweeps frequently
failed at grasp/lift because the grasp choreography is not robust across all randomized cup
placements.

### Balanced proven-state generator

`scripts/generate_pour_marbles_variants.py` maintains an exact direction balance and uses two
proven anchors:

- `demo_seed_000000.npz`: reset seed 0, right cup is source;
- `demo_seed_000030.npz`: reset seed 30, left cup is source.

The final reliable bulk command was:

```bash
conda run -n bimanual-rope python scripts/generate_pour_marbles_variants.py \
  --count 100 --workers 4 --translation-mm 0
```

Earlier accepted pilots used small translations and account for part of the existing scene and
trajectory variance. Repeated larger, then sub-millimeter perturbation trials had poor yield.
Even small changes can cause a grasp miss, dropped marble, cup placement miss, or terminal tip.

On the original collection machine, four concurrent simulators consumed roughly 12 GiB of a
14 GiB system. Do not increase `--workers` beyond 4 without measuring memory use. The complete
balanced fill took about 62.5 minutes for 78 remaining demonstrations.

### Legacy migration

`scripts/migrate_pour_marbles_v2.py` was used once to normalize 18 successful pose-only files to
v3. Those runs had been initialized through the legacy load path, which explicitly set object
velocities to zero, so adding zero velocity arrays preserved their executed initial conditions.
The current 100-file corpus requires no migration.

## Known failure modes and decisions

- **Unsettled reset state:** position-only replay can diverge. Preserve captured object velocity.
- **Random grasp failures:** fail fast after lift; do not continue and mislabel the rollout.
- **Wrong rim height:** using logical containment height rather than physical collision rim makes
  marbles miss the receiver.
- **Sequential-looking setup:** avoid long source-only then target-only positioning moves.
- **Receiver tracking:** the receiving cup must follow the source outlet during the tilt; fixed
  placement was less reliable.
- **Release tipping:** vertical gripper retreat can snag the cup; use outward/upward retreat.
- **Motion noise:** endpoint and release-clearance perturbations often invalidate placement.
  `--motion-variant` is currently identifying metadata, while `motion_parameters` stay nominal.
- **Record identity:** `seed` denotes the actual reset seed. Use `record_id` for the unique dataset
  sample ID; do not overload one with the other.

## Suggested continuation prompt

After cloning the repository and restoring any required NPZ data, a new Codex session can begin
with:

> Read `AGENTS.md` and `docs/CODEX_HANDOFF.md`, inspect the current branch and relevant scripts,
> then summarize the pour-marbles controller invariants and dataset status before making changes.

This gives the new session the durable information from the original work without depending on
machine-local Codex memories or the original chat transcript.
