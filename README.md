# intelligent_proprioceptive_synchrony

Learning from Demonstration for bimanual manipulation with two Franka FR3 robots. This
repository provides a reproducible DuoBench + RCS + MuJoCo + Gymnasium simulation development
environment and a first custom task, [`bimanual_rope/sling_hook`](#custom-environment-bimanual_ropesling_hook).

## Environment setup

Target platform: Linux x86-64 (developed on Ubuntu 26.04 LTS). Creating the environment
installs DuoBench, which pulls in Robot Control Stack (RCS, published as the `rcs-core`
wheel), MuJoCo and Gymnasium.

```bash
# Create the Conda environment (single documented command, reproducible on any compatible Linux host)
conda env create -f environment.yml

# Activate it
conda activate bimanual-rope

# Run the headless smoke test (imports the stack, resets one DuoBench task, closes it)
python scripts/smoke_test_duobench.py

# Update the environment after editing environment.yml
conda env update -f environment.yml --prune

# Remove the environment
conda env remove -n bimanual-rope
```

### Assets

DuoBench and RCS download version-matched assets on first use (first `import`). By default
they are cached under:

- `~/.duobench`
- `~/.rcs`

These caches live outside the repository and must not be committed. To store them elsewhere,
export the overrides before running (do not commit machine-specific paths):

```bash
export DUOBENCH_PREFIX=/path/to/duobench-assets
export RCS_PREFIX=/path/to/rcs-assets
```

> **DuoBench 0.1.0 quirk — repeated asset download.** DuoBench re-downloads its assets on *every*
> import because its "assets present" check looks for an RCS-only marker file
> (`assets/scenes/empty_world/scene.xml`) inside `~/.duobench`, which its own assets never
> contain. Create that marker once to stop the repeated download (DuoBench never loads it):
>
> ```bash
> mkdir -p ~/.duobench/assets/scenes/empty_world
> cp ~/.rcs/assets/scenes/empty_world/scene.xml ~/.duobench/assets/scenes/empty_world/scene.xml
> ```

### Smoke-test options

By default the smoke test runs headless and takes 10 random-action steps after reset. Flags:

```bash
python scripts/smoke_test_duobench.py --steps 50   # headless, take 50 random-action steps
python scripts/smoke_test_duobench.py --render     # open the interactive MuJoCo viewer
python scripts/smoke_test_duobench.py --render --steps 300   # watch it move for longer
```

`--render` opens RCS's on-screen MuJoCo viewer, so it needs a **local display** (a desktop
session) and does not work over a plain headless SSH connection. Leave `MUJOCO_GL` unset (or
`glfw`) when rendering — do not force `egl`, which is for offscreen rendering.

### Headless rendering

The default (headless, no camera observations) needs no GPU or GL context. If you later enable
rendered camera *observations* on a headless host, select a MuJoCo GL backend:

- `MUJOCO_GL=egl` on a host with a compatible GPU and the EGL runtime.
- `MUJOCO_GL=osmesa` for CPU rendering when OSMesa (`libosmesa6`) is installed.

On-screen GUI rendering (`--render`) additionally requires system OpenGL/GLFW libraries and a
display.

## Custom environment: `bimanual_rope/sling_hook`

A local, repository-owned DuoBench/RCS task for attaching a flexible closed sling to a hook.
The Gymnasium id is `bimanual_rope/sling_hook`. The package (`src/bimanual_rope/`) is installed
editable via `environment.yml`; if you set the env up before this task existed, run
`conda env update -n bimanual-rope -f environment.yml --prune` (or `pip install -e .`) once.

Run the headless smoke test (imports the task, resets, settles, checks stability, closes):

```bash
conda activate bimanual-rope
python scripts/smoke_test_sling_hook.py
```

**Scene contents.** The existing dual-FR3 Vention base scene plus:

- a **rigid, fixed hook** in the central bimanual workspace (`src/bimanual_rope/assets/sling_hook/sling_hook.xml`),
  with query sites `hook_tip`, `hook_throat`, `hook_support`. A dynamically swinging hook is
  deferred to a later phase.
- a **genuine closed 1-D MuJoCo deformable flex** "sling" (a real deformable loop, not a rigid
  torus). MuJoCo 3.2.6 corrupts a `flexcomp` when it is inserted through `MjSpec.attach` (which
  RCS uses), so the sling is built directly with named vertex bodies + `MjSpec.add_flex`.

**Provisional material parameters** (stability-tuned for the scene's 2 ms timestep; not
calibrated textile values): 24 loop vertices, loop radius 0.075 m, collision radius 0.006 m,
total mass 0.05 kg, soft edge stiffness 2500 N/m, edge damping 30 N·s/m.

**Privileged stage diagnostics.** The task exposes standard DuoBench stage info
(`stage`/`max_stage`/`instruction`/…) with five latched, monotonic stages: 0 resting → 1 lifted
→ 2 hook throat inside the loop aperture → 3 captured → 4 released and left hanging. These are
simulation diagnostics for reward/evaluation development, **not** policy observations. Known
limitation: the stage-2 aperture test fits a plane to the loop and does a planar point-in-polygon
check, which becomes unreliable when the sling is heavily folded or self-intersecting.

**No task controller or policy exists yet** — this phase only establishes the scene, a
deterministic reset, and the stage predicates.

## Replaying recorded trajectories

The DuoBench study publishes a LeRobot dataset of demonstrations on the Hugging Face Hub
([RobotControlStack/duobench](https://huggingface.co/datasets/RobotControlStack/duobench)).
`scripts/replay_trajectory.py` downloads one episode's trajectory (parquet only, not the videos)
and executes its recorded joint-space actions in the matching DuoBench environment while
rendering. It needs `huggingface_hub` (in `environment.yml`).

```bash
conda activate bimanual-rope
python scripts/replay_trajectory.py                        # ball_maze, episode 0, interactive viewer
python scripts/replay_trajectory.py --task pour_marbles --episode 3
python scripts/replay_trajectory.py --list-episodes        # list episodes for the task
python scripts/replay_trajectory.py --headless             # no window (CI / headless host)
```

Rendering opens RCS's interactive MuJoCo viewer, so it needs a local display. Available tasks:
`ball_maze`, `bin_sort`, `block_balance`, `carry_pot`, `hinge_chest`, `join_blocks`,
`pour_marbles`, `spring_door`, `transfer_cube`, `transfer_gate`, `transfer_reorient`.

**What is (and isn't) reproduced.** The recorded `action` is absolute joint targets (7 joints +
gripper per arm), so the two arms track the demonstration exactly (< 1e-3 rad). The LeRobot
dataset stores only robot state/action (plus camera video) — **not** the full simulator state or
the reset seed — so this is an open-loop *action* replay: the arms reproduce the demonstrated
motion, but manipulated objects (and DuoBench's randomized object placement on reset) will not
match the original scene, so task success is generally not reproduced. Faithful, state-restoring
replay would require RCS's own recording format (`rcs.sim.replayer`), not this LeRobot export.