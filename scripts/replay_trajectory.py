#!/usr/bin/env python3
"""Replay a recorded DuoBench trajectory in the simulator while rendering.

Downloads one LeRobot episode from the DuoBench Hugging Face dataset
(https://huggingface.co/datasets/RobotControlStack/duobench) and executes its recorded
joint-space actions in the matching DuoBench Gymnasium environment. The recorded ``action`` is
``[left_joints(7), left_gripper, right_joints(7), right_gripper]`` (absolute joint targets), so
the env is driven in ``ControlMode.JOINTS`` with ``RelativeTo.NONE`` and the arms track the
demonstration (tracking error < 1e-3 rad).

By default it opens RCS's interactive MuJoCo viewer (needs a local display); ``--headless``
runs without a window. Only the parquet trajectory files are downloaded (not the videos).

Usage:
    python scripts/replay_trajectory.py                          # ball_maze, episode 0, rendered
    python scripts/replay_trajectory.py --task pour_marbles --episode 3
    python scripts/replay_trajectory.py --list-episodes          # show available episodes
    python scripts/replay_trajectory.py --headless               # no window (CI / headless host)

Exit code 0 on success, non-zero on failure.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

REPO_ID = "RobotControlStack/duobench"
TASKS = [
    "ball_maze", "bin_sort", "block_balance", "carry_pot", "hinge_chest", "join_blocks",
    "pour_marbles", "spring_door", "transfer_cube", "transfer_gate", "transfer_reorient",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="ball_maze", choices=TASKS, help="DuoBench task (default: ball_maze).")
    p.add_argument("--episode", type=int, default=0, help="Episode index to replay (default: 0).")
    p.add_argument("--headless", action="store_true", help="Run without the interactive viewer (default renders).")
    p.add_argument("--fps", type=int, default=30, help="Control frequency / playback rate (default: 30).")
    p.add_argument("--hold", type=float, default=3.0, help="Seconds to keep the viewer open after replay.")
    p.add_argument("--list-episodes", action="store_true", help="List episodes for the task and exit.")
    return p.parse_args()


def _download_parquets(task: str) -> tuple[str, str]:
    """Fetch the task's LeRobot parquet trajectory + meta files (no videos). Returns globs."""
    from huggingface_hub import snapshot_download

    root = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=[f"{task}/sim/data/*/*.parquet", f"{task}/sim/meta/*.parquet"],
    )
    data_glob = os.path.join(root, task, "sim", "data", "*", "*.parquet")
    tasks_parquet = os.path.join(root, task, "sim", "meta", "tasks.parquet")
    return data_glob, tasks_parquet


def _to_action(flat: np.ndarray) -> dict:
    """Map a flat 16-D recorded action to the JOINTS-mode nested action dict."""
    return {
        "left": {"joints": flat[0:7], "gripper": flat[7:8]},
        "right": {"joints": flat[8:15], "gripper": flat[15:16]},
    }


def main() -> int:
    args = _parse_args()

    import duckdb

    data_glob, tasks_parquet = _download_parquets(args.task)
    connection = duckdb.connect()

    if args.list_episodes:
        rows = connection.execute(
            "SELECT episode_index, count(*) FROM read_parquet(?) GROUP BY episode_index ORDER BY episode_index",
            [data_glob],
        ).fetchall()
        print(f"{args.task}: {len(rows)} episodes")
        for ep, n in rows:
            print(f"  episode {int(ep):3d}: {int(n)} frames")
        return 0

    rows = connection.execute(
        'SELECT action, task_index FROM read_parquet(?) WHERE episode_index = ? ORDER BY frame_index',
        [data_glob, args.episode],
    ).fetchall()
    if not rows:
        raise SystemExit(f"Episode {args.episode} not found for task {args.task!r}.")
    actions = [np.asarray(r[0], dtype=np.float32) for r in rows]
    instruction_row = connection.execute(
        "SELECT task FROM read_parquet(?) WHERE task_index = ?", [tasks_parquet, int(rows[0][1])]
    ).fetchone()
    instruction = instruction_row[0] if instruction_row else "(unknown)"

    import gymnasium as gym
    import mujoco  # noqa: F401
    import rcs  # noqa: F401
    import duobench  # noqa: F401
    from importlib import import_module
    from rcs._core.sim import SimConfig
    from rcs.envs.base import ControlMode, RelativeTo

    import_module(f"duobench.tasks.{args.task}")  # registers duobench/<task>
    env_id = f"duobench/{args.task}"
    assert env_id in gym.registry, f"{env_id!r} was not registered"

    print(f"Task        : {args.task}  (env {env_id})")
    print(f"Episode     : {args.episode}  ({len(actions)} frames @ {args.fps} fps)")
    print(f"Instruction : {instruction}")
    print(f"Rendering   : {'headless (no window)' if args.headless else 'interactive viewer (needs a display)'}")

    creator = gym.spec(env_id).entry_point
    cfg = creator.config()
    cfg.control_mode = ControlMode.JOINTS      # recorded actions are absolute joint targets
    cfg.relative_to = RelativeTo.NONE
    cfg.headless = args.headless
    cfg.camera_cfgs = None
    cfg.camera_adds = None
    cfg.sim_cfg = SimConfig(
        async_control=False, realtime=not args.headless, frequency=args.fps, max_convergence_steps=500
    )
    env = creator.create_env(cfg)
    try:
        env.reset()
        sim = env.get_wrapper_attr("sim")
        for flat in actions:
            _, _, terminated, truncated, info = env.step(_to_action(flat))
            if not args.headless:
                sim.sync_gui()
        print(f"Replayed {len(actions)} frames. Final stage {info.get('stage')}/{info.get('max_stage')}, "
              f"success={info.get('success')}")
        if not args.headless and args.hold > 0:
            time.sleep(args.hold)
    finally:
        env.close()

    print("\nTRAJECTORY REPLAY DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
