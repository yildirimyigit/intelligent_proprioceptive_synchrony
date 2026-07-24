#!/usr/bin/env python3
"""Headless-by-default smoke test for the DuoBench simulation environment.

Verifies that the `bimanual-rope` Conda environment can import the full stack
(MuJoCo, Gymnasium, DuoBench, RCS), instantiate one existing DuoBench task through
Gymnasium, reset it, take a few random-action steps, and close it cleanly.

Usage:
    python scripts/smoke_test_duobench.py                 # headless (default), 10 steps
    python scripts/smoke_test_duobench.py --steps 50      # headless, 50 steps
    python scripts/smoke_test_duobench.py --render        # open the interactive viewer
    python scripts/smoke_test_duobench.py --render --steps 300

`--render` opens RCS's interactive MuJoCo viewer and therefore needs a local display
(a desktop session); it does not work over a plain headless SSH connection.

Exit code 0 on success, non-zero on any failure.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as dist_version

ENV_ID = "duobench/ball_maze"
REQUIRED_INFO_KEYS = ("instruction", "stage", "max_stage")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Open the interactive MuJoCo viewer (requires a local display).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of random-action steps to take after reset (default: 10).",
    )
    return parser.parse_args()


def _version(module: object, dist_name: str) -> str:
    """Best-effort version: module `__version__`, else installed distribution version."""
    exposed = getattr(module, "__version__", None)
    if exposed:
        return str(exposed)
    try:
        return dist_version(dist_name)
    except PackageNotFoundError:
        return "unknown"


def main() -> int:
    args = _parse_args()

    # Select a MuJoCo GL backend before importing RCS (it bootstraps a GL context on import).
    # The headless path uses offscreen EGL; when rendering we leave the backend to the
    # environment so the on-screen GLFW viewer is used. `setdefault` keeps a user-set value.
    # Imports live inside main() so the viewer's spawned GUI subprocess (which re-imports
    # this module) does not re-run the heavy import/registration work.
    if not args.render:
        os.environ.setdefault("MUJOCO_GL", "egl")

    import gymnasium as gym
    import mujoco
    import rcs  # RCS is published as the `rcs-core` wheel.

    import duobench
    import duobench.tasks.ball_maze  # import registers the gym env id

    print("== Versions ==")
    print(f"  Python     : {platform.python_version()}")
    print(f"  DuoBench   : {_version(duobench, 'duobench')}")
    print(f"  MuJoCo     : {_version(mujoco, 'mujoco')}")
    print(f"  Gymnasium  : {_version(gym, 'gymnasium')}")
    print(f"  RCS        : {_version(rcs, 'rcs-core')}")
    print(f"  MUJOCO_GL  : {os.environ.get('MUJOCO_GL', '<unset>')}")
    print(f"  Render     : {args.render}")

    assert ENV_ID in gym.registry, f"{ENV_ID!r} was not registered with Gymnasium"
    print(f"\n== Environment ==\n  Selected env id: {ENV_ID}")

    # The registered entry point is an RCS SimEnvCreator. Its default config renders
    # (headless=False); build a config with the requested mode and no camera observations,
    # then pass it through gym.make.
    creator = gym.spec(ENV_ID).entry_point
    cfg = creator.config()
    cfg.headless = not args.render
    cfg.camera_cfgs = None
    cfg.camera_adds = None

    env = gym.make(ENV_ID, cfg=cfg, disable_env_checker=True)
    try:
        obs, info = env.reset(seed=0)

        missing = [k for k in REQUIRED_INFO_KEYS if k not in info]
        assert not missing, f"reset() info is missing required keys: {missing}"

        if isinstance(obs, Mapping):
            print(f"  Observation    : Mapping, top-level keys = {sorted(obs.keys())}")
        else:
            print(f"  Observation    : {type(obs).__name__}")
        print(f"  Instruction    : {info['instruction']}")
        print(f"  Stage/MaxStage : {info['stage']} / {info['max_stage']}")

        # Exercise the step() contract with a few random actions from the action space.
        print(f"\n== Stepping ({args.steps} random actions) ==")
        reward = 0.0
        for _ in range(args.steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if args.render:
                time.sleep(1.0 / 30.0)  # pace so the viewer motion is watchable
            if terminated or truncated:
                obs, info = env.reset(seed=0)
        print(
            f"  ran {args.steps} steps; final stage {info['stage']}/{info['max_stage']}, "
            f"last reward {reward:.3f}"
        )
    finally:
        env.close()

    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
