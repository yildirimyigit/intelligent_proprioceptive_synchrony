#!/usr/bin/env python3
"""Headless smoke test for the custom bimanual_rope/sling_hook environment.

Verifies that the environment registers, uses the dual-FR3 Vention base scene, contains the
fixed hook (body + sites) and a genuine closed deformable-flex sling, resets deterministically
in headless mode, stays numerically stable during unactuated settling, returns the standard
DuoBench stage info, and closes cleanly. Only the initial stage is verified (the task is not
autonomously solved here).

Usage:
    python scripts/smoke_test_sling_hook.py
    python scripts/smoke_test_sling_hook.py --steps 300
    python scripts/smoke_test_sling_hook.py --render      # interactive viewer (needs a display)
    python scripts/smoke_test_sling_hook.py --save-image artifacts/sling_hook_reset.png

Headless by default (the required mode). Exit code 0 on success, non-zero on any failure.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as dist_version

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
ENV_ID = "bimanual_rope/sling_hook"
REQUIRED_INFO_KEYS = ("instruction", "stage", "max_stage")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=200, help="Unactuated settling steps for the stability check.")
    p.add_argument("--render", action="store_true",
                   help="Open the interactive MuJoCo viewer (needs a local display); default is headless.")
    p.add_argument("--hold", type=float, default=8.0,
                   help="With --render, seconds to keep the viewer open after the checks.")
    p.add_argument("--save-image", metavar="PATH", default=None,
                   help="Optional: render one reset frame to PATH (needs the working EGL renderer).")
    return p.parse_args()


def _version(module: object, dist_name: str) -> str:
    exposed = getattr(module, "__version__", None)
    if exposed:
        return str(exposed)
    try:
        return dist_version(dist_name)
    except PackageNotFoundError:
        return "unknown"


def _try_save_image(mujoco, model, data, path: str, target) -> None:
    """Best-effort single-frame render; never fatal to the smoke test."""
    try:
        from PIL import Image

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        height = min(480, int(model.vis.global_.offheight))
        width = min(640, int(model.vis.global_.offwidth))
        renderer = mujoco.Renderer(model, height=height, width=width)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        cam.lookat[:] = target
        cam.distance, cam.azimuth, cam.elevation = 1.8, 135.0, -20.0
        renderer.update_scene(data, camera=cam)
        Image.fromarray(renderer.render()).save(path)
        renderer.close()
        print(f"  saved reset image -> {path}")
    except Exception as exc:  # optional feature; report and continue
        print(f"  --save-image skipped ({type(exc).__name__}: {exc})")


def main() -> int:
    args = _parse_args()

    import gymnasium as gym
    import mujoco
    import rcs

    import duobench  # noqa: F401
    from bimanual_rope.tasks import sling_hook  # noqa: F401  (import registers the env)
    from bimanual_rope.utils.sling_state import SlingState

    assert ENV_ID in gym.registry, f"{ENV_ID!r} was not registered with Gymnasium"
    creator = gym.spec(ENV_ID).entry_point
    cfg = creator.config()
    cfg.headless = not args.render  # headless is the required default; --render opens the viewer
    cfg.camera_cfgs = None
    cfg.camera_adds = None

    env = gym.make(ENV_ID, cfg=cfg, disable_env_checker=True)
    site = mujoco.mjtObj.mjOBJ_SITE
    body = mujoco.mjtObj.mjOBJ_BODY
    try:
        obs, info = env.reset(seed=0)
        missing = [k for k in REQUIRED_INFO_KEYS if k not in info]
        assert not missing, f"reset() info missing required keys: {missing}"

        sim = env.unwrapped.get_wrapper_attr("sim")
        model, data = sim.model, sim.data
        sling = SlingState(model, data)  # __init__ validates a clean closed loop

        # Hook body + sites resolve.
        hook_bid = mujoco.mj_name2id(model, body, "sling_hook_hook")
        assert hook_bid >= 0, "hook body not found"
        hook_sites = {}
        for key in ("tip", "throat", "support"):
            sid = mujoco.mj_name2id(model, site, f"sling_hook_hook_{key}")
            assert sid >= 0, f"hook site {key} not found"
            hook_sites[key] = data.site_xpos[sid].copy()

        # Sling is a non-trivial closed flex.
        assert sling.vert_num > 8, f"sling has too few vertices ({sling.vert_num})"
        assert sling.edge_num == sling.vert_num, "sling loop is not closed"
        centroid_initial = sling.centroid().copy()
        stage, max_stage = int(info["stage"]), int(info["max_stage"])
        assert 0 <= stage <= max_stage, f"stage {stage} out of range [0,{max_stage}]"
        assert stage == 0, f"initial stage should be 0 (resting), got {stage}"

        # Advance the raw sim (robots held, not commanded) and check stability.
        hook_xpos0, hook_xquat0 = data.xpos[hook_bid].copy(), data.xquat[hook_bid].copy()
        nflexvert0 = int(model.nflexvert)
        for _ in range(args.steps):
            mujoco.mj_step(model, data)

        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all(), "non-finite qpos/qvel"
        assert np.isfinite(sling.vertices()).all(), "non-finite sling vertices"
        max_speed = sling.max_speed()
        assert max_speed < 5.0, f"sling velocity grew too large ({max_speed:.2f} m/s)"
        assert int(model.nflexvert) == nflexvert0, "flex vertex count changed"
        edges, rest = sling.edge_lengths(), sling.rest_edge_lengths()
        assert edges.min() > 0.3 * rest.min(), f"sling edge collapsed ({edges.min():.4f})"
        assert edges.max() < 3.0 * rest.max(), f"sling edge over-stretched ({edges.max():.4f})"
        hook_moved = float(np.linalg.norm(data.xpos[hook_bid] - hook_xpos0)) + float(
            np.linalg.norm(data.xquat[hook_bid] - hook_xquat0)
        )
        assert hook_moved < 1e-6, f"fixed hook moved by {hook_moved:.2e}"
        centroid_settled = sling.centroid().copy()

        # Second deterministic reset restores the sling near its initial settled state.
        env.reset(seed=0)
        centroid_reset2 = sling.centroid().copy()
        reset_delta = float(np.linalg.norm(centroid_reset2 - centroid_initial))
        assert reset_delta < 0.01, f"2nd reset diverged from 1st by {reset_delta:.4f} m"

        if args.save_image:
            _try_save_image(mujoco, model, data, args.save_image,
                            target=0.5 * (centroid_initial + hook_sites["throat"]))

        print("== Versions ==")
        print(f"  Python {platform.python_version()} | DuoBench {_version(duobench, 'duobench')} | "
              f"RCS {_version(rcs, 'rcs-core')} | MuJoCo {_version(mujoco, 'mujoco')} | "
              f"Gymnasium {_version(gym, 'gymnasium')}")
        print("== Environment ==")
        print(f"  Env id           : {ENV_ID}")
        print(f"  Headless         : {cfg.headless}")
        print(f"  Control mode     : {cfg.control_mode}")
        for key, pos in hook_sites.items():
            print(f"  hook_{key:<8s}   : {np.round(pos, 3)}")
        print(f"  Sling flex id    : {sling.flex_id}  (vertices={sling.vert_num}, edges={sling.edge_num})")
        print(f"  Sling centroid   : initial {np.round(centroid_initial, 3)} -> "
              f"settled {np.round(centroid_settled, 3)}")
        print(f"  Max sling speed  : {max_speed:.4f} m/s over {args.steps} settle steps")
        print(f"  Stage / max      : {stage} / {max_stage}")
        print(f"  Subinstruction   : {info['current_subinstruction']}")
        print(f"  2nd-reset delta  : {reset_delta:.5f} m")

        if args.render:
            print(f"  Rendering: holding the interactive viewer for {args.hold:.0f}s "
                  "(the sling should rest on the table below the hook)...")
            deadline = time.time() + args.hold
            while time.time() < deadline:
                mujoco.mj_step(model, data)
                time.sleep(1.0 / 60.0)
    finally:
        env.close()

    print("\nSLING-HOOK SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
