#!/usr/bin/env python3
"""Online, closed-loop execution of the two cross-conditioned pour-marbles CNMPs.

Runs the paper's execution loop (Fig. 3) extended to a *coupled pair* of CNMPs: the left arm's
CNMP predicts the left joints conditioned on the right arm's momentary joints (time-varying
gamma), and vice versa. Because each arm's conditioning is the OTHER arm's future, we generate
the next-N steps with a short fixed-point (Gauss-Seidel) iteration at every replan, and re-anchor
on the freshly MEASURED joints of both arms -- i.e. momentary proprioception drives an iterative,
receding-horizon generation, not a one-shot offline rollout.

Per replan (every ``--horizon`` control steps):
  * context = the skill goals (self start/mid/end from the chosen demo) + the current measured
    self joints, with the partner's joints at those same times as gamma;
  * query   = the next-N normalized times;
  * couple  = iterate  left <- CNMP(.. gamma=right_future);  right <- CNMP(.. gamma=left_future);
  * execute = command the predicted joints (de-normalized) + the demo's gripper schedule, step the
    env, and read back the realized joints as the next proprioceptive anchor.

The scene (cup/marble state) is restored from the chosen demo so the run is comparable to it; the
gripper open/close schedule is taken from the demo (the CNMPs predict joints only). Success is read
from the DuoBench stage info at the end.

Usage:
    python scripts/execute_cnmp_pour.py \
        --left-ckpt output/cnmp_pour/left/<ts>/cnmp_best.pt \
        --right-ckpt output/cnmp_pour/right/<ts>/cnmp_best.pt \
        --source left --demo 0 --horizon 25 --debug-frames /tmp/exec_frames
    # add --render for the interactive viewer (needs a display)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cnmp import CNMP
from cnmp_data import ARMS, DATA_DIR, DEFAULT_T, N_JOINTS, load_demo, make_cross_conditioned_dataset, \
    partner_of, resample
from pour_marbles_controller import CUP_JOINTS, MARBLE_JOINTS, make_env

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--left-ckpt", required=True)
    p.add_argument("--right-ckpt", required=True)
    p.add_argument("--source", choices=["left", "right"], default=None,
                   help="use the same fixed-role subset the models were trained on (if any)")
    p.add_argument("--demo", type=int, default=0, help="which TEST demo to reproduce (index into the test split)")
    p.add_argument("--horizon", type=int, default=25, help="N: steps generated per replan (receding horizon)")
    p.add_argument("--coupling-iters", type=int, default=3, help="fixed-point iterations resolving the left<->right gamma")
    p.add_argument("--offline", action="store_true",
                   help="one-shot generation with the demo's TRUE partner (no coupling, no re-anchoring), open-loop")
    p.add_argument("--T", type=int, default=DEFAULT_T)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--render", action="store_true", help="interactive viewer (needs a display)")
    p.add_argument("--debug-frames", metavar="DIR", default=None, help="save offscreen frames to DIR")
    return p.parse_args()


def load_skill(name: str, T: int):
    """Return the chosen demo's normalized joints, gripper schedule (T,), and object state."""
    npz = np.load(DATA_DIR / name, allow_pickle=False)
    demo = load_demo(DATA_DIR / name)
    rs = resample(demo, T)  # {arm: (T,7)} raw
    grid = np.linspace(0.0, 1.0, T)
    grip = {a: np.interp(grid, demo["t"], np.asarray(npz[f"{a}_gripper_commands"])) for a in ARMS}
    obj = {k: np.asarray(npz[k]) for k in ("cup_qpos", "marble_qpos", "cup_qvel", "marble_qvel")}
    return rs, grip, obj


def restore_scene(sim, obj) -> None:
    import mujoco as mj

    d = sim.data
    for i, jn in enumerate(CUP_JOINTS):
        d.joint(jn).qpos[:] = obj["cup_qpos"][i]
        d.joint(jn).qvel[:] = obj["cup_qvel"][i]
    for i, jn in enumerate(MARBLE_JOINTS):
        d.joint(jn).qpos[:] = obj["marble_qpos"][i]
        d.joint(jn).qvel[:] = obj["marble_qvel"][i]
    mj.mj_forward(sim.model, sim.data)


class Executor:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.T = args.T
        self.model = {"left": CNMP.load(args.left_ckpt, device=self.device),
                      "right": CNMP.load(args.right_ckpt, device=self.device)}
        self.param_dim = self.model["left"].param_dim
        self.cup_dim = self.param_dim - N_JOINTS       # 0, or 4 if the checkpoint was trained with cup-gamma
        # one dataset gives the (train-fit) normalizer + the test-split demo list; norm is shared by both arms
        self.ds = make_cross_conditioned_dataset("left", T=self.T, split_seed=args.seed,
                                                 device="cpu", only_source=args.source,
                                                 cup_gamma=self.cup_dim > 0)
        self.norm = self.ds.norm
        test_names = [self.ds.names[i] for i in self.ds.test_ids.tolist()]
        self.name = test_names[args.demo % len(test_names)]

        rs, self.grip, self.obj = load_skill(self.name, self.T)
        self.demoN = {a: self.norm.apply(rs[a], a).astype(np.float32) for a in ARMS}  # (T,7) normalized
        self.cups_n = None
        if self.cup_dim:  # this demo's own cup (x,y) -> the static gamma tail
            self.cups_n = self.norm.apply_cups(self.obj["cup_qpos"][:, :2].reshape(-1)).astype(np.float32)
        self.grid = np.linspace(0.0, 1.0, self.T, dtype=np.float32)
        self.G = np.array([0, self.T // 2, self.T - 1])                                # goal indices

        self.env = make_env(args.render)
        self.sim = self.env.get_wrapper_attr("sim")
        self.jn = {a: [f"robot{a}_fr3_joint{i + 1}" for i in range(N_JOINTS)] for a in ARMS}
        self._renderer = None

    # --- helpers ---
    def measure(self, arm):
        return np.array([self.sim.data.joint(n).qpos[0] for n in self.jn[arm]], dtype=np.float32)

    def measN(self, arm):
        return self.norm.apply(self.measure(arm), arm).astype(np.float32)

    def _predict(self, arm, obs_t, obs_self, obs_partner, tar_t, tar_partner):
        if self.cup_dim:  # append the static cup (x,y) so gamma matches the model's param_dim
            obs_partner = np.concatenate([obs_partner, np.broadcast_to(self.cups_n, (len(obs_partner), self.cup_dim))], axis=1)
            tar_partner = np.concatenate([tar_partner, np.broadcast_to(self.cups_n, (len(tar_partner), self.cup_dim))], axis=1)
        f = lambda x, d: torch.as_tensor(np.asarray(x, np.float32).reshape(1, -1, d), device=self.device)
        mean, _ = self.model[arm].predict(
            obs_x=f(obs_t, 1), obs_y=f(obs_self, N_JOINTS), tar_x=f(tar_t, 1),
            obs_mask=torch.ones(1, len(obs_t), dtype=torch.bool, device=self.device),
            gamma=f(obs_partner, self.param_dim), tar_gamma=f(tar_partner, self.param_dim))
        return mean[0].cpu().numpy()

    def _step(self, cmd, grip):
        action = {a: {"joints": cmd[a].astype(np.float64), "gripper": np.array([grip[a]], np.float32)}
                  for a in ARMS}
        _, _, _, _, info = self.env.step(action)
        if self.args.render:
            self.sim.sync_gui()
        return info

    def frame(self, tag):
        if not self.args.debug_frames:
            return
        import mujoco as mj
        from PIL import Image

        if self._renderer is None:
            self._renderer = mj.Renderer(self.sim.model, height=480, width=640)
        cam = mj.MjvCamera(); mj.mjv_defaultFreeCamera(self.sim.model, cam)
        cam.lookat[:] = [0.5, 0.0, 0.95]; cam.distance, cam.azimuth, cam.elevation = 1.6, 135.0, -10.0
        self._renderer.update_scene(self.sim.data, camera=cam)
        os.makedirs(self.args.debug_frames, exist_ok=True)
        Image.fromarray(self._renderer.render()).save(os.path.join(self.args.debug_frames, f"{tag}.png"))

    # --- main loop ---
    def run(self):
        _, info = self.env.reset(seed=self.args.seed)
        restore_scene(self.sim, self.obj)
        self.frame("00_start")
        mode = "offline (true partner, open-loop)" if self.args.offline \
            else f"online (horizon={self.args.horizon}, coupling_iters={self.args.coupling_iters})"
        print(f"executing demo {self.args.demo} ({self.name}) -- {mode}")

        goalN = {a: self.demoN[a][self.G] for a in ARMS}   # (3,7) normalized self goals

        if self.args.offline:  # one-shot generation with the demo's TRUE partner; no coupling / re-anchor
            pred = {a: self._predict(a, self.grid[self.G], goalN[a], goalN[partner_of(a)],
                                     self.grid, self.demoN[partner_of(a)]) for a in ARMS}
            for i in range(1, self.T):
                cmd = {a: self.norm.invert(pred[a][i], a).astype(np.float32) for a in ARMS}
                info = self._step(cmd, {a: float(self.grip[a][i]) for a in ARMS})
                if i % self.args.horizon == 0:
                    self.frame(f"t{i:03d}")
        else:
            t = 0
            while t < self.T - 1:
                H = np.arange(t + 1, min(t + self.args.horizon, self.T - 1) + 1)
                tar_t = self.grid[H]
                obs_t = np.array([self.grid[self.G[0]], self.grid[self.G[1]], self.grid[self.G[2]], self.grid[t]])
                obs = {a: np.vstack([goalN[a], self.measN(a)]) for a in ARMS}   # self obs incl. current (4,7)

                # resolve the left<->right coupling over this horizon (Gauss-Seidel, seeded from current state)
                fut = {a: np.tile(self.measN(a), (len(H), 1)) for a in ARMS}
                for _ in range(self.args.coupling_iters):
                    for a in ARMS:  # partner obs/gamma is the OTHER arm's obs / predicted future
                        fut[a] = self._predict(a, obs_t, obs[a], obs[partner_of(a)], tar_t, fut[partner_of(a)])

                # execute the horizon; realized joints become the next anchor
                for j, i in enumerate(H):
                    cmd = {a: self.norm.invert(fut[a][j], a).astype(np.float32) for a in ARMS}
                    info = self._step(cmd, {a: float(self.grip[a][i]) for a in ARMS})
                t = int(H[-1])
                self.frame(f"t{t:03d}")

        stage, mx = int(info.get("stage", -1)), int(info.get("max_stage", 6))
        print(f"\nFINISHED: stage {stage}/{mx}  success={bool(info.get('success', stage >= mx))}  "
              f"marblesL/R={info.get('marbles_in_left_cup')}/{info.get('marbles_in_right_cup')}  "
              f"placeL/R={info.get('left_cup_in_place')}/{info.get('right_cup_in_place')}  "
              f"uprightL/R={info.get('left_cup_upright')}/{info.get('right_cup_upright')}")
        self.frame("99_end")
        self.env.close()


def main() -> int:
    Executor(parse_args()).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
