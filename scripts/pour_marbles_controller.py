#!/usr/bin/env python3
"""Scripted joint-space oracle for the DuoBench `pour_marbles` task.

Reads the privileged cup/marble state from the simulator, then drives the two FR3 arms
(JOINTS control via IK) to grasp both cups, lift them, pour the marbles from the full cup into
the empty cup, and place both cups back in their green target squares upright -- i.e. it aims
for DuoBench's full stage-6 success.

Extras:
  * ``--record PATH``    save a *successful* run (initial cup/marble state, commanded targets,
                         and measured joint trajectories for both arms) to an .npz.
  * ``--replay PATH``    restore the saved object state and replay the recorded joints.
  * ``--save-state P``   reset once and save only the object state (cups + marbles) to P.
  * ``--load-state P``   restore object state from P after reset, then run the oracle on it.
  * ``--render``         open the interactive MuJoCo viewer (needs a local display).
  * ``--debug-frames D`` save offscreen frames at each phase to directory D (for tuning).

Only object positions and velocities are restored (not robot kinodynamics), so loading always
starts from the environment's standard robot reset. Legacy pose-only files use zero velocity.

Usage:
    python scripts/pour_marbles_controller.py --render
    python scripts/pour_marbles_controller.py --seed 3 --record runs/pour_ok.npz
    python scripts/pour_marbles_controller.py --replay runs/pour_ok.npz --render
    python scripts/pour_marbles_controller.py --seed 0 --save-state runs/initial.npz
    python scripts/pour_marbles_controller.py --load-state runs/initial.npz --record runs/trajectory.npz
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

ENV_ID = "duobench/pour_marbles"
N_MARBLES = 20
CUP_RADIUS = 0.032
# The task's logical marble-containment check extends to z=0.105, but the MuJoCo
# collision walls actually end at 0.04 + 0.0325 = 0.0725 m.  Pour geometry must
# use the physical rim: using 0.105 leaves the outlet both high and off-centre.
CUP_RIM_HEIGHT = 0.0725
CUP_JOINTS = ("leftteacup_joint", "rightteacup_joint")
MARBLE_JOINTS = tuple(f"{i}marble" for i in range(N_MARBLES))
ARMS = ("left", "right")
FILE_FORMAT_VERSION = 3


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0, help="Env reset seed.")
    p.add_argument(
        "--record-id",
        type=int,
        default=None,
        help="Optional unique dataset ID, separate from the environment reset seed.",
    )
    p.add_argument("--render", action="store_true", help="Open the interactive viewer (needs a display).")
    p.add_argument("--record", metavar="PATH", default=None, help="Save a successful run to PATH (.npz).")
    p.add_argument("--replay", metavar="PATH", default=None, help="Replay a saved run from PATH (.npz).")
    p.add_argument("--save-state", metavar="PATH", default=None, help="Reset, save object state to PATH, exit.")
    p.add_argument("--load-state", metavar="PATH", default=None, help="Restore object state from PATH, then run.")
    p.add_argument("--debug-frames", metavar="DIR", default=None, help="Save offscreen frames per phase to DIR.")
    p.add_argument("--verbose", action="store_true", help="Print stage after each phase.")
    p.add_argument("--test-grasp", action="store_true", help="Stop after grasp+lift (for tuning the grasp).")
    p.add_argument(
        "--motion-variant",
        type=int,
        default=0,
        help="Dataset variant identifier stored in recordings.",
    )
    return p.parse_args()


# --------------------------------------------------------------------------------------------
# Initial object-state capture / restore  --  robot state is untouched.
# --------------------------------------------------------------------------------------------
def capture_object_state(sim) -> dict[str, np.ndarray]:
    d = sim.data
    return {name: d.joint(name).qpos.copy() for name in (*CUP_JOINTS, *MARBLE_JOINTS)}


def capture_object_velocity(sim) -> dict[str, np.ndarray]:
    d = sim.data
    return {name: d.joint(name).qvel.copy() for name in (*CUP_JOINTS, *MARBLE_JOINTS)}


def restore_object_state(
    sim,
    state: dict[str, np.ndarray],
    velocity: dict[str, np.ndarray] | None = None,
) -> None:
    import mujoco as mj

    d = sim.data
    for name, qpos in state.items():
        d.joint(name).qpos[:] = qpos
        d.joint(name).qvel[:] = 0.0 if velocity is None else velocity[name]
    mj.mj_forward(sim.model, sim.data)


def object_state_payload(
    state: dict[str, np.ndarray],
    velocity: dict[str, np.ndarray],
    seed: int,
    source_cup: str,
) -> dict[str, np.ndarray]:
    """Return the canonical, pickle-free NPZ representation of an initial scene."""
    return {
        "format_version": np.asarray(FILE_FORMAT_VERSION, dtype=np.int32),
        "file_type": np.asarray("pour_marbles_initial_state"),
        "cup_joint_names": np.asarray(CUP_JOINTS),
        "marble_joint_names": np.asarray(MARBLE_JOINTS),
        "cup_qpos": np.stack([state[j] for j in CUP_JOINTS]),
        "marble_qpos": np.stack([state[j] for j in MARBLE_JOINTS]),
        "cup_qvel": np.stack([velocity[j] for j in CUP_JOINTS]),
        "marble_qvel": np.stack([velocity[j] for j in MARBLE_JOINTS]),
        "seed": np.asarray(seed, dtype=np.int64),
        "source_cup": np.asarray(source_cup),
    }


# --------------------------------------------------------------------------------------------
# Oracle
# --------------------------------------------------------------------------------------------
class PourOracle:
    def __init__(
        self,
        env,
        verbose=False,
        render=False,
        frames_dir=None,
        test_grasp=False,
        motion_variant=0,
    ):
        import mujoco as mj
        import rcs
        from rcs._core.common import GripperType, RobotType
        from scipy.spatial.transform import Rotation

        self.env, self.verbose, self.render, self.frames_dir = env, verbose, render, frames_dir
        self.test_grasp = test_grasp
        self.sim = env.get_wrapper_attr("sim")
        self.m, self.d = self.sim.model, self.sim.data
        self._mj, self._rcs, self._Rot = mj, rcs, Rotation
        self._renderer = None

        fr3 = rcs.ROBOTS[RobotType.FR3]
        self.tcp_offset = rcs.GRIPPER_OFFSETS[GripperType("Robotiq2F85")]
        self.ik = {a: rcs.common.Pin(fr3.mjcf_model_path, fr3.attachment_site) for a in ARMS}
        lo, hi = fr3.joint_limits
        self._jlo, self._jhi = np.array(lo, dtype=float), np.array(hi, dtype=float)
        self._rng = np.random.default_rng(0)
        self.motion_variant = int(motion_variant)
        # Task-critical motion and release clearance stay on the proven nominal path.  Dataset
        # variance comes from separately validated initial scenes; forcing additional endpoint
        # noise here can bump an already placed cup and invalidate an otherwise good rollout.
        self.motion_parameters = np.array([-0.5, 60, 0.12, 0.10, 10], dtype=float)
        self.base_pose = {a: self._body_pose(f"robot{a}_base") for a in ARMS}
        self.site_id = {a: mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_SITE, f"robot{a}_{fr3.attachment_site}") for a in ARMS}
        self.joint_names = {a: [f"robot{a}_fr3_joint{i + 1}" for i in range(7)] for a in ARMS}
        self._table_bid = mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_BODY, "vention_table")
        self._arm_bids = {
            a: {i for i in range(self.m.nbody)
                if (mj.mj_id2name(self.m, mj.mjtObj.mjOBJ_BODY, i) or "").startswith((f"robot{a}_", f"gripper{a}_"))}
            for a in ARMS
        }
        self._table_z = float(self.d.geom("left_target").xpos[2])
        self.cur_q = {a: self._read_joints(a) for a in ARMS}
        self.grip = {a: 1.0 for a in ARMS}  # 1 = open, 0 = closed
        self.cup_T_tcp: dict = {}
        self.cup_upright: dict = {}
        self.initial_joint_qpos = {a: self._read_joints(a) for a in ARMS}
        self.traj: list[np.ndarray] = []
        self.joint_qpos_traj: dict[str, list[np.ndarray]] = {a: [] for a in ARMS}
        self.timestamps_s: list[float] = []
        self._trajectory_t0 = float(self.d.time)
        self.last_info: dict = {}

    # --- pose helpers ---
    def _pose(self, pos, quat_xyzw):
        return self._rcs.common.Pose(translation=np.asarray(pos, float), quaternion=np.asarray(quat_xyzw, float))

    def _body_pose(self, name):
        bid = self._mj.mj_name2id(self.m, self._mj.mjtObj.mjOBJ_BODY, name)
        q = self.d.xquat[bid]  # wxyz
        return self._pose(self.d.xpos[bid].copy(), [q[1], q[2], q[3], q[0]])

    def _tcp_world(self, arm):
        r = self._Rot.from_matrix(self.d.site_xmat[self.site_id[arm]].reshape(3, 3))
        return self._pose(self.d.site_xpos[self.site_id[arm]].copy(), r.as_quat()) * self.tcp_offset

    def _read_joints(self, arm):
        return np.array([self.d.joint(n).qpos[0] for n in self.joint_names[arm]])

    def _down_quat(self, yaw_deg=0.0):
        return (self._Rot.from_euler("z", yaw_deg, degrees=True) * self._Rot.from_euler("x", 180, degrees=True)).as_quat()

    def cup_pos(self, arm):
        return self.d.body(f"{arm}teacup_body").xpos.copy()

    def green_pos(self, arm):
        return self.d.geom(f"{arm}_target").xpos.copy()

    # --- IK + stepping ---
    def _table_metrics(self, arm, q):
        """(collision, clearance) for holding this arm at joint config q (everything else as-is).
        collision is checked via real contacts, not a height heuristic (a link's radius means
        "origin above the table" isn't sufficient to rule out contact). clearance is the lowest
        forearm/wrist link height above the table -- FR3 is 7-DOF/redundant, so many joint
        configs can reach the exact same TCP pose; ranking candidates by clearance instead of
        just accepting the first collision-free one favors holding the elbow up, not skimming it
        just above the table with a compensating wrist twist."""
        names = self.joint_names[arm]
        saved = np.array([self.d.joint(n).qpos[0] for n in names])
        for n, v in zip(names, q):
            self.d.joint(n).qpos[0] = v
        self._mj.mj_forward(self.m, self.d)
        collision = False
        for i in range(self.d.ncon):
            con = self.d.contact[i]
            b1, b2 = self.m.geom_bodyid[con.geom1], self.m.geom_bodyid[con.geom2]
            if (b1 == self._table_bid and b2 in self._arm_bids[arm]) or \
               (b2 == self._table_bid and b1 in self._arm_bids[arm]):
                collision = True
                break
        clearance = min(self.d.body(f"robot{arm}_fr3_link{i}").xpos[2] for i in range(4, 8)) - self._table_z
        for n, v in zip(names, saved):
            self.d.joint(n).qpos[0] = v
        self._mj.mj_forward(self.m, self.d)
        return collision, clearance

    def _ik(self, arm, tcp_world_pose, seed):
        # Prefer the natural continuation (seeded from the current joints); only fall back to
        # randomized restarts if that fails, so a distant alternate solution branch never hijacks
        # a perfectly good, continuous one (random restarts are needed for the 90 deg side grasp,
        # which the natural seed can't reach from a resting posture). Candidates that would rest
        # the arm on the table are rejected outright, no matter how low their position error.
        target_base = self.base_pose[arm].inverse() * tcp_world_pose
        tgt = target_base.translation()
        tgt_q = np.asarray(target_base.rotation_q(), float)
        POS_TOL, ANG_TOL = 0.02, 0.15  # 2 cm, ~8.6 deg -- position alone isn't enough: a solution
        # can plant the TCP at the right spot through a twisted wrist/elbow that doesn't actually
        # face the cup perpendicular, so orientation must be checked too.

        def try_seed(s):
            sol = self.ik[arm].inverse(target_base, s, tcp_offset=self.tcp_offset)
            if sol is None:
                return None, None
            sol = np.asarray(sol, float)
            if np.any(sol < self._jlo) or np.any(sol > self._jhi):
                return None, None  # Pinocchio doesn't respect joint limits during iteration
            fk = self.ik[arm].forward(sol) * self.tcp_offset
            pos_err = float(np.linalg.norm(fk.translation() - tgt))
            fk_q = np.asarray(fk.rotation_q(), float)
            ang_err = 2.0 * float(np.arccos(np.clip(abs(np.dot(fk_q, tgt_q)), -1.0, 1.0)))
            if pos_err >= POS_TOL or ang_err >= ANG_TOL:
                return None, None
            collision, clearance = self._table_metrics(arm, sol)
            if collision:
                return None, None
            return sol, clearance

        # Among all valid (accurate, non-colliding) candidates found, keep the one with the most
        # table clearance rather than the first one -- stop early only once clearance is generous.
        best, best_clear = try_seed(np.asarray(seed, float))
        if best is not None and best_clear > 0.08:
            return best
        for _ in range(24):
            s = self._jlo + (self._jhi - self._jlo) * self._rng.random(7)
            sol, clearance = try_seed(s)
            if sol is not None and (best is None or clearance > best_clear):
                best, best_clear = sol, clearance
            if best_clear is not None and best_clear > 0.08:
                break
        return best

    def _step_once(self):
        action = {a: {"joints": self.cur_q[a].astype(np.float64), "gripper": np.array([self.grip[a]], np.float32)}
                  for a in ARMS}
        _, _, _, _, info = self.env.step(action)
        self.last_info = info
        # ``traj`` is the command sent for this control step.  Joint states are sampled after
        # env.step, so row i in each measured trajectory is the state resulting from action i.
        self.traj.append(np.concatenate([self.cur_q["left"], [self.grip["left"]],
                                         self.cur_q["right"], [self.grip["right"]]]).astype(np.float32))
        for a in ARMS:
            self.joint_qpos_traj[a].append(self._read_joints(a).astype(np.float32))
        self.timestamps_s.append(float(self.d.time) - self._trajectory_t0)
        if self.render:
            self.sim.sync_gui()
        return info

    def move_tcp(self, targets, steps=40, grip=None):
        """targets[arm] = rcs Pose (world TCP target) or None (hold)."""
        for a in ARMS:
            if grip is not None and a in grip:
                self.grip[a] = grip[a]
            if targets.get(a) is not None:
                sol = self._ik(a, targets[a], self.cur_q[a])
                if sol is None:
                    print(f"  [warn] IK failed for {a}", flush=True)
                else:
                    self.cur_q[a] = np.asarray(sol, float)
        info = {}
        for _ in range(steps):
            info = self._step_once()
        return info

    def move_cup(self, targets, steps=40, grip=None):
        """targets[arm] = (cup_pos, cup_quat_xyzw) or None. Commands via the measured grasp offset."""
        tcp = {}
        for a in ARMS:
            if targets.get(a) is not None:
                pos, quat = targets[a]
                tcp[a] = self._pose(pos, quat) * self.cup_T_tcp[a]
            else:
                tcp[a] = None
        return self.move_tcp(tcp, steps=steps, grip=grip)

    def move_tcp_smooth(self, targets, transitions=12, settle_steps=1, grip=None):
        """Move to TCP targets through small absolute-joint target increments.

        ``SimConfig(async_control=False)`` makes each environment action converge toward the
        supplied target.  Repeating one distant target for many actions, as ``move_tcp`` does,
        does not make the move gradual: the entire jump is commanded by its first action.  This
        helper preserves JOINTS + RelativeTo.NONE control while explicitly interpolating the
        accepted IK solution.  A cubic smoothstep gives zero endpoint velocity and avoids
        shaking loose objects in an open cup.
        """
        starts = {a: self.cur_q[a].copy() for a in ARMS}
        goals = {a: starts[a].copy() for a in ARMS}
        for a in ARMS:
            if grip is not None and a in grip:
                self.grip[a] = grip[a]
            if targets.get(a) is not None:
                sol = self._ik(a, targets[a], starts[a])
                if sol is None:
                    print(f"  [warn] IK failed for {a}", flush=True)
                else:
                    goals[a] = np.asarray(sol, float)

        info = {}
        transitions = max(1, int(transitions))
        for i in range(1, transitions + 1):
            u = i / transitions
            alpha = u * u * (3.0 - 2.0 * u)
            for a in ARMS:
                self.cur_q[a] = starts[a] + alpha * (goals[a] - starts[a])
            info = self._step_once()
        for _ in range(max(0, int(settle_steps))):
            info = self._step_once()
        return info

    def move_cup_smooth(self, targets, transitions=12, settle_steps=1, grip=None):
        """Smooth joint-space counterpart of :meth:`move_cup`."""
        tcp = {}
        for a in ARMS:
            if targets.get(a) is not None:
                pos, quat = targets[a]
                tcp[a] = self._pose(pos, quat) * self.cup_T_tcp[a]
            else:
                tcp[a] = None
        return self.move_tcp_smooth(
            tcp, transitions=transitions, settle_steps=settle_steps, grip=grip
        )

    def set_grip(self, left=None, right=None, steps=25):
        g = {}
        if left is not None:
            g["left"] = left
        if right is not None:
            g["right"] = right
        return self.move_tcp({a: None for a in ARMS}, steps=steps, grip=g)

    # --- debug rendering ---
    def frame(self, tag):
        if not self.frames_dir:
            return
        try:
            from PIL import Image

            if self._renderer is None:
                h = min(480, int(self.m.vis.global_.offheight))
                w = min(640, int(self.m.vis.global_.offwidth))
                self._renderer = self._mj.Renderer(self.m, height=h, width=w)
            cam = self._mj.MjvCamera()
            self._mj.mjv_defaultFreeCamera(self.m, cam)
            cam.lookat[:] = [0.5, 0.0, 0.95]
            cam.distance, cam.azimuth, cam.elevation = 1.6, 75.0, -80.0
            self._renderer.update_scene(self.d, camera=cam)
            os.makedirs(self.frames_dir, exist_ok=True)
            Image.fromarray(self._renderer.render()).save(os.path.join(self.frames_dir, f"{tag}.png"))
        except Exception as exc:
            print(f"  [frame {tag} skipped: {exc}]", flush=True)

    def _log(self, tag):
        i = self.last_info
        if self.verbose:
            print(f"  {tag:12s} stage {i.get('stage')}  marblesL/R={i.get('marbles_in_left_cup')}/"
                  f"{i.get('marbles_in_right_cup')}  graspL/R={i.get('left_gripper_grasps_left_cup')}/"
                  f"{i.get('right_gripper_grasps_right_cup')}  placeL/R={i.get('left_cup_in_place')}/"
                  f"{i.get('right_cup_in_place')}  uprightL/R={i.get('left_cup_upright')}/"
                  f"{i.get('right_cup_upright')}", flush=True)
        self.frame(tag)

    # --- choreography ---
    def run(self) -> dict:
        src = self.last_info.get("source_cup", "right")   # full cup
        tgt = "left" if src == "right" else "right"        # empty cup
        cup = {a: self.cup_pos(a) for a in ARMS}

        if self.render:
            time.sleep(3)

        self.frame("00_start")
        # 1) SIDE grasp: approach the LEFT cup from its LEFT (+y) side and the RIGHT cup from its
        #    RIGHT (-y) side, gripper horizontal (90 deg from vertical), fingers (tool y) closing
        #    along x around the cup wall.
        GA = np.radians(90.0)

        def side_pose(a, over):  # over = offset of the TCP along the approach axis (tool z)
            side = 1.0 if cup[a][1] >= 0.0 else -1.0                     # +1 left cup, -1 right cup
            toolz = np.array([0.0, -side * np.sin(GA), -np.cos(GA)])     # from the outer y side toward the cup
            tooly = np.array([-side, 0.0, 0.0])                         # fingers close along x; sign keeps the
                                                                          # mount bracket rolled up (clear of the
                                                                          # table) on both sides, not just the right
            toolx = np.cross(tooly, toolz)
            r = self._Rot.from_matrix(np.column_stack([toolx, tooly, toolz]))
            pos = np.array([cup[a][0], cup[a][1], cup[a][2] + 0.05]) + over * toolz
            return self._pose(pos, r.as_quat())
        
        pregrasp_over, pregrasp_steps = self.motion_parameters[:2]
        self.move_tcp(
            {a: side_pose(a, pregrasp_over) for a in ARMS},
            steps=int(pregrasp_steps),
            grip={"left": 1.0, "right": 1.0},
        )

        # Ease in gradually: commanding the full approach in one jump moves fast enough to smack
        # into the cups (both arms) before they're actually grasped.
        for over in np.linspace(-0.1, 0.02, 7)[1:]:
            self.move_tcp({a: side_pose(a, over) for a in ARMS}, steps=25)

        if self.verbose:
            for a in ARMS:
                lp = self.d.body(f"gripper{a}_left_pad").xpos
                rp = self.d.body(f"gripper{a}_right_pad").xpos
                tcp = self._tcp_world(a).translation()
                print(f"    geom {a}: cup={np.round(cup[a], 3)} tcp={np.round(tcp, 3)} "
                      f"Lpad={np.round(lp, 3)} Rpad={np.round(rp, 3)}", flush=True)
        self.set_grip(left=0.0, right=0.0, steps=50)
        self._log("01_grasp")

        # measure grasp offset (cup -> tcp) now that cups are held
        for a in ARMS:
            self.cup_T_tcp[a] = self._body_pose(f"{a}teacup_body").inverse() * self._tcp_world(a)
            self.cup_upright[a] = self._Rot.from_matrix(self.d.body(f"{a}teacup_body").xmat.reshape(3, 3))

        def cup_quat(arm, tilt_deg=0.0, tilt_axis="x"):
            r = self._Rot.from_euler(tilt_axis, tilt_deg, degrees=True) * self.cup_upright[arm]
            return r.as_quat()

        # 4) Lift both cups gently.  This is also the endpoint for --test-grasp.
        self.move_cup_smooth(
            {a: (cup[a] + [0, 0, 0.1], cup_quat(a)) for a in ARMS},
            transitions=10,
            settle_steps=2,
        )
        self._log("02_lift")
        grasp_success = bool(
            self.last_info.get("left_gripper_grasps_left_cup")
            and self.last_info.get("right_gripper_grasps_right_cup")
            and self.last_info.get("left_cup_lifted")
            and self.last_info.get("right_cup_lifted")
        )
        if not grasp_success:
            print("  [abort] grasp/lift failed; skipping pour choreography", flush=True)
            return self.last_info
        if self.test_grasp:
            return self.last_info

        # 5/6) Bring both held cups into the pour setup together.  The source first rises while
        #      the receiver moves inward, then both cross toward the centre with generous
        #      vertical separation.  Only the source's short final vertical drop remains after
        #      the receiver arrives.  This removes the visibly sequential long moves while
        #      preserving the original, known-stable final pour alignment and grasp calibration.
        d = 1.0 if tgt == "left" else -1.0
        green = {a: self.green_pos(a) for a in ARMS}
        pre_pour_spot = np.array([0.5, 0.15, self._table_z + 0.17])
        pour_spot = np.array([0.5, 0.0, self._table_z + 0.20])
        # Pour around the cup's PHYSICAL rim.  The old path used 0.105 m as the
        #    rim height (that is only the task's logical containment height).  At 130
        #    degrees that placed the physical outlet 45 mm to one side of the receiving
        #    centre, so nearly every marble missed.
        #
        #    Keep the source body's endpoint fixed on one known-stable IK branch, but choose
        #    its position from the complete lower-lip sweep.  A marble can first
        #    clear the wall at atan(rim_height / radius), about 66 degrees.  Between
        #    there and the draining angle of 130 degrees, the low lip travels only
        #    44 mm in y.  Centre that interval over the receiver: its extrema are then
        #    +/-22 mm, within the 27 mm effective catch radius for a 5 mm marble.
        pour_ang = 130.0
        spill_ang = np.degrees(np.arctan2(CUP_RIM_HEIGHT, CUP_RADIUS))

        def lower_lip_y(angle_deg):
            angle = np.radians(angle_deg)
            return CUP_RADIUS * np.cos(angle) + CUP_RIM_HEIGHT * np.sin(angle)

        lip_y_spill = lower_lip_y(spill_ang)
        lip_y_final = lower_lip_y(pour_ang)
        # Bias 5% toward the final outlet, where the bulk of the marbles leave.
        lip_y_mid = 0.45 * lip_y_spill + 0.55 * lip_y_final
        pour_body = pour_spot + np.array([0.0, -d * lip_y_mid, 0.17])
        start = self.cup_pos(src)
        self.move_cup_smooth(
            {
                src: (start + [0.0, 0.0, 0.12], cup_quat(src)),
                tgt: (pre_pour_spot, cup_quat(tgt)),
            },
            transitions=8,
            settle_steps=1,
        )
        self.move_cup_smooth(
            {
                src: (pour_body + [0.0, 0.0, 0.08], cup_quat(src)),
                tgt: (pour_spot, cup_quat(tgt)),
            },
            transitions=12,
            settle_steps=1,
        )
        # As in the original successful sequence, derive the last source waypoint from the
        # receiver body's measured (not merely commanded) position.
        receiver_setup = self.cup_pos(tgt)
        pour_body = receiver_setup + np.array([0.0, -d * lip_y_mid, 0.17])
        self.move_cup_smooth(
            {src: (pour_body, cup_quat(src))},
            transitions=6,
            settle_steps=2,
        )
        self._log("04a_pour_setup")
        tilt = -pour_ang * d

        def lowest_rim_point(cup_pose):
            cup_r = self._Rot.from_quat(np.asarray(cup_pose.rotation_q(), float))
            rim_angles = np.linspace(0.0, 2.0 * np.pi, 361)
            local_rim = np.column_stack(
                [
                    CUP_RADIUS * np.cos(rim_angles),
                    CUP_RADIUS * np.sin(rim_angles),
                    np.full_like(rim_angles, CUP_RIM_HEIGHT),
                ]
            )
            world_rim = cup_pose.translation() + cup_r.apply(local_rim)
            return world_rim[np.argmin(world_rim[:, 2])]

        final_quat = cup_quat(src, tilt, "x")
        tcp_target = self._pose(pour_body, final_quat) * self.cup_T_tcp[src]
        goal_q = self._ik(src, tcp_target, self.cur_q[src])
        pour_steps = 84
        if goal_q is None:
            print(f"  [warn] pour IK failed for {src}", flush=True)
        else:
            start_q = self.cur_q[src].copy()
            goal_q = np.asarray(goal_q, float)
            measured_grasp = (
                self._body_pose(f"{src}teacup_body").inverse() * self._tcp_world(src)
            )
            predicted_path = []
            for i in range(1, pour_steps + 1):
                u = i / pour_steps
                alpha = u * u * (3.0 - 2.0 * u)
                q = start_q + alpha * (goal_q - start_q)
                tcp_base = self.ik[src].forward(q) * self.tcp_offset
                predicted_cup = self.base_pose[src] * tcp_base * measured_grasp.inverse()
                predicted_r = self._Rot.from_quat(
                    np.asarray(predicted_cup.rotation_q(), float)
                )
                predicted_tilt = np.degrees(
                    np.arccos(
                        np.clip(predicted_r.apply([0.0, 0.0, 1.0])[2], -1.0, 1.0)
                    )
                )
                if predicted_tilt >= spill_ang:
                    predicted_path.append((i, lowest_rim_point(predicted_cup)))

            if predicted_path:
                path_points = np.asarray([p for _, p in predicted_path])
                path_xy = path_points[:, :2]
                fixed_center = 0.5 * (path_xy.min(axis=0) + path_xy.max(axis=0))
                fixed_radius = float(
                    np.max(np.linalg.norm(path_xy - fixed_center, axis=1))
                )
                key_indices = np.unique(
                    np.rint(np.linspace(0, len(predicted_path) - 1, 5)).astype(int)
                )

                def receiver_body_for(path_index):
                    progress = path_index / max(1, len(predicted_path) - 1)
                    if src == "left":
                        # The mirrored left-source branch has a compact outlet path but releases
                        # its last layer as one large batch.  Keep its receiver closer to suppress
                        # rim rebounds; the wider right-source path needs more early clearance.
                        gap = 0.022 * (1.0 - progress) + 0.010 * progress
                    else:
                        gap = 0.040 * (1.0 - progress) + 0.012 * progress
                    outlet = predicted_path[path_index][1]
                    return outlet - np.array([0.0, 0.0, gap + CUP_RIM_HEIGHT])

                # Pre-position the empty receiver under the first spill point while the
                # source is upright, then re-measure the receiver grasp for accurate tracking.
                first_body = receiver_body_for(int(key_indices[0]))
                self.move_cup_smooth(
                    {tgt: (first_body, cup_quat(tgt))},
                    transitions=8,
                    settle_steps=2,
                )
                self.cup_T_tcp[tgt] = (
                    self._body_pose(f"{tgt}teacup_body").inverse() * self._tcp_world(tgt)
                )
                self.move_cup_smooth(
                    {tgt: (first_body, cup_quat(tgt))},
                    transitions=4,
                    settle_steps=2,
                )

                target_keyframes = [(predicted_path[int(key_indices[0])][0], self.cur_q[tgt].copy())]
                target_seed = self.cur_q[tgt].copy()
                for path_index in key_indices[1:]:
                    path_index = int(path_index)
                    body = receiver_body_for(path_index)
                    target_tcp = (
                        self._pose(body, cup_quat(tgt)) * self.cup_T_tcp[tgt]
                    )
                    target_q = self._ik(tgt, target_tcp, target_seed)
                    if target_q is None:
                        print(
                            f"  [warn] receiver tracking IK failed at path index {path_index}",
                            flush=True,
                        )
                        target_q = target_seed
                    target_seed = np.asarray(target_q, float)
                    target_keyframes.append(
                        (predicted_path[path_index][0], target_seed.copy())
                    )
                if self.verbose:
                    print(
                        f"    fixed-receiver path radius={fixed_radius:.4f}m; "
                        f"tracking with {len(target_keyframes)} keyframes",
                        flush=True,
                    )
            else:
                target_keyframes = []

            tracking_errors = []
            tracking_start_step = (
                target_keyframes[0][0] if target_keyframes else pour_steps + 1
            )
            previous_count = self.last_info.get(f"marbles_in_{tgt}_cup", 0)
            final_catch_adjusted = False
            for i in range(1, pour_steps + 1):
                u = i / pour_steps
                alpha = u * u * (3.0 - 2.0 * u)
                self.cur_q[src] = start_q + alpha * (goal_q - start_q)
                if target_keyframes:
                    if i <= target_keyframes[0][0]:
                        self.cur_q[tgt] = target_keyframes[0][1].copy()
                    elif i >= target_keyframes[-1][0]:
                        self.cur_q[tgt] = target_keyframes[-1][1].copy()
                    else:
                        for (i0, q0), (i1, q1) in zip(
                            target_keyframes[:-1], target_keyframes[1:]
                        ):
                            if i0 <= i <= i1:
                                beta = (i - i0) / (i1 - i0)
                                self.cur_q[tgt] = q0 + beta * (q1 - q0)
                                break
                self._step_once()
                actual_outlet = lowest_rim_point(
                    self._body_pose(f"{src}teacup_body")
                )
                tgt_body_now = self.d.body(f"{tgt}teacup_body")
                tgt_r_now = self._Rot.from_matrix(tgt_body_now.xmat.reshape(3, 3))
                actual_receiver = tgt_body_now.xpos + tgt_r_now.apply(
                    [0.0, 0.0, CUP_RIM_HEIGHT]
                )
                step_error = actual_outlet - actual_receiver
                if i >= tracking_start_step:
                    tracking_errors.append(step_error)
                current_count = self.last_info.get(f"marbles_in_{tgt}_cup", 0)
                if self.verbose and current_count != previous_count:
                    print(
                        f"    transfer step={i:02d} count={current_count} "
                        f"outlet_delta={np.round(step_error, 4)}",
                        flush=True,
                    )
                previous_count = current_count
                if current_count == N_MARBLES - 1 and not final_catch_adjusted:
                    # The final marble often drains a few steps after the main group.  Stop
                    # moving the nearly-full receiver along its approximate keyframes and put
                    # it directly under the measured outlet for the remainder of the drain.
                    self.cup_T_tcp[tgt] = (
                        self._body_pose(f"{tgt}teacup_body").inverse()
                        * self._tcp_world(tgt)
                    )
                    catch_body = actual_outlet - np.array(
                        [0.0, 0.0, CUP_RIM_HEIGHT + 0.012]
                    )
                    self.move_cup_smooth(
                        {tgt: (catch_body, cup_quat(tgt))},
                        transitions=2,
                        settle_steps=2,
                    )
                    # Re-measure once under load and close the residual rim-position error.
                    self.cup_T_tcp[tgt] = (
                        self._body_pose(f"{tgt}teacup_body").inverse()
                        * self._tcp_world(tgt)
                    )
                    catch_body = lowest_rim_point(
                        self._body_pose(f"{src}teacup_body")
                    ) - np.array([0.0, 0.0, CUP_RIM_HEIGHT + 0.012])
                    self.move_cup_smooth(
                        {tgt: (catch_body, cup_quat(tgt))},
                        transitions=2,
                        settle_steps=4,
                    )
                    target_keyframes = []
                    final_catch_adjusted = True
                    previous_count = self.last_info.get(
                        f"marbles_in_{tgt}_cup", previous_count
                    )
                    if self.verbose:
                        print(
                            f"    final catch count={previous_count} "
                            f"receiver_body={np.round(catch_body, 4)}",
                            flush=True,
                        )
            for _ in range(5):
                self._step_once()
            if self.verbose and tracking_errors:
                tracking_errors = np.asarray(tracking_errors)
                print(
                    f"    max outlet xy tracking error="
                    f"{np.max(np.linalg.norm(tracking_errors[:, :2], axis=1)):.4f}m",
                    flush=True,
                )
        if self.verbose:
            outlet = lowest_rim_point(self._body_pose(f"{src}teacup_body"))
            tgt_body = self.d.body(f"{tgt}teacup_body")
            tgt_r = self._Rot.from_matrix(tgt_body.xmat.reshape(3, 3))
            receiver = tgt_body.xpos + tgt_r.apply([0.0, 0.0, CUP_RIM_HEIGHT])
            print(
                f"    measured outlet={np.round(outlet, 4)} "
                f"receiver={np.round(receiver, 4)} "
                f"delta={np.round(outlet - receiver, 4)}",
                flush=True,
            )
        grasp_key = f"{src}_gripper_grasps_{src}_cup"
        if self.verbose:
            print(f"    pour: grasp={self.last_info.get(grasp_key)} "
                  f"tgt_marbles={self.last_info.get(f'marbles_in_{tgt}_cup')}", flush=True)
        self._log("04_pour")
        for a in ARMS:
            if self.last_info.get(f"{a}_gripper_grasps_{a}_cup"):
                self.cup_T_tcp[a] = (
                    self._body_pose(f"{a}teacup_body").inverse() * self._tcp_world(a)
                )
        # Retreat up and toward the source arm FIRST, still tilted, to clear the target cup
        # entirely, THEN untilt in place.  A straight 0.18 m lift was outside the right arm's
        # reachable branch on seed 0; this shorter diagonal retreat provides more separation
        # while staying reachable.
        clear_high = self.cup_pos(src) + [0.0, -0.10 * d, 0.08]
        self.move_cup_smooth(
            {src: (clear_high, cup_quat(src, tilt, "x"))},
            transitions=8,
            settle_steps=1,
        )
        for tf in (0.4, 0.7, 1.0):
            self.move_cup_smooth(
                {src: (clear_high, cup_quat(src, tilt * (1 - tf), "x"))},
                transitions=6,
                settle_steps=1,
            )
        self._log("04c_untilt")
        for a in ARMS:
            if self.last_info.get(f"{a}_gripper_grasps_{a}_cup"):
                self.cup_T_tcp[a] = (
                    self._body_pose(f"{a}teacup_body").inverse() * self._tcp_world(a)
                )

        # 7) carry both still-held cups back to their own green squares, place them down upright,
        #    and only now release -- neither cup ever touched the table before this.
        home_start = {a: self.cup_pos(a) for a in ARMS}
        home_high = {a: green[a] + [0.0, 0.0, 0.22] for a in ARMS}
        for f in (1.0 / 3.0, 2.0 / 3.0, 1.0):
            self.move_cup_smooth(
                {
                    a: (home_start[a] * (1.0 - f) + home_high[a] * f, cup_quat(a))
                    for a in ARMS
                },
                transitions=7,
                settle_steps=1,
            )
        self.move_cup_smooth(
            {a: (green[a] + [0.0, 0.0, 0.06], cup_quat(a)) for a in ARMS},
            transitions=8,
            settle_steps=1,
        )
        self.move_cup_smooth(
            {a: (green[a] + [0.0, 0.0, 0.004], cup_quat(a)) for a in ARMS},
            transitions=6,
            settle_steps=3,
        )
        self._log("05a_touch")
        self.set_grip(left=1.0, right=1.0, steps=25)
        self._log("05b_release")
        # These are side grasps, so a vertical retreat can drag a finger or wrist bracket up the
        # cup wall and tip it after an otherwise clean release.  Move outward from each cup while
        # rising instead: +y for the left arm, -y for the right arm.
        retract_y, retract_z, retract_transitions = self.motion_parameters[2:]
        retract = {
            a: self._pose(
                self._tcp_world(a).translation()
                + [0.0, retract_y if a == "left" else -retract_y, retract_z],
                self._tcp_world(a).rotation_q(),
            )
            for a in ARMS
        }
        self.move_tcp_smooth(retract, transitions=int(retract_transitions), settle_steps=3)
        self._log("05_place")
        return self.last_info


# --------------------------------------------------------------------------------------------
# Env setup + record/replay
# --------------------------------------------------------------------------------------------
def make_env(render: bool):
    import gymnasium as gym
    import rcs  # noqa: F401
    import duobench  # noqa: F401
    import duobench.tasks.pour_marbles  # noqa: F401  (registers the env)
    from rcs._core.sim import SimConfig
    from rcs.envs.base import ControlMode, RelativeTo

    creator = gym.spec(ENV_ID).entry_point
    cfg = creator.config()
    cfg.control_mode = ControlMode.JOINTS
    cfg.relative_to = RelativeTo.NONE
    cfg.headless = not render
    cfg.camera_cfgs = None
    cfg.camera_adds = None
    cfg.sim_cfg = SimConfig(async_control=False, realtime=render, frequency=30, max_convergence_steps=300)
    return creator.create_env(cfg)


def state_from_npz(npz) -> dict[str, np.ndarray]:
    """Read canonical v2, legacy recording, or legacy named-joint state files."""
    if "cup_qpos" in npz and "marble_qpos" in npz:
        cup_qpos = np.asarray(npz["cup_qpos"])
        marble_qpos = np.asarray(npz["marble_qpos"])
        if cup_qpos.shape != (len(CUP_JOINTS), 7):
            raise ValueError(f"cup_qpos must have shape ({len(CUP_JOINTS)}, 7), got {cup_qpos.shape}")
        if marble_qpos.shape != (len(MARBLE_JOINTS), 7):
            raise ValueError(
                f"marble_qpos must have shape ({len(MARBLE_JOINTS)}, 7), got {marble_qpos.shape}"
            )
        state = {j: cup_qpos[i].copy() for i, j in enumerate(CUP_JOINTS)}
        state.update({j: marble_qpos[i].copy() for i, j in enumerate(MARBLE_JOINTS)})
        return state

    missing = [j for j in (*CUP_JOINTS, *MARBLE_JOINTS) if j not in npz]
    if missing:
        raise ValueError(
            "Not a pour_marbles initial-state file; missing object joints: "
            + ", ".join(missing[:3])
            + (" ..." if len(missing) > 3 else "")
        )
    return {j: np.asarray(npz[j]).copy() for j in (*CUP_JOINTS, *MARBLE_JOINTS)}


def velocity_from_npz(npz) -> dict[str, np.ndarray] | None:
    """Read v3 object velocities; older pose-only files intentionally fall back to zero."""
    if "cup_qvel" not in npz or "marble_qvel" not in npz:
        return None
    cup_qvel = np.asarray(npz["cup_qvel"])
    marble_qvel = np.asarray(npz["marble_qvel"])
    if cup_qvel.shape != (len(CUP_JOINTS), 6):
        raise ValueError(f"cup_qvel must have shape ({len(CUP_JOINTS)}, 6), got {cup_qvel.shape}")
    if marble_qvel.shape != (len(MARBLE_JOINTS), 6):
        raise ValueError(
            f"marble_qvel must have shape ({len(MARBLE_JOINTS)}, 6), got {marble_qvel.shape}"
        )
    velocity = {j: cup_qvel[i].copy() for i, j in enumerate(CUP_JOINTS)}
    velocity.update({j: marble_qvel[i].copy() for i, j in enumerate(MARBLE_JOINTS)})
    return velocity


def recording_payload(
    oracle: PourOracle,
    object_state: dict[str, np.ndarray],
    object_velocity: dict[str, np.ndarray],
    seed: int,
    record_id: int,
    source_cup: str,
    final_info: dict,
) -> dict[str, np.ndarray]:
    """Build a self-describing recording while retaining legacy ``actions`` replay support."""
    actions = np.asarray(oracle.traj, dtype=np.float32)
    left_qpos = np.asarray(oracle.joint_qpos_traj["left"], dtype=np.float32)
    right_qpos = np.asarray(oracle.joint_qpos_traj["right"], dtype=np.float32)
    timestamps = np.asarray(oracle.timestamps_s, dtype=np.float64)
    if not (len(actions) == len(left_qpos) == len(right_qpos) == len(timestamps)):
        raise RuntimeError("Command and measured trajectory lengths do not match")

    payload = object_state_payload(
        object_state,
        object_velocity,
        seed=seed,
        source_cup=source_cup,
    )
    payload.update({
        "file_type": np.asarray("pour_marbles_recording"),
        "record_id": np.asarray(record_id, dtype=np.int64),
        "actions": actions,
        "timestamps_s": timestamps,
        "joint_state_sample": np.asarray("post_step"),
        "left_joint_names": np.asarray(oracle.joint_names["left"]),
        "right_joint_names": np.asarray(oracle.joint_names["right"]),
        "initial_left_joint_qpos": oracle.initial_joint_qpos["left"].astype(np.float32),
        "initial_right_joint_qpos": oracle.initial_joint_qpos["right"].astype(np.float32),
        "left_joint_qpos": left_qpos,
        "right_joint_qpos": right_qpos,
        "left_joint_targets": actions[:, 0:7],
        "right_joint_targets": actions[:, 8:15],
        "left_gripper_commands": actions[:, 7],
        "right_gripper_commands": actions[:, 15],
        "final_stage": np.asarray(final_info.get("stage", -1), dtype=np.int32),
        "final_max_stage": np.asarray(final_info.get("max_stage", 6), dtype=np.int32),
        "final_success": np.asarray(bool(final_info.get("success", False))),
        "final_marbles_left": np.asarray(
            final_info.get("marbles_in_left_cup", -1), dtype=np.int32
        ),
        "final_marbles_right": np.asarray(
            final_info.get("marbles_in_right_cup", -1), dtype=np.int32
        ),
        "final_left_cup_in_place": np.asarray(
            bool(final_info.get("left_cup_in_place", False))
        ),
        "final_right_cup_in_place": np.asarray(
            bool(final_info.get("right_cup_in_place", False))
        ),
        "final_left_cup_upright": np.asarray(
            bool(final_info.get("left_cup_upright", False))
        ),
        "final_right_cup_upright": np.asarray(
            bool(final_info.get("right_cup_upright", False))
        ),
        "strict_success": np.asarray(strict_demonstration_success(final_info)),
        "motion_variant": np.asarray(oracle.motion_variant, dtype=np.int64),
        "motion_parameters": oracle.motion_parameters.astype(np.float32),
    })
    return payload


def strict_demonstration_success(info: dict) -> bool:
    """Higher bar for training data than the task's cumulative stage success flag."""
    src = str(info.get("source_cup", ""))
    if src not in ARMS:
        return False
    tgt = "left" if src == "right" else "right"
    stage = int(info.get("stage", -1))
    max_stage = int(info.get("max_stage", 6))
    return bool(
        info.get("success", stage >= max_stage)
        and stage >= max_stage
        and info.get(f"marbles_in_{tgt}_cup") == N_MARBLES
        and info.get(f"marbles_in_{src}_cup") == 0
        and info.get("left_cup_in_place")
        and info.get("right_cup_in_place")
        and info.get("left_cup_upright")
        and info.get("right_cup_upright")
    )


def main() -> int:
    args = _parse_args()
    env = make_env(args.render)
    sim = env.get_wrapper_attr("sim")
    try:
        _, info = env.reset(seed=args.seed)

        if args.save_state:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_state)) or ".", exist_ok=True)
            state = capture_object_state(sim)
            velocity = capture_object_velocity(sim)
            np.savez(
                args.save_state,
                **object_state_payload(
                    state,
                    velocity,
                    seed=args.seed,
                    source_cup=str(info.get("source_cup", "")),
                ),
            )
            print(f"Saved object state (seed {args.seed}) -> {args.save_state}")
            return 0

        if args.replay:
            with np.load(args.replay, allow_pickle=False) as npz:
                state = state_from_npz(npz)
                velocity = velocity_from_npz(npz)
                actions = np.asarray(npz["actions"]).copy()
                source_cup = str(npz["source_cup"])
            restore_object_state(sim, state, velocity)
            env.get_wrapper_attr("stage_tracker").update_internal_state(sim)
            print(f"Replaying {len(actions)} steps (source_cup={source_cup}) ...")
            for a in actions:
                _, _, _, _, info = env.step({
                    "left": {"joints": a[0:7].astype(np.float64), "gripper": a[7:8].astype(np.float32)},
                    "right": {"joints": a[8:15].astype(np.float64), "gripper": a[15:16].astype(np.float32)},
                })
                if args.render:
                    sim.sync_gui()
            print(
                f"Replay final stage {info.get('stage')}/{info.get('max_stage')}, "
                f"success={info.get('success')}  "
                f"placeL/R={info.get('left_cup_in_place')}/{info.get('right_cup_in_place')}  "
                f"uprightL/R={info.get('left_cup_upright')}/{info.get('right_cup_upright')}"
            )
            return 0

        if args.load_state:
            with np.load(args.load_state, allow_pickle=False) as npz:
                state = state_from_npz(npz)
                velocity = velocity_from_npz(npz)
            restore_object_state(sim, state, velocity)
            env.get_wrapper_attr("stage_tracker").update_internal_state(sim)
            info = {**info, **env.get_wrapper_attr("stage_tracker").info}
            print(f"Loaded object state from {args.load_state}")

        obj_state = capture_object_state(sim)
        obj_velocity = capture_object_velocity(sim)
        oracle = PourOracle(
            env,
            verbose=args.verbose,
            render=args.render,
            frames_dir=args.debug_frames,
            test_grasp=args.test_grasp,
            motion_variant=args.motion_variant,
        )
        oracle.last_info = info
        info = oracle.run()

        if args.test_grasp:
            grasp_success = bool(
                info.get("left_gripper_grasps_left_cup")
                and info.get("right_gripper_grasps_right_cup")
                and info.get("left_cup_lifted")
                and info.get("right_cup_lifted")
            )
            print(
                f"\nGrasp test: success={grasp_success}  "
                f"graspL/R={info.get('left_gripper_grasps_left_cup')}/"
                f"{info.get('right_gripper_grasps_right_cup')}  "
                f"liftedL/R={info.get('left_cup_lifted')}/"
                f"{info.get('right_cup_lifted')}",
                flush=True,
            )
            return 0 if grasp_success else 1

        stage, max_stage = int(info.get("stage", -1)), int(info.get("max_stage", 6))
        success = bool(info.get("success", stage >= max_stage))
        strict_success = strict_demonstration_success(info)
        print(f"\nFinished: stage {stage}/{max_stage}  success={success}  "
              f"marblesL/R={info.get('marbles_in_left_cup')}/{info.get('marbles_in_right_cup')}  "
              f"placeL/R={info.get('left_cup_in_place')}/{info.get('right_cup_in_place')}  "
              f"uprightL/R={info.get('left_cup_upright')}/{info.get('right_cup_upright')}",
              flush=True)

        if args.record:
            if strict_success:
                os.makedirs(os.path.dirname(os.path.abspath(args.record)) or ".", exist_ok=True)
                np.savez(
                    args.record,
                    **recording_payload(
                        oracle,
                        obj_state,
                        obj_velocity,
                        seed=args.seed,
                        record_id=args.seed if args.record_id is None else args.record_id,
                        source_cup=str(info.get("source_cup", "")),
                        final_info=info,
                    ),
                )
                print(f"Saved recording -> {args.record}")
            else:
                print(
                    "Not recording: run did not meet strict demonstration criteria "
                    f"(stage {stage}/{max_stage}, task_success={success})."
                )
        return 0 if (strict_success if args.record else success) else 1
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(main())
