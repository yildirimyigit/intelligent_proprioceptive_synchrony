#!/usr/bin/env python3
"""Read-only data layer for training cross-conditioned CNMPs on the pour_marbles demos.

Turns the 100 strict-success recordings in ``data/duo/pour_marbles`` into the tensors the CNMP
of ``cnmp.py`` + the ``TrajectorySampler`` of ``cnmp_test_data.py`` consume -- WITHOUT modifying
any ``.npz``.

Design (user-confirmed):
* **Cross-conditioned, one CNMP per arm.** For arm A (partner B), the model predicts A's realized
  joints from time, conditioned on B's *momentary* realized joints:
      input   x      = time                     (input_dim = 1)
      output  SM     = q_A  (7 realized joints)  (output_dim = 7)
      gamma          = q_B  (7 realized joints), **time-varying** (param_dim = 7)
  The partner joints ride in the model's ``gamma`` (context values in ``gamma``, query values in
  ``tar_gamma``) via the per-point-gamma extension to the sampler and model.
* **Realized joints only** (no gripper), per the project decision.
* **Downsample to T=500** points (largest round number below the 508-step minimum) by linear
  interpolation on normalized time [0, 1]. The ``.npz`` files are never touched.
* **Leakage-safe 80/10/10 split**: whole initial *scenes* stay within one split, stratified by
  ``source_cup`` (identical split for the left- and right-arm models).
* **Train-only per-arm, per-joint z-score normalization.**

Usage:
    python scripts/cnmp_data.py                 # write split + norm JSON, print shapes
    from cnmp_data import make_cross_conditioned_dataset
    ds = make_cross_conditioned_dataset("left", T=500)   # ds.Y, ds.gamma, ds.x, ds.train_ids, ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "duo" / "pour_marbles"
SPLIT_JSON = ROOT / "data" / "duo" / "pour_marbles_cnmp_split.json"
NORM_JSON = ROOT / "data" / "duo" / "pour_marbles_cnmp_norm.json"
ARMS = ("left", "right")
N_JOINTS = 7
DEFAULT_T = 500


def partner_of(arm: str) -> str:
    return "right" if arm == "left" else "left"


# --------------------------------------------------------------------------------------------
# Per-demo loading (read-only) + resampling
# --------------------------------------------------------------------------------------------
def _scene_hash(d) -> str:
    return hashlib.sha256(
        np.asarray(d["cup_qpos"]).tobytes() + np.asarray(d["marble_qpos"]).tobytes()
    ).hexdigest()


def load_demo(path: str | Path) -> dict:
    """Load one recording (native length) into plain numpy arrays. Does not modify the file."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as d:
        t = np.asarray(d["timestamps_s"], dtype=np.float64)
        t_norm = (t - t[0]) / (t[-1] - t[0])  # -> [0, 1]; timestamps are strictly increasing
        return {
            "name": path.name,
            "source_cup": str(d["source_cup"]),
            "scene_hash": _scene_hash(d),
            "t": t_norm,  # (T_i,)
            "q": {a: np.asarray(d[f"{a}_joint_qpos"], np.float64) for a in ARMS},  # (T_i,7) realized
        }


def resample(demo: dict, T: int) -> dict[str, np.ndarray]:
    """Linear-interpolate a demo's realized joints onto a uniform T-point grid on [0, 1]."""
    grid = np.linspace(0.0, 1.0, T)
    return {a: np.stack([np.interp(grid, demo["t"], demo["q"][a][:, j]) for j in range(N_JOINTS)],
                        axis=1) for a in ARMS}  # (T, 7) each


# --------------------------------------------------------------------------------------------
# Leakage-safe 80/10/10 split: whole scenes stay together, stratified by source.
# --------------------------------------------------------------------------------------------
def build_split(data_dir: Path = DATA_DIR, ratios=(0.8, 0.1, 0.1), seed: int = 0) -> dict:
    files = sorted(data_dir.glob("demo_seed_*.npz"))
    groups: dict[tuple[str, str], list[str]] = {}
    for f in files:
        with np.load(f, allow_pickle=False) as d:
            key = (str(d["source_cup"]), _scene_hash(d))
        groups.setdefault(key, []).append(f.name)

    rng = np.random.default_rng(seed)
    split = {"train": [], "val": [], "test": []}
    for source in ARMS:
        src_groups = [v for (s, _), v in sorted(groups.items()) if s == source]
        n_files = sum(len(g) for g in src_groups)
        n_test, n_val = round(n_files * ratios[2]), round(n_files * ratios[1])
        # multi-file scenes (reused anchor layouts) go to train, so val/test are clean, distinct
        # single-scene generalization cases; singletons fill test/val to the target counts.
        singles = [g for g in src_groups if len(g) == 1]
        for g in [g for g in src_groups if len(g) > 1]:
            split["train"] += g
        rng.shuffle(singles)
        for i, g in enumerate(singles):
            bucket = "test" if i < n_test else "val" if i < n_test + n_val else "train"
            split[bucket] += g
    return {k: sorted(v) for k, v in split.items()}


def load_split(path: Path = SPLIT_JSON) -> dict:
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------------------------------
# Train-only normalization (per-arm, per-joint z-score).
# --------------------------------------------------------------------------------------------
class Normalizer:
    def __init__(self, mean: dict, std: dict, cup_mean=None, cup_std=None):
        self.mean = {a: np.asarray(mean[a], np.float64) for a in ARMS}
        self.std = {a: np.asarray(std[a], np.float64) for a in ARMS}
        self.cup_mean = None if cup_mean is None else np.asarray(cup_mean, np.float64)
        self.cup_std = None if cup_std is None else np.asarray(cup_std, np.float64)

    @classmethod
    def fit(cls, resampled: dict[str, np.ndarray], train_ids: np.ndarray, cups=None) -> "Normalizer":
        """Fit on the TRAIN split only. ``resampled[a]`` is (num_traj, T, 7); ``cups`` is (num_traj, 4)."""
        mean, std = {}, {}
        for a in ARMS:
            frames = resampled[a][train_ids].reshape(-1, N_JOINTS)  # (n_train*T, 7)
            mean[a], std[a] = frames.mean(0), frames.std(0) + 1e-6
        cup_mean = cup_std = None
        if cups is not None:
            cup_mean, cup_std = cups[train_ids].mean(0), cups[train_ids].std(0) + 1e-6
        return cls(mean, std, cup_mean, cup_std)

    def apply(self, q: np.ndarray, arm: str) -> np.ndarray:
        return (q - self.mean[arm]) / self.std[arm]

    def invert(self, q_norm: np.ndarray, arm: str) -> np.ndarray:
        return q_norm * self.std[arm] + self.mean[arm]

    def apply_cups(self, cups: np.ndarray) -> np.ndarray:
        return (cups - self.cup_mean) / self.cup_std

    def to_json(self) -> dict:
        j = {"mean": {a: self.mean[a].tolist() for a in ARMS},
             "std": {a: self.std[a].tolist() for a in ARMS}}
        if self.cup_mean is not None:
            j["cup_mean"], j["cup_std"] = self.cup_mean.tolist(), self.cup_std.tolist()
        return j

    @classmethod
    def from_json(cls, path: Path = NORM_JSON) -> "Normalizer":
        j = json.loads(Path(path).read_text())
        return cls(j["mean"], j["std"], j.get("cup_mean"), j.get("cup_std"))


# --------------------------------------------------------------------------------------------
# Cross-conditioned dataset builder
# --------------------------------------------------------------------------------------------
@dataclass
class CrossConditionedData:
    arm: str                    # the "self" arm this dataset predicts
    partner: str
    T: int
    x: torch.Tensor            # (T, 1) shared normalized-time grid
    Y: torch.Tensor            # (num_traj, T, 7) self realized joints (normalized)
    gamma: torch.Tensor        # (num_traj, T, 7) partner realized joints (normalized), time-varying
    train_ids: torch.Tensor    # long, index into num_traj
    val_ids: torch.Tensor
    test_ids: torch.Tensor
    names: list[str]           # file name per num_traj index
    norm: Normalizer
    cup_dim: int = 0           # extra STATIC gamma dims from the cup positions (0 if disabled)


def _load_cup_targets(names: list[str]) -> np.ndarray:
    """Initial (x, y) of the left and right cups per demo -> (num_traj, 4) = [Lx, Ly, Rx, Ry]."""
    cups = []
    for n in names:
        with np.load(DATA_DIR / n, allow_pickle=False) as d:
            cups.append(np.asarray(d["cup_qpos"])[:, :2].reshape(-1))
    return np.stack(cups).astype(np.float64)


def _load_resampled(T: int) -> tuple[list[str], dict[str, np.ndarray], list[str]]:
    """Return (names, {arm: (num_traj, T, 7)}, source_cup per traj) in sorted file order."""
    files = sorted(DATA_DIR.glob("demo_seed_*.npz"))
    names, sources = [], []
    stacks = {a: [] for a in ARMS}
    for f in files:
        demo = load_demo(f)
        rs = resample(demo, T)
        names.append(demo["name"])
        sources.append(demo["source_cup"])
        for a in ARMS:
            stacks[a].append(rs[a])
    resampled = {a: np.stack(stacks[a], axis=0).astype(np.float64) for a in ARMS}  # (num_traj,T,7)
    return names, resampled, sources


def make_cross_conditioned_dataset(arm: str, T: int = DEFAULT_T, split_seed: int = 0,
                                   device: torch.device | str = "cpu",
                                   only_source: str | None = None,
                                   cup_gamma: bool = False) -> CrossConditionedData:
    """Build the (Y=self, gamma=partner) tensors + leakage-safe split for one arm's CNMP.

    ``only_source`` restricts to demos whose source cup is that arm (fixed-role PoC): with
    ``only_source="left"`` the left arm always pours and the right always holds, which drops the
    high-dispersion right-arm pour trajectories so the right CNMP only has to learn the (easy)
    holding motion. The split stays leakage-safe; it is simply the single-source half of it.

    ``cup_gamma`` appends the two cups' initial ``(x, y)`` -- a per-demo STATIC task parameter --
    to the (time-varying) partner gamma, so ``param_dim`` becomes 7 + 4 = 11. This gives each
    self-CNMP its own grasp target directly, rather than having to infer it from the partner.
    """
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    if only_source is not None and only_source not in ARMS:
        raise ValueError(f"only_source must be one of {ARMS} or None")
    partner = partner_of(arm)

    names, resampled, sources = _load_resampled(T)
    index = {n: i for i, n in enumerate(names)}
    split = build_split(seed=split_seed)
    if only_source is not None:  # keep only the fixed-role (single source-cup) demos
        src_by_name = dict(zip(names, sources))
        split = {k: [n for n in v if src_by_name[n] == only_source] for k, v in split.items()}
    ids = {k: np.array([index[n] for n in split[k]], dtype=np.int64) for k in split}

    cups = _load_cup_targets(names) if cup_gamma else None      # (num_traj, 4) raw
    norm = Normalizer.fit(resampled, ids["train"], cups=cups)
    Y = norm.apply(resampled[arm], arm)              # (num_traj, T, 7) self
    gamma = norm.apply(resampled[partner], partner)  # (num_traj, T, 7) partner (time-varying)
    cup_dim = 0
    if cup_gamma:
        cups_n = norm.apply_cups(cups)               # (num_traj, 4) static, normalized
        cup_dim = cups_n.shape[1]
        cup_chan = np.broadcast_to(cups_n[:, None, :], (gamma.shape[0], T, cup_dim))
        gamma = np.concatenate([gamma, cup_chan], axis=-1)   # (num_traj, T, 7+4) partner + static cups
    x = np.linspace(0.0, 1.0, T)[:, None]            # (T, 1)

    to_t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)
    return CrossConditionedData(
        arm=arm, partner=partner, T=T,
        x=to_t(x), Y=to_t(Y), gamma=to_t(gamma),
        train_ids=torch.as_tensor(ids["train"], device=device),
        val_ids=torch.as_tensor(ids["val"], device=device),
        test_ids=torch.as_tensor(ids["test"], device=device),
        names=names, norm=norm, cup_dim=cup_dim,
    )


# --------------------------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--T", type=int, default=DEFAULT_T)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    split = build_split(seed=args.seed)
    SPLIT_JSON.write_text(json.dumps(split, indent=1))

    ds = make_cross_conditioned_dataset("left", T=args.T, split_seed=args.seed)
    NORM_JSON.write_text(json.dumps(ds.norm.to_json(), indent=1))

    # split summary + leakage check (whole scenes must not straddle splits)
    name_scene = {}
    for f in sorted(DATA_DIR.glob("demo_seed_*.npz")):
        with np.load(f, allow_pickle=False) as d:
            name_scene[f.name] = (str(d["source_cup"]), _scene_hash(d))
    scene_sets = {k: {name_scene[n][1] for n in split[k]} for k in split}
    leak = (scene_sets["train"] & scene_sets["val"]) | (scene_sets["train"] & scene_sets["test"]) | \
           (scene_sets["val"] & scene_sets["test"])

    print(f"== cross-conditioned dataset (T={args.T}, downsampled from native 508-527) ==")
    for k in ("train", "val", "test"):
        src = {a: sum(name_scene[n][0] == a for n in split[k]) for a in ARMS}
        print(f"  {k:5s}: {len(split[k]):3d} files  source={src}  unique_scenes={len(scene_sets[k])}")
    print(f"  scene leakage across splits: {len(leak)} (must be 0)")
    print(f"  arm=left model tensors: x={tuple(ds.x.shape)}  Y(self)={tuple(ds.Y.shape)}  "
          f"gamma(partner)={tuple(ds.gamma.shape)}")
    print(f"  -> CNMP dims: input_dim=1 (time), output_dim={N_JOINTS} (self), "
          f"param_dim={N_JOINTS} (partner, time-varying)")
    print(f"  norm left  mean[:3]={np.round(ds.norm.mean['left'][:3], 3)} "
          f"std[:3]={np.round(ds.norm.std['left'][:3], 3)}")
    print(f"\nwrote split -> {SPLIT_JSON.relative_to(ROOT)}")
    print(f"wrote norm  -> {NORM_JSON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
