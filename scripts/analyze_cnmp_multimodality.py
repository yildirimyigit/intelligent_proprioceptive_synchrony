#!/usr/bin/env python3
"""Quantify per-arm trajectory multimodality in the pour_marbles demos (read-only).

Tests the hypothesis that the right-arm CNMP is harder to fit because the right arm's joint
trajectories are MULTIMODAL given the conditioning -- similar (self start/mid/end + partner),
but different interior path, i.e. IK branch/path variation -- which a unimodal-Gaussian CNMP
cannot represent (it averages the branches, inflating MSE).

Metrics per arm (self = that arm), using the same normalized T=500 tensors the CNMP trains on:
  * kNN conditional dispersion: for each demo, the mean full self-trajectory MSE to its K nearest
    neighbours in CONDITIONING space. High = similar conditioning yet dispersed trajectories.
  * branch fraction: demos whose single nearest-conditioning neighbour nonetheless has a
    top-quartile self-trajectory distance (same conditioning -> very different path).
Both are broken down by role (holder vs pourer), where the left/right asymmetry lives.

Conditioning per demo = self and partner joints at 5 phases (start .. end). Trajectory distance =
mean squared difference over the whole self trajectory.
"""

from __future__ import annotations

import numpy as np

from cnmp_data import make_cross_conditioned_dataset, _load_resampled

GI = [0, 125, 250, 375, 499]  # phases used for the conditioning proxy
K = 5


def _pairwise_sq(a: np.ndarray) -> np.ndarray:
    """Row-wise pairwise squared Euclidean distances, diagonal set to +inf."""
    g = a @ a.T
    d = np.diag(g)[:, None] + np.diag(g)[None, :] - 2 * g
    np.fill_diagonal(d, np.inf)
    return np.maximum(d, 0.0)


def analyze(arm: str, sources: np.ndarray) -> None:
    ds = make_cross_conditioned_dataset(arm, T=500)
    Y = ds.Y.numpy()          # (100, 500, 7) self, normalized
    G = ds.gamma.numpy()      # (100, 500, 7) partner, normalized
    n = Y.shape[0]

    cond = np.concatenate([Y[:, GI, :].reshape(n, -1), G[:, GI, :].reshape(n, -1)], axis=1)
    cond_d = _pairwise_sq(cond)                      # conditioning distance
    traj_mse = _pairwise_sq(Y.reshape(n, -1)) / Y[0].size  # full-trajectory MSE between demos

    nn = cond_d.argsort(1)[:, :K]                    # K nearest by conditioning
    disp = np.take_along_axis(traj_mse, nn, 1).mean(1)   # local trajectory dispersion per demo
    nearest = cond_d.argmin(1)
    thr = np.quantile(traj_mse[np.isfinite(traj_mse)], 0.75)
    branch = traj_mse[np.arange(n), nearest] > thr   # nearest-conditioning twin has a far path

    role = np.where(sources == arm, "pour", "hold")  # self pours when it is the source cup
    print(f"{arm:5s} arm:  kNN-conditional dispersion={disp.mean():.3f}   branch fraction={branch.mean():.2f}")
    for r in ("hold", "pour"):
        m = role == r
        print(f"    role={r} (n={m.sum():2d}):  dispersion={disp[m].mean():.3f}   branch fraction={branch[m].mean():.2f}")


def main() -> int:
    _, _, sources = _load_resampled(500)
    sources = np.array(sources)
    print(f"multimodality (higher = more branch/path variation given similar conditioning); K={K}\n")
    for arm in ("left", "right"):
        analyze(arm, sources)
    print("\nInterpretation: a large right>left gap in dispersion / branch fraction -- especially in the")
    print("trivial 'hold' role -- confirms the right arm takes different joint paths for similar")
    print("conditioning, which a unimodal-Gaussian CNMP cannot fit (hence its higher val MSE).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
