#!/usr/bin/env python3
"""Branch-consistent IK selection for demonstration-generating oracles.

WHY THIS EXISTS
---------------
The pour_marbles oracle's ``_ik`` runs multi-seed IK and, among the valid candidates, keeps the
one with the most table clearance. For a redundant (7-DOF) arm many joint configs reach the same
TCP pose, so "max clearance" lets the solver hop between IK solutions from one demo (or one step)
to the next. That injects *conditional trajectory variance*: two demos with near-identical task
geometry end up with different joint paths. A unimodal-Gaussian CNMP cannot represent that spread,
so it shows up as irreducible val MSE (this is exactly why the right-arm pour was harder to learn).

THE FIX
-------
When generating demonstrations for the new oracles (carry_pot, transfer_cube, transfer_gate),
select the valid IK solution **closest in joint space to a reference config** -- normally the
*previous control step's* joints. Called every step with ``seed_q = self.cur_q[arm]``, this keeps
each arm on a single, smooth, REPEATABLE branch across steps and across demos, which lowers the
conditional trajectory variance the CNMP has to fit.

This is a drop-in replacement for the max-clearance ranking; validity (accuracy + joint limits +
an optional collision/user check) is still enforced -- continuity only decides *among* valid
candidates. Duck-typed so it needs no rcs import: ``target`` and ``forward(q)`` return anything
with ``.translation()`` (xyz) and ``.rotation_q()`` (xyzw); ``ik.inverse(target, seed, tcp_offset)``
returns a joint vector or None.

    # inside an oracle, per control step:
    q = solve_ik_continuous(self.ik[a], target_base, seed_q=self.cur_q[a],
                            tcp_offset=self.tcp_offset, jlo=self._jlo, jhi=self._jhi,
                            forward=lambda q: self.ik[a].forward(q) * self.tcp_offset,
                            is_valid=lambda q: self._table_metrics(a, q)[0] is False, rng=self._rng)
    if q is not None:
        self.cur_q[a] = q
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def _quat_angle(qa: np.ndarray, qb: np.ndarray) -> float:
    """Geodesic angle (rad) between two xyzw quaternions."""
    return 2.0 * float(np.arccos(np.clip(abs(float(np.dot(qa, qb))), -1.0, 1.0)))


def solve_ik_continuous(
    ik,
    target,
    seed_q: np.ndarray,
    *,
    tcp_offset,
    jlo: np.ndarray,
    jhi: np.ndarray,
    forward: Callable[[np.ndarray], object],
    is_valid: Optional[Callable[[np.ndarray], bool]] = None,
    n_restarts: int = 24,
    rng: Optional[np.random.Generator] = None,
    pos_tol: float = 0.02,
    ang_tol: float = 0.15,
) -> Optional[np.ndarray]:
    """Return the valid IK solution for ``target`` closest in joint space to ``seed_q``.

    ``target`` is the desired pose in the arm base frame. Candidates come from the natural seed
    (``seed_q``) plus up to ``n_restarts`` random restarts within [jlo, jhi]. A candidate is valid
    iff it is within joint limits, its FK reaches ``target`` within ``pos_tol`` / ``ang_tol``, and
    ``is_valid(q)`` (if given) is True. Among valid candidates the one minimizing ``||q - seed_q||``
    is returned -- enforcing branch continuity. Returns None if no candidate is valid.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    jlo, jhi, seed_q = np.asarray(jlo, float), np.asarray(jhi, float), np.asarray(seed_q, float)
    tgt_p = np.asarray(target.translation(), float)
    tgt_q = np.asarray(target.rotation_q(), float)

    def valid(q: np.ndarray) -> bool:
        if np.any(q < jlo) or np.any(q > jhi):  # Pinocchio ignores joint limits while iterating
            return False
        fk = forward(q)
        if np.linalg.norm(np.asarray(fk.translation(), float) - tgt_p) >= pos_tol:
            return False
        if _quat_angle(np.asarray(fk.rotation_q(), float), tgt_q) >= ang_tol:
            return False
        return is_valid is None or bool(is_valid(q))

    best, best_dist = None, np.inf
    seeds = [seed_q] + [jlo + (jhi - jlo) * rng.random(seed_q.shape[0]) for _ in range(n_restarts)]
    for s in seeds:
        sol = ik.inverse(target, s, tcp_offset=tcp_offset)
        if sol is None:
            continue
        sol = np.asarray(sol, float)
        if not valid(sol):
            continue
        dist = float(np.linalg.norm(sol - seed_q))  # continuity: closeness to the reference config
        if dist < best_dist:
            best, best_dist = sol, dist
    return best


# ------------------------------------------------------------------ self-test (no rcs needed)
def _self_test() -> None:
    """Verify the selector keeps the branch closest to the seed among valid candidates."""

    class _Pose:  # duck-typed stand-in for an rcs Pose
        def __init__(self, p, q):
            self._p, self._q = np.asarray(p, float), np.asarray(q, float)

        def translation(self):
            return self._p

        def rotation_q(self):
            return self._q

    target = _Pose([0.5, 0.0, 0.3], [0, 0, 0, 1])
    q_near = np.array([0.10, 0.0, 0.0, -1.0, 0.0, 1.5, 0.0])   # close to the seed, reaches target
    q_far = np.array([2.00, 0.0, 0.0, -1.0, 0.0, 1.5, 0.0])    # another branch, also reaches target
    q_bad = np.array([0.05, 0.0, 0.0, -1.0, 0.0, 1.5, 9.9])    # out of joint limits
    seed_q = np.array([0.0, 0.0, 0.0, -1.0, 0.0, 1.5, 0.0])
    jlo, jhi = np.full(7, -3.0), np.full(7, 3.0)

    class _IK:  # returns the seed itself if it is one of the known solutions, else cycles branches
        def __init__(self):
            self._pool = [q_near, q_far, q_bad]
            self._i = 0

        def inverse(self, target, seed, tcp_offset=None):
            for cand in (q_near, q_far, q_bad):
                if np.allclose(seed, cand):
                    return cand
            out = self._pool[self._i % len(self._pool)]
            self._i += 1
            return out

        def forward(self, q):  # all three "solutions" reach the target pose; q_bad is limit-rejected
            return _Pose([0.5, 0.0, 0.3], [0, 0, 0, 1])

    got = solve_ik_continuous(_IK(), target, seed_q, tcp_offset=None, jlo=jlo, jhi=jhi,
                              forward=lambda q: _IK().forward(q), n_restarts=8,
                              rng=np.random.default_rng(0))
    assert got is not None and np.allclose(got, q_near), got  # near branch chosen, bad one rejected
    print("branch_consistent_ik self-test OK: picked the seed-closest valid branch, rejected the "
          "out-of-limits one.")


if __name__ == "__main__":
    _self_test()
