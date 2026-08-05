"""Batching utilities and a small synthetic dataset for CNMP.

The sampler builds *padded* batches: every batch element gets its own number of
observations ``n_i ~ U[1, n_max]`` and targets ``m_i ~ U[1, m_max]``, and the
unused slots are flagged in ``obs_mask`` / ``tar_mask``. Sampling is fully
vectorised over the batch (no Python loop over batch elements).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "Batch",
    "TrajectorySampler",
    "build_query",
    "generate_avoidance_trajectories",
]


# --------------------------------------------------------------------- batches


@dataclass
class Batch:
    """A padded, masked CNMP batch."""

    obs_x: torch.Tensor  # (B, N, dx)
    obs_y: torch.Tensor  # (B, N, dy)
    obs_mask: torch.Tensor  # (B, N) bool
    tar_x: torch.Tensor  # (B, M, dx)
    tar_y: torch.Tensor  # (B, M, dy)
    tar_mask: torch.Tensor  # (B, M) bool
    gamma: Optional[torch.Tensor] = None  # (B, dg) static, or (B, N, dg) context values
    obs_value_mask: Optional[torch.Tensor] = None  # (B, N, dy) bool
    tar_value_mask: Optional[torch.Tensor] = None  # (B, M, dy) bool
    tar_gamma: Optional[torch.Tensor] = None  # (B, M, dg) query values for time-varying gamma

    def inputs(self) -> Dict[str, torch.Tensor]:
        """Keyword arguments for ``CNMP.forward``."""
        return dict(
            obs_x=self.obs_x,
            obs_y=self.obs_y,
            tar_x=self.tar_x,
            obs_mask=self.obs_mask,
            gamma=self.gamma,
            obs_value_mask=self.obs_value_mask,
            tar_gamma=self.tar_gamma,
        )

    def targets(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Positional arguments for ``CNMP.loss`` after ``(mean, std)``."""
        return self.tar_y, self.tar_mask, self.tar_value_mask


# --------------------------------------------------------------------- sampler


class TrajectorySampler:
    """Samples masked context/target batches from a set of demonstrations.

    Parameters
    ----------
    y:
        Demonstrations, ``(num_traj, t_steps, dy)``.
    x:
        Query inputs, ``(t_steps, dx)`` (shared by all demonstrations) or
        ``(num_traj, t_steps, dx)``. Defaults to a uniform grid on [0, 1], i.e.
        the time-invariance scaling used in the paper.
    gamma:
        Task parameters, ``(num_traj, dg)``, or None.
    n_max / m_max:
        Padded (and maximum) number of observations / targets per element.
    context_in_target:
        If True, context and target index sets are drawn independently, so a
        target may coincide with a context point (the original CNP formulation
        draws targets as a superset of the contexts). If False (default) they are
        forced to be disjoint, which requires ``n_max + m_max <= t_steps``.
        Measured over three seeds the two settings are indistinguishable here;
        see README.md.
    value_masking:
        If True the sampler also produces ``obs_value_mask``: each context point
        reveals a random non-empty subset of the ``dy`` output dimensions.
    value_mask_prob:
        Probability that a given output dimension is revealed. At least one
        dimension per context point is always revealed.
    """

    def __init__(
        self,
        y: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        gamma: Optional[torch.Tensor] = None,
        n_max: int = 10,
        m_max: int = 10,
        context_in_target: bool = False,
        value_masking: bool = False,
        value_mask_prob: float = 0.5,
        device: torch.device | str = "cpu",
        seed: Optional[int] = None,
    ):
        self.device = torch.device(device)
        self.y = y.to(self.device)
        self.num_traj, self.t_steps, self.output_dim = self.y.shape

        if x is None:
            x = torch.linspace(0.0, 1.0, self.t_steps).unsqueeze(-1)
        x = x.to(self.device)
        if x.dim() == 2:  # shared grid -> broadcast without copying
            x = x.unsqueeze(0).expand(self.num_traj, -1, -1)
        if x.shape[:2] != (self.num_traj, self.t_steps):
            raise ValueError("x must be (t_steps, dx) or (num_traj, t_steps, dx)")
        self.x = x
        self.input_dim = x.shape[-1]

        self.time_varying_gamma = False
        if gamma is not None:
            gamma = gamma.to(self.device)
            if gamma.dim() == 1:
                gamma = gamma.unsqueeze(-1)  # (num_traj,) -> (num_traj, 1) static scalar param
            if gamma.shape[0] != self.num_traj:
                raise ValueError("gamma must have one row per trajectory")
            if gamma.dim() == 3:  # (num_traj, t_steps, dg): a per-timestep (time-varying) parameter
                if gamma.shape[1] != self.t_steps:
                    raise ValueError("time-varying gamma must be (num_traj, t_steps, dg)")
                self.time_varying_gamma = True
            elif gamma.dim() != 2:
                raise ValueError("gamma must be (num_traj,), (num_traj, dg) or (num_traj, t_steps, dg)")
        self.gamma = gamma
        self.param_dim = 0 if gamma is None else gamma.shape[-1]

        if not 1 <= n_max <= self.t_steps or not 1 <= m_max <= self.t_steps:
            raise ValueError("n_max and m_max must be in [1, t_steps]")
        if not context_in_target and n_max + m_max > self.t_steps:
            raise ValueError("disjoint sampling needs n_max + m_max <= t_steps")

        self.n_max = n_max
        self.m_max = m_max
        self.context_in_target = context_in_target
        self.value_masking = value_masking
        self.value_mask_prob = value_mask_prob

        self.generator: Optional[torch.Generator] = None
        if seed is not None:
            self.generator = torch.Generator(device=self.device).manual_seed(seed)

    # -- internals ---------------------------------------------------------

    def _rand(self, *shape: int) -> torch.Tensor:
        return torch.rand(*shape, device=self.device, generator=self.generator)

    def _randint(self, high: int, shape: Tuple[int, ...]) -> torch.Tensor:
        return torch.randint(high, shape, device=self.device, generator=self.generator)

    def _gather(self, source: torch.Tensor, traj_ids: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """source (num_traj, T, d), idx (B, K) -> (B, K, d)."""
        rows = source[traj_ids]  # (B, T, d)
        return torch.gather(rows, 1, idx.unsqueeze(-1).expand(-1, -1, rows.shape[-1]))

    def _value_mask(self, batch: int, n_points: int) -> torch.Tensor:
        keep = self._rand(batch, n_points, self.output_dim) < self.value_mask_prob
        forced = F.one_hot(self._randint(self.output_dim, (batch, n_points)), self.output_dim).bool()
        return keep | forced

    # -- public API --------------------------------------------------------

    def sample(self, traj_ids: torch.Tensor) -> Batch:
        """Draw one training batch; one demonstration per element of ``traj_ids``."""
        traj_ids = torch.as_tensor(traj_ids, device=self.device, dtype=torch.long)
        batch = traj_ids.numel()

        # one random permutation of the time steps per batch element
        perm = torch.argsort(self._rand(batch, self.t_steps), dim=1)
        obs_idx = perm[:, : self.n_max]
        if self.context_in_target:
            tar_idx = torch.argsort(self._rand(batch, self.t_steps), dim=1)[:, : self.m_max]
        else:
            tar_idx = perm[:, self.n_max : self.n_max + self.m_max]

        n_obs = self._randint(self.n_max, (batch, 1)) + 1  # U[1, n_max]
        n_tar = self._randint(self.m_max, (batch, 1)) + 1
        obs_mask = torch.arange(self.n_max, device=self.device).unsqueeze(0) < n_obs
        tar_mask = torch.arange(self.m_max, device=self.device).unsqueeze(0) < n_tar

        obs_x = self._gather(self.x, traj_ids, obs_idx) * obs_mask.unsqueeze(-1)
        obs_y = self._gather(self.y, traj_ids, obs_idx) * obs_mask.unsqueeze(-1)
        tar_x = self._gather(self.x, traj_ids, tar_idx) * tar_mask.unsqueeze(-1)
        tar_y = self._gather(self.y, traj_ids, tar_idx) * tar_mask.unsqueeze(-1)

        value_mask = None
        if self.value_masking:
            value_mask = self._value_mask(batch, self.n_max) & obs_mask.unsqueeze(-1)

        if self.gamma is None:
            obs_gamma, tar_gamma = None, None
        elif self.time_varying_gamma:  # gather the parameter at the context and query times
            obs_gamma = self._gather(self.gamma, traj_ids, obs_idx) * obs_mask.unsqueeze(-1)
            tar_gamma = self._gather(self.gamma, traj_ids, tar_idx) * tar_mask.unsqueeze(-1)
        else:
            obs_gamma, tar_gamma = self.gamma[traj_ids], None

        return Batch(
            obs_x=obs_x,
            obs_y=obs_y,
            obs_mask=obs_mask,
            tar_x=tar_x,
            tar_y=tar_y,
            tar_mask=tar_mask,
            gamma=obs_gamma,
            obs_value_mask=value_mask,
            tar_gamma=tar_gamma,
        )

    def random_batch(self, batch_size: int) -> Batch:
        return self.sample(self._randint(self.num_traj, (batch_size,)))

    def epoch_batches(self, batch_size: int, drop_last: bool = True):
        """Iterate over a shuffled epoch of the demonstrations."""
        order = torch.randperm(self.num_traj, device=self.device, generator=self.generator)
        for start in range(0, self.num_traj, batch_size):
            ids = order[start : start + batch_size]
            if drop_last and ids.numel() < batch_size:
                break
            yield self.sample(ids)

    def full_target_batch(
        self,
        traj_ids: torch.Tensor,
        context_idx: torch.Tensor,
        value_mask: Optional[torch.Tensor] = None,
    ) -> Batch:
        """Evaluation batch: explicit context indices, targets = the whole trajectory.

        ``context_idx`` is ``(K,)`` (same context steps for every element) or
        ``(B, K)``. No padding is needed because every element uses ``K`` points.
        """
        traj_ids = torch.as_tensor(traj_ids, device=self.device, dtype=torch.long)
        batch = traj_ids.numel()
        context_idx = torch.as_tensor(context_idx, device=self.device, dtype=torch.long)
        if context_idx.dim() == 1:
            context_idx = context_idx.unsqueeze(0).expand(batch, -1)

        obs_mask = torch.ones(context_idx.shape, dtype=torch.bool, device=self.device)
        tar_x = self.x[traj_ids]
        tar_y = self.y[traj_ids]
        tar_mask = torch.ones(tar_x.shape[:2], dtype=torch.bool, device=self.device)

        if self.gamma is None:
            obs_gamma, tar_gamma = None, None
        elif self.time_varying_gamma:
            obs_gamma = self._gather(self.gamma, traj_ids, context_idx)
            tar_gamma = self.gamma[traj_ids]  # targets are the whole trajectory
        else:
            obs_gamma, tar_gamma = self.gamma[traj_ids], None

        return Batch(
            obs_x=self._gather(self.x, traj_ids, context_idx),
            obs_y=self._gather(self.y, traj_ids, context_idx),
            obs_mask=obs_mask,
            tar_x=tar_x,
            tar_y=tar_y,
            tar_mask=tar_mask,
            gamma=obs_gamma,
            obs_value_mask=None if value_mask is None else value_mask.to(self.device),
            tar_gamma=tar_gamma,
        )


def build_query(
    obs_x: Sequence | torch.Tensor,
    obs_y: Sequence | torch.Tensor,
    tar_x: Sequence | torch.Tensor,
    gamma: Optional[Sequence | torch.Tensor] = None,
    obs_value_mask: Optional[Sequence | torch.Tensor] = None,
    device: torch.device | str = "cpu",
    tar_gamma: Optional[Sequence | torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Assemble model inputs from hand-specified conditioning points.

    Accepts unbatched ``(K, dx)`` / ``(M, dx)`` / ``(dg,)`` inputs and adds the
    batch dimension, or already-batched tensors which are passed through.
    Returns a dict ready for ``model(**query)``.
    """
    device = torch.device(device)

    def prep(v, ndim):
        if v is None:
            return None
        t = torch.as_tensor(v, dtype=torch.float32, device=device)
        while t.dim() < ndim:
            t = t.unsqueeze(0)
        return t

    obs_x_t = prep(obs_x, 3)
    obs_y_t = prep(obs_y, 3)
    tar_x_t = prep(tar_x, 3)
    # gamma is prepped to 2-D for the static (dg,) case; an already-batched per-point
    # (1, K, dg) tensor passes through unchanged. tar_gamma (query-time values) is per-point.
    query = dict(
        obs_x=obs_x_t,
        obs_y=obs_y_t,
        tar_x=tar_x_t,
        obs_mask=torch.ones(obs_x_t.shape[:2], dtype=torch.bool, device=device),
        gamma=prep(gamma, 2),
        obs_value_mask=None if obs_value_mask is None else prep(obs_value_mask, 3).bool(),
        tar_gamma=prep(tar_gamma, 3),
    )
    return query


# --------------------------------------------------------------------- dataset


def generate_avoidance_trajectories(
    num_traj: int,
    t_steps: int = 200,
    noise: float = 0.0,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Toy 2-D obstacle-avoidance demonstrations, in the spirit of Fig. 4 of the paper.

    Each demonstration starts at the left, passes the (invisible) obstacle either
    above or below it at t = 0.5, and ends on the right at the same height it
    started. The obstacle height is exposed as a task parameter; which side the
    trajectory passes is **not** exposed, so the skill is bimodal given gamma.

    Returns
    -------
    y : (num_traj, t_steps, 2)   sensorimotor trajectories, dim 0 = x, dim 1 = y
    gamma : (num_traj, 1)        obstacle height in [0, 1]
    mode : (num_traj,)           +1 above / -1 below (ground truth, for plots only)
    """
    gen = torch.Generator().manual_seed(seed) if seed is not None else None

    def uniform(low, high, *shape):
        return torch.rand(*shape, generator=gen) * (high - low) + low

    t = torch.linspace(0.0, 1.0, t_steps).view(1, t_steps)

    height = uniform(0.0, 1.0, num_traj, 1)  # gamma
    mode = torch.where(torch.rand(num_traj, 1, generator=gen) < 0.5, -1.0, 1.0)
    offset = uniform(-0.1, 0.1, num_traj, 1)  # start/end height
    speed = uniform(-0.06, 0.06, num_traj, 1)  # velocity profile of the x axis

    traj_x = t + speed * torch.sin(torch.pi * t)
    # clamp_min: sin(pi) is a tiny *negative* float, and (-eps) ** 1.4 is NaN
    bump = torch.sin(torch.pi * t).clamp_min(0.0) ** 1.4
    traj_y = offset + mode * (0.25 + 0.55 * height) * bump

    y = torch.stack([traj_x, traj_y], dim=-1)
    if noise > 0:
        y = y + noise * torch.randn(y.shape, generator=gen)
    return y, height, mode.squeeze(-1)
