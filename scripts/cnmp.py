"""Conditional Neural Movement Primitives (CNMP).

A batched implementation of the CNMP model of Seker et al. (RSS 2019), which is a
Conditional Neural Process (Garnelo et al., ICML 2018) applied to sensorimotor
trajectories.

Given a set of observations (context) ``{(t_i, SM(t_i))}`` sampled from a single
demonstration, plus optional task parameters ``gamma``, the model predicts a
Gaussian distribution ``N(mu_q, sigma_q)`` over ``SM(t_q)`` for arbitrary query
times ``t_q``.

Shape conventions
-----------------
``B`` batch size (one demonstration per batch element), ``N`` padded number of
observations, ``M`` padded number of targets, ``dx`` input dim (usually 1: time),
``dg`` task-parameter dim, ``dy`` sensorimotor dim.

    obs_x   (B, N, dx)      obs_y   (B, N, dy)      obs_mask   (B, N)   bool
    tar_x   (B, M, dx)      tar_y   (B, M, dy)      tar_mask   (B, M)   bool
    gamma   (B, dg) or (B, 1, dg) or (B, N/M, dg)

Every batch element may use a different number of observations and targets; the
padded slots are ignored by both the aggregation and the loss.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CNMP"]

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "elu": nn.ELU,
}


def _mlp(
    in_dim: int,
    hidden_dims: Sequence[int],
    out_dim: int,
    activation: str = "relu",
    layer_norm: bool = False,
) -> nn.Sequential:
    """MLP with `len(hidden_dims)` hidden layers and a linear output layer."""
    if len(hidden_dims) < 1:
        raise ValueError("hidden_dims must contain at least one layer")
    act = _ACTIVATIONS[activation]
    dims = [in_dim, *hidden_dims]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if layer_norm:
            layers.append(nn.LayerNorm(dims[i + 1]))
        layers.append(act())
    layers.append(nn.Linear(dims[-1], out_dim))
    return nn.Sequential(*layers)


class CNMP(nn.Module):
    """Conditional Neural Movement Primitive.

    Parameters
    ----------
    input_dim:
        Dimensionality of the query input ``x`` (1 for plain time).
    output_dim:
        Dimensionality of the sensorimotor vector ``SM``.
    param_dim:
        Dimensionality of the task parameters ``gamma``. ``gamma`` is
        concatenated to *both* the encoder and the decoder input, as in Sec. III-B
        of the paper. Use 0 to disable.
    latent_dim:
        Size of the per-observation representation ``r_i`` (and hence of the
        aggregated ``r``). Decoupled from the encoder width.
    encoder_hidden_dims / decoder_hidden_dims:
        Hidden layer widths of the two MLPs.
    min_std:
        Lower bound on the predicted standard deviation. Keeps the Gaussian NLL
        from diverging to -inf on (near) noiseless demonstrations. Lowering this
        below ~1e-3 tends to destabilise training.
    value_masking:
        If True, the model accepts a per-observation mask over the *output*
        dimensions, so a context point may specify only a subset of ``SM``
        (e.g. force but not joint position, cf. Sec. III-B). The encoder then
        receives the masked values and the mask itself as extra channels, so it
        can tell "value is 0" from "value is unknown". Off by default: turning it
        on changes the encoder input size, so a model must be *trained* with the
        flag set for it to be usable.
    layer_norm:
        Insert LayerNorm after every hidden layer. Helps with deep encoders.
    """

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        param_dim: int = 0,
        latent_dim: int = 128,
        encoder_hidden_dims: Sequence[int] = (128, 128),
        decoder_hidden_dims: Sequence[int] = (128, 128),
        min_std: float = 0.01,
        value_masking: bool = False,
        activation: str = "relu",
        layer_norm: bool = False,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(_ACTIVATIONS)}")
        if min_std <= 0:
            raise ValueError("min_std must be positive")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.param_dim = int(param_dim)
        self.latent_dim = int(latent_dim)
        self.encoder_hidden_dims = tuple(encoder_hidden_dims)
        self.decoder_hidden_dims = tuple(decoder_hidden_dims)
        self.min_std = float(min_std)
        self.value_masking = bool(value_masking)
        self.activation = activation
        self.layer_norm = bool(layer_norm)

        # (x, gamma, y) -> r_i ; with value masking also the 0/1 mask over y.
        enc_in = self.input_dim + self.param_dim + self.output_dim * (2 if self.value_masking else 1)
        self.encoder = _mlp(enc_in, self.encoder_hidden_dims, self.latent_dim, activation, layer_norm)

        # (r, x_q, gamma) -> (mu, raw_sigma)
        dec_in = self.latent_dim + self.input_dim + self.param_dim
        self.decoder = _mlp(dec_in, self.decoder_hidden_dims, 2 * self.output_dim, activation, layer_norm)

        self.device = torch.device(device)
        self.to(self.device)

    # ------------------------------------------------------------------ utils

    @property
    def config(self) -> Dict:
        """Constructor arguments, so a checkpoint can rebuild the model."""
        return dict(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            param_dim=self.param_dim,
            latent_dim=self.latent_dim,
            encoder_hidden_dims=self.encoder_hidden_dims,
            decoder_hidden_dims=self.decoder_hidden_dims,
            min_std=self.min_std,
            value_masking=self.value_masking,
            activation=self.activation,
            layer_norm=self.layer_norm,
        )

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _expand_gamma(self, gamma: Optional[torch.Tensor], batch: int, n_points: int) -> torch.Tensor:
        if self.param_dim == 0:
            raise RuntimeError("model was built with param_dim=0 but gamma was given")
        if gamma is None:
            raise ValueError(f"model expects gamma with {self.param_dim} dims, got None")
        if gamma.dim() == 2:  # (B, dg) -> one parameter vector per demonstration
            gamma = gamma.unsqueeze(1)
        if gamma.shape[-1] != self.param_dim:
            raise ValueError(f"gamma last dim {gamma.shape[-1]} != param_dim {self.param_dim}")
        return gamma.expand(batch, n_points, self.param_dim)

    # --------------------------------------------------------------- encoding

    def encode(
        self,
        obs_x: torch.Tensor,
        obs_y: torch.Tensor,
        obs_mask: Optional[torch.Tensor] = None,
        gamma: Optional[torch.Tensor] = None,
        obs_value_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode observations into the general representation ``r`` (B, latent_dim).

        Padded observations (``obs_mask == False``) are excluded from the mean, so
        batch elements with different numbers of observations are handled exactly
        as if they had been encoded on their own.
        """
        batch, n_points, _ = obs_x.shape
        parts = [obs_x]
        if self.param_dim:
            parts.append(self._expand_gamma(gamma, batch, n_points))
        if self.value_masking:
            if obs_value_mask is None:
                value_mask = torch.ones_like(obs_y)
            else:
                value_mask = obs_value_mask.to(obs_y.dtype)
            parts.extend([obs_y * value_mask, value_mask])
        else:
            if obs_value_mask is not None:
                raise RuntimeError("obs_value_mask given but model was built with value_masking=False")
            parts.append(obs_y)

        repr_i = self.encoder(torch.cat(parts, dim=-1))  # (B, N, latent_dim)

        if obs_mask is None:
            return repr_i.mean(dim=1)
        weights = obs_mask.unsqueeze(-1).to(repr_i.dtype)  # (B, N, 1)
        counts = weights.sum(dim=1)  # (B, 1)
        # clamp keeps the gradient finite if a row happens to have no observation
        return (repr_i * weights).sum(dim=1) / counts.clamp(min=1.0)

    # --------------------------------------------------------------- decoding

    def decode(
        self,
        repr_r: torch.Tensor,
        tar_x: torch.Tensor,
        gamma: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query the decoder at ``tar_x``; returns ``(mean, std)``, each (B, M, dy)."""
        batch, n_points, _ = tar_x.shape
        parts = [repr_r.unsqueeze(1).expand(batch, n_points, self.latent_dim), tar_x]
        if self.param_dim:
            parts.append(self._expand_gamma(gamma, batch, n_points))
        out = self.decoder(torch.cat(parts, dim=-1))
        mean, raw_std = out.split(self.output_dim, dim=-1)
        std = F.softplus(raw_std) + self.min_std
        return mean, std

    def forward(
        self,
        obs_x: torch.Tensor,
        obs_y: torch.Tensor,
        tar_x: torch.Tensor,
        obs_mask: Optional[torch.Tensor] = None,
        gamma: Optional[torch.Tensor] = None,
        obs_value_mask: Optional[torch.Tensor] = None,
        return_repr: bool = False,
        tar_gamma: Optional[torch.Tensor] = None,
    ):
        """Predict ``p(SM(tar_x) | observations)`` as a diagonal Gaussian.

        ``gamma`` conditions the encoder (context points). By default the same
        ``gamma`` also conditions the decoder (query points), which is the static
        per-demonstration task-parameter case of the paper. For a **time-varying**
        parameter -- one whose value differs between context and query times, e.g.
        the partner arm's momentary joints -- pass the query-time values as
        ``tar_gamma`` (B, M, dg); then ``gamma`` should carry the context-time
        values (B, N, dg). ``_expand_gamma`` accepts (B, K, dg) per-point tensors
        as well as the (B, dg) static form.

        Returns ``(mean, std)`` of shape (B, M, output_dim), or
        ``(mean, std, r)`` if ``return_repr``.
        """
        repr_r = self.encode(obs_x, obs_y, obs_mask, gamma, obs_value_mask)
        mean, std = self.decode(repr_r, tar_x, gamma if tar_gamma is None else tar_gamma)
        if return_repr:
            return mean, std, repr_r
        return mean, std

    @torch.no_grad()
    def predict(self, *args, **kwargs):
        """`forward` in eval mode without gradients; restores the previous mode."""
        was_training = self.training
        self.eval()
        try:
            return self.forward(*args, **kwargs)
        finally:
            self.train(was_training)

    # ------------------------------------------------------------------ losses

    @staticmethod
    def _weights(
        like: torch.Tensor,
        tar_mask: Optional[torch.Tensor],
        tar_value_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        weights = torch.ones_like(like)
        if tar_mask is not None:
            weights = weights * tar_mask.unsqueeze(-1).to(like.dtype)
        if tar_value_mask is not None:
            weights = weights * tar_value_mask.to(like.dtype)
        return weights

    def loss(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        tar_y: torch.Tensor,
        tar_mask: Optional[torch.Tensor] = None,
        tar_value_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Masked Gaussian negative log-likelihood (Eq. 1 of the paper).

        Averaged over valid (target, output-dim) entries within each batch element
        and then over the batch, so that batch elements with many targets do not
        dominate the gradient.
        """
        dist = torch.distributions.Normal(mean, std, validate_args=False)
        log_prob = dist.log_prob(tar_y)
        weights = self._weights(log_prob, tar_mask, tar_value_mask)
        # `where` rather than a product: keeps padded slots harmless even if they
        # contain garbage (0 * inf = nan would poison the whole batch).
        log_prob = torch.where(weights > 0, log_prob, torch.zeros_like(log_prob))
        total = (log_prob * weights).sum(dim=(1, 2))
        count = weights.sum(dim=(1, 2)).clamp(min=1.0)
        return -(total / count).mean()

    def mse(
        self,
        mean: torch.Tensor,
        tar_y: torch.Tensor,
        tar_mask: Optional[torch.Tensor] = None,
        tar_value_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Masked mean squared error of the predicted mean (reporting metric)."""
        sq_err = (mean - tar_y) ** 2
        weights = self._weights(sq_err, tar_mask, tar_value_mask)
        return (sq_err * weights).sum() / weights.sum().clamp(min=1.0)

    # -------------------------------------------------------- (de)serialisation

    def save(self, path: str) -> None:
        torch.save({"config": self.config, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, device: torch.device | str = "cpu") -> "CNMP":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(**ckpt["config"], device=device)
        model.load_state_dict(ckpt["state_dict"])
        return model
