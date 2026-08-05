"""Sanity checks for the CNMP implementation. Run with: python test_cnmp.py"""

import torch

from cnmp import CNMP
from cnmp_data import TrajectorySampler, build_query, generate_avoidance_trajectories

torch.manual_seed(0)


def test_shapes_and_masking_invariance():
    """Padded observations must not influence the representation."""
    model = CNMP(input_dim=1, output_dim=2, param_dim=1, latent_dim=32,
                 encoder_hidden_dims=(32, 32), decoder_hidden_dims=(32, 32))
    b, n, m = 4, 7, 5
    obs_x = torch.rand(b, n, 1)
    obs_y = torch.rand(b, n, 2)
    tar_x = torch.rand(b, m, 1)
    gamma = torch.rand(b, 1)
    obs_mask = torch.zeros(b, n, dtype=torch.bool)
    counts = [1, 3, 7, 4]
    for i, c in enumerate(counts):
        obs_mask[i, :c] = True

    mean, std = model(obs_x, obs_y, tar_x, obs_mask, gamma)
    assert mean.shape == (b, m, 2) and std.shape == (b, m, 2)
    assert (std > 0).all()

    r_batched = model.encode(obs_x, obs_y, obs_mask, gamma)
    for i, c in enumerate(counts):
        r_single = model.encode(obs_x[i : i + 1, :c], obs_y[i : i + 1, :c], None, gamma[i : i + 1])
        assert torch.allclose(r_batched[i], r_single[0], atol=1e-6), i
    # garbage in the padded slots changes nothing
    obs_y_dirty = obs_y.clone()
    for i, c in enumerate(counts):
        obs_y_dirty[i, c:] = 1e3
    assert torch.allclose(r_batched, model.encode(obs_x, obs_y_dirty, obs_mask, gamma), atol=1e-4)
    print("ok: shapes + masking invariance")


def test_target_count_flexibility():
    """The number of queries at test time is independent of anything seen in training."""
    model = CNMP(input_dim=1, output_dim=3, latent_dim=16,
                 encoder_hidden_dims=(16,), decoder_hidden_dims=(16,))
    r = model.encode(torch.rand(2, 5, 1), torch.rand(2, 5, 3))
    for m in (1, 13, 977):
        mean, std = model.decode(r, torch.rand(2, m, 1))
        assert mean.shape == (2, m, 3)
    print("ok: arbitrary target counts")


def test_loss_ignores_padding():
    model = CNMP(input_dim=1, output_dim=2, latent_dim=16,
                 encoder_hidden_dims=(16,), decoder_hidden_dims=(16,))
    mean = torch.zeros(2, 4, 2)
    std = torch.ones(2, 4, 2)
    tar_y = torch.zeros(2, 4, 2)
    tar_mask = torch.tensor([[True, True, False, False], [True, False, False, False]])
    base = model.loss(mean, std, tar_y, tar_mask)
    tar_y_dirty = tar_y.clone()
    tar_y_dirty[:, 2:] = 50.0
    assert torch.allclose(base, model.loss(mean, std, tar_y_dirty, tar_mask))
    assert torch.isfinite(base)
    # all-padded row must not produce NaN
    empty = torch.zeros(2, 4, dtype=torch.bool)
    assert torch.isfinite(model.loss(mean, std, tar_y, empty))
    print("ok: loss ignores padding")


def test_value_masking():
    model = CNMP(input_dim=1, output_dim=3, latent_dim=16, value_masking=True,
                 encoder_hidden_dims=(16,), decoder_hidden_dims=(16,))
    obs_x, obs_y = torch.rand(2, 4, 1), torch.rand(2, 4, 3)
    vm = torch.tensor([[1, 0, 0], [1, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=torch.bool)
    vm = vm.unsqueeze(0).expand(2, -1, -1)
    r = model.encode(obs_x, obs_y, None, None, vm)
    # hidden output dims must not leak into the representation
    obs_y_dirty = obs_y.clone()
    obs_y_dirty[~vm] = 99.0
    assert torch.allclose(r, model.encode(obs_x, obs_y_dirty, None, None, vm), atol=1e-5)
    print("ok: value masking hides unobserved dims")


def test_sampler_and_training_step():
    y, gamma, _ = generate_avoidance_trajectories(32, t_steps=64, seed=1)
    sampler = TrajectorySampler(y, gamma=gamma, n_max=6, m_max=8, seed=1)
    batch = sampler.random_batch(5)
    assert batch.obs_x.shape == (5, 6, 1)
    assert batch.tar_y.shape == (5, 8, 2)
    assert batch.obs_mask.sum(1).min() >= 1
    assert batch.gamma.shape == (5, 1)

    disjoint = TrajectorySampler(y, gamma=gamma, n_max=6, m_max=8, context_in_target=False, seed=1)
    d = disjoint.sample(torch.arange(4))
    for i in range(4):
        o = d.obs_x[i][d.obs_mask[i]].flatten()
        t = d.tar_x[i][d.tar_mask[i]].flatten()
        assert len(set(o.tolist()) & set(t.tolist())) == 0

    model = CNMP(input_dim=1, output_dim=2, param_dim=1, latent_dim=32,
                 encoder_hidden_dims=(32, 32), decoder_hidden_dims=(32, 32))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = last = None
    for step in range(400):
        b = sampler.random_batch(16)
        mean, std = model(**b.inputs())
        loss = model.loss(mean, std, *b.targets())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    assert last < first, (first, last)
    print(f"ok: training step ({first:.3f} -> {last:.3f})")

    ev = sampler.full_target_batch(torch.arange(3), context_idx=torch.tensor([0, 31, 63]))
    mean, std = model.predict(**ev.inputs())
    assert mean.shape == (3, 64, 2) and model.training is True
    print("ok: eval batch")


def test_value_mask_sampler_and_query_helper():
    y, gamma, _ = generate_avoidance_trajectories(16, t_steps=32, seed=2)
    s = TrajectorySampler(y, gamma=gamma, n_max=5, m_max=5, value_masking=True, seed=2)
    b = s.random_batch(4)
    assert b.obs_value_mask.shape == (4, 5, 2)
    # every valid observation reveals at least one dimension
    valid = b.obs_value_mask[b.obs_mask]
    assert valid.sum(-1).min() >= 1

    model = CNMP(input_dim=1, output_dim=2, param_dim=1, latent_dim=16, value_masking=True,
                 encoder_hidden_dims=(16,), decoder_hidden_dims=(16,))
    q = build_query(obs_x=[[0.5]], obs_y=[[0.0, 0.4]], tar_x=torch.linspace(0, 1, 40).unsqueeze(-1),
                    gamma=[0.5], obs_value_mask=[[0, 1]])
    mean, std = model.predict(**q)
    assert mean.shape == (1, 40, 2)
    print("ok: value-mask sampler + build_query")


def test_save_load(tmp="/tmp/cnmp_ckpt.pt"):
    model = CNMP(input_dim=1, output_dim=2, param_dim=1, latent_dim=16,
                 encoder_hidden_dims=(16,), decoder_hidden_dims=(16,))
    model.save(tmp)
    other = CNMP.load(tmp)
    x, y_, tx, g = torch.rand(1, 3, 1), torch.rand(1, 3, 2), torch.rand(1, 9, 1), torch.rand(1, 1)
    a, _ = model.predict(x, y_, tx, gamma=g)
    b, _ = other.predict(x, y_, tx, gamma=g)
    assert torch.allclose(a, b)
    print("ok: save/load round trip")


if __name__ == "__main__":
    test_shapes_and_masking_invariance()
    test_target_count_flexibility()
    test_loss_ignores_padding()
    test_value_masking()
    test_sampler_and_training_step()
    test_value_mask_sampler_and_query_helper()
    test_save_load()
    print("\nall tests passed")
