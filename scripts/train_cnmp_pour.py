#!/usr/bin/env python3
"""Train cross-conditioned CNMPs on the pour_marbles demonstrations.

One CNMP per arm: it predicts that arm's realized joint trajectory from time, conditioned on the
partner arm's *momentary* realized joints, which are fed as the model's time-varying ``gamma``.

    input_dim = 1        (normalized time)
    output_dim = 7       (self realized joints)
    param_dim = 7        (partner realized joints, time-varying gamma)

The training loop, objective (Gaussian NLL) and reporting (val NLL + start/middle/end MSE) mirror
scripts/cnmp_demo.ipynb. Data, the 80/10/10 split and normalization come from cnmp_data.py.

Usage:
    python scripts/train_cnmp_pour.py --arm both --steps 100000
    python scripts/train_cnmp_pour.py --arm left --steps 50000 --device cuda:0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make cnmp*.py importable from any cwd

from cnmp import CNMP
from cnmp_data import DEFAULT_T, N_JOINTS, make_cross_conditioned_dataset
from cnmp_test_data import TrajectorySampler

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=["left", "right", "both"], default="both")
    p.add_argument("--source", choices=["left", "right"], default=None,
                   help="fixed-role PoC: restrict to this source cup ('left' = left pours, right holds)")
    p.add_argument("--cup-gamma", action="store_true",
                   help="append the two cups' initial (x,y) as static gamma (param_dim 7 -> 11)")
    p.add_argument("--steps", type=int, default=250_000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--n-max", type=int, default=32, help="max context points per element")
    p.add_argument("--m-max", type=int, default=32, help="max target points per element")
    p.add_argument("--T", type=int, default=DEFAULT_T, help="downsampled trajectory length")
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-std", type=float, default=0.01, help="floor on the predicted sigma")
    p.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay (0 = none)")
    p.add_argument("--select-on", choices=["val_mse", "val_nll"], default="val_mse",
                   help="metric selecting the best checkpoint (lower is better)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=str(ROOT / "output" / "cnmp_pour"))
    return p.parse_args()


def train_one_arm(arm: str, args: argparse.Namespace) -> Path:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    ds = make_cross_conditioned_dataset(arm, T=args.T, split_seed=args.seed, device=device,
                                        only_source=args.source, cup_gamma=args.cup_gamma)
    n_val = ds.val_ids.numel()
    role = "" if args.source is None else f" | source={args.source} (self {'pours' if arm == args.source else 'holds'})"
    print(f"\n=== arm={arm} (partner={ds.partner}){role} | device={device} | "
          f"train/val/test = {ds.train_ids.numel()}/{n_val}/{ds.test_ids.numel()} ===")

    train_sampler = TrajectorySampler(
        ds.Y[ds.train_ids], x=ds.x, gamma=ds.gamma[ds.train_ids],
        n_max=args.n_max, m_max=args.m_max, device=device, seed=args.seed,
    )
    val_sampler = TrajectorySampler(
        ds.Y[ds.val_ids], x=ds.x, gamma=ds.gamma[ds.val_ids],
        n_max=args.n_max, m_max=args.m_max, device=device, seed=args.seed + 1,
    )

    param_dim = int(ds.gamma.shape[-1])   # 7 (partner joints) or 11 (+ 4 static cup dims)
    model = CNMP(
        input_dim=1, output_dim=N_JOINTS, param_dim=param_dim,
        latent_dim=args.latent_dim,
        encoder_hidden_dims=(args.hidden, args.hidden),
        decoder_hidden_dims=(args.hidden, args.hidden),
        activation="gelu", min_std=args.min_std, device=device,
    )
    print(f"param_dim={param_dim} (cup_gamma={args.cup_gamma}, cup_dim={ds.cup_dim}) | "
          f"trainable parameters: {model.num_parameters()}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-4)

    ckpt_dir = Path(args.out) / arm / str(int(time.time()))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "cnmp_best.pt"

    # fixed validation batches -> comparable curve; + a start/middle/end -> whole-trajectory MSE
    val_batches = [val_sampler.random_batch(n_val) for _ in range(8)]
    goal_idx = torch.tensor([0, args.T // 2, args.T - 1])
    val_goal_batch = val_sampler.full_target_batch(torch.arange(n_val), goal_idx)

    def evaluate():
        nll = 0.0
        for vb in val_batches:
            mean, std = model.predict(**vb.inputs())
            nll += model.loss(mean, std, *vb.targets()).item()
        mean, _ = model.predict(**val_goal_batch.inputs())
        mse = model.mse(mean, val_goal_batch.tar_y, val_goal_batch.tar_mask).item()
        return nll / len(val_batches), mse

    history = {"step": [], "train_nll": [], "val_nll": [], "val_mse": []}
    best_metric, running, t0 = float("inf"), 0.0, time.time()

    for step in range(1, args.steps + 1):
        batch = train_sampler.random_batch(args.batch)
        mean, std = model(**batch.inputs())
        loss = model.loss(mean, std, *batch.targets())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        running += loss.item()

        if step % args.eval_every == 0:
            val_nll, val_mse = evaluate()
            history["step"].append(step)
            history["train_nll"].append(running / args.eval_every)
            history["val_nll"].append(val_nll)
            history["val_mse"].append(val_mse)
            metric = val_mse if args.select_on == "val_mse" else val_nll
            if metric < best_metric:  # lower is better for both NLL and MSE
                best_metric = metric
                model.save(best_path)
            print(f"step {step:6d} | train NLL {running / args.eval_every:8.3f} | "
                  f"val NLL {val_nll:8.3f} | val MSE {val_mse:.5f} | {time.time() - t0:5.0f}s")
            running = 0.0

    model.save(ckpt_dir / "cnmp_final.pt")
    torch.save(history, ckpt_dir / "history.pt")
    print(f"arm={arm}: best {args.select_on}={best_metric:.5f} | checkpoints -> {ckpt_dir}")
    return best_path


def main() -> int:
    args = parse_args()
    arms = ["left", "right"] if args.arm == "both" else [args.arm]
    for arm in arms:
        train_one_arm(arm, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
