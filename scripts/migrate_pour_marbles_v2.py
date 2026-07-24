#!/usr/bin/env python3
"""Normalize pose-only v2 pour recordings whose restored object velocity was zero."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "duo" / "pour_marbles"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, nargs="?", default=DEFAULT_DATASET)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Atomically rewrite v2 files; without this flag, only report them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(args.dataset.resolve().glob("demo_seed_*.npz"))
    migrated = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            version = int(np.asarray(data["format_version"]).item())
            if version != 2:
                continue
            if "cup_qvel" in data or "marble_qvel" in data:
                raise ValueError(f"{path.name}: v2 file unexpectedly has velocity fields")
            payload = {key: np.asarray(data[key]).copy() for key in data.files}

        print(path.name)
        if not args.apply:
            continue

        payload["format_version"] = np.asarray(3, dtype=np.int32)
        payload["cup_qvel"] = np.zeros((2, 6), dtype=np.float64)
        payload["marble_qvel"] = np.zeros((20, 6), dtype=np.float64)
        if "record_id" not in payload:
            payload["record_id"] = np.asarray(int(payload["seed"]), dtype=np.int64)

        temporary = path.with_name(f".{path.stem}.v3.npz")
        np.savez(temporary, **payload)
        with np.load(temporary, allow_pickle=False) as check:
            if int(check["format_version"]) != 3:
                raise RuntimeError(f"{temporary.name}: verification failed")
            if check["cup_qvel"].shape != (2, 6) or check["marble_qvel"].shape != (20, 6):
                raise RuntimeError(f"{temporary.name}: velocity verification failed")
        temporary.replace(path)
        migrated += 1

    action = "Migrated" if args.apply else "Found"
    print(f"{action} {migrated if args.apply else sum(1 for p in paths if _is_v2(p))} v2 files")
    return 0


def _is_v2(path: Path) -> bool:
    with np.load(path, allow_pickle=False) as data:
        return int(np.asarray(data["format_version"]).item()) == 2


if __name__ == "__main__":
    raise SystemExit(main())
