#!/usr/bin/env python3
"""Generate an exact number of strictly successful pour_marbles demonstrations."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
import os
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = ROOT / "scripts" / "pour_marbles_controller.py"
DEFAULT_OUTPUT = ROOT / "data" / "duo" / "pour_marbles"
REQUIRED_BOOL_KEYS = (
    "final_success",
    "final_left_cup_in_place",
    "final_right_cup_in_place",
    "final_left_cup_upright",
    "final_right_cup_upright",
    "strict_success",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def validate_recording(path: Path) -> tuple[bool, str]:
    try:
        with np.load(path, allow_pickle=False) as data:
            missing = [key for key in REQUIRED_BOOL_KEYS if key not in data]
            if missing:
                return False, f"missing metadata: {', '.join(missing)}"
            if not all(bool(data[key]) for key in REQUIRED_BOOL_KEYS):
                return False, "strict success metadata is false"

            source = str(data["source_cup"])
            if source not in ("left", "right"):
                return False, f"invalid source_cup={source!r}"
            expected = (0, 20) if source == "left" else (20, 0)
            counts = (int(data["final_marbles_left"]), int(data["final_marbles_right"]))
            if counts != expected:
                return False, f"final marble counts {counts}, expected {expected}"

            actions = np.asarray(data["actions"])
            left = np.asarray(data["left_joint_qpos"])
            right = np.asarray(data["right_joint_qpos"])
            timestamps = np.asarray(data["timestamps_s"])
            steps = len(actions)
            if actions.shape != (steps, 16):
                return False, f"actions shape is {actions.shape}"
            if left.shape != (steps, 7) or right.shape != (steps, 7):
                return False, f"joint shapes are {left.shape} and {right.shape}"
            if timestamps.shape != (steps,) or not np.all(np.diff(timestamps) > 0):
                return False, "timestamps are missing or not strictly increasing"
            if not all(np.isfinite(x).all() for x in (actions, left, right, timestamps)):
                return False, "trajectory contains non-finite values"
            if np.asarray(data["cup_qpos"]).shape != (2, 7):
                return False, "invalid initial cup state"
            if np.asarray(data["marble_qpos"]).shape != (20, 7):
                return False, "invalid initial marble state"
            if "cup_qvel" in data and np.asarray(data["cup_qvel"]).shape != (2, 6):
                return False, "invalid initial cup velocity"
            if "marble_qvel" in data and np.asarray(data["marble_qvel"]).shape != (20, 6):
                return False, "invalid initial marble velocity"
            return True, source
    except Exception as exc:
        return False, f"could not read recording: {exc}"


def run_seed(
    seed: int,
    output: Path,
    log_dir: Path,
    load_state: Path | None = None,
    motion_variant: int | None = None,
    reset_seed: int | None = None,
) -> tuple[int, bool, str, float]:
    final_path = output / f"demo_seed_{seed:06d}.npz"
    existing_ok, existing_detail = validate_recording(final_path) if final_path.exists() else (False, "")
    if existing_ok:
        return seed, True, existing_detail, 0.0

    candidate = output / f".candidate_seed_{seed:06d}.npz"
    log_path = log_dir / f"seed_{seed:06d}.log"
    if candidate.exists():
        candidate.unlink()

    started = time.monotonic()
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[variable] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        command = [
            sys.executable,
            str(CONTROLLER),
            "--seed",
            str(seed if reset_seed is None else reset_seed),
            "--record-id",
            str(seed),
        ]
        if load_state is not None:
            command.extend(["--load-state", str(load_state)])
        if motion_variant is not None:
            command.extend(["--motion-variant", str(motion_variant)])
        command.extend(["--record", str(candidate)])
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.monotonic() - started

    valid, detail = validate_recording(candidate) if candidate.exists() else (False, "no recording")
    if result.returncode == 0 and valid:
        candidate.replace(final_path)
        return seed, True, detail, elapsed
    if candidate.exists():
        candidate.unlink()
    return seed, False, f"exit={result.returncode}, {detail}, log={log_path}", elapsed


def main() -> int:
    args = parse_args()
    if args.count <= 0 or args.workers <= 0 or args.max_attempts <= 0:
        raise ValueError("count, workers, and max-attempts must be positive")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    log_dir = Path("/tmp/pour_marbles_demo_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    successes: dict[int, str] = {}
    for path in sorted(output.glob("demo_seed_*.npz")):
        valid, detail = validate_recording(path)
        if valid:
            with np.load(path, allow_pickle=False) as data:
                identifier = int(data["record_id"]) if "record_id" in data else int(data["seed"])
                successes[identifier] = detail
    if len(successes) > args.count:
        raise RuntimeError(
            f"Found {len(successes)} valid demonstrations, more than requested count={args.count}"
        )

    next_seed = args.start_seed
    attempts = 0
    inflight: dict[Future, int] = {}
    started = time.monotonic()
    print(
        f"Generating {args.count - len(successes)} demonstrations with {args.workers} workers "
        f"in {output}",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while len(successes) < args.count:
            while (
                len(inflight) < args.workers
                and len(successes) + len(inflight) < args.count
                and attempts < args.max_attempts
            ):
                while next_seed in successes or (output / f"demo_seed_{next_seed:06d}.npz").exists():
                    next_seed += 1
                future = pool.submit(run_seed, next_seed, output, log_dir)
                inflight[future] = next_seed
                next_seed += 1
                attempts += 1

            if not inflight:
                raise RuntimeError(
                    f"Stopped after {attempts} attempts with only {len(successes)} successes"
                )

            done, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for future in done:
                inflight.pop(future)
                seed, ok, detail, elapsed = future.result()
                if ok:
                    successes[seed] = detail
                    print(
                        f"[{len(successes):03d}/{args.count}] seed={seed} source={detail} "
                        f"elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                else:
                    print(f"[failed] seed={seed} elapsed={elapsed:.1f}s {detail}", flush=True)

    total = time.monotonic() - started
    print(
        f"Complete: {len(successes)} strictly successful demonstrations from {attempts} attempts "
        f"in {total / 60.0:.1f} min",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
