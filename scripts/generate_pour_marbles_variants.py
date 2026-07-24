#!/usr/bin/env python3
"""Generate balanced, executed scene variants from proven left/right pour layouts."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
import time

import numpy as np

from generate_pour_marbles_demos import (
    DEFAULT_OUTPUT,
    ROOT,
    run_seed,
    validate_recording,
)


BASE_RECORDINGS = {
    "right": DEFAULT_OUTPUT / "demo_seed_000000.npz",
    "left": DEFAULT_OUTPUT / "demo_seed_000030.npz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--translation-mm", type=float, default=3.0)
    parser.add_argument("--left-translation-mm", type=float, default=None)
    parser.add_argument("--right-translation-mm", type=float, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-attempts", type=int, default=1000)
    return parser.parse_args()


def load_base(path: Path) -> dict[str, np.ndarray | str]:
    valid, source = validate_recording(path)
    if not valid:
        raise ValueError(f"Invalid base recording {path}: {source}")
    with np.load(path, allow_pickle=False) as data:
        return {
            "source_cup": source,
            "seed": int(data["seed"]),
            "cup_qpos": np.asarray(data["cup_qpos"]).copy(),
            "marble_qpos": np.asarray(data["marble_qpos"]).copy(),
            "cup_qvel": np.asarray(data["cup_qvel"]).copy(),
            "marble_qvel": np.asarray(data["marble_qvel"]).copy(),
        }


def write_variant_state(
    base: dict[str, np.ndarray | str], variant_id: int, radius_m: float, state_dir: Path
) -> Path:
    rng = np.random.default_rng(0x5EED + variant_id)
    cup_qpos = np.asarray(base["cup_qpos"]).copy()
    marble_qpos = np.asarray(base["marble_qpos"]).copy()

    # Independent planar translations make the bimanual paths vary.  All source marbles move
    # rigidly with their containing cup so every rollout begins from a physically equivalent,
    # settled state rather than a synthetic overlap or mid-air checkpoint.
    cup_delta = rng.uniform(-radius_m, radius_m, size=(2, 2))
    cup_qpos[:, :2] += cup_delta
    source = str(base["source_cup"])
    source_index = 0 if source == "left" else 1
    marble_qpos[:, :2] += cup_delta[source_index]

    path = state_dir / f"variant_{variant_id:06d}.npz"
    np.savez(
        path,
        format_version=np.asarray(3, dtype=np.int32),
        file_type=np.asarray("pour_marbles_initial_state"),
        cup_qpos=cup_qpos,
        marble_qpos=marble_qpos,
        cup_qvel=np.asarray(base["cup_qvel"]),
        marble_qvel=np.asarray(base["marble_qvel"]),
        seed=np.asarray(variant_id, dtype=np.int64),
        source_cup=np.asarray(source),
    )
    return path


def run_variant(
    variant_id: int,
    base: dict[str, np.ndarray | str],
    radius_m: float,
    output: Path,
    state_dir: Path,
    log_dir: Path,
) -> tuple[int, bool, str, float]:
    state_path = write_variant_state(base, variant_id, radius_m, state_dir)
    try:
        return run_seed(
            variant_id,
            output,
            log_dir,
            load_state=state_path,
            motion_variant=variant_id,
            reset_seed=int(base["seed"]),
        )
    finally:
        state_path.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    if args.count < 2 or args.workers <= 0 or args.translation_mm < 0:
        raise ValueError("count must be >= 2; workers positive; translation-mm non-negative")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    bases = {source: load_base(path.resolve()) for source, path in BASE_RECORDINGS.items()}
    target_by_source = {"right": args.count // 2, "left": args.count - args.count // 2}

    successes: dict[int, str] = {}
    for path in sorted(output.glob("demo_seed_*.npz")):
        valid, source = validate_recording(path)
        if valid and source in target_by_source:
            with np.load(path, allow_pickle=False) as data:
                identifier = int(data["record_id"]) if "record_id" in data else int(data["seed"])
                successes[identifier] = source
    counts = Counter(successes.values())
    if any(counts[source] > target for source, target in target_by_source.items()):
        raise RuntimeError(f"Existing source counts {dict(counts)} exceed targets {target_by_source}")

    state_dir = Path("/tmp/pour_marbles_variant_states")
    log_dir = Path("/tmp/pour_marbles_demo_logs")
    state_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    radius_mm = {
        "left": args.left_translation_mm or args.translation_mm,
        "right": args.right_translation_mm or args.translation_mm,
    }
    if any(radius < 0 for radius in radius_mm.values()):
        raise ValueError("per-source translation radii must be non-negative")
    next_id = 1_000_000
    attempts = 0
    inflight: dict[Future, tuple[int, str]] = {}
    started = time.monotonic()
    print(
        f"Generating balanced variants to {target_by_source} with radii_mm={radius_mm}",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while len(successes) < args.count:
            inflight_counts = Counter(source for _, source in inflight.values())
            while len(inflight) < args.workers and attempts < args.max_attempts:
                deficits = {
                    source: target_by_source[source] - counts[source] - inflight_counts[source]
                    for source in target_by_source
                }
                source = max(deficits, key=deficits.get)
                if deficits[source] <= 0:
                    break
                while (output / f"demo_seed_{next_id:06d}.npz").exists():
                    next_id += 1
                future = pool.submit(
                    run_variant,
                    next_id,
                    bases[source],
                    radius_mm[source] / 1000.0,
                    output,
                    state_dir,
                    log_dir,
                )
                inflight[future] = (next_id, source)
                inflight_counts[source] += 1
                next_id += 1
                attempts += 1

            if not inflight:
                raise RuntimeError(
                    f"Stopped at {dict(counts)} after {attempts} attempts; targets={target_by_source}"
                )

            done, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for future in done:
                variant_id, intended_source = inflight.pop(future)
                seed, ok, source, elapsed = future.result()
                if ok and source == intended_source:
                    successes[seed] = source
                    counts[source] += 1
                    print(
                        f"[{len(successes):03d}/{args.count}] id={variant_id} source={source} "
                        f"counts={dict(counts)} elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                else:
                    print(
                        f"[failed] id={variant_id} intended={intended_source} result={source} "
                        f"elapsed={elapsed:.1f}s",
                        flush=True,
                    )

    elapsed = time.monotonic() - started
    print(
        f"Complete: {len(successes)} demonstrations, source counts={dict(counts)}, "
        f"attempts={attempts}, elapsed={elapsed / 60.0:.1f} min",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
