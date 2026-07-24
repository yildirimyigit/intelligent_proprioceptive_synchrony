#!/usr/bin/env python3
"""Independently audit a directory of pour-marbles demonstration archives."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "duo" / "pour_marbles"
REQUIRED_TRUE = (
    "final_success",
    "final_left_cup_in_place",
    "final_right_cup_in_place",
    "final_left_cup_upright",
    "final_right_cup_upright",
    "strict_success",
)
TRAJECTORY_FIELDS = {
    "actions": 16,
    "left_joint_qpos": 7,
    "right_joint_qpos": 7,
    "left_joint_targets": 7,
    "right_joint_targets": 7,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, nargs="?", default=DEFAULT_DATASET)
    parser.add_argument("--expected-count", type=int, default=100)
    return parser.parse_args()


def scalar(data: np.lib.npyio.NpzFile, key: str):
    value = np.asarray(data[key])
    if value.shape != ():
        raise ValueError(f"{key} must be scalar, got {value.shape}")
    return value.item()


def require_finite(name: str, value: np.ndarray) -> None:
    if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
        raise ValueError(f"{name} is non-numeric or contains non-finite values")


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    paths = sorted(dataset.glob("demo_seed_*.npz"))
    all_npz = sorted(dataset.glob("*.npz"))
    unexpected = sorted(set(all_npz) - set(paths))
    hidden_candidates = sorted(dataset.glob(".candidate*.npz"))

    errors: list[str] = []
    if len(paths) != args.expected_count:
        errors.append(f"expected {args.expected_count} demonstrations, found {len(paths)}")
    if unexpected:
        errors.append("unexpected NPZ files: " + ", ".join(path.name for path in unexpected))
    if hidden_candidates:
        errors.append("candidate files remain: " + ", ".join(path.name for path in hidden_candidates))

    source_counts: Counter[str] = Counter()
    version_counts: Counter[int] = Counter()
    record_ids: list[int] = []
    lengths: list[int] = []
    durations: list[float] = []
    cup_xy: list[np.ndarray] = []
    terminal_by_source: dict[str, list[np.ndarray]] = {"left": [], "right": []}
    trajectory_hashes: set[str] = set()
    scene_hashes: set[str] = set()
    files_with_initial_velocity = 0

    for path in paths:
        try:
            with np.load(path, allow_pickle=False) as data:
                if str(scalar(data, "file_type")) != "pour_marbles_recording":
                    raise ValueError("invalid file_type")
                version = int(scalar(data, "format_version"))
                if version != 3:
                    raise ValueError(f"format_version={version}, expected 3")
                version_counts[version] += 1

                if not all(bool(scalar(data, key)) for key in REQUIRED_TRUE):
                    raise ValueError("one or more strict success flags are false")
                if int(scalar(data, "final_stage")) != 6:
                    raise ValueError("final_stage is not 6")
                if int(scalar(data, "final_max_stage")) != 6:
                    raise ValueError("final_max_stage is not 6")

                source = str(scalar(data, "source_cup"))
                if source not in source_counts and source not in ("left", "right"):
                    raise ValueError(f"invalid source_cup={source!r}")
                expected_marbles = (0, 20) if source == "left" else (20, 0)
                actual_marbles = (
                    int(scalar(data, "final_marbles_left")),
                    int(scalar(data, "final_marbles_right")),
                )
                if actual_marbles != expected_marbles:
                    raise ValueError(
                        f"final marble counts are {actual_marbles}, expected {expected_marbles}"
                    )
                source_counts[source] += 1

                record_id = int(scalar(data, "record_id")) if "record_id" in data else int(
                    scalar(data, "seed")
                )
                record_ids.append(record_id)

                actions = np.asarray(data["actions"])
                steps = len(actions)
                if steps <= 0:
                    raise ValueError("empty trajectory")
                for field, width in TRAJECTORY_FIELDS.items():
                    values = np.asarray(data[field])
                    if values.shape != (steps, width):
                        raise ValueError(f"{field} shape is {values.shape}, expected {(steps, width)}")
                    require_finite(field, values)
                for field in ("left_gripper_commands", "right_gripper_commands", "timestamps_s"):
                    values = np.asarray(data[field])
                    if values.shape != (steps,):
                        raise ValueError(f"{field} shape is {values.shape}, expected {(steps,)}")
                    require_finite(field, values)

                timestamps = np.asarray(data["timestamps_s"])
                if not np.all(np.diff(timestamps) > 0):
                    raise ValueError("timestamps are not strictly increasing")

                initial_left = np.asarray(data["initial_left_joint_qpos"])
                initial_right = np.asarray(data["initial_right_joint_qpos"])
                if initial_left.shape != (7,) or initial_right.shape != (7,):
                    raise ValueError("invalid initial robot joint-state shape")
                require_finite("initial_left_joint_qpos", initial_left)
                require_finite("initial_right_joint_qpos", initial_right)

                cup_qpos = np.asarray(data["cup_qpos"])
                marble_qpos = np.asarray(data["marble_qpos"])
                if cup_qpos.shape != (2, 7) or marble_qpos.shape != (20, 7):
                    raise ValueError("invalid initial object-position shape")
                require_finite("cup_qpos", cup_qpos)
                require_finite("marble_qpos", marble_qpos)

                velocity_keys = ("cup_qvel" in data, "marble_qvel" in data)
                if velocity_keys[0] != velocity_keys[1]:
                    raise ValueError("only one of cup_qvel/marble_qvel is present")
                if all(velocity_keys):
                    cup_qvel = np.asarray(data["cup_qvel"])
                    marble_qvel = np.asarray(data["marble_qvel"])
                    if cup_qvel.shape != (2, 6) or marble_qvel.shape != (20, 6):
                        raise ValueError("invalid initial object-velocity shape")
                    require_finite("cup_qvel", cup_qvel)
                    require_finite("marble_qvel", marble_qvel)
                    files_with_initial_velocity += 1
                else:
                    raise ValueError("recording lacks initial object velocities")

                left_qpos = np.asarray(data["left_joint_qpos"])
                right_qpos = np.asarray(data["right_joint_qpos"])
                trajectory_hashes.add(
                    hashlib.sha256(left_qpos.tobytes() + right_qpos.tobytes()).hexdigest()
                )
                scene_hashes.add(
                    hashlib.sha256(cup_qpos.tobytes() + marble_qpos.tobytes()).hexdigest()
                )
                terminal_by_source[source].append(np.concatenate((left_qpos[-1], right_qpos[-1])))
                cup_xy.append(cup_qpos[:, :2])
                lengths.append(steps)
                durations.append(float(timestamps[-1] - timestamps[0]))
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    if len(record_ids) != len(set(record_ids)):
        duplicates = [item for item, count in Counter(record_ids).items() if count > 1]
        errors.append(f"duplicate record IDs: {duplicates}")
    expected_per_source = args.expected_count // 2
    expected_sources = {"left": args.expected_count - expected_per_source, "right": expected_per_source}
    if dict(source_counts) != expected_sources:
        errors.append(f"source balance is {dict(source_counts)}, expected {expected_sources}")

    if errors:
        print("AUDIT FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    cup_xy_values = np.stack(cup_xy)
    endpoint_spans = {
        source: float(np.ptp(np.stack(values), axis=0).max())
        for source, values in terminal_by_source.items()
    }
    total_bytes = sum(path.stat().st_size for path in paths)
    print("AUDIT PASSED")
    print(f"files: {len(paths)}")
    print(f"source_counts: {dict(source_counts)}")
    print(f"format_versions: {dict(version_counts)}")
    print(f"files_with_initial_object_velocity: {files_with_initial_velocity}")
    print(f"steps: min={min(lengths)} max={max(lengths)} unique={len(set(lengths))} total={sum(lengths)}")
    print(f"durations_s: min={min(durations):.3f} max={max(durations):.3f}")
    print(f"unique_initial_scenes: {len(scene_hashes)}")
    print(f"unique_joint_trajectories: {len(trajectory_hashes)}")
    print(
        "initial_cup_xy_span_m: "
        f"min={cup_xy_values.min(axis=0).tolist()} max={cup_xy_values.max(axis=0).tolist()}"
    )
    print(f"max_terminal_joint_span_rad_by_source: {endpoint_spans}")
    print(f"total_size_mib: {total_bytes / (1024 * 1024):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
