#!/usr/bin/env python3
"""Reduce PASTIS Sentinel-2 time series (43 dates) to 2 FTW-compatible dates."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PASTIS_ROOT = Path(r"E:\PASTIS\PASTIS")
INPUT_DIR = PASTIS_ROOT / "DATA_S2"

EXPECTED_SHAPE_TAIL = (10, 128, 128)
FTW_BANDS = (1, 2, 3, 7)  # Blue, Green, Red, NIR (0-indexed Sentinel-2 bands)
RED_BAND = 3
NIR_BAND = 7
NDVI_EPS = 1e-6

RANDOM_SEED = 42
N_VIZ_EXAMPLES = 5
VIZ_FILENAME = "temporal_selection_examples.png"
SUMMARY_FILENAME = "temporal_reduction_summary.json"

TemporalStrategy = Literal["ndvi_minmax", "random", "first_last", "median_ndvi"]
STRATEGY_CHOICES: tuple[TemporalStrategy, ...] = (
    "ndvi_minmax",
    "random",
    "first_last",
    "median_ndvi",
)

# FTW stacks window_b then window_a (date1 then date2 in the reduced array).
STACK_ORDER = ("date1", "date2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reduce PASTIS S2 time series to 2 FTW-compatible dates"
    )
    parser.add_argument(
        "--temporal_strategy",
        choices=STRATEGY_CHOICES,
        default="ndvi_minmax",
        help=(
            "How to pick 2 timesteps per patch "
            "(default: ndvi_minmax = argmin + argmax mean NDVI)"
        ),
    )
    return parser.parse_args()


def output_dir_for_strategy(strategy: TemporalStrategy) -> Path:
    return PASTIS_ROOT / f"DATA_S2_2DATE_{strategy}"


def patch_id_from_path(path: Path) -> int:
    return int(path.stem.split("_")[1])


def mean_ndvi_timeseries(s2: np.ndarray) -> np.ndarray:
    """Mean NDVI over all pixels for each timestep."""
    red = s2[:, RED_BAND].astype(np.float64)
    nir = s2[:, NIR_BAND].astype(np.float64)
    ndvi = (nir - red) / (nir + red + NDVI_EPS)
    return ndvi.mean(axis=(1, 2))


def select_dates_ndvi_minmax(mean_ndvi: np.ndarray) -> tuple[int, int]:
    """Return (low-NDVI timestep, high-NDVI timestep) indices."""
    peak = int(np.argmax(mean_ndvi))
    off_season = int(np.argmin(mean_ndvi))
    return off_season, peak


def select_dates_random(n_timesteps: int, rng: random.Random) -> tuple[int, int]:
    """Return two distinct timesteps in chronological order."""
    idx1, idx2 = rng.sample(range(n_timesteps), 2)
    return min(idx1, idx2), max(idx1, idx2)


def select_dates_first_last(n_timesteps: int) -> tuple[int, int]:
    """Return first and last timestep indices."""
    return 0, n_timesteps - 1


def select_dates_median_ndvi(mean_ndvi: np.ndarray) -> tuple[int, int]:
    """Return timestep closest to median NDVI and farthest from median NDVI."""
    median_val = float(np.median(mean_ndvi))
    dist = np.abs(mean_ndvi - median_val)
    closest = int(np.argmin(dist))
    farthest = int(np.argmax(dist))
    if closest == farthest and len(mean_ndvi) > 1:
        order = np.argsort(dist)[::-1]
        farthest = int(order[1])
    return closest, farthest


def select_dates(
    strategy: TemporalStrategy,
    *,
    n_timesteps: int,
    mean_ndvi: np.ndarray,
    rng: random.Random,
) -> tuple[int, int]:
    if strategy == "ndvi_minmax":
        return select_dates_ndvi_minmax(mean_ndvi)
    if strategy == "random":
        return select_dates_random(n_timesteps, rng)
    if strategy == "first_last":
        return select_dates_first_last(n_timesteps)
    if strategy == "median_ndvi":
        return select_dates_median_ndvi(mean_ndvi)
    raise ValueError(f"Unknown temporal strategy: {strategy!r}")


def date_labels(strategy: TemporalStrategy) -> tuple[str, str]:
    if strategy == "ndvi_minmax":
        return "off-season (min NDVI)", "peak (max NDVI)"
    if strategy == "random":
        return "random date1", "random date2"
    if strategy == "first_last":
        return "first timestep", "last timestep"
    if strategy == "median_ndvi":
        return "closest to median NDVI", "farthest from median NDVI"
    raise ValueError(f"Unknown temporal strategy: {strategy!r}")


def extract_ftw_stack(s2: np.ndarray, date1_idx: int, date2_idx: int) -> np.ndarray:
    """Stack 4 bands from date1 and date2 into shape (8, 128, 128)."""
    date1 = s2[date1_idx, FTW_BANDS, :, :]
    date2 = s2[date2_idx, FTW_BANDS, :, :]
    return np.concatenate([date1, date2], axis=0)


def print_index_distribution(label: str, indices: list[int]) -> None:
    counts = np.bincount(indices)
    print(f"\n{label} (timestep index -> count, {len(indices)} patches):")
    for idx, count in enumerate(counts):
        if count:
            print(f"  t={idx:2d}: {count}")


def plot_selection_examples(
    examples: list[tuple[int, np.ndarray, int, int]],
    out_path: Path,
    *,
    strategy: TemporalStrategy,
    label1: str,
    label2: str,
) -> None:
    """Plot mean NDVI curves for example patches with selected dates marked."""
    fig, axes = plt.subplots(1, len(examples), figsize=(4 * len(examples), 3.5))
    if len(examples) == 1:
        axes = [axes]

    for ax, (patch_id, mean_ndvi, date1_idx, date2_idx) in zip(axes, examples):
        t_axis = np.arange(mean_ndvi.shape[0])
        ax.plot(t_axis, mean_ndvi, color="0.35", linewidth=1.5, label="mean NDVI")
        ax.scatter(
            [date1_idx],
            [mean_ndvi[date1_idx]],
            color="tab:blue",
            s=60,
            zorder=3,
            label=f"{label1} (t={date1_idx})",
        )
        ax.scatter(
            [date2_idx],
            [mean_ndvi[date2_idx]],
            color="tab:green",
            s=60,
            zorder=3,
            label=f"{label2} (t={date2_idx})",
        )
        ax.set_xlabel("Timestep index")
        ax.set_ylabel("Mean NDVI")
        ax.set_title(f"Patch {patch_id}")
        ax.set_xlim(-0.5, mean_ndvi.shape[0] - 0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        f"PASTIS temporal reduction ({strategy}): selected timesteps",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved visualization: {out_path}")


def process_all(strategy: TemporalStrategy) -> dict[str, dict[str, int]]:
    """Process every S2 patch and write reduced .npy arrays."""
    output_dir = output_dir_for_strategy(strategy)
    label1, label2 = date_labels(strategy)

    s2_files = sorted(INPUT_DIR.glob("S2_*.npy"))
    if not s2_files:
        raise FileNotFoundError(f"No S2_*.npy files in {INPUT_DIR}")

    output_dir.mkdir(parents=True, exist_ok=True)

    date1_indices: list[int] = []
    date2_indices: list[int] = []
    n_timesteps_list: list[int] = []
    selections: dict[str, dict[str, int]] = {}
    viz_candidates: list[tuple[int, np.ndarray, int, int]] = []
    rng = random.Random(RANDOM_SEED)

    print(f"Temporal strategy: {strategy}")
    print(f"Processing {len(s2_files)} patches from {INPUT_DIR}")
    print(f"Writing reduced arrays to {output_dir}")

    for i, s2_path in enumerate(s2_files):
        patch_id = patch_id_from_path(s2_path)
        s2 = np.load(s2_path)

        if s2.ndim != 4 or s2.shape[1:] != EXPECTED_SHAPE_TAIL:
            raise ValueError(
                f"Unexpected shape {s2.shape} in {s2_path.name}; "
                f"expected (T, {EXPECTED_SHAPE_TAIL[0]}, 128, 128)"
            )
        if s2.shape[0] < 2:
            raise ValueError(f"Need at least 2 timesteps in {s2_path.name}, got {s2.shape[0]}")

        n_timesteps = s2.shape[0]
        n_timesteps_list.append(n_timesteps)
        mean_ndvi = mean_ndvi_timeseries(s2)
        date1_idx, date2_idx = select_dates(
            strategy,
            n_timesteps=n_timesteps,
            mean_ndvi=mean_ndvi,
            rng=rng,
        )
        reduced = extract_ftw_stack(s2, date1_idx, date2_idx)

        if reduced.shape != (8, 128, 128):
            raise ValueError(f"Reduced shape {reduced.shape} != (8, 128, 128)")

        np.save(output_dir / s2_path.name, reduced)

        date1_indices.append(date1_idx)
        date2_indices.append(date2_idx)
        selections[str(patch_id)] = {
            "n_timesteps": int(n_timesteps),
            "date1_timestep": date1_idx,
            "date2_timestep": date2_idx,
        }
        if strategy == "ndvi_minmax":
            selections[str(patch_id)]["off_season_timestep"] = date1_idx
            selections[str(patch_id)]["peak_timestep"] = date2_idx
        viz_candidates.append((patch_id, mean_ndvi, date1_idx, date2_idx))

        if (i + 1) % 250 == 0 or i + 1 == len(s2_files):
            print(f"  {i + 1}/{len(s2_files)} patches done")

    print("\n" + "=" * 60)
    print("Timesteps per patch")
    print("=" * 60)
    ts_counts = np.bincount(n_timesteps_list)
    for n_ts, count in enumerate(ts_counts):
        if count:
            print(f"  T={n_ts}: {count} patches")

    print("\n" + "=" * 60)
    print(f"Selected timestep index distributions ({strategy})")
    print("=" * 60)
    print_index_distribution(f"Date 1 ({label1})", date1_indices)
    print_index_distribution(f"Date 2 ({label2})", date2_indices)

    viz_rng = random.Random(RANDOM_SEED)
    viz_examples = viz_rng.sample(viz_candidates, min(N_VIZ_EXAMPLES, len(viz_candidates)))
    plot_selection_examples(
        viz_examples,
        output_dir / VIZ_FILENAME,
        strategy=strategy,
        label1=label1,
        label2=label2,
    )

    summary = {
        "temporal_strategy": strategy,
        "input_dir": str(INPUT_DIR),
        "output_dir": str(output_dir),
        "n_patches": len(s2_files),
        "timestep_counts": {str(n): int(c) for n, c in enumerate(ts_counts) if c},
        "ftw_bands_extracted": list(FTW_BANDS),
        "ndvi_bands": {"red": RED_BAND, "nir": NIR_BAND},
        "stack_order": list(STACK_ORDER),
        "date1_label": label1,
        "date2_label": label2,
        "output_shape": [8, 128, 128],
        "random_seed": RANDOM_SEED,
        "patches": selections,
    }
    summary_path = output_dir / SUMMARY_FILENAME
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON: {summary_path}")

    return selections


def main() -> None:
    args = parse_args()
    process_all(args.temporal_strategy)
    print("\nDone.")


if __name__ == "__main__":
    main()
