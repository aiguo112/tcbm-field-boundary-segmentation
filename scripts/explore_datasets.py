#!/usr/bin/env python3
"""Explore PASTIS and FTW remote sensing datasets for research paper analysis."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

# ---------------------------------------------------------------------------
# Paths (edit if your data lives elsewhere)
# ---------------------------------------------------------------------------
PASTIS_ROOT = Path(r"E:\PASTIS\PASTIS")
FTW_ROOT = Path(r"E:\FTW\ftw-baselines\data\ftw")
FTW_COUNTRY = "france"
OUTPUT_DIR = Path("outputs")

# FTW on disk: s2_images/window_{a,b} (4 bands each) + label_masks/semantic_3class
FTW_FRANCE = FTW_ROOT / FTW_COUNTRY
FTW_WINDOW_A = FTW_FRANCE / "s2_images" / "window_a"
FTW_WINDOW_B = FTW_FRANCE / "s2_images" / "window_b"
FTW_MASKS = FTW_FRANCE / "label_masks" / "semantic_3class"

# PASTIS Sentinel-2 bands (0-indexed): B02, B03, B04 at indices 0, 1, 2
RGB_BANDS = (2, 1, 0)  # display order: B04, B03, B02

RANDOM_SEED = 42
N_SAMPLES = 3

FTW_CLASS_NAMES = {
    0: "background",
    1: "field",
    2: "boundary",
    3: "ignore",
}


def stretch_rgb(rgb: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0) -> np.ndarray:
    """Percentile stretch for display."""
    lo, hi = np.percentile(rgb, (low_pct, high_pct))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((rgb.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def pastis_patch_id(path: Path) -> int:
    return int(path.stem.split("_")[1])


def explore_pastis_patches(n_samples: int = N_SAMPLES) -> tuple[Path, np.ndarray, np.ndarray]:
    """Load random PASTIS patches and print summary statistics."""
    print("=" * 60)
    print("1. PASTIS patches (DATA_S2 + ANNOTATIONS)")
    print("=" * 60)

    s2_dir = PASTIS_ROOT / "DATA_S2"
    ann_dir = PASTIS_ROOT / "ANNOTATIONS"
    s2_files = sorted(s2_dir.glob("S2_*.npy"))
    if not s2_files:
        raise FileNotFoundError(f"No S2_*.npy files found in {s2_dir}")

    rng = random.Random(RANDOM_SEED)
    chosen = rng.sample(s2_files, min(n_samples, len(s2_files)))

    first_path = chosen[0]
    first_s2: np.ndarray | None = None
    first_labels: np.ndarray | None = None

    for s2_path in chosen:
        patch_id = pastis_patch_id(s2_path)
        ann_path = ann_dir / f"ParcelIDs_{patch_id}.npy"

        s2 = np.load(s2_path)
        labels = np.load(ann_path)
        unique_labels = np.unique(labels)

        print(f"\nPatch {patch_id}:")
        print(f"  S2 file:         {s2_path.name}")
        print(f"  S2 shape:        {s2.shape}")
        print(f"  S2 dtype:        {s2.dtype}")
        print(f"  S2 min/max:      {s2.min()} / {s2.max()}")
        print(f"  Labels file:     {ann_path.name}")
        print(f"  Labels shape:    {labels.shape}")
        print(f"  Labels dtype:    {labels.dtype}")
        print(f"  Label min/max:   {labels.min()} / {labels.max()}")
        print(f"  Unique label IDs ({len(unique_labels)}): {unique_labels.tolist()}")

        if first_s2 is None:
            first_path, first_s2, first_labels = s2_path, s2, labels

    assert first_s2 is not None and first_labels is not None
    return first_path, first_s2, first_labels


def explore_pastis_norm() -> None:
    """Print structure of PASTIS normalization statistics."""
    print("\n" + "=" * 60)
    print("2. PASTIS NORM_S2_patch.json")
    print("=" * 60)

    norm_path = PASTIS_ROOT / "NORM_S2_patch.json"
    with norm_path.open(encoding="utf-8") as f:
        norm = json.load(f)

    print(f"File: {norm_path}")
    print(f"Top-level keys (folds): {list(norm.keys())}")
    for fold_name, stats in norm.items():
        print(f"\n  [{fold_name}]")
        print(f"    keys: {list(stats.keys())}")
        print(f"    mean: {len(stats['mean'])} values, first 3 = {stats['mean'][:3]}")
        print(f"    std:  {len(stats['std'])} values, first 3 = {stats['std'][:3]}")


def explore_pastis_folds() -> None:
    """Count PASTIS patches per cross-validation fold."""
    print("\n" + "=" * 60)
    print("3. PASTIS metadata.geojson folds")
    print("=" * 60)

    meta_path = PASTIS_ROOT / "metadata.geojson"
    with meta_path.open(encoding="utf-8") as f:
        metadata = json.load(f)

    fold_counts = Counter(
        feature["properties"]["Fold"] for feature in metadata["features"]
    )
    print(f"File: {meta_path}")
    print(f"Total patches: {len(metadata['features'])}")
    for fold in sorted(fold_counts):
        print(f"  Fold {fold}: {fold_counts[fold]} patches")


def list_ftw_chips() -> list[str]:
    """Return chip IDs with both temporal windows and a semantic mask."""
    chips = []
    for mask_path in FTW_MASKS.glob("*.tif"):
        chip_id = mask_path.stem
        if (FTW_WINDOW_A / f"{chip_id}.tif").exists() and (
            FTW_WINDOW_B / f"{chip_id}.tif"
        ).exists():
            chips.append(chip_id)
    return sorted(chips)


def load_ftw_chip(chip_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Load stacked 8-band image (window_b + window_a) and mask."""
    with rasterio.open(FTW_WINDOW_B / f"{chip_id}.tif") as src:
        window_b = src.read()
    with rasterio.open(FTW_WINDOW_A / f"{chip_id}.tif") as src:
        window_a = src.read()
    with rasterio.open(FTW_MASKS / f"{chip_id}.tif") as src:
        mask = src.read(1)

    image = np.concatenate([window_b, window_a], axis=0)
    return image, mask


def explore_ftw_chips(n_samples: int = N_SAMPLES) -> tuple[str, np.ndarray, np.ndarray]:
    """Load random FTW chips from France and print summary statistics."""
    print("\n" + "=" * 60)
    print(f"4. FTW chips ({FTW_COUNTRY})")
    print("=" * 60)
    print(f"  Images: window_b + window_a -> (8, 256, 256)")
    print(f"  Masks:  {FTW_MASKS}")

    chips = list_ftw_chips()
    if not chips:
        raise FileNotFoundError(f"No complete FTW chips found under {FTW_FRANCE}")

    rng = random.Random(RANDOM_SEED)
    chosen = rng.sample(chips, min(n_samples, len(chips)))

    first_chip_id = chosen[0]
    first_image: np.ndarray | None = None
    first_mask: np.ndarray | None = None

    for chip_id in chosen:
        image, mask = load_ftw_chip(chip_id)

        print(f"\nChip {chip_id}:")
        print(f"  Image shape:  {image.shape}")
        print(f"  Image dtype:  {image.dtype}")
        print(f"  Image min/max: {image.min()} / {image.max()}")
        print(f"  Mask shape:   {mask.shape}")
        print(f"  Mask dtype:   {mask.dtype}")
        print(f"  Mask min/max: {mask.min()} / {mask.max()}")

        if first_image is None:
            first_chip_id, first_image, first_mask = chip_id, image, mask

    assert first_image is not None and first_mask is not None
    return first_chip_id, first_image, first_mask


def explore_ftw_mask_classes() -> None:
    """Print unique FTW mask classes and pixel counts across all France chips."""
    print("\n" + "=" * 60)
    print("5. FTW mask class distribution (all France chips)")
    print("=" * 60)

    mask_files = sorted(FTW_MASKS.glob("*.tif"))
    total_counts: Counter[int] = Counter()

    for mask_path in mask_files:
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
        values, counts = np.unique(mask, return_counts=True)
        for value, count in zip(values, counts):
            total_counts[int(value)] += int(count)

    print(f"Chips scanned: {len(mask_files)}")
    print(f"Unique class values: {sorted(total_counts.keys())}")
    for class_id in sorted(total_counts):
        name = FTW_CLASS_NAMES.get(class_id, "unknown")
        print(f"  class {class_id} ({name}): {total_counts[class_id]:,} pixels")


def visualize_pastis(
    s2: np.ndarray, labels: np.ndarray, patch_id: int, out_path: Path
) -> None:
    """Save side-by-side PASTIS RGB (first timestep) and parcel label map."""
    rgb = s2[0, list(RGB_BANDS), :, :].transpose(1, 2, 0)
    rgb_display = stretch_rgb(rgb)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(rgb_display)
    axes[0].set_title(f"PASTIS patch {patch_id}\nRGB (timestep 0)")
    axes[0].axis("off")

    label_plot = axes[1].imshow(labels, cmap="tab20")
    axes[1].set_title("Parcel ID map")
    axes[1].axis("off")
    fig.colorbar(label_plot, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def visualize_ftw(
    image: np.ndarray, mask: np.ndarray, chip_id: str, out_path: Path
) -> None:
    """Save side-by-side FTW RGB (first 3 bands of stacked image) and semantic mask."""
    rgb = image[:3].transpose(1, 2, 0)
    rgb_display = stretch_rgb(rgb)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(rgb_display)
    axes[0].set_title(f"FTW chip {chip_id}\nRGB (window B: B04, B03, B02)")
    axes[0].axis("off")

    mask_plot = axes[1].imshow(mask, cmap="viridis", vmin=0, vmax=3)
    axes[1].set_title("Semantic mask (0=bg, 1=field, 2=boundary, 3=ignore)")
    axes[1].axis("off")
    fig.colorbar(mask_plot, ax=axes[1], ticks=[0, 1, 2, 3], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    s2_path, s2, labels = explore_pastis_patches()
    explore_pastis_norm()
    explore_pastis_folds()

    chip_id, ftw_image, ftw_mask = explore_ftw_chips()
    explore_ftw_mask_classes()

    print("\n" + "=" * 60)
    print("6–7. Visualizations")
    print("=" * 60)

    patch_id = pastis_patch_id(s2_path)
    visualize_pastis(s2, labels, patch_id, OUTPUT_DIR / "pastis_sample.png")
    visualize_ftw(ftw_image, ftw_mask, chip_id, OUTPUT_DIR / "ftw_sample.png")

    print("\nDone. Outputs written to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
