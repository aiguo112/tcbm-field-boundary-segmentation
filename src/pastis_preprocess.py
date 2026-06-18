#!/usr/bin/env python3
"""Convert PASTIS parcel-ID annotations to FTW-style 3-class boundary masks."""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import ndimage

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PASTIS_ROOT = Path(r"E:\PASTIS\PASTIS")
ANNOTATIONS_DIR = PASTIS_ROOT / "ANNOTATIONS"
OUTPUT_MASK_DIR = PASTIS_ROOT / "BOUNDARY_MASKS"
VIS_OUTPUT_DIR = Path("outputs") / "label_conversion"

RANDOM_SEED = 42
N_VIS_EXAMPLES = 5

# FTW semantic classes
CLASS_BACKGROUND = 0
CLASS_FIELD = 1
CLASS_BOUNDARY = 2

CLASS_NAMES = {
    CLASS_BACKGROUND: "background",
    CLASS_FIELD: "field interior",
    CLASS_BOUNDARY: "boundary",
}

# 4-connected neighbor shifts for scipy.ndimage.shift (sy, sx)
NEIGHBOR_SHIFTS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def parcel_boundaries_4conn(parcel_ids: np.ndarray) -> np.ndarray:
    """Mark pixels with a 4-connected neighbor carrying a different parcel ID."""
    parcel_ids = np.asarray(parcel_ids)

    is_boundary = np.zeros(parcel_ids.shape, dtype=bool)
    for shift in NEIGHBOR_SHIFTS:
        neighbor = ndimage.shift(
            parcel_ids,
            shift=shift,
            order=0,
            prefilter=False,
            mode="nearest",
        )
        is_boundary |= neighbor != parcel_ids

    return is_boundary


def parcel_ids_to_boundary_mask(parcel_ids: np.ndarray) -> np.ndarray:
    """Convert a parcel-ID map to a 3-class FTW-compatible mask."""
    parcel_ids = np.asarray(parcel_ids)

    mask = np.zeros(parcel_ids.shape, dtype=np.uint8)
    mask[parcel_ids > 0] = CLASS_FIELD
    mask[parcel_boundaries_4conn(parcel_ids)] = CLASS_BOUNDARY
    return mask


def class_percentages(mask: np.ndarray) -> dict[int, float]:
    """Return per-class pixel percentages for one patch."""
    total = mask.size
    return {
        cls: 100.0 * np.count_nonzero(mask == cls) / total
        for cls in (CLASS_BACKGROUND, CLASS_FIELD, CLASS_BOUNDARY)
    }


def process_all_annotations() -> tuple[dict[int, float], list[tuple[Path, np.ndarray, np.ndarray]]]:
    """Convert every annotation file and return averaged class percentages."""
    annotation_files = sorted(ANNOTATIONS_DIR.glob("ParcelIDs_*.npy"))
    if not annotation_files:
        raise FileNotFoundError(f"No ParcelIDs_*.npy files found in {ANNOTATIONS_DIR}")

    OUTPUT_MASK_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(RANDOM_SEED)
    vis_paths = set(rng.sample(annotation_files, min(N_VIS_EXAMPLES, len(annotation_files))))

    class_pixel_counts = np.zeros(3, dtype=np.int64)
    total_pixels = 0
    vis_examples: list[tuple[Path, np.ndarray, np.ndarray]] = []

    print(f"Processing {len(annotation_files)} annotation files...")
    for i, ann_path in enumerate(annotation_files, start=1):
        parcel_ids = np.load(ann_path)
        mask = parcel_ids_to_boundary_mask(parcel_ids)

        out_path = OUTPUT_MASK_DIR / ann_path.name
        np.save(out_path, mask)

        for cls in (CLASS_BACKGROUND, CLASS_FIELD, CLASS_BOUNDARY):
            class_pixel_counts[cls] += np.count_nonzero(mask == cls)
        total_pixels += mask.size

        if ann_path in vis_paths:
            vis_examples.append((ann_path, parcel_ids, mask))

        if i % 500 == 0 or i == len(annotation_files):
            print(f"  {i}/{len(annotation_files)} done")

    avg_pct = {
        cls: 100.0 * class_pixel_counts[cls] / total_pixels
        for cls in (CLASS_BACKGROUND, CLASS_FIELD, CLASS_BOUNDARY)
    }
    return avg_pct, vis_examples


def print_class_distribution(avg_pct: dict[int, float], n_patches: int) -> None:
    """Print averaged class distribution across all patches."""
    print("\n" + "=" * 60)
    print(f"Class distribution (averaged over {n_patches} patches)")
    print("=" * 60)
    for cls in (CLASS_BACKGROUND, CLASS_FIELD, CLASS_BOUNDARY):
        print(f"  class {cls} ({CLASS_NAMES[cls]}): {avg_pct[cls]:.2f}%")


def visualize_examples(examples: list[tuple[Path, np.ndarray, np.ndarray]]) -> None:
    """Save side-by-side parcel-ID and boundary-mask plots."""
    VIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    boundary_cmap = plt.matplotlib.colors.ListedColormap(["#000000", "#2ca02c", "#d62728"])
    boundary_norm = plt.matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], boundary_cmap.N)

    for ann_path, parcel_ids, mask in examples:
        patch_id = ann_path.stem.replace("ParcelIDs_", "")
        pct = class_percentages(mask)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        parcel_plot = axes[0].imshow(parcel_ids, cmap="tab20")
        axes[0].set_title(f"Parcel IDs (patch {patch_id})")
        axes[0].axis("off")
        fig.colorbar(parcel_plot, ax=axes[0], fraction=0.046, pad=0.04)

        axes[1].imshow(mask, cmap=boundary_cmap, norm=boundary_norm)
        axes[1].set_title(
            "Boundary mask\n"
            f"bg {pct[0]:.1f}% | field {pct[1]:.1f}% | boundary {pct[2]:.1f}%"
        )
        axes[1].axis("off")
        legend_patches = [
            mpatches.Patch(color=boundary_cmap(i), label=f"{i}={CLASS_NAMES[i]}")
            for i in (CLASS_BACKGROUND, CLASS_FIELD, CLASS_BOUNDARY)
        ]
        axes[1].legend(handles=legend_patches, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=3)

        fig.tight_layout()
        out_path = VIS_OUTPUT_DIR / f"conversion_{patch_id}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


def main() -> None:
    n_files = len(list(ANNOTATIONS_DIR.glob("ParcelIDs_*.npy")))
    avg_pct, vis_examples = process_all_annotations()
    print_class_distribution(avg_pct, n_files)
    visualize_examples(vis_examples)
    print(f"\nDone. Masks saved to {OUTPUT_MASK_DIR.resolve()}")


if __name__ == "__main__":
    main()
