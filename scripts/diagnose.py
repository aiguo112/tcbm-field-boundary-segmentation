#!/usr/bin/env python3
"""Diagnose FTW mask/image layout and FTWDataset sample contents."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import rasterio

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import FTWDataset, remap_ftw_mask

FTW_FRANCE = Path(r"E:\FTW\ftw-baselines\data\ftw\france")
MASKS_DIR_USER = FTW_FRANCE / "masks"
MASKS_DIR_FTW = FTW_FRANCE / "label_masks" / "semantic_3class"
WINDOW_A_DIR = FTW_FRANCE / "s2_images" / "window_a"
WINDOW_B_DIR = FTW_FRANCE / "s2_images" / "window_b"

RANDOM_SEED = 42
N_RANDOM_MASKS = 10


def find_mask_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(root.rglob("*.tif")) + sorted(root.rglob("*.tiff"))


def analyze_mask(path: Path) -> dict:
    with rasterio.open(path, "r") as src:
        data = src.read()
        band0 = data[0] if data.ndim == 3 else data

    values, counts = np.unique(band0, return_counts=True)
    total = band0.size
    return {
        "path": path,
        "n_bands": data.shape[0] if data.ndim == 3 else 1,
        "dtype": str(band0.dtype),
        "shape": tuple(band0.shape),
        "values": values.astype(int),
        "counts": counts.astype(np.int64),
        "total_pixels": int(total),
    }


def print_mask_report(title: str, info: dict) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print(f"File: {info['path'].name}")
    print(f"  Bands:  {info['n_bands']}")
    print(f"  Dtype:  {info['dtype']}")
    print(f"  Shape:  {info['shape']}")
    print(f"  Unique values: {info['values'].tolist()}")
    print("  Value distribution:")
    for v, c in zip(info["values"], info["counts"]):
        pct = 100.0 * c / info["total_pixels"]
        print(f"    class {int(v):3d}: {int(c):>10,} px ({pct:6.2f}%)")


def print_average_distribution(mask_paths: list[Path]) -> None:
    rng = random.Random(RANDOM_SEED)
    chosen = rng.sample(mask_paths, min(N_RANDOM_MASKS, len(mask_paths)))

    # Track counts for values 0-3 and any extras
    value_totals: dict[int, int] = {}
    pixel_total = 0

    print(f"\n{'=' * 60}")
    print(f"Average class distribution ({len(chosen)} random masks)")
    for path in chosen:
        info = analyze_mask(path)
        pixel_total += info["total_pixels"]
        for v, c in zip(info["values"], info["counts"]):
            value_totals[int(v)] = value_totals.get(int(v), 0) + int(c)

    all_values = sorted(value_totals)
    print(f"  Unique values seen across sample: {all_values}")
    for v in all_values:
        c = value_totals[v]
        pct = 100.0 * c / max(pixel_total, 1)
        print(f"    class {v:3d}: {c:>12,} px ({pct:6.2f}%)")


def analyze_image_for_chip(chip_id: str) -> None:
    print(f"\n{'=' * 60}")
    print("Corresponding S2 image (window_b + window_a)")
    paths = [
        WINDOW_B_DIR / f"{chip_id}.tif",
        WINDOW_A_DIR / f"{chip_id}.tif",
    ]
    for label, img_path in zip(("window_b", "window_a"), paths):
        if not img_path.exists():
            print(f"  {label}: MISSING {img_path}")
            continue
        with rasterio.open(img_path, "r") as src:
            arr = src.read()
        print(f"  {label}: {img_path.name}")
        print(f"    shape={arr.shape}, dtype={arr.dtype}, min={arr.min()}, max={arr.max()}")


def analyze_ftw_dataset_sample() -> None:
    print(f"\n{'=' * 60}")
    print("FTWDataset sample (split=train, index=0)")
    ds = FTWDataset(split="train", augment=False, cache_in_memory=False)
    sample = ds[0]

    image = sample["image"]
    mask = sample["mask"]
    print(f"  chip_id:  {sample['chip_id']}")
    print(f"  country:  {sample['country']}")
    print(
        f"  image tensor: shape={tuple(image.shape)}, dtype={image.dtype}, "
        f"min={image.min().item():.4f}, max={image.max().item():.4f}"
    )
    mask_unique = torch_unique_sorted(mask)
    print(f"  mask tensor:  shape={tuple(mask.shape)}, dtype={mask.dtype}")
    print(f"  mask unique values: {mask_unique}")
    has_class_0 = 0 in mask_unique
    print(f"  class 0 (background) present in mask: {has_class_0}")
    if has_class_0:
        n0 = (mask == 0).sum().item()
        print(f"  class 0 pixel count: {n0:,} ({100.0 * n0 / mask.numel():.2f}%)")


def torch_unique_sorted(tensor) -> list[int]:
    import torch

    return torch.unique(tensor).cpu().tolist()


def main() -> None:
    print("FTW France diagnostic")
    print(f"Requested mask dir: {MASKS_DIR_USER}")

    mask_files = find_mask_files(MASKS_DIR_USER)
    print(f"\nMask files under france/masks/: {len(mask_files)}")

    if not mask_files:
        print("\nNOTE: france/masks/ does not exist or has no .tif files.")
        print(f"      FTWDataset uses: {MASKS_DIR_FTW}")
        mask_files = find_mask_files(MASKS_DIR_FTW)
        print(f"Mask files under label_masks/semantic_3class/: {len(mask_files)}")
        if not mask_files:
            raise SystemExit("No mask files found in either location.")

    first = mask_files[0]
    print_mask_report("First mask file", analyze_mask(first))

    print_average_distribution(mask_files)

    chip_id = first.stem
    analyze_image_for_chip(chip_id)

    analyze_ftw_dataset_sample()

    # Also show raw vs remapped for first semantic mask if using FTW path
    if MASKS_DIR_FTW in first.parents or first.parent == MASKS_DIR_FTW:
        with rasterio.open(first, "r") as src:
            raw = src.read(1)
        remapped = remap_ftw_mask(raw)
        raw_u = np.unique(raw).astype(int).tolist()
        remap_u = np.unique(remapped).astype(int).tolist()
        print(f"\n{'=' * 60}")
        print("Remap check on first mask (dataset.remap_ftw_mask)")
        print(f"  raw unique:      {raw_u}")
        print(f"  remapped unique: {remap_u}")
        print(f"  class 0 preserved: {np.array_equal(raw == 0, remapped == 0)}")


if __name__ == "__main__":
    main()
