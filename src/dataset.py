"""PyTorch datasets for FTW and PASTIS field boundary segmentation."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Sequence

# Larger GDAL raster cache speeds repeated GeoTIFF reads (per worker process).
if "GDAL_CACHEMAX" not in os.environ:
    os.environ["GDAL_CACHEMAX"] = "512"

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FTW_ROOT = Path(r"E:\FTW\ftw-baselines\data\ftw")
PASTIS_ROOT = Path(r"E:\PASTIS\PASTIS")

FTW_VAL_COUNTRIES = ("india",)
FTW_TEST_COUNTRIES = ("vietnam", "cambodia")
FTW_IGNORE_CLASS = 3
IGNORE_INDEX = 255

# Sentinel-2 band indices used when reducing PASTIS to FTW-compatible 4-band stacks.
PASTIS_BAND_INDICES = (1, 2, 3, 7)

RGB_BANDS = (2, 1, 0)  # R, G, B within each 4-band temporal window (B04, B03, B02)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def apply_augmentations(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    augment: bool,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply paired flips and 90-degree rotations to image (C,H,W) and mask (H,W)."""
    if not augment:
        return image, mask

    if rng.random() < 0.5:
        image = np.flip(image, axis=2).copy()
        mask = np.flip(mask, axis=1).copy()
    if rng.random() < 0.5:
        image = np.flip(image, axis=1).copy()
        mask = np.flip(mask, axis=0).copy()

    k = rng.randint(0, 3)
    if k:
        image = np.rot90(image, k, axes=(1, 2)).copy()
        mask = np.rot90(mask, k).copy()

    return image, mask


def normalize_image(image: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Per-channel z-score normalization."""
    mean = mean.reshape(-1, 1, 1)
    std = std.reshape(-1, 1, 1)
    return ((image.astype(np.float32) - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def remap_ftw_mask(mask: np.ndarray) -> np.ndarray:
    """Remap FTW ignore class 3 to 255 for CrossEntropyLoss. Class 0 is unchanged."""
    mask = mask.astype(np.int64, copy=True)
    mask[mask == FTW_IGNORE_CLASS] = IGNORE_INDEX
    return mask


FTW_CLASS_NAMES = ("background", "field", "boundary", "ignore")
FTW_EXPECTED_RAW_VALUES = {0, 1, 2, FTW_IGNORE_CLASS}


def read_ftw_mask_raw(record: dict[str, str]) -> np.ndarray:
    """Read semantic mask (H, W) from GeoTIFF without remapping."""
    with rasterio.open(record["mask"], "r") as src:
        return src.read(1)


def audit_ftw_mask_labels(
    samples: list[dict[str, str]],
    n_masks: int = 100,
    seed: int = 42,
) -> None:
    """Verify remap rules and warn on unexpected raw label values."""
    if not samples:
        return

    rng = random.Random(seed)
    chosen = rng.sample(samples, min(n_masks, len(samples)))
    unexpected_raw: set[int] = set()
    background_ok = True

    print(f"\nFTW mask audit ({len(chosen)} samples):", flush=True)
    for record in chosen:
        raw = read_ftw_mask_raw(record)
        raw_unique = set(np.unique(raw).astype(int).tolist())
        unexpected_raw |= raw_unique - FTW_EXPECTED_RAW_VALUES

        remapped = remap_ftw_mask(raw)
        remapped_unique = set(np.unique(remapped).astype(int).tolist())
        if 0 not in raw_unique and 0 not in remapped_unique:
            background_ok = False

        if raw_unique - {FTW_IGNORE_CLASS} and 0 not in raw_unique:
            # chips can be all-field; only flag systemic absence of bg in remapped space
            pass

        # Class 0 must never become ignore_index
        if np.any((raw == 0) & (remapped != 0)):
            print(
                f"  ERROR: background (0) altered for {record['chip_id']} — "
                f"raw {sorted(raw_unique)} -> remapped {sorted(remapped_unique)}",
                flush=True,
            )
            background_ok = False

        if FTW_IGNORE_CLASS in raw_unique and np.any(remapped == FTW_IGNORE_CLASS):
            print(
                f"  ERROR: ignore class 3 not remapped to 255 for {record['chip_id']}",
                flush=True,
            )

    print("  Remap rule: class 0 (background) unchanged, class 3 -> 255 only.", flush=True)
    if unexpected_raw:
        print(
            f"  WARNING: unexpected raw class values seen: {sorted(unexpected_raw)} "
            f"(expected subset of {sorted(FTW_EXPECTED_RAW_VALUES)})",
            flush=True,
        )
    else:
        print("  Raw values OK: only classes 0, 1, 2, 3 observed.", flush=True)
    if background_ok:
        print("  Background class 0 preserved after remap.", flush=True)


def sample_ftw_class_distribution(
    samples: list[dict[str, str]],
    n_masks: int = 500,
    seed: int = 42,
) -> dict[str, np.ndarray | float]:
    """Count pixels per class on raw and remapped masks (ignore excluded from training counts)."""
    if not samples:
        raise ValueError("Cannot sample class distribution from empty sample list.")

    rng = random.Random(seed)
    chosen = rng.sample(samples, min(n_masks, len(samples)))

    raw_counts = np.zeros(4, dtype=np.int64)  # 0,1,2,3
    train_counts = np.zeros(3, dtype=np.int64)  # 0,1,2 after remap (no ignore)

    for record in chosen:
        raw = read_ftw_mask_raw(record)
        for cls in range(4):
            raw_counts[cls] += int(np.sum(raw == cls))

        remapped = remap_ftw_mask(raw)
        valid = remapped != IGNORE_INDEX
        for cls in range(3):
            train_counts[cls] += int(np.sum((remapped == cls) & valid))

    raw_total = raw_counts.sum()
    train_total = train_counts.sum()
    return {
        "n_masks": len(chosen),
        "raw_counts": raw_counts,
        "raw_freq": raw_counts / max(raw_total, 1),
        "train_counts": train_counts,
        "train_freq": train_counts / max(train_total, 1),
    }


def inverse_frequency_class_weights(
    class_counts: np.ndarray,
    num_classes: int = 3,
    eps: float = 1e-6,
) -> np.ndarray:
    """Inverse-frequency weights normalized to mean 1."""
    counts = class_counts[:num_classes].astype(np.float64)
    freqs = counts / max(counts.sum(), 1.0)
    weights = 1.0 / (freqs + eps)
    weights = weights * (num_classes / weights.sum())
    return weights.astype(np.float32)


def stretch_rgb(rgb: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0) -> np.ndarray:
    lo, hi = np.percentile(rgb, (low_pct, high_pct))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((rgb.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def compute_ftw_norm_stats(
    samples: list[dict[str, str]],
    sample_size: int = 128,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate per-channel mean/std from a random subset of FTW training chips."""
    if not samples:
        raise ValueError("Cannot compute normalization stats from an empty sample list.")

    rng = random.Random(seed)
    chosen = rng.sample(samples, min(sample_size, len(samples)))

    channel_sum = np.zeros(8, dtype=np.float64)
    channel_sq_sum = np.zeros(8, dtype=np.float64)
    pixel_count = 0

    for record in chosen:
        with rasterio.open(record["window_b"]) as src:
            window_b = src.read().astype(np.float64)
        with rasterio.open(record["window_a"]) as src:
            window_a = src.read().astype(np.float64)
        stacked = np.concatenate([window_b, window_a], axis=0)

        channel_sum += stacked.reshape(stacked.shape[0], -1).sum(axis=1)
        channel_sq_sum += (stacked.reshape(stacked.shape[0], -1) ** 2).sum(axis=1)
        pixel_count += stacked.shape[1] * stacked.shape[2]

    mean = (channel_sum / pixel_count).astype(np.float32)
    var = channel_sq_sum / pixel_count - mean.astype(np.float64) ** 2
    std = np.sqrt(np.maximum(var, 0.0)).astype(np.float32)
    return mean, std


def load_pastis_norm_stats(norm_path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load fold-specific mean/std for 8-channel PASTIS stacks."""
    with norm_path.open() as f:
        raw = json.load(f)

    stats: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for fold in range(1, 6):
        fold_stats = raw[f"Fold_{fold}"]
        mean10 = np.array(fold_stats["mean"], dtype=np.float32)
        std10 = np.array(fold_stats["std"], dtype=np.float32)
        mean4 = mean10[list(PASTIS_BAND_INDICES)]
        std4 = std10[list(PASTIS_BAND_INDICES)]
        stats[fold] = (np.concatenate([mean4, mean4]), np.concatenate([std4, std4]))
    return stats


def load_pastis_patch_folds(metadata_path: Path) -> dict[int, int]:
    with metadata_path.open() as f:
        metadata = json.load(f)
    return {
        int(feature["properties"]["ID_PATCH"]): int(feature["properties"]["Fold"])
        for feature in metadata["features"]
    }


# ---------------------------------------------------------------------------
# FTW
# ---------------------------------------------------------------------------
def read_ftw_chip_arrays(record: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    """Read stacked S2 windows (8, H, W) and mask (H, W) from GeoTIFF paths."""
    with rasterio.open(record["window_b"], "r") as src:
        window_b = src.read()
    with rasterio.open(record["window_a"], "r") as src:
        window_a = src.read()
    image = np.concatenate([window_b, window_a], axis=0)

    with rasterio.open(record["mask"], "r") as src:
        mask = src.read(1)
    return image, mask


# Reuse computed mean/std when building val/test from the same FTW root.
_FTW_NORM_STATS_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


class FTWDataset(Dataset):
    """FTW field boundary segmentation dataset."""

    def __init__(
        self,
        root: str | Path = FTW_ROOT,
        split: str = "train",
        augment: bool = False,
        norm_sample_size: int = 128,
        seed: int = 42,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        cache_in_memory: bool | None = None,
        cache_max_samples: int = 5000,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.augment = augment
        self.seed = seed

        if split == "train":
            excluded = set(FTW_VAL_COUNTRIES + FTW_TEST_COUNTRIES)
            countries = sorted(
                p.name for p in self.root.iterdir() if p.is_dir() and p.name not in excluded
            )
        elif split == "val":
            countries = list(FTW_VAL_COUNTRIES)
        elif split == "test":
            countries = list(FTW_TEST_COUNTRIES)
        else:
            raise ValueError(f"Invalid split {split!r}; expected train, val, or test.")

        self.samples = self._collect_samples(countries)
        if not self.samples:
            raise RuntimeError(f"No FTW samples found for split={split!r} under {self.root}")

        if split == "train":
            audit_ftw_mask_labels(self.samples, n_masks=100, seed=seed)

        if mean is None or std is None:
            cache_key = f"{self.root.resolve()}::train_norm"
            if cache_key in _FTW_NORM_STATS_CACHE:
                self.mean, self.std = _FTW_NORM_STATS_CACHE[cache_key]
            elif split == "train":
                self.mean, self.std = compute_ftw_norm_stats(
                    self.samples, sample_size=norm_sample_size, seed=seed
                )
                _FTW_NORM_STATS_CACHE[cache_key] = (self.mean, self.std)
            else:
                train_countries = sorted(
                    p.name
                    for p in self.root.iterdir()
                    if p.is_dir()
                    and p.name not in set(FTW_VAL_COUNTRIES + FTW_TEST_COUNTRIES)
                )
                train_samples = self._collect_samples(train_countries)
                self.mean, self.std = compute_ftw_norm_stats(
                    train_samples, sample_size=norm_sample_size, seed=seed
                )
                _FTW_NORM_STATS_CACHE[cache_key] = (self.mean, self.std)
        else:
            self.mean = np.asarray(mean, dtype=np.float32)
            self.std = np.asarray(std, dtype=np.float32)

        if cache_in_memory is None:
            cache_in_memory = len(self.samples) <= cache_max_samples
        self.cache_in_memory = cache_in_memory
        self._chip_cache: dict[int, tuple[np.ndarray, np.ndarray]] | None = None
        if self.cache_in_memory:
            self._preload_chip_cache()

    def _collect_samples(self, countries: Sequence[str]) -> list[dict[str, str]]:
        samples: list[dict[str, str]] = []
        for country in countries:
            country_root = self.root / country
            mask_dir = country_root / "label_masks" / "semantic_3class"
            window_a_dir = country_root / "s2_images" / "window_a"
            window_b_dir = country_root / "s2_images" / "window_b"
            if not mask_dir.is_dir():
                continue

            for mask_path in sorted(mask_dir.glob("*.tif")):
                chip_id = mask_path.stem
                window_a = window_a_dir / f"{chip_id}.tif"
                window_b = window_b_dir / f"{chip_id}.tif"
                if window_a.exists() and window_b.exists():
                    samples.append(
                        {
                            "country": country,
                            "chip_id": chip_id,
                            "window_a": str(window_a),
                            "window_b": str(window_b),
                            "mask": str(mask_path),
                        }
                    )
        return samples

    def _preload_chip_cache(self) -> None:
        """Load chips as numpy arrays only (no torch tensors — safe for Windows workers)."""
        print(
            f"FTWDataset [{self.split}]: caching {len(self.samples)} chips in memory "
            "(numpy only)...",
            flush=True,
        )
        cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for index, record in enumerate(self.samples):
            image, mask = self._load_chip_normalized(record)
            if not isinstance(image, np.ndarray) or not isinstance(mask, np.ndarray):
                raise TypeError("In-memory cache must store numpy arrays only.")
            cache[index] = (image, mask)
            if (index + 1) % 500 == 0 or index + 1 == len(self.samples):
                print(f"  cached {index + 1}/{len(self.samples)}", flush=True)
        self._chip_cache = cache
        print(f"FTWDataset [{self.split}]: in-memory cache ready.", flush=True)

    def _load_chip_normalized(self, record: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
        image, mask = read_ftw_chip_arrays(record)
        image = normalize_image(image, self.mean, self.std).astype(np.float32, copy=False)
        mask = remap_ftw_mask(mask).astype(np.int64, copy=False)
        return image, mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.samples[index]
        rng = random.Random(self.seed + index) if self.augment else None

        if self._chip_cache is not None:
            image, mask = self._chip_cache[index]
            image = np.array(image, copy=True)
            mask = np.array(mask, copy=True)
        else:
            image, mask = self._load_chip_normalized(record)

        if self.augment:
            image, mask = apply_augmentations(image, mask, augment=True, rng=rng)

        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask).long(),
            "country": record["country"],
            "chip_id": record["chip_id"],
        }


# ---------------------------------------------------------------------------
# PASTIS
# ---------------------------------------------------------------------------
class PASTISDataset(Dataset):
    """PASTIS field boundary segmentation dataset (2-date reduced S2 + boundary masks)."""

    def __init__(
        self,
        root: str | Path = PASTIS_ROOT,
        folds: Sequence[int] = (1, 2, 3, 4),
        augment: bool = False,
        seed: int = 42,
        data_subdir: str = "DATA_S2_2DATE",
    ) -> None:
        self.root = Path(root)
        self.augment = augment
        self.seed = seed
        self.data_subdir = data_subdir
        self.image_dir = self.root / data_subdir
        self.mask_dir = self.root / "BOUNDARY_MASKS"

        if not self.image_dir.is_dir():
            raise FileNotFoundError(
                f"PASTIS image directory not found: {self.image_dir}. "
                "Run pastis_temporal_reduction.py first."
            )
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(
                f"PASTIS mask directory not found: {self.mask_dir}. "
                "Run pastis_preprocess.py first."
            )

        patch_folds = load_pastis_patch_folds(self.root / "metadata.geojson")
        fold_set = set(folds)
        self.norm_stats = load_pastis_norm_stats(self.root / "NORM_S2_patch.json")

        self.samples: list[dict[str, Any]] = []
        for image_path in sorted(self.image_dir.glob("S2_*.npy")):
            patch_id = int(image_path.stem.split("_")[1])
            fold = patch_folds.get(patch_id)
            if fold is None or fold not in fold_set:
                continue

            mask_path = self.mask_dir / f"ParcelIDs_{patch_id}.npy"
            if not mask_path.exists():
                continue

            self.samples.append(
                {
                    "patch_id": patch_id,
                    "fold": fold,
                    "image": str(image_path),
                    "mask": str(mask_path),
                }
            )

        if not self.samples:
            raise RuntimeError(
                f"No PASTIS samples found for folds={tuple(folds)} under {self.root}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.samples[index]
        rng = random.Random(self.seed + index) if self.augment else None

        image = np.load(record["image"])
        mask = np.load(record["mask"])

        if image.shape != (8, 128, 128):
            raise ValueError(
                f"Expected image shape (8, 128, 128) for patch {record['patch_id']}, "
                f"got {image.shape}"
            )

        if self.augment:
            image, mask = apply_augmentations(image, mask, augment=True, rng=rng)

        mean, std = self.norm_stats[record["fold"]]
        image = normalize_image(image, mean, std)
        mask = mask.astype(np.int64)

        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask).long(),
            "patch_id": record["patch_id"],
            "fold": record["fold"],
        }


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------
class CombinedDataset(Dataset):
    """Sample jointly from FTW and PASTIS according to a configurable ratio."""

    def __init__(
        self,
        ftw_dataset: FTWDataset,
        pastis_dataset: PASTISDataset,
        ftw_ratio: float = 0.7,
        length: int | None = None,
        seed: int = 42,
        pastis_data_subdir: str | None = None,
    ) -> None:
        if not 0.0 < ftw_ratio < 1.0:
            raise ValueError(f"ftw_ratio must be in (0, 1), got {ftw_ratio}")

        self.ftw = ftw_dataset
        self.pastis = pastis_dataset
        self.pastis_data_subdir = pastis_data_subdir or pastis_dataset.data_subdir
        self.ftw_ratio = ftw_ratio
        self.seed = seed

        if length is None:
            ftw_len = len(self.ftw) / ftw_ratio
            pastis_len = len(self.pastis) / (1.0 - ftw_ratio)
            self.length = int(max(ftw_len, pastis_len))
        else:
            self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = random.Random(self.seed + index)
        if rng.random() < self.ftw_ratio:
            sample_idx = rng.randrange(len(self.ftw))
            sample = self.ftw[sample_idx]
            sample = dict(sample)
            sample["source"] = "ftw"
            return sample

        sample_idx = rng.randrange(len(self.pastis))
        sample = self.pastis[sample_idx]
        sample = dict(sample)
        sample["source"] = "pastis"
        return sample


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def visualize_sample(
    sample: dict[str, Any],
    title: str,
    out_path: Path,
) -> None:
    image = sample["image"].numpy()
    mask = sample["mask"].numpy()

    rgb = image[list(RGB_BANDS), :, :].transpose(1, 2, 0)
    rgb_display = stretch_rgb(rgb)

    boundary_cmap = plt.matplotlib.colors.ListedColormap(["#000000", "#2ca02c", "#d62728"])
    boundary_norm = plt.matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], boundary_cmap.N)

    display_mask = np.where(mask == IGNORE_INDEX, np.nan, mask)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(rgb_display)
    axes[0].set_title(f"{title}\nRGB (first temporal window)")
    axes[0].axis("off")

    axes[1].imshow(display_mask, cmap=boundary_cmap, norm=boundary_norm)
    axes[1].set_title("Boundary mask (0=bg, 1=field, 2=boundary)")
    axes[1].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _print_sample_stats(name: str, sample: dict[str, Any]) -> None:
    image = sample["image"]
    mask = sample["mask"]
    print(f"\n{name} sample:")
    print(f"  image shape: {tuple(image.shape)}, dtype: {image.dtype}")
    print(
        f"  image range: [{image.min().item():.4f}, {image.max().item():.4f}]"
    )
    unique_vals = torch.unique(mask).tolist()
    unique_vals = sorted(v for v in unique_vals if v != IGNORE_INDEX)
    print(f"  mask shape:  {tuple(mask.shape)}, dtype: {mask.dtype}")
    print(
        f"  mask range:  [{mask.min().item()}, {mask.max().item()}], "
        f"unique (excl. ignore): {unique_vals}"
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def main() -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FTWDataset")
    print("=" * 60)
    ftw_train = FTWDataset(split="train", augment=True)
    ftw_val = FTWDataset(split="val", mean=ftw_train.mean, std=ftw_train.std)
    ftw_test = FTWDataset(split="test", mean=ftw_train.mean, std=ftw_train.std)
    print(f"train: {len(ftw_train)} | val: {len(ftw_val)} | test: {len(ftw_test)}")
    print(f"FTW norm mean: {ftw_train.mean.round(2).tolist()}")
    print(f"FTW norm std:  {ftw_train.std.round(2).tolist()}")

    ftw_sample = ftw_train[0]
    _print_sample_stats("FTW train", ftw_sample)
    visualize_sample(ftw_sample, "FTW", output_dir / "dataset_ftw_sample.png")
    print(f"Saved {output_dir / 'dataset_ftw_sample.png'}")

    print("\n" + "=" * 60)
    print("PASTISDataset")
    print("=" * 60)
    pastis_train = PASTISDataset(folds=(1, 2, 3, 4), augment=True)
    pastis_val = PASTISDataset(folds=(5,), augment=False)
    print(f"train folds 1-4: {len(pastis_train)} | val fold 5: {len(pastis_val)}")

    pastis_sample = pastis_train[0]
    _print_sample_stats("PASTIS train", pastis_sample)
    visualize_sample(pastis_sample, "PASTIS", output_dir / "dataset_pastis_sample.png")
    print(f"Saved {output_dir / 'dataset_pastis_sample.png'}")

    print("\n" + "=" * 60)
    print("CombinedDataset (0.7 FTW + 0.3 PASTIS)")
    print("=" * 60)
    combined = CombinedDataset(ftw_train, pastis_train, ftw_ratio=0.7, length=1000)
    print(f"virtual epoch length: {len(combined)}")

    source_counts = {"ftw": 0, "pastis": 0}
    for i in range(len(combined)):
        source_counts[combined[i]["source"]] += 1
    print(
        f"source mix over {len(combined)} draws: "
        f"FTW={source_counts['ftw']} ({100 * source_counts['ftw'] / len(combined):.1f}%), "
        f"PASTIS={source_counts['pastis']} ({100 * source_counts['pastis'] / len(combined):.1f}%)"
    )

    combined_sample = combined[0]
    _print_sample_stats(f"Combined ({combined_sample['source']})", combined_sample)

    print("\nDone.")


if __name__ == "__main__":
    main()
