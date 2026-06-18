#!/usr/bin/env python3
"""Train field-boundary segmentation models on FTW and/or PASTIS."""

from __future__ import annotations

import argparse
import csv
import functools
import multiprocessing as mp
import platform
import subprocess
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset import (
    FTW_CLASS_NAMES,
    IGNORE_INDEX,
    CombinedDataset,
    FTWDataset,
    PASTISDataset,
    sample_ftw_class_distribution,
)

CLASS_WEIGHT_MASK_SAMPLES = 200
from models import NUM_CLASSES, get_model, segmentation_logits

BOUNDARY_CLASS = 2
DIST_LOSS_WEIGHT = 0.3

LOSS_WINDOW = 20
PLATEAU_WARN_EPOCHS = 10
TIMING_PROBE_BATCHES = 5
VAL_TIME_FACTOR = 0.25  # val forward-only vs train step (probe is train-like)
BATCH_SIZE_CANDIDATES = [8, 16, 24, 32, 48]

PASTIS_STRATEGY_DATA_SUBDIRS = {
    "ndvi_minmax": "DATA_S2_2DATE",
    "random": "DATA_S2_2DATE_random",
    "first_last": "DATA_S2_2DATE_first_last",
}


def pastis_data_subdir_for_strategy(strategy: str) -> str:
    if strategy not in PASTIS_STRATEGY_DATA_SUBDIRS:
        choices = ", ".join(PASTIS_STRATEGY_DATA_SUBDIRS)
        raise ValueError(f"Unknown pastis_strategy {strategy!r}; expected one of: {choices}")
    return PASTIS_STRATEGY_DATA_SUBDIRS[strategy]

# ---------------------------------------------------------------------------
# Config (edit defaults here or override via CLI)
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    model_name: str = "unet"
    dataset: str = "ftw"
    ftw_ratio: float = 0.7
    epochs: int = 50
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    prefetch_factor: int = 4
    persistent_workers: bool = True
    pin_memory: bool = platform.system() != "Windows"
    output_dir: str = r"E:\FTW\runs"
    resume: str = ""
    mixed_precision: bool = True
    train_size: int = 256
    seed: int = 42
    class_weights: tuple[float, float, float] | None = None
    grad_accumulation_steps: int = 1
    batch_size_finder: bool = False
    compile_model: bool = True
    cache_ftw_in_memory: bool | None = None
    use_dist_loss: bool | None = None
    dist_loss_weight: float = DIST_LOSS_WEIGHT
    pastis_strategy: str = "ndvi_minmax"


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train FTW/PASTIS segmentation models")
    parser.add_argument(
        "--model_name",
        default=TrainConfig.model_name,
        choices=["unet", "segformer", "utae", "utae_dist", "unet_concat", "tcbm_unet"],
    )
    parser.add_argument("--dataset", default=TrainConfig.dataset, choices=["ftw", "pastis", "combined"])
    parser.add_argument("--ftw_ratio", type=float, default=TrainConfig.ftw_ratio)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--weight_decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--num_workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--prefetch_factor", type=int, default=TrainConfig.prefetch_factor)
    parser.add_argument(
        "--persistent_workers",
        action=argparse.BooleanOptionalAction,
        default=TrainConfig.persistent_workers,
    )
    parser.add_argument(
        "--pin_memory",
        action=argparse.BooleanOptionalAction,
        default=TrainConfig.pin_memory,
    )
    parser.add_argument("--grad_accumulation_steps", type=int, default=TrainConfig.grad_accumulation_steps)
    parser.add_argument("--batch_size_finder", action="store_true")
    parser.add_argument(
        "--compile_model",
        action=argparse.BooleanOptionalAction,
        default=TrainConfig.compile_model,
    )
    parser.add_argument(
        "--cache_ftw_in_memory",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--output_dir", default=TrainConfig.output_dir)
    parser.add_argument("--resume", default=TrainConfig.resume)
    parser.add_argument("--mixed_precision", action=argparse.BooleanOptionalAction, default=TrainConfig.mixed_precision)
    parser.add_argument("--train_size", type=int, default=TrainConfig.train_size)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument(
        "--class_weights",
        nargs=3,
        type=float,
        default=None,
        metavar=("W_BG", "W_FIELD", "W_BOUNDARY"),
        help="Override automatic inverse-frequency weights (background, field, boundary)",
    )
    parser.add_argument(
        "--use_dist_loss",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Auxiliary distance-transform MSE (default: on for utae_dist, off otherwise)",
    )
    parser.add_argument(
        "--pastis_strategy",
        default=TrainConfig.pastis_strategy,
        choices=list(PASTIS_STRATEGY_DATA_SUBDIRS),
        help="PASTIS temporal reduction strategy (selects DATA_S2_2DATE_* folder)",
    )
    args = parser.parse_args()
    return TrainConfig(
        model_name=args.model_name,
        dataset=args.dataset,
        ftw_ratio=args.ftw_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        pin_memory=args.pin_memory,
        grad_accumulation_steps=max(1, args.grad_accumulation_steps),
        batch_size_finder=args.batch_size_finder,
        compile_model=args.compile_model,
        cache_ftw_in_memory=args.cache_ftw_in_memory,
        output_dir=args.output_dir,
        resume=args.resume,
        mixed_precision=args.mixed_precision,
        train_size=args.train_size,
        seed=args.seed,
        class_weights=tuple(args.class_weights) if args.class_weights is not None else None,
        use_dist_loss=args.use_dist_loss,
        pastis_strategy=args.pastis_strategy,
    )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def compute_distance_transform_targets(
    masks: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """
    Per-pixel distance to nearest boundary (class 2), normalized to [0, 1].
    Background (class 0) pixels are set to 0.
    """
    b, h, w = masks.shape
    device = masks.device
    out = np.zeros((b, h, w), dtype=np.float32)
    masks_np = masks.detach().cpu().numpy()

    for i in range(b):
        m = masks_np[i]
        boundary = m == BOUNDARY_CLASS
        edt_mask = np.ones((h, w), dtype=bool)
        edt_mask[boundary] = False
        raw = distance_transform_edt(edt_mask).astype(np.float32)
        raw[m == 0] = 0.0
        raw[m == ignore_index] = 0.0
        mx = float(raw.max())
        if mx > 0:
            raw /= mx
        out[i] = raw

    return torch.from_numpy(out).to(device=device)


class CombinedSegLoss(nn.Module):
    """Cross-entropy segmentation loss plus optional distance-transform MSE."""

    def __init__(
        self,
        ce_loss: nn.Module,
        *,
        use_dist_loss: bool = False,
        dist_loss_weight: float = DIST_LOSS_WEIGHT,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        super().__init__()
        self.ce_loss = ce_loss
        self.use_dist_loss = use_dist_loss
        self.dist_loss_weight = dist_loss_weight
        self.ignore_index = ignore_index

    def forward(
        self,
        outputs: torch.Tensor | dict[str, torch.Tensor],
        masks: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(outputs, dict):
            logits = outputs["seg"]
            dist_pred = outputs.get("dist")
        else:
            logits = outputs
            dist_pred = None

        loss = self.ce_loss(logits, masks)

        if self.use_dist_loss and dist_pred is not None:
            dist_tgt = compute_distance_transform_targets(masks, self.ignore_index)
            valid = masks != self.ignore_index
            if valid.any():
                loss = loss + self.dist_loss_weight * F.mse_loss(
                    dist_pred[valid], dist_tgt[valid]
                )
        return loss


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_iou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Per-class IoU on valid pixels only (targets != ignore_index).

    Returns per-class IoU, mean IoU, and per-class intersection/union counts
    so callers can aggregate across batches globally (not average batch IoUs).
    """
    if preds.shape != targets.shape:
        raise ValueError(
            f"pred/target shape mismatch: preds {tuple(preds.shape)} vs "
            f"targets {tuple(targets.shape)}"
        )

    preds = preds.detach().reshape(-1).long()
    targets = targets.detach().reshape(-1).long()
    valid = targets != ignore_index
    preds = preds[valid]
    targets = targets[valid]

    intersect = np.zeros(num_classes, dtype=np.int64)
    union = np.zeros(num_classes, dtype=np.int64)
    for cls in range(num_classes):
        pred_c = preds == cls
        target_c = targets == cls
        intersect[cls] = int((pred_c & target_c).sum().item())
        union[cls] = int((pred_c | target_c).sum().item())

    ious = np.full(num_classes, np.nan, dtype=np.float64)
    for cls in range(num_classes):
        if union[cls] > 0:
            ious[cls] = intersect[cls] / union[cls]

    mean_iou = float(np.nanmean(ious))
    return ious, mean_iou, intersect, union


def debug_first_val_batch(
    logits: torch.Tensor,
    preds: torch.Tensor,
    masks: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> None:
    """One-time debug dump for the first validation batch."""
    log("\n--- IoU debug (first validation batch only) ---")
    log(f"  logits shape:      {tuple(logits.shape)}")
    log(
        f"  logits range:      [{logits.min().item():.4f}, {logits.max().item():.4f}] "
        f"(argmax dim=1 over {logits.shape[1]} classes)"
    )
    log(f"  preds shape:       {tuple(preds.shape)} (dtype={preds.dtype})")
    log(f"  masks shape:       {tuple(masks.shape)} (dtype={masks.dtype})")

    preds_flat = preds.reshape(-1).long()
    masks_flat = masks.reshape(-1).long()
    valid = masks_flat != ignore_index
    n_valid = int(valid.sum().item())
    n_total = masks_flat.numel()

    log(f"  ignore_index:      {ignore_index}")
    log(f"  valid pixels:      {n_valid:,} / {n_total:,} ({100.0 * n_valid / n_total:.2f}%)")

    def class_distribution(tensor: torch.Tensor, label: str) -> None:
        unique, counts = torch.unique(tensor, return_counts=True)
        total = counts.sum().item()
        log(f"  {label} unique classes: {unique.tolist()}")
        for u, c in zip(unique.tolist(), counts.tolist()):
            pct = 100.0 * c / max(total, 1)
            log(f"    class {int(u):3d}: {c:>10,} px ({pct:6.2f}%)")

    class_distribution(preds_flat, "Predicted (all pixels, argmax)")
    class_distribution(masks_flat, "Ground truth (all pixels)")
    if n_valid > 0:
        class_distribution(preds_flat[valid], "Predicted (valid pixels only)")
        class_distribution(masks_flat[valid], "Ground truth (valid pixels only)")
    else:
        log("  WARNING: no valid pixels in batch — IoU will be undefined.")

    batch_ious, _, intersect, union = compute_iou(preds, masks, NUM_CLASSES, ignore_index)
    log("  Per-class IoU (this batch, valid pixels only):")
    for cls, name in enumerate(FTW_CLASS_NAMES[:3]):
        if np.isnan(batch_ious[cls]):
            log(f"    {name:10s} (class {cls}): n/a (union=0)")
        else:
            log(
                f"    {name:10s} (class {cls}): {batch_ious[cls]:.4f} "
                f"(intersect={intersect[cls]:,}, union={union[cls]:,})"
            )
    log("--- end IoU debug ---\n")


def compute_pixel_accuracy(
    preds: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = 255,
) -> float:
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)
    valid = targets != ignore_index
    if valid.sum() == 0:
        return 0.0
    correct = (preds[valid] == targets[valid]).sum().item()
    return correct / valid.sum().item()


# ---------------------------------------------------------------------------
# Data / training helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random_seed = seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)


def resize_sample(
    image: torch.Tensor,
    mask: torch.Tensor,
    size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize one sample to (C, size, size) and (size, size) mask."""
    if image.shape[-1] == size and image.shape[-2] == size:
        return image, mask

    image = torch.nn.functional.interpolate(
        image.unsqueeze(0),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    mask = torch.nn.functional.interpolate(
        mask.unsqueeze(0).unsqueeze(0).float(),
        size=(size, size),
        mode="nearest",
    ).squeeze(0).squeeze(0).long()
    return image, mask


def collate_batch(
    samples: list[dict],
    train_size: int = 256,
) -> dict[str, torch.Tensor]:
    """Stack batch; upsample PASTIS (128) to train_size when mixed with FTW (256)."""
    images: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for sample in samples:
        image, mask = resize_sample(sample["image"], sample["mask"], train_size)
        images.append(image)
        masks.append(mask)
    return {"image": torch.stack(images), "mask": torch.stack(masks)}


def resize_batch(
    images: torch.Tensor,
    masks: torch.Tensor,
    size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if images.shape[-1] == size and images.shape[-2] == size:
        return images, masks

    images = torch.nn.functional.interpolate(
        images, size=(size, size), mode="bilinear", align_corners=False
    )
    masks = torch.nn.functional.interpolate(
        masks.unsqueeze(1).float(),
        size=(size, size),
        mode="nearest",
    ).squeeze(1).long()
    return images, masks


def build_train_dataset(config: TrainConfig) -> tuple[torch.utils.data.Dataset, FTWDataset]:
    """Return training dataset and FTW train set (for shared normalization stats)."""
    ftw_train = FTWDataset(
        split="train",
        augment=True,
        seed=config.seed,
        cache_in_memory=config.cache_ftw_in_memory,
    )

    if config.dataset == "ftw":
        return ftw_train, ftw_train

    pastis_data_subdir = pastis_data_subdir_for_strategy(config.pastis_strategy)
    pastis_train = PASTISDataset(
        folds=(1, 2, 3, 4),
        augment=True,
        seed=config.seed,
        data_subdir=pastis_data_subdir,
    )

    if config.dataset == "pastis":
        return pastis_train, ftw_train

    combined = CombinedDataset(
        ftw_train,
        pastis_train,
        ftw_ratio=config.ftw_ratio,
        seed=config.seed,
        pastis_data_subdir=pastis_data_subdir,
    )
    return combined, ftw_train


def dataloader_kwargs(config: TrainConfig) -> dict:
    """DataLoader settings tuned for Windows (spawn) and Linux."""
    kwargs: dict = {
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory and torch.cuda.is_available(),
        "collate_fn": functools.partial(collate_batch, train_size=config.train_size),
    }
    if config.num_workers > 0:
        if platform.system() == "Windows":
            kwargs["multiprocessing_context"] = mp.get_context("spawn")
        if config.num_workers >= 2:
            kwargs["prefetch_factor"] = config.prefetch_factor
            kwargs["persistent_workers"] = config.persistent_workers
    return kwargs


def compute_inverse_frequency_weights(
    class_counts: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> tuple[float, float, float]:
    """
    weight_c = total_pixels / (n_classes * count_c), then scale so min weight = 1.0.
    """
    counts = class_counts[:num_classes].astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return (1.0, 1.0, 1.0)

    raw = total / (num_classes * np.maximum(counts, 1.0))
    raw = raw / raw.min()
    return (float(raw[0]), float(raw[1]), float(raw[2]))


def resolve_class_weights(
    ftw_train: FTWDataset,
    config: TrainConfig,
) -> tuple[tuple[float, float, float], dict]:
    """Compute loss weights from sampled training masks; print distribution."""
    dist = sample_ftw_class_distribution(
        ftw_train.samples,
        n_masks=CLASS_WEIGHT_MASK_SAMPLES,
        seed=config.seed,
    )

    log(f"\nClass distribution ({CLASS_WEIGHT_MASK_SAMPLES} sampled FTW training masks):")
    log(f"  Masks sampled: {dist['n_masks']}")
    log("  Training pixel counts (after remap, ignore=255 excluded):")
    for i, name in enumerate(FTW_CLASS_NAMES[:3]):
        cnt = int(dist["train_counts"][i])
        pct = 100.0 * dist["train_freq"][i]
        log(f"    {name:10s} (class {i}): {cnt:>12,} pixels ({pct:5.2f}%)")

    if config.class_weights is not None:
        weights = config.class_weights
        log("\nCrossEntropyLoss class weights (manual --class_weights):")
    else:
        weights = compute_inverse_frequency_weights(dist["train_counts"])
        log("\nCrossEntropyLoss class weights (inverse frequency, min=1.0):")

    log(
        f"  background (0) = {weights[0]:.4f}\n"
        f"  field      (1) = {weights[1]:.4f}\n"
        f"  boundary   (2) = {weights[2]:.4f}"
    )

    return weights, dist


def log_val_prediction_distribution(pred_counts: np.ndarray) -> None:
    """Print predicted class percentages across the full validation set."""
    total = pred_counts.sum()
    if total == 0:
        return
    log("  Val prediction distribution (all pixels):")
    for cls, name in enumerate(FTW_CLASS_NAMES[:3]):
        pct = 100.0 * pred_counts[cls] / total
        log(f"    pred {name:10s} (class {cls}): {pct:6.2f}%")


def build_val_loader(ftw_train: FTWDataset, config: TrainConfig) -> DataLoader:
    val_set = FTWDataset(
        split="val",
        augment=False,
        mean=ftw_train.mean,
        std=ftw_train.std,
        seed=config.seed,
        cache_in_memory=config.cache_ftw_in_memory,
    )
    return DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        **dataloader_kwargs(config),
    )


def build_train_loader(train_set: torch.utils.data.Dataset, config: TrainConfig) -> DataLoader:
    return DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=len(train_set) >= config.batch_size * config.grad_accumulation_steps,
        **dataloader_kwargs(config),
    )


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    epoch: int,
    best_miou: float,
    config: TrainConfig,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_miou": best_miou,
            "config": asdict(config),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
) -> tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return int(ckpt.get("epoch", 0)), float(ckpt.get("best_miou", 0.0))


def log(msg: str) -> None:
    print(msg, flush=True)


def configure_cuda_backends() -> None:
    """Enable cuDNN autotune and TF32 for faster matmul/conv on Ampere+ GPUs."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def get_gpu_utilization_pct(device: torch.device) -> int | None:
    """Best-effort GPU utilization % (NVML API or nvidia-smi)."""
    if device.type != "cuda":
        return None

    if hasattr(torch.cuda, "utilization"):
        try:
            util = torch.cuda.utilization(device)
            if isinstance(util, (int, float)):
                return int(util)
        except Exception:
            pass

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return int(result.stdout.strip().split("\n")[0].strip())
    except Exception:
        return None


def maybe_compile_model(model: nn.Module) -> nn.Module:
    if not hasattr(torch, "compile"):
        log("torch.compile not available; using eager mode.")
        return model
    try:
        compiled = torch.compile(model)
        log("Model wrapped with torch.compile().")
        return compiled
    except Exception as exc:
        log(f"torch.compile failed ({exc}); using eager mode.")
        return model


def find_max_batch_size(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: TrainConfig,
) -> int:
    """Binary search over candidate batch sizes using one forward+backward pass."""
    if device.type != "cuda":
        log("batch_size_finder: CUDA not available; keeping batch_size unchanged.")
        return config.batch_size

    log("batch_size_finder: probing batch sizes " + str(BATCH_SIZE_CANDIDATES))
    max_ok = BATCH_SIZE_CANDIDATES[0]

    for batch_size in BATCH_SIZE_CANDIDATES:
        torch.cuda.empty_cache()
        try:
            images = torch.randn(
                batch_size, 8, config.train_size, config.train_size, device=device
            )
            masks = torch.randint(
                0, NUM_CLASSES, (batch_size, config.train_size, config.train_size), device=device
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                enabled=config.mixed_precision,
            ):
                logits = model(images)
                loss = criterion(logits, masks) / config.grad_accumulation_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            max_ok = batch_size
            log(f"  batch_size={batch_size}: OK")
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                log(f"  batch_size={batch_size}: OOM")
                torch.cuda.empty_cache()
                break
            raise

    log(f"batch_size_finder: selected batch_size={max_ok}")
    return max_ok


def format_duration(seconds: float) -> str:
    """Format seconds as Xh Ym Zs or Xm Ys."""
    if seconds < 0 or not np.isfinite(seconds):
        return "unknown"
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def gpu_memory_gb(device: torch.device) -> tuple[float, float]:
    """Return (reserved_gb, total_gb)."""
    if device.type != "cuda":
        return 0.0, 0.0
    reserved = torch.cuda.memory_reserved(device) / (1024**3)
    total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    return reserved, total


def forward_backward_batch(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    config: TrainConfig,
    *,
    check_nan: bool = False,
) -> float:
    """Forward + backward for one batch (loss scaled for gradient accumulation)."""
    images = batch["image"].to(device, non_blocking=True)
    masks = batch["mask"].to(device, non_blocking=True)
    images, masks = resize_batch(images, masks, config.train_size)

    with torch.autocast(
        device_type=device.type,
        enabled=config.mixed_precision and device.type == "cuda",
    ):
        outputs = model(images)
        loss = criterion(outputs, masks) / config.grad_accumulation_steps

    if check_nan:
        logits = segmentation_logits(outputs)
        if torch.isnan(logits).any():
            print("WARNING: NaN in logits after forward pass")
        if torch.isnan(loss).any():
            print("WARNING: NaN in loss")

    if scaler.is_enabled():
        scaler.scale(loss).backward()
    else:
        loss.backward()

    return loss.item() * config.grad_accumulation_steps


def training_module(model: nn.Module) -> nn.Module:
    """Unwrap torch.compile() so parameter names match the source module."""
    return getattr(model, "_orig_mod", model)


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def print_tcbm_gradient_check(model: nn.Module) -> None:
    """Report mean |gradient| per TCBMUNet block after the first backward pass."""
    group_grads: dict[str, list[torch.Tensor]] = {
        "encoder": [],
        "TCBMFusion": [],
        "decoder": [],
    }
    for name, param in training_module(model).named_parameters():
        if param.grad is None:
            continue
        if name.startswith("date_encoder"):
            group_grads["encoder"].append(param.grad)
        elif name.startswith("temporal_fuse"):
            group_grads["TCBMFusion"].append(param.grad)
        elif name.startswith("decoder") or name.startswith("seg_head"):
            group_grads["decoder"].append(param.grad)

    log("TCBM gradient check (first batch):")
    for group_name, grads in group_grads.items():
        if not grads:
            log(f"  {group_name}: no gradients (possible flow issue)")
            continue
        mean_abs = torch.stack([g.detach().abs().mean() for g in grads]).mean().item()
        log(f"  {group_name}: mean |grad| = {mean_abs:.6e}")


def optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> None:
    if scaler.is_enabled():
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def run_timing_probe(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    config: TrainConfig,
    n_batches: int = TIMING_PROBE_BATCHES,
) -> float:
    """Run a few batches and return mean seconds per batch."""
    model.train()
    timings: list[float] = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= n_batches:
            break
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        forward_backward_batch(model, batch, criterion, device, scaler, config)
        optimizer_step(model, optimizer, scaler)
        timings.append(time.perf_counter() - t0)
    return float(np.mean(timings)) if timings else 0.0


def estimate_training_duration(
    avg_batch_sec: float,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
) -> tuple[float, float]:
    """Return (seconds per epoch, total seconds)."""
    train_batches = max(len(train_loader), 1)
    val_batches = max(len(val_loader), 1)
    sec_per_epoch = avg_batch_sec * (train_batches + val_batches * VAL_TIME_FACTOR)
    total_sec = sec_per_epoch * epochs
    return sec_per_epoch, total_sec


def run_epoch_train(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    config: TrainConfig,
    epoch: int,
    total_epochs: int,
    scheduler: CosineAnnealingLR | None = None,
    *,
    check_tcbm_gradients: bool = False,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    optimizer_steps = 0
    recent_losses: deque[float] = deque(maxlen=LOSS_WINDOW)
    epoch_t0 = time.perf_counter()

    gpu_util = get_gpu_utilization_pct(device)
    util_str = f"{gpu_util}%" if gpu_util is not None else "n/a"
    log(
        f"Epoch {epoch}/{total_epochs} start | lr={current_lr(optimizer):.2e} | "
        f"GPU util: {util_str} | "
        f"batch={config.batch_size} x accum={config.grad_accumulation_steps} "
        f"(effective={config.batch_size * config.grad_accumulation_steps})"
    )

    pbar = tqdm(
        loader,
        total=len(loader),
        desc=f"Epoch {epoch}/{total_epochs}",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    )

    optimizer.zero_grad(set_to_none=True)
    accum_steps = config.grad_accumulation_steps

    for batch_idx, batch in enumerate(pbar):
        loss_val = forward_backward_batch(
            model,
            batch,
            criterion,
            device,
            scaler,
            config,
            check_nan=(batch_idx == 0),
        )
        total_loss += loss_val
        n_batches += 1
        recent_losses.append(loss_val)

        if check_tcbm_gradients and batch_idx == 0:
            print_tcbm_gradient_check(model)

        is_accum_step = (batch_idx + 1) % accum_steps == 0
        is_last_batch = batch_idx + 1 == len(loader)
        if is_accum_step or is_last_batch:
            optimizer_step(model, optimizer, scaler)
            optimizer_steps += 1

        reserved_gb, total_gb = gpu_memory_gb(device)
        elapsed = time.perf_counter() - epoch_t0
        bps = (batch_idx + 1) / elapsed if elapsed > 0 else 0.0
        avg_loss = sum(recent_losses) / len(recent_losses)

        postfix = {
            "loss": f"{avg_loss:.4f}",
            "b/s": f"{bps:.2f}",
        }
        if device.type == "cuda":
            postfix["gpu"] = f"{reserved_gb:.1f}/{total_gb:.1f}GB"
            if gpu_util is not None:
                postfix["util"] = f"{gpu_util}%"
        pbar.set_postfix(postfix, refresh=True)

    if scheduler is not None and optimizer_steps > 0:
        scheduler.step()

    return total_loss / max(n_batches, 1)


def print_epoch_summary(
    *,
    epoch: int,
    total_epochs: int,
    elapsed_sec: float,
    train_loss: float,
    val_metrics: dict[str, float],
    is_best: bool,
    device: torch.device,
    eta_sec: float | None,
    epochs_without_improvement: int,
) -> None:
    reserved_gb, total_gb = gpu_memory_gb(device)
    sep = "═" * 47

    miou_line = f"Val mIoU:    {val_metrics['miou']:.3f}"
    if is_best:
        miou_line += "  ← best so far ✓"

    def fmt_iou(v: float) -> str:
        return f"{v:.2f}" if np.isfinite(v) else "n/a"

    lines = [
        sep,
        f"Epoch {epoch}/{total_epochs} completed in {format_duration(elapsed_sec)}",
        f"Train loss:  {train_loss:.3f}",
        f"Val loss:    {val_metrics['loss']:.3f}",
        miou_line,
        (
            "Per-class IoU:  "
            f"bg={fmt_iou(val_metrics['iou_0'])}  "
            f"field={fmt_iou(val_metrics['iou_1'])}  "
            f"boundary={fmt_iou(val_metrics['iou_2'])}"
        ),
    ]
    if device.type == "cuda":
        lines.append(f"GPU mem:     {reserved_gb:.1f} / {total_gb:.1f} GB")
    if eta_sec is not None:
        lines.append(f"ETA:         {format_duration(eta_sec)} remaining")
    lines.append(sep)

    for line in lines:
        log(line)

    if epochs_without_improvement >= PLATEAU_WARN_EPOCHS:
        log(
            f"⚠ No improvement for {epochs_without_improvement} epochs — consider stopping"
        )


@torch.no_grad()
def run_epoch_eval(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: TrainConfig,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    miou_list: list[float] = []
    acc_list: list[float] = []
    pred_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    total_intersect = np.zeros(NUM_CLASSES, dtype=np.int64)
    total_union = np.zeros(NUM_CLASSES, dtype=np.int64)
    debug_printed = False

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=config.mixed_precision and device.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, masks)

        total_loss += loss.item()
        n_batches += 1

        logits = segmentation_logits(outputs)
        preds = logits.argmax(dim=1)
        if not debug_printed:
            debug_first_val_batch(logits, preds, masks, IGNORE_INDEX)
            debug_printed = True

        class_ious, mean_iou, intersect, union = compute_iou(
            preds, masks, NUM_CLASSES, IGNORE_INDEX
        )
        total_intersect += intersect
        total_union += union
        miou_list.append(mean_iou)
        acc_list.append(compute_pixel_accuracy(preds, masks, IGNORE_INDEX))

        valid = masks.reshape(-1) != IGNORE_INDEX
        preds_flat = preds.reshape(-1)
        for cls in range(NUM_CLASSES):
            pred_counts[cls] += int((preds_flat[valid] == cls).sum().item())

    log_val_prediction_distribution(pred_counts)

    per_class_iou = np.full(NUM_CLASSES, np.nan, dtype=np.float64)
    for cls in range(NUM_CLASSES):
        if total_union[cls] > 0:
            per_class_iou[cls] = total_intersect[cls] / total_union[cls]
    global_miou = float(np.nanmean(per_class_iou))
    return {
        "loss": total_loss / max(n_batches, 1),
        "miou": global_miou,
        "iou_0": float(per_class_iou[0]),
        "iou_1": float(per_class_iou[1]),
        "iou_2": float(per_class_iou[2]),
        "pixel_acc": float(np.mean(acc_list)) if acc_list else 0.0,
    }


def save_plots(history: list[dict], out_dir: Path) -> None:
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    val_miou = [row["val_miou"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, val_miou, color="tab:green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mIoU")
    axes[1].set_title("Validation mIoU (FTW India)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved {out_dir / 'training_curves.png'}")


def write_csv_row(csv_path: Path, row: dict, write_header: bool) -> None:
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    config = parse_args()
    if config.use_dist_loss is None:
        config.use_dist_loss = config.model_name == "utae_dist"
    set_seed(config.seed)
    configure_cuda_backends()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and config.mixed_precision:
        log("CUDA not available; disabling mixed precision.")
        config.mixed_precision = False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.output_dir) / f"{config.model_name}_{config.dataset}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log("Building datasets (FTW norm stats computed on first train load)...")
    train_set, ftw_train = build_train_dataset(config)

    log(f"Run directory: {run_dir}")
    log(f"Device: {device}")
    log(f"Train samples: {len(train_set)}")
    dl_kw = dataloader_kwargs(config)
    log(
        f"DataLoader: workers={config.num_workers}, pin_memory={config.pin_memory}, "
        f"prefetch={dl_kw.get('prefetch_factor', 'off')}, "
        f"persistent_workers={dl_kw.get('persistent_workers', False)}, "
        f"mp_context={'spawn' if platform.system() == 'Windows' and config.num_workers > 0 else 'default'}"
    )
    log(f"Config: {asdict(config)}")

    weight_tuple, _ = resolve_class_weights(ftw_train, config)

    model = get_model(config.model_name).to(device)
    if config.compile_model:
        model = maybe_compile_model(model)

    class_weights = torch.tensor(weight_tuple, dtype=torch.float32, device=device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)
    criterion = CombinedSegLoss(
        ce_loss,
        use_dist_loss=config.use_dist_loss,
        dist_loss_weight=config.dist_loss_weight,
    )
    if config.use_dist_loss:
        log(f"Distance-transform loss enabled (weight={config.dist_loss_weight})")

    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=config.mixed_precision and device.type == "cuda",
    )

    if config.batch_size_finder:
        config.batch_size = find_max_batch_size(
            model, criterion, optimizer, scaler, device, config
        )

    val_loader = build_val_loader(ftw_train, config)
    train_loader = build_train_loader(train_set, config)
    log(f"Val (FTW India): {len(val_loader.dataset)} samples")

    start_epoch = 1
    best_miou = 0.0
    if config.resume:
        start_epoch, best_miou = load_checkpoint(
            Path(config.resume), model, optimizer, scheduler
        )
        start_epoch += 1
        log(f"Resumed from {config.resume} (epoch {start_epoch - 1}, best mIoU {best_miou:.4f})")

    csv_path = run_dir / "training_log.csv"
    history: list[dict] = []
    epoch_durations: list[float] = []
    epochs_without_improvement = 0

    epochs_to_run = config.epochs - start_epoch + 1
    log(f"\nRunning {TIMING_PROBE_BATCHES}-batch timing probe...")
    avg_batch_sec = run_timing_probe(
        model, train_loader, criterion, optimizer, device, scaler, config
    )
    sec_per_epoch, total_est = estimate_training_duration(
        avg_batch_sec, train_loader, val_loader, epochs_to_run
    )
    log(f"  Avg train batch time: {avg_batch_sec:.3f}s")
    log(f"  Estimated per epoch:  {format_duration(sec_per_epoch)}")
    log(f"  Estimated total:    {format_duration(total_est)} ({epochs_to_run} epochs)\n")

    for epoch in range(start_epoch, config.epochs + 1):
        t0 = time.perf_counter()

        train_loss = run_epoch_train(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            config,
            epoch,
            config.epochs,
            scheduler,
            check_tcbm_gradients=(
                config.model_name == "tcbm_unet" and epoch == start_epoch
            ),
        )
        val_metrics = run_epoch_eval(model, val_loader, criterion, device, config)

        elapsed = time.perf_counter() - t0
        epoch_durations.append(elapsed)

        is_best = val_metrics["miou"] > best_miou
        if is_best:
            best_miou = val_metrics["miou"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        remaining_epochs = config.epochs - epoch
        avg_epoch_time = sum(epoch_durations) / len(epoch_durations)
        eta_sec = avg_epoch_time * remaining_epochs if remaining_epochs > 0 else 0.0

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_miou": round(val_metrics["miou"], 6),
            "val_iou_0": round(val_metrics["iou_0"], 6),
            "val_iou_1": round(val_metrics["iou_1"], 6),
            "val_iou_2": round(val_metrics["iou_2"], 6),
            "val_pixel_acc": round(val_metrics["pixel_acc"], 6),
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_sec": round(elapsed, 2),
        }
        history.append(row)
        write_csv_row(csv_path, row, write_header=(epoch == start_epoch))

        print_epoch_summary(
            epoch=epoch,
            total_epochs=config.epochs,
            elapsed_sec=elapsed,
            train_loss=train_loss,
            val_metrics=val_metrics,
            is_best=is_best,
            device=device,
            eta_sec=eta_sec,
            epochs_without_improvement=epochs_without_improvement,
        )

        save_checkpoint(
            run_dir / "last_model.pth",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_miou=best_miou,
            config=config,
        )

        if is_best:
            save_checkpoint(
                run_dir / "best_model.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_miou=best_miou,
                config=config,
            )

    save_plots(history, run_dir)
    log(f"\nTraining complete. Best val mIoU: {best_miou:.4f}")
    log(f"Logs: {csv_path}")
    log(f"Checkpoints: {run_dir / 'best_model.pth'}, {run_dir / 'last_model.pth'}")


if __name__ == "__main__":
    main()
