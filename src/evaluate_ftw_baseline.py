#!/usr/bin/env python3
"""Evaluate the official FTW pretrained U-Net baseline from HuggingFace."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from dataset import FTWDataset, IGNORE_INDEX, read_ftw_chip_arrays, remap_ftw_mask
from evaluate import (
    BOUNDARY_CLASS,
    collate_eval,
    compute_boundary_f1,
    count_parameters,
    print_metrics,
)
from models import NUM_CLASSES
from train import compute_iou, compute_pixel_accuracy

HF_REPO_ID = "torchgeo/fields-of-the-world"
HF_WEIGHTS_FILE = "ftw-3class-full_unet-efficientnetb3_rgbnir_1ba4e1bd.pth"
MODEL_DISPLAY_NAME = "U-Net (FTW Official Pretrained)"
RUN_NAME = "ftw_official_baseline"
EVAL_ROOT = Path(r"E:\FTW\evaluation")
OUT_DIR = EVAL_ROOT / RUN_NAME
BATCH_SIZE = 32
NUM_WORKERS = 0
OFFICIAL_RADIANCE_SCALE = 3000.0


class FTWOfficialEvalDataset(Dataset):
    """FTW val/test chips with official /3000 radiance scaling (no z-score)."""

    def __init__(self, split: str) -> None:
        if split not in ("val", "test"):
            raise ValueError(f"Official baseline eval expects val or test split, got {split!r}")
        reference = FTWDataset(
            split=split,
            augment=False,
            mean=np.zeros(8, dtype=np.float32),
            std=np.ones(8, dtype=np.float32),
            cache_in_memory=False,
        )
        self.samples = reference.samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        record = self.samples[index]
        image, mask = read_ftw_chip_arrays(record)
        image = image.astype(np.float32) / OFFICIAL_RADIANCE_SCALE
        mask = remap_ftw_mask(mask).astype(np.int64, copy=False)
        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask).long(),
            "country": record["country"],
            "chip_id": record["chip_id"],
        }


@dataclass
class BaselineEvalConfig:
    model_path: str = ""


def parse_args() -> BaselineEvalConfig:
    parser = argparse.ArgumentParser(
        description="Evaluate the official FTW pretrained U-Net baseline"
    )
    parser.add_argument(
        "--model_path",
        default="",
        help="Local path to pretrained .pth weights (skips HuggingFace download)",
    )
    args = parser.parse_args()
    return BaselineEvalConfig(model_path=args.model_path)


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_huggingface_hub() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        log("huggingface_hub not found; installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        log("huggingface_hub installed.")


def download_pretrained_weights() -> Path:
    ensure_huggingface_hub()
    from huggingface_hub import hf_hub_download

    log(f"Downloading {HF_REPO_ID}/{HF_WEIGHTS_FILE} (cached after first run)...")
    cached_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_WEIGHTS_FILE)
    log(f"  weights: {cached_path}")
    return Path(cached_path)


def resolve_weights_path(config: BaselineEvalConfig) -> Path:
    if config.model_path:
        weights_path = Path(config.model_path)
        if not weights_path.is_file():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")
        log(f"Using local weights: {weights_path}")
        return weights_path.resolve()
    return download_pretrained_weights()


def _strip_loss_weights(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = dict(state_dict)
    cleaned.pop("criterion.weight", None)
    cleaned.pop("ce_loss.weight", None)
    return cleaned


def build_unet() -> nn.Module:
    return smp.Unet(
        encoder_name="efficientnet-b3",
        encoder_weights=None,
        in_channels=8,
        classes=3,
    )


def load_pretrained_model(weights_path: Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    model = build_unet()

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = {
            key.replace("model.", ""): value
            for key, value in ckpt["state_dict"].items()
        }
        state_dict = _strip_loss_weights(state_dict)
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        if any(key.startswith("net.") for key in state_dict):
            state_dict = {
                key.replace("net.", "", 1): value for key, value in state_dict.items()
            }
        state_dict = _strip_loss_weights(state_dict)
    elif isinstance(ckpt, dict):
        state_dict = _strip_loss_weights(ckpt)
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(ckpt)!r}")

    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    mixed_precision: bool = True,
) -> dict[str, float]:
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            enabled=mixed_precision and device.type == "cuda",
        ):
            logits = model(images)

        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())

    preds_cat = torch.cat(all_preds, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    class_ious, mean_iou, _, _ = compute_iou(
        preds_cat, targets_cat, NUM_CLASSES, IGNORE_INDEX
    )

    return {
        "mean_iou": mean_iou,
        "pixel_accuracy": compute_pixel_accuracy(preds_cat, targets_cat, IGNORE_INDEX),
        "boundary_f1": compute_boundary_f1(
            preds_cat, targets_cat, BOUNDARY_CLASS, IGNORE_INDEX
        ),
        "iou_background": float(class_ious[0]) if not np.isnan(class_ious[0]) else 0.0,
        "iou_field": float(class_ious[1]) if not np.isnan(class_ious[1]) else 0.0,
        "iou_boundary": float(class_ious[2]) if not np.isnan(class_ious[2]) else 0.0,
    }


def build_loader(split: str, device: torch.device) -> DataLoader:
    dataset = FTWOfficialEvalDataset(split=split)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        collate_fn=collate_eval,
    )


def append_results_table(row: dict, output_dir: Path) -> None:
    fieldnames = [
        "model",
        "run_name",
        "train_dataset",
        "params",
        "val_miou",
        "test_miou",
        "val_boundary_iou",
        "test_boundary_iou",
        "val_boundary_f1",
        "test_boundary_f1",
    ]

    csv_path = output_dir / "results_table.csv"
    rows: list[dict] = []
    if csv_path.exists():
        with csv_path.open(newline="") as f:
            rows = list(csv.DictReader(f))

    rows = [r for r in rows if r.get("run_name") != row["run_name"]]
    rows.append({key: row.get(key, "") for key in fieldnames})

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log(f"\nUpdated {csv_path}")

    md_path = output_dir / "results_table.md"
    lines = [
        "# Evaluation results",
        "",
        "| Model | Params | Val mIoU | Test mIoU | Val boundary IoU | Test boundary IoU |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for table_row in rows:
        params = int(float(table_row["params"]))
        lines.append(
            f"| {table_row['model']} | {params:,} | "
            f"{float(table_row['val_miou']):.4f} | {float(table_row['test_miou']):.4f} | "
            f"{float(table_row['val_boundary_iou']):.4f} | "
            f"{float(table_row['test_boundary_iou']):.4f} |"
        )
    lines.extend(
        [
            "",
            "Detailed CSV includes run names, boundary F1, and train dataset.",
            "",
            f"_{len(rows)} checkpoint(s) evaluated._",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"Updated {md_path}")


def main() -> None:
    config = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mixed_precision = device.type == "cuda"

    log(f"Device: {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)

    weights_path = resolve_weights_path(config)
    model = load_pretrained_model(weights_path, device)
    n_params = count_parameters(model)

    log(
        "Using official FTW input normalization: "
        f"raw radiance / {OFFICIAL_RADIANCE_SCALE:.1f} (no mean subtraction)"
    )

    log(f"\n{'=' * 60}")
    log(f"Evaluating: {MODEL_DISPLAY_NAME}")
    log(f"  weights:  {weights_path}")
    log(f"  params:   {n_params:,}")

    results: dict = {
        "model_name": MODEL_DISPLAY_NAME,
        "repo_id": HF_REPO_ID if not config.model_path else None,
        "weights_file": HF_WEIGHTS_FILE if not config.model_path else weights_path.name,
        "weights_path": str(weights_path),
        "normalization": f"divide_by_{int(OFFICIAL_RADIANCE_SCALE)}",
        "params": n_params,
    }

    for split in ("val", "test"):
        loader = build_loader(split, device)
        log(f"\n  {split}: {len(loader.dataset)} samples")
        metrics = evaluate_split(model, loader, device, mixed_precision=mixed_precision)
        print_metrics(split, metrics)
        results[split] = metrics

    metrics_path = OUT_DIR / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(results, f, indent=2)
    log(f"\nSaved {metrics_path}")

    table_row = {
        "model": MODEL_DISPLAY_NAME,
        "run_name": RUN_NAME,
        "train_dataset": "official-pretrained",
        "params": n_params,
        "val_miou": results["val"]["mean_iou"],
        "test_miou": results["test"]["mean_iou"],
        "val_boundary_iou": results["val"]["iou_boundary"],
        "test_boundary_iou": results["test"]["iou_boundary"],
        "val_boundary_f1": results["val"]["boundary_f1"],
        "test_boundary_f1": results["test"]["boundary_f1"],
    }
    append_results_table(table_row, EVAL_ROOT)
    log("\nEvaluation complete.")


if __name__ == "__main__":
    main()
