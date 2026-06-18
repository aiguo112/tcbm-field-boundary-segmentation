#!/usr/bin/env python3
"""Evaluate trained segmentation checkpoints on FTW val and test splits."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import FTWDataset, IGNORE_INDEX, RGB_BANDS, stretch_rgb
from models import NUM_CLASSES, get_model, segmentation_logits

EVAL_ROOT = Path(r"E:\FTW\evaluation")
RUNS_DIR = Path(r"E:\FTW\runs")
CLASS_NAMES = ("background", "field", "boundary")
BOUNDARY_CLASS = 2
N_WORST = 10


@dataclass
class EvalConfig:
    checkpoints: list[str]
    runs_dir: str = str(RUNS_DIR)
    output_dir: str = str(EVAL_ROOT)
    batch_size: int = 8
    num_workers: int = 4
    mixed_precision: bool = True


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description="Evaluate FTW segmentation checkpoints")
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=[],
        help="Paths to .pth checkpoints (best_model.pth or last_model.pth)",
    )
    parser.add_argument(
        "--runs_dir",
        default=str(RUNS_DIR),
        help="If set, evaluate best_model.pth under each run subdirectory",
    )
    parser.add_argument("--output_dir", default=str(EVAL_ROOT))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--mixed_precision",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    checkpoints = list(args.checkpoints)
    if not checkpoints:
        runs_dir = Path(args.runs_dir)
        checkpoints = sorted(
            str(p) for p in runs_dir.glob("*/best_model.pth") if p.is_file()
        )

    if not checkpoints:
        raise SystemExit(
            "No checkpoints found. Pass --checkpoints or place runs under --runs_dir."
        )

    return EvalConfig(
        checkpoints=checkpoints,
        runs_dir=args.runs_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mixed_precision=args.mixed_precision,
    )


def log(msg: str) -> None:
    print(msg, flush=True)


def compute_iou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[np.ndarray, float]:
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)
    valid = targets != ignore_index
    preds = preds[valid]
    targets = targets[valid]

    ious = np.full(num_classes, np.nan, dtype=np.float64)
    for cls in range(num_classes):
        pred_c = preds == cls
        target_c = targets == cls
        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        if union > 0:
            ious[cls] = intersection / union

    return ious, float(np.nanmean(ious))


def compute_pixel_accuracy(
    preds: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> float:
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)
    valid = targets != ignore_index
    if valid.sum() == 0:
        return 0.0
    return (preds[valid] == targets[valid]).sum().item() / valid.sum().item()


def compute_boundary_f1(
    preds: torch.Tensor,
    targets: torch.Tensor,
    boundary_class: int = BOUNDARY_CLASS,
    ignore_index: int = IGNORE_INDEX,
) -> float:
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)
    valid = targets != ignore_index
    preds = preds[valid]
    targets = targets[valid]

    pred_b = preds == boundary_class
    target_b = targets == boundary_class
    tp = (pred_b & target_b).sum().item()
    fp = (pred_b & ~target_b).sum().item()
    fn = (~pred_b & target_b).sum().item()

    if tp + fp + fn == 0:
        return 0.0
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def collate_eval(samples: list[dict]) -> dict:
    return {
        "image": torch.stack([s["image"] for s in samples]),
        "mask": torch.stack([s["mask"] for s in samples]),
        "chip_id": [s["chip_id"] for s in samples],
        "country": [s["country"] for s in samples],
    }


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    model_name = config.get("model_name", "unet")
    model = get_model(model_name).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config


def run_name_from_checkpoint(path: Path) -> str:
    if path.name in ("best_model.pth", "last_model.pth"):
        return path.parent.name
    return path.stem


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mixed_precision: bool,
) -> tuple[dict[str, float], list[dict]]:
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    chip_records: list[dict] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            enabled=mixed_precision and device.type == "cuda",
        ):
            outputs = model(images)
            logits = segmentation_logits(outputs)

        preds = logits.argmax(dim=1)

        for i in range(preds.shape[0]):
            pred_i = preds[i]
            mask_i = masks[i]
            _, chip_miou = compute_iou(pred_i, mask_i, NUM_CLASSES, IGNORE_INDEX)
            chip_records.append(
                {
                    "chip_id": batch["chip_id"][i],
                    "country": batch["country"][i],
                    "miou": chip_miou,
                    "pred": pred_i.cpu(),
                    "target": mask_i.cpu(),
                    "image": batch["image"][i].cpu(),
                }
            )

        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())

    preds_cat = torch.cat(all_preds, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    class_ious, mean_iou = compute_iou(preds_cat, targets_cat, NUM_CLASSES, IGNORE_INDEX)

    metrics = {
        "mean_iou": mean_iou,
        "pixel_accuracy": compute_pixel_accuracy(preds_cat, targets_cat, IGNORE_INDEX),
        "boundary_f1": compute_boundary_f1(preds_cat, targets_cat, BOUNDARY_CLASS, IGNORE_INDEX),
        "iou_background": float(class_ious[0]) if not np.isnan(class_ious[0]) else 0.0,
        "iou_field": float(class_ious[1]) if not np.isnan(class_ious[1]) else 0.0,
        "iou_boundary": float(class_ious[2]) if not np.isnan(class_ious[2]) else 0.0,
    }
    return metrics, chip_records


def print_metrics(split: str, metrics: dict[str, float]) -> None:
    log(f"\n  [{split}]")
    log(f"    mean IoU:        {metrics['mean_iou']:.4f}")
    log(f"    pixel accuracy:  {metrics['pixel_accuracy']:.4f}")
    log(f"    boundary F1:     {metrics['boundary_f1']:.4f}")
    log(f"    IoU background:  {metrics['iou_background']:.4f}")
    log(f"    IoU field:       {metrics['iou_field']:.4f}")
    log(f"    IoU boundary:    {metrics['iou_boundary']:.4f}")


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Map classes 0,1,2 to RGB for visualization."""
    colors = np.array(
        [[0, 0, 0], [46, 139, 87], [220, 20, 60]], dtype=np.float32
    )  # bg, field, boundary
    mask = np.where(mask == IGNORE_INDEX, 0, mask)
    mask = np.clip(mask, 0, 2).astype(np.int32)
    return colors[mask]


def save_worst_chip_figure(record: dict, out_path: Path, title: str) -> None:
    image = record["image"].numpy()
    target = record["target"].numpy()
    pred = record["pred"].numpy()

    rgb = image[list(RGB_BANDS), :, :].transpose(1, 2, 0)
    rgb_display = stretch_rgb(rgb)
    gt_rgb = mask_to_rgb(target)
    pred_rgb = mask_to_rgb(pred)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(rgb_display)
    axes[0].set_title("Image (RGB)")
    axes[0].axis("off")

    axes[1].imshow(gt_rgb / 255.0)
    axes[1].set_title("Ground truth")
    axes[1].axis("off")

    axes[2].imshow(pred_rgb / 255.0)
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    legend_colors = [
        (0.0, 0.0, 0.0),
        (46 / 255, 139 / 255, 87 / 255),
        (220 / 255, 20 / 255, 60 / 255),
    ]
    patches = [
        mpatches.Patch(color=color, label=name)
        for color, name in zip(legend_colors, CLASS_NAMES)
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_worst_chips(
    chip_records: list[dict],
    out_dir: Path,
    split: str,
    n: int = N_WORST,
) -> None:
    worst = sorted(chip_records, key=lambda r: r["miou"])[:n]
    for rank, record in enumerate(worst, start=1):
        chip_id = record["chip_id"]
        country = record["country"]
        title = (
            f"{split} | {chip_id} ({country}) | "
            f"mIoU={record['miou']:.4f} | rank {rank}/{n}"
        )
        out_path = out_dir / f"{rank:02d}_{chip_id}_{country}.png"
        save_worst_chip_figure(record, out_path, title)
    log(f"    Saved {len(worst)} worst-chip figures to {out_dir}")


def evaluate_checkpoint(
    checkpoint_path: Path,
    config: EvalConfig,
    device: torch.device,
    ftw_train: FTWDataset,
) -> dict:
    model, train_config = load_checkpoint(checkpoint_path, device)
    run_name = run_name_from_checkpoint(checkpoint_path)
    n_params = count_parameters(model)

    log(f"\n{'=' * 60}")
    log(f"Evaluating: {run_name}")
    log(f"  checkpoint: {checkpoint_path}")
    log(f"  model:      {train_config.get('model_name', 'unknown')}")
    log(f"  dataset:    {train_config.get('dataset', 'unknown')}")
    log(f"  params:     {n_params:,}")

    out_run_dir = Path(config.output_dir) / run_name
    out_run_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "run_name": run_name,
        "checkpoint": str(checkpoint_path),
        "model_name": train_config.get("model_name", "unknown"),
        "train_dataset": train_config.get("dataset", "unknown"),
        "params": n_params,
        "epoch": train_config.get("epoch") if isinstance(train_config.get("epoch"), int) else None,
    }

    for split in ("val", "test"):
        dataset = FTWDataset(
            split=split,
            augment=False,
            mean=ftw_train.mean,
            std=ftw_train.std,
        )
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate_eval,
        )

        metrics, chip_records = evaluate_loader(
            model, loader, device, config.mixed_precision
        )
        print_metrics(split, metrics)
        results[f"{split}"] = metrics

        metrics_path = out_run_dir / f"metrics_{split}.json"
        with metrics_path.open("w") as f:
            json.dump(metrics, f, indent=2)

        visualize_worst_chips(
            chip_records,
            out_run_dir / f"worst_{split}",
            split=split,
            n=N_WORST,
        )

    summary_path = out_run_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(results, f, indent=2)

    return {
        "model": results["model_name"],
        "run_name": run_name,
        "train_dataset": results["train_dataset"],
        "params": n_params,
        "val_miou": results["val"]["mean_iou"],
        "test_miou": results["test"]["mean_iou"],
        "val_boundary_iou": results["val"]["iou_boundary"],
        "test_boundary_iou": results["test"]["iou_boundary"],
        "val_boundary_f1": results["val"]["boundary_f1"],
        "test_boundary_f1": results["test"]["boundary_f1"],
    }


def save_results_table(rows: list[dict], output_dir: Path) -> None:
    if not rows:
        return

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
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    log(f"\nSaved {csv_path}")

    md_path = output_dir / "results_table.md"
    lines = [
        "# Evaluation results",
        "",
        "| Model | Params | Val mIoU | Test mIoU | Val boundary IoU | Test boundary IoU |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['params']:,} | {row['val_miou']:.4f} | "
            f"{row['test_miou']:.4f} | {row['val_boundary_iou']:.4f} | "
            f"{row['test_boundary_iou']:.4f} |"
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
    log(f"Saved {md_path}")


def main() -> None:
    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    log(f"Output: {output_dir}")
    log(f"Checkpoints ({len(config.checkpoints)}):")
    for ckpt in config.checkpoints:
        log(f"  - {ckpt}")

    log("\nLoading FTW train set for normalization stats...")
    ftw_train = FTWDataset(split="train", augment=False)

    table_rows: list[dict] = []
    for ckpt_str in config.checkpoints:
        ckpt_path = Path(ckpt_str)
        if not ckpt_path.exists():
            log(f"WARNING: skipping missing checkpoint {ckpt_path}")
            continue
        row = evaluate_checkpoint(ckpt_path, config, device, ftw_train)
        table_rows.append(row)

    save_results_table(table_rows, output_dir)
    log("\nEvaluation complete.")


if __name__ == "__main__":
    main()
