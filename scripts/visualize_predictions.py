#!/usr/bin/env python3
"""Generate paper-quality qualitative comparison figures on FTW test chips."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import (
    FTWDataset,
    IGNORE_INDEX,
    read_ftw_chip_arrays,
    stretch_rgb,
)
from evaluate import collate_eval, compute_iou
from evaluate_ftw_baseline import OFFICIAL_RADIANCE_SCALE, load_pretrained_model
from models import get_model, segmentation_logits

QUAL_ROOT = Path(r"E:\FTW\evaluation\qualitative")
PAPER_FIGURE_PATH = Path(r"E:\FTW\paper\figures\paper_figure_main.png")
DPI = 300

OFFICIAL_WEIGHTS = Path(
    r"C:\Users\Arbi\.cache\huggingface\hub"
    r"\models--torchgeo--fields-of-the-world\snapshots"
    r"\ae66b65af560948315bd71f6421f8a14098a9443"
    r"\ftw-3class-full_unet-efficientnetb3_rgbnir_1ba4e1bd.pth"
)

CHECKPOINTS: dict[str, tuple[str, Path | None]] = {
    "unet_official": ("U-Net Official", OFFICIAL_WEIGHTS),
    "segformer": (
        "SegFormer",
        Path(r"E:\FTW\runs\segformer_ftw_20260602_162621\best_model.pth"),
    ),
    "utae_ftw": (
        "TA-UNet FTW",
        Path(r"E:\FTW\runs\utae_ftw_20260605_174329\best_model.pth"),
    ),
    "utae_pastis": (
        "TA-UNet+PASTIS",
        Path(r"E:\FTW\runs\utae_combined_20260603_203227\best_model.pth"),
    ),
    "tcbm_unet": (
        "TCBM-UNet (Ours)",
        Path(r"E:\FTW\runs\tcbm_unet_combined_20260612_192019\best_model.pth"),
    ),
}

MODEL_KEYS = ("unet_official", "segformer", "utae_ftw", "utae_pastis", "tcbm_unet")

COLUMN_HEADERS = (
    "RGB",
    "Ground Truth",
    "U-Net Official",
    "SegFormer",
    "TA-UNet FTW",
    "TA-UNet+PASTIS",
    "TCBM-UNet (Ours)",
)

N_COLUMNS = len(COLUMN_HEADERS)
TCBM_COLUMN_INDEX = N_COLUMNS - 1

MASK_COLORS: dict[int, tuple[int, int, int]] = {
    0: (255, 255, 255),
    1: (144, 238, 144),
    2: (220, 50, 50),
    IGNORE_INDEX: (200, 200, 200),
}

LEGEND_LABELS = ("Background", "Field", "Boundary", "Ignored")


def log(msg: str) -> None:
    print(msg, flush=True)


def remap_legacy_utae_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map pre-refactor UTAE checkpoints (decoder.5 head) to seg_head keys."""
    if "decoder.5.weight" in state and "seg_head.weight" not in state:
        state = dict(state)
        state["seg_head.weight"] = state.pop("decoder.5.weight")
        state["seg_head.bias"] = state.pop("decoder.5.bias")
    return state


def load_model_checkpoint(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict, int | str]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    model_name = config.get("model_name", "unet")
    state = remap_legacy_utae_state(ckpt["model"])
    model = get_model(model_name).to(device)
    model.load_state_dict(state)
    model.eval()
    epoch = ckpt.get("epoch", "?")
    return model, config, epoch


def load_models(device: torch.device) -> dict[str, torch.nn.Module]:
    models: dict[str, torch.nn.Module] = {}

    official_path = CHECKPOINTS["unet_official"][1]
    if official_path is None or not official_path.is_file():
        raise FileNotFoundError(f"Missing official U-Net weights: {official_path}")
    models["unet_official"] = load_pretrained_model(official_path, device)
    log(f"  U-Net Official: {official_path.name} (divide by {OFFICIAL_RADIANCE_SCALE:.0f})")

    for key in MODEL_KEYS:
        if key == "unet_official":
            continue
        display_name, path = CHECKPOINTS[key]
        if path is None or not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint for {display_name}: {path}")
        model, _config, epoch = load_model_checkpoint(path, device)
        models[key] = model
        log(f"  {display_name}: {path.parent.name} (epoch {epoch})")

    return models


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Colorize segmentation mask with paper color scheme."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in MASK_COLORS.items():
        rgb[mask == cls] = color
    return rgb


def window_a_rgb_display(record: dict[str, str]) -> np.ndarray:
    """RGB composite from raw window_a bands 0, 1, 2 (percentile-stretched)."""
    image, _ = read_ftw_chip_arrays(record)
    window_a = image[4:8]
    rgb = window_a[[0, 1, 2], :, :].transpose(1, 2, 0).astype(np.float32)
    return stretch_rgb(rgb)


def chip_miou(pred: np.ndarray, target: np.ndarray) -> float:
    _, mean_iou = compute_iou(
        torch.from_numpy(pred),
        torch.from_numpy(target),
        num_classes=3,
        ignore_index=IGNORE_INDEX,
    )
    return float(mean_iou)


def official_input_from_zscore(
    images_zscore: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Recover raw radiance from z-score inputs, then apply official /3000 scaling."""
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)
    raw = images_zscore.float() * std + mean
    return raw / OFFICIAL_RADIANCE_SCALE


@torch.no_grad()
def predict_batch(
    model: torch.nn.Module,
    images: torch.Tensor,
    *,
    mixed_precision: bool,
    device: torch.device,
) -> np.ndarray:
    with torch.autocast(
        device_type=device.type,
        enabled=mixed_precision and device.type == "cuda",
    ):
        outputs = model(images)
        logits = segmentation_logits(outputs)
        return logits.argmax(dim=1).cpu().numpy()


@torch.no_grad()
def collect_chip_results(
    models: dict[str, torch.nn.Module],
    loader: DataLoader,
    device: torch.device,
    mixed_precision: bool,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> list[dict]:
    """Run all models on the test loader and aggregate per-chip predictions."""
    sample_index = {
        (s["chip_id"], s["country"]): s for s in loader.dataset.samples
    }
    pending: dict[tuple[str, str], dict] = {}

    for batch in loader:
        images_zscore = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        images_official = official_input_from_zscore(images_zscore, mean, std)

        batch_preds: dict[str, np.ndarray] = {}
        for key, model in models.items():
            if key == "unet_official":
                batch_preds[key] = predict_batch(
                    model,
                    images_official,
                    mixed_precision=mixed_precision,
                    device=device,
                )
            else:
                batch_preds[key] = predict_batch(
                    model,
                    images_zscore,
                    mixed_precision=mixed_precision,
                    device=device,
                )

        for i in range(images_zscore.shape[0]):
            chip_id = batch["chip_id"][i]
            country = batch["country"][i]
            key = (chip_id, country)
            target = masks[i].cpu().numpy()

            entry = pending.get(key)
            if entry is None:
                record = sample_index[key]
                entry = {
                    "chip_id": chip_id,
                    "country": country,
                    "record": record,
                    "target": target,
                    "preds": {},
                    "ious": {},
                }
                pending[key] = entry

            for model_key in MODEL_KEYS:
                pred = batch_preds[model_key][i]
                entry["preds"][model_key] = pred
                entry["ious"][model_key] = chip_miou(pred, target)

    return list(pending.values())


def select_success_cases(chips: list[dict], n: int = 3) -> list[dict]:
    """Chips where TCBM-UNet beats SegFormer by the largest margin."""
    ranked = sorted(
        chips,
        key=lambda c: c["ious"]["tcbm_unet"] - c["ious"]["segformer"],
        reverse=True,
    )
    return ranked[:n]


def select_failure_cases(chips: list[dict], n: int = 4) -> list[dict]:
    """Chips where every model mIoU is below 0.2."""
    hard = [
        c
        for c in chips
        if all(c["ious"][k] < 0.2 for k in MODEL_KEYS)
    ]
    hard.sort(key=lambda c: max(c["ious"][k] for k in MODEL_KEYS))
    return hard[:n]


def add_legend(fig: plt.Figure) -> None:
    colors = [
        tuple(c / 255.0 for c in MASK_COLORS[0]),
        tuple(c / 255.0 for c in MASK_COLORS[1]),
        tuple(c / 255.0 for c in MASK_COLORS[2]),
        tuple(c / 255.0 for c in MASK_COLORS[IGNORE_INDEX]),
    ]
    patches = [
        mpatches.Patch(facecolor=color, edgecolor="0.4", linewidth=0.5, label=label)
        for color, label in zip(colors, LEGEND_LABELS)
    ]
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=4,
        fontsize=9,
        frameon=True,
        fancybox=False,
        edgecolor="0.7",
    )


def set_column_title(ax: plt.Axes, col: int) -> None:
    title = COLUMN_HEADERS[col]
    if col == TCBM_COLUMN_INDEX:
        ax.set_title(title, fontsize=10, fontweight="bold", color="red", pad=6)
    else:
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)


def draw_chip_row(
    axes: np.ndarray,
    chip: dict,
    *,
    show_headers: bool,
    rgb_cache: dict[tuple[str, str], np.ndarray] | None = None,
) -> None:
    """Fill one row of seven panels for a single chip."""
    record = chip["record"]
    cache_key = (chip["chip_id"], chip["country"])

    if rgb_cache is not None and cache_key in rgb_cache:
        rgb = rgb_cache[cache_key]
    else:
        rgb = window_a_rgb_display(record)
        if rgb_cache is not None:
            rgb_cache[cache_key] = rgb

    target_rgb = mask_to_rgb(chip["target"])

    panels: list[tuple[str, np.ndarray, float | None]] = [
        ("rgb", rgb, None),
        ("gt", target_rgb / 255.0, None),
    ]
    for model_key in MODEL_KEYS:
        pred_rgb = mask_to_rgb(chip["preds"][model_key]) / 255.0
        panels.append(("pred", pred_rgb, chip["ious"][model_key]))

    for col, (kind, img, iou) in enumerate(panels):
        ax = axes[col]
        ax.imshow(img)

        if show_headers:
            set_column_title(ax, col)

        if iou is not None:
            ax.set_xlabel(f"mIoU = {iou:.3f}", fontsize=9, labelpad=4)
        elif col == 0:
            ax.set_xlabel(
                f"{chip['chip_id']} ({chip['country']})",
                fontsize=8,
                labelpad=4,
            )

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)


def save_single_chip_figure(chip: dict, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, N_COLUMNS, figsize=(21.0, 3.0))
    draw_chip_row(axes, chip, show_headers=True)
    add_legend(fig)
    fig.suptitle(title, fontsize=11, y=1.02)
    fig.subplots_adjust(bottom=0.14, wspace=0.05, top=0.82)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_paper_figure(chips: list[dict], out_path: Path) -> None:
    n_rows = len(chips)
    fig, axes = plt.subplots(n_rows, N_COLUMNS, figsize=(21.0, 3.0 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    rgb_cache: dict[tuple[str, str], np.ndarray] = {}
    for row, chip in enumerate(chips):
        draw_chip_row(
            axes[row],
            chip,
            show_headers=(row == 0),
            rgb_cache=rgb_cache,
        )

    add_legend(fig)
    fig.subplots_adjust(bottom=0.08, hspace=0.30, wspace=0.05, top=0.90)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate qualitative FTW test comparison figures"
    )
    parser.add_argument("--output_dir", type=Path, default=QUAL_ROOT)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--mixed_precision",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--n_success", type=int, default=3)
    parser.add_argument("--n_failure", type=int, default=4)
    parser.add_argument("--n_paper_rows", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and args.mixed_precision:
        args.mixed_precision = False

    log(f"Device: {device}")
    log("Loading checkpoints...")
    models = load_models(device)

    ftw_train = FTWDataset(split="train", augment=False)
    mean = torch.from_numpy(ftw_train.mean).to(device)
    std = torch.from_numpy(ftw_train.std).to(device)

    test_set = FTWDataset(
        split="test",
        augment=False,
        mean=ftw_train.mean,
        std=ftw_train.std,
        cache_in_memory=len(ftw_train.samples) <= 5000,
    )
    loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_eval,
    )
    log(f"FTW test split: {len(test_set)} chips (Vietnam + Cambodia)")
    log(
        "Normalization: official U-Net uses image/3000; "
        "all other models use FTWDataset z-score stats."
    )

    log("Running inference with all models...")
    chips = collect_chip_results(
        models,
        loader,
        device,
        mixed_precision=args.mixed_precision,
        mean=mean,
        std=std,
    )
    log(f"Collected predictions for {len(chips)} chips")

    success_cases = select_success_cases(chips, n=args.n_success)
    failure_cases = select_failure_cases(chips, n=args.n_failure)

    log(
        f"\nSuccess cases (top {len(success_cases)} "
        "TCBM-UNet vs SegFormer margin):"
    )
    for i, chip in enumerate(success_cases, start=1):
        margin = chip["ious"]["tcbm_unet"] - chip["ious"]["segformer"]
        log(
            f"  {i}. {chip['chip_id']} ({chip['country']}) "
            f"margin={margin:.3f} | tcbm={chip['ious']['tcbm_unet']:.3f} "
            f"segformer={chip['ious']['segformer']:.3f}"
        )

    log(f"\nFailure cases (all models mIoU < 0.2, n={len(failure_cases)}):")
    for i, chip in enumerate(failure_cases, start=1):
        ious = ", ".join(
            f"{CHECKPOINTS[k][0]}={chip['ious'][k]:.3f}" for k in MODEL_KEYS
        )
        log(f"  {i}. {chip['chip_id']} ({chip['country']}) | {ious}")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, chip in enumerate(success_cases, start=1):
        margin = chip["ious"]["tcbm_unet"] - chip["ious"]["segformer"]
        title = (
            f"Success case {i}: {chip['chip_id']} ({chip['country']}) | "
            f"TCBM-UNet advantage over SegFormer: +{margin:.3f} mIoU"
        )
        path = out_dir / f"success_case_{i}.png"
        save_single_chip_figure(chip, path, title)
        log(f"Saved {path}")

    for i, chip in enumerate(failure_cases, start=1):
        max_iou = max(chip["ious"][k] for k in MODEL_KEYS)
        title = (
            f"Failure case {i}: {chip['chip_id']} ({chip['country']}) | "
            f"best model mIoU = {max_iou:.3f}"
        )
        path = out_dir / f"failure_case_{i}.png"
        save_single_chip_figure(chip, path, title)
        log(f"Saved {path}")

    paper_chips = success_cases[: args.n_paper_rows]
    if paper_chips:
        paper_v2_path = out_dir / "paper_figure_main_v2.png"
        save_paper_figure(paper_chips, paper_v2_path)
        log(f"Saved {paper_v2_path}")

        save_paper_figure(paper_chips, PAPER_FIGURE_PATH)
        log(f"Saved {PAPER_FIGURE_PATH}")

    log("\nDone.")


if __name__ == "__main__":
    main()
