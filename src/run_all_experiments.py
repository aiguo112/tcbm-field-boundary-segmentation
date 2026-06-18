#!/usr/bin/env python3
"""Train SegFormer + U-TAE, then evaluate all three models (incl. existing U-Net)."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import torch

PYTHON = Path(r"C:\Users\Arbi\.conda\envs\seg\python.exe")
ROOT = Path(r"E:\FTW")
RUNS_DIR = ROOT / "runs"
EVAL_DIR = ROOT / "evaluation"
SUMMARY_PATH = RUNS_DIR / "experiment_summary.txt"

EXISTING_UNET_CHECKPOINT = (
    RUNS_DIR / "unet_ftw_20260531_224618" / "best_model.pth"
)

TRAIN_BATCH_SIZE = 8
EPOCHS = 50
NUM_WORKERS = 4
PIN_MEMORY = True

# Only train these two; U-Net checkpoint is reused from a prior run.
TRAIN_EXPERIMENTS = [
    {
        "label": "segformer_ftw",
        "train_args": ["--model_name", "segformer", "--dataset", "ftw"],
        "glob": "segformer_ftw_*",
    },
    {
        "label": "utae_combined",
        "train_args": [
            "--model_name",
            "utae",
            "--dataset",
            "combined",
            "--ftw_ratio",
            "0.7",
        ],
        "glob": "utae_combined_*",
    },
]


def detect_gpu_memory_gb() -> tuple[float | None, str]:
    """Print GPU info; training uses fixed TRAIN_BATCH_SIZE."""
    if not torch.cuda.is_available():
        return None, "CUDA not available."

    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / (1024**3)
    log_text = f"GPU: {props.name}\n  Total memory: {total_gb:.2f} GB"
    return total_gb, log_text


def write_summary(lines: list[str], *, append: bool = True) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with SUMMARY_PATH.open(mode, encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def run_cmd(args: list[str], step: str) -> None:
    cmd = [str(PYTHON), *args]
    print(f"\n{'=' * 60}", flush=True)
    print(f"{step}", flush=True)
    print(" ".join(cmd), flush=True)
    print("=" * 60, flush=True)
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"{step} failed with exit code {result.returncode}")


def latest_checkpoint(glob_pattern: str) -> Path | None:
    candidates = sorted(RUNS_DIR.glob(f"{glob_pattern}/best_model.pth"))
    if not candidates:
        return None
    return candidates[-1]


def train_args_for_experiment(exp_train_args: list[str]) -> list[str]:
    return [
        str(ROOT / "train.py"),
        *exp_train_args,
        "--epochs",
        str(EPOCHS),
        "--batch_size",
        str(TRAIN_BATCH_SIZE),
        "--num_workers",
        str(NUM_WORKERS),
        "--pin_memory" if PIN_MEMORY else "--no-pin_memory",
    ]


def read_eval_metrics(checkpoint: Path) -> tuple[float | None, float | None]:
    """Read val/test mIoU written by evaluate.py for this checkpoint's run folder."""
    run_name = checkpoint.parent.name
    val_path = EVAL_DIR / run_name / "metrics_val.json"
    test_path = EVAL_DIR / run_name / "metrics_test.json"

    val_miou = test_miou = None
    if val_path.exists():
        with val_path.open(encoding="utf-8") as f:
            val_miou = json.load(f).get("mean_iou")
    if test_path.exists():
        with test_path.open(encoding="utf-8") as f:
            test_miou = json.load(f).get("mean_iou")
    return val_miou, test_miou


def evaluate_checkpoint(checkpoint: Path) -> None:
    run_cmd(
        [
            str(ROOT / "evaluate.py"),
            "--output_dir",
            str(EVAL_DIR),
            "--checkpoints",
            str(checkpoint),
            "--batch_size",
            str(TRAIN_BATCH_SIZE),
            "--num_workers",
            str(NUM_WORKERS),
        ],
        step=f"Evaluate: {checkpoint.parent.name}",
    )


def append_metrics_to_summary(label: str, checkpoint: Path) -> None:
    val_miou, test_miou = read_eval_metrics(checkpoint)
    val_str = f"{val_miou:.4f}" if val_miou is not None else "n/a"
    test_str = f"{test_miou:.4f}" if test_miou is not None else "n/a"
    write_summary(
        [
            f"Checkpoint: {checkpoint}",
            f"Val mIoU: {val_str}",
            f"Test mIoU: {test_str}",
            "",
        ]
    )
    print(f"  {label} -> val mIoU={val_str}, test mIoU={test_str}", flush=True)


def main() -> None:
    if not PYTHON.exists():
        raise SystemExit(f"Python not found: {PYTHON}")

    if not EXISTING_UNET_CHECKPOINT.is_file():
        raise SystemExit(f"Existing U-Net checkpoint not found: {EXISTING_UNET_CHECKPOINT}")

    _, gpu_log = detect_gpu_memory_gb()
    run_started = datetime.now()

    write_summary(
        [
            f"Experiment run started: {run_started.isoformat(timespec='seconds')}",
            gpu_log,
            f"batch_size: {TRAIN_BATCH_SIZE} (fixed)",
            f"epochs: {EPOCHS}",
            f"num_workers: {NUM_WORKERS}",
            f"pin_memory: {PIN_MEMORY}",
            "",
            "--- unet_ftw (skipped training) ---",
            f"Using existing checkpoint: {EXISTING_UNET_CHECKPOINT}",
            "",
        ],
        append=False,
    )

    print(gpu_log or "CUDA not available.", flush=True)
    print(f"batch_size={TRAIN_BATCH_SIZE}, epochs={EPOCHS}", flush=True)
    print(f"num_workers={NUM_WORKERS}, pin_memory={PIN_MEMORY}", flush=True)
    print(f"Skipping U-Net training; using {EXISTING_UNET_CHECKPOINT}", flush=True)

    all_checkpoints: list[str] = [str(EXISTING_UNET_CHECKPOINT)]

    for exp in TRAIN_EXPERIMENTS:
        exp_started = datetime.now()
        write_summary(
            [
                f"--- {exp['label']} ---",
                f"Training started: {exp_started.isoformat(timespec='seconds')}",
            ]
        )

        run_cmd(
            train_args_for_experiment(exp["train_args"]),
            step=f"Training: {exp['label']}",
        )

        ckpt = latest_checkpoint(exp["glob"])
        if ckpt is None:
            raise SystemExit(
                f"No best_model.pth found for pattern {exp['glob']} under {RUNS_DIR}"
            )
        print(f"  -> checkpoint: {ckpt}", flush=True)
        all_checkpoints.append(str(ckpt))

        exp_train_done = datetime.now()
        write_summary(
            [f"Training finished: {exp_train_done.isoformat(timespec='seconds')}"]
        )

        evaluate_checkpoint(ckpt)
        append_metrics_to_summary(exp["label"], ckpt)

    write_summary(["--- Final evaluation (all 3 models) ---"])
    run_cmd(
        [
            str(ROOT / "evaluate.py"),
            "--output_dir",
            str(EVAL_DIR),
            "--checkpoints",
            *all_checkpoints,
            "--batch_size",
            str(TRAIN_BATCH_SIZE),
            "--num_workers",
            str(NUM_WORKERS),
        ],
        step="Evaluation: U-Net + SegFormer + U-TAE results table",
    )

    write_summary(["--- unet_ftw (pretrained) metrics ---"])
    append_metrics_to_summary("unet_ftw (pretrained)", EXISTING_UNET_CHECKPOINT)

    run_ended = datetime.now()
    write_summary(
        [
            f"Experiment run ended: {run_ended.isoformat(timespec='seconds')}",
            f"Total duration: {run_ended - run_started}",
            f"Checkpoints evaluated: {', '.join(all_checkpoints)}",
            f"Summary log: {SUMMARY_PATH}",
        ]
    )

    print("\nAll experiments finished.", flush=True)
    print(f"Models in results table: {len(all_checkpoints)}", flush=True)
    for ckpt in all_checkpoints:
        print(f"  - {ckpt}", flush=True)
    print(f"Summary log: {SUMMARY_PATH}", flush=True)
    print(f"Results: {EVAL_DIR / 'results_table.csv'}", flush=True)
    print(f"         {EVAL_DIR / 'results_table.md'}", flush=True)


if __name__ == "__main__":
    main()
