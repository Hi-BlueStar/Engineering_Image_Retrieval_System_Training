#!/usr/bin/env python
"""
Train a SimSiam encoder with self-supervised learning and export training metrics.

Example:
    python -m src.model.train_simsiam --train_dir data/train --val_dir data/val --epochs 100
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass, fields
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from src.model.simsiam import (
    SimSiam,
    TransformTwice,
    UnlabeledImages,
    evaluate,
    make_norm,
    train_one_epoch,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class TrainingConfig:
    train_dir: Path
    val_dir: Path
    output_dir: Path
    epochs: int = 100
    batch_size: int = 128
    val_batch_size: int = 256
    img_size: int = 224
    backbone: str = "resnet50"
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    seed: int = 42
    device: str | None = None
    metrics_path: Path | None = None
    log_interval: int = 1


@dataclass
class TrainingResult:
    config: TrainingConfig
    train_losses: list[float]
    val_losses: list[float]
    best_epoch: int
    best_val_loss: float
    final_train_loss: float
    final_val_loss: float
    duration_seconds: float
    device: str
    model_path: Path
    chart_path: Path
    metrics_path: Path


def list_images(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise FileNotFoundError(
            f"No images with extensions {sorted(IMAGE_EXTS)} under {root}"
        )
    return files


def make_val_transform(img_size: int):
    base = T.Compose(
        [
            T.Resize(int(img_size * 1.14), interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(img_size),
        ]
    )

    class _ValTransform:
        def __init__(self, base_transform):
            self.base = base_transform

        def __call__(self, img):
            v = self.base(img)
            return v, v

    return _ValTransform(base)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SimSiam on unlabeled images.")
    parser.add_argument(
        "--train_dir", required=True, type=Path, help="Directory of training images."
    )
    parser.add_argument(
        "--val_dir", required=True, type=Path, help="Directory of validation images."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results"),
        help="Directory to store outputs.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Training batch size."
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=256, help="Validation batch size."
    )
    parser.add_argument(
        "--img_size", type=int, default=224, help="Input resolution for the backbone."
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet50"],
        help="Encoder backbone.",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="AdamW learning rate.")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="AdamW weight decay."
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="DataLoader worker processes."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Force device placement (cpu or cuda)."
    )
    parser.add_argument(
        "--metrics_path",
        type=Path,
        default=None,
        help="Optional path to store training metrics JSON.",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=1,
        help="Epoch interval for console logging.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot_losses(
    train_losses: list[float], val_losses: list[float], output_path: Path
) -> None:
    if not train_losses:
        return
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (negative cosine similarity)")
    plt.title("SimSiam Training Curve")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _config_to_dict(config: TrainingConfig) -> dict:
    data = {}
    for field in fields(config):
        value = getattr(config, field.name)
        if isinstance(value, Path):
            data[field.name] = str(value)
        else:
            data[field.name] = value
    return data


def run_training(config: TrainingConfig) -> TrainingResult:
    setup_seed(config.seed)

    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = isinstance(device, str) and device.startswith("cuda")

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "simsiam_model.pth"
    chart_path = output_dir / "loss_chart.png"
    metrics_path = config.metrics_path or (output_dir / "metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    norm = make_norm(mean, std)

    train_paths = list_images(config.train_dir)
    val_paths = list_images(config.val_dir)

    train_ds = UnlabeledImages(
        train_paths, transform=TransformTwice(config.img_size), norm=norm
    )
    val_ds = UnlabeledImages(
        val_paths, transform=make_val_transform(config.img_size), norm=norm
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = SimSiam(backbone=config.backbone).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    train_losses, val_losses = [], []
    history = []
    best_val = float("inf")
    best_epoch = 0

    start_time = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        val_loss = evaluate(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch

        if epoch % max(1, config.log_interval) == 0:
            print(
                f"[Epoch {epoch:03d}/{config.epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
            )

    duration = time.perf_counter() - start_time

    torch.save(model.state_dict(), model_path)
    plot_losses(train_losses, val_losses, chart_path)

    metrics = {
        "config": _config_to_dict(config),
        "device": device,
        "epochs": config.epochs,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
        "duration_seconds": duration,
        "model_path": str(model_path),
        "chart_path": str(chart_path),
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return TrainingResult(
        config=config,
        train_losses=train_losses,
        val_losses=val_losses,
        best_epoch=best_epoch,
        best_val_loss=best_val,
        final_train_loss=train_losses[-1] if train_losses else float("nan"),
        final_val_loss=val_losses[-1] if val_losses else float("nan"),
        duration_seconds=duration,
        device=device,
        model_path=model_path,
        chart_path=chart_path,
        metrics_path=metrics_path,
    )


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    config = TrainingConfig(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        img_size=args.img_size,
        backbone=args.backbone,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        metrics_path=args.metrics_path,
        log_interval=args.log_interval,
    )

    result = run_training(config)
    print(
        f"Training complete in {result.duration_seconds:.1f}s on device={result.device}"
    )
    print(f"Best val loss {result.best_val_loss:.4f} at epoch {result.best_epoch}")
    print(f"Model saved to {result.model_path}")
    print(f"Loss chart saved to {result.chart_path}")
    print(f"Metrics JSON saved to {result.metrics_path}")


if __name__ == "__main__":
    main()

# python -m src.model.train_simsiam --train_dir results/batch/engineering_images_100dpi_flat/train --val_dir results\batch\engineering_images_100dpi_flat\val --output_dir runs --img_size 1024 --backbone resnet18 --device cuda --epochs 100
