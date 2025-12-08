#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run hyper-parameter sweeps for SimSiam training.

Example:
    python -m src.model.simsiam_sweep \
        --train_dir data/train --val_dir data/val \
        --grid_file configs/simsiam_grid.yaml \
        --output_dir results/sweeps/resnet50
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

try:
    import yaml
except ImportError:  # pragma: no cover - dependency declared in pyproject
    yaml = None

from src.model.train_simsiam import TrainingConfig, TrainingResult, run_training


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search for SimSiam training.")
    parser.add_argument("--train_dir", required=True, type=Path, help="Directory of training images.")
    parser.add_argument("--val_dir", required=True, type=Path, help="Directory of validation images.")
    parser.add_argument("--output_dir", type=Path, default=Path("results") / "sweeps", help="Base directory to store sweep runs.")
    parser.add_argument("--grid_file", type=Path, help="YAML/JSON file describing hyper-parameter lists.")
    parser.add_argument("--grid", type=str, help="Inline JSON string describing the grid (e.g. '{\"lr\":[3e-4,1e-3]}').")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs (unless overridden in grid).")
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--val_batch_size", type=int, default=256, help="Validation batch size.")
    parser.add_argument("--img_size", type=int, default=224, help="Input resolution for the backbone.")
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet18", "resnet50"], help="Encoder backbone.")
    parser.add_argument("--lr", type=float, default=3e-4, help="AdamW learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker processes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (can be swept).")
    parser.add_argument("--device", type=str, default=None, help="Force device placement (cpu or cuda).")
    parser.add_argument("--log_interval", type=int, default=1, help="Epoch interval for console logging.")
    parser.add_argument("--run_prefix", type=str, default="run", help="Folder prefix for each sweep run.")
    parser.add_argument("--max_runs", type=int, default=None, help="Optional cap on number of combinations to execute.")
    return parser.parse_args(list(argv) if argv is not None else None)


def load_grid(args: argparse.Namespace) -> Dict[str, List]:
    if args.grid and args.grid_file:
        raise ValueError("Provide either --grid or --grid_file, not both.")
    if args.grid:
        data = json.loads(args.grid)
    elif args.grid_file:
        text = args.grid_file.read_text(encoding="utf-8")
        if args.grid_file.suffix.lower() in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("PyYAML is required to read YAML grids.")
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
    else:
        raise ValueError("A hyper-parameter grid must be supplied via --grid or --grid_file.")

    if not isinstance(data, dict):
        raise ValueError("Grid specification must be a mapping from parameter names to value lists.")

    grid: Dict[str, List] = {}
    for key, value in data.items():
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError(f"Grid values must be sequences. Key '{key}' has invalid value: {value!r}")
        if len(value) == 0:
            raise ValueError(f"Grid list for '{key}' is empty.")
        grid[key] = list(value)
    return grid


def make_base_config(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        output_dir=args.output_dir,  # will be replaced per-run
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
        metrics_path=None,
        log_interval=args.log_interval,
    )


def cast_config_value(name: str, value):
    """Convert grid values to the proper type expected by TrainingConfig."""
    field_types = {f.name: f.type for f in TrainingConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    expected_type = field_types.get(name)
    if expected_type is None:
        return value
    if expected_type in (Path, Path | None):  # type: ignore[comparison-overlap]
        return Path(value)
    return value


def ensure_valid_param(name: str) -> None:
    if name not in TrainingConfig.__dataclass_fields__:  # type: ignore[attr-defined]
        raise ValueError(f"Unknown hyper-parameter '{name}'. Valid keys: {list(TrainingConfig.__dataclass_fields__.keys())}")


def build_run_config(base: TrainingConfig, params: Dict[str, object], run_dir: Path) -> TrainingConfig:
    base_dict = asdict(base)
    base_dict.update(params)
    base_dict["output_dir"] = run_dir
    base_dict["metrics_path"] = None
    for key, value in list(base_dict.items()):
        if isinstance(getattr(TrainingConfig, key, None), Path):  # pragma: no cover
            base_dict[key] = Path(value)
    return TrainingConfig(**{k: cast_config_value(k, v) for k, v in base_dict.items()})


def make_serializable(data: Dict) -> Dict:
    serializable = {}
    for key, value in data.items():
        if isinstance(value, Path):
            serializable[key] = str(value)
        else:
            serializable[key] = value
    return serializable


def write_summary_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    keys = set()
    for row in rows:
        keys.update(row.keys())
    fieldnames = ["run_name"] + sorted(k for k in keys if k != "run_name")
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    grid = load_grid(args)
    for key in grid:
        ensure_valid_param(key)

    base_config = make_base_config(args)
    base_output = args.output_dir
    base_output.mkdir(parents=True, exist_ok=True)

    keys = list(grid.keys())
    values_lists = [grid[k] for k in keys]
    combinations = list(product(*values_lists))
    if args.max_runs is not None:
        combinations = combinations[: args.max_runs]

    print(f"Executing {len(combinations)} SimSiam runs (grid size {len(combinations)}).")

    summary_rows: List[Dict[str, object]] = []
    for idx, values in enumerate(combinations, start=1):
        params = dict(zip(keys, values))
        run_name = f"{args.run_prefix}_{idx:03d}"
        run_dir = base_output / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        config = build_run_config(base_config, params, run_dir)
        params_path = run_dir / "params.json"
        params_payload = {
            "base_config": make_serializable(asdict(base_config)),
            "overrides": make_serializable(params),
        }
        with params_path.open("w", encoding="utf-8") as f:
            json.dump(params_payload, f, indent=2)

        print(f"\n=== [{run_name}] params={params} ===")
        result: TrainingResult = run_training(config)

        summary = {
            "run_name": run_name,
            **make_serializable(params),
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
            "final_train_loss": result.final_train_loss,
            "final_val_loss": result.final_val_loss,
            "duration_seconds": result.duration_seconds,
            "model_path": str(result.model_path),
            "metrics_path": str(result.metrics_path),
        }
        summary_rows.append(summary)

    summary_csv = base_output / "sweep_summary.csv"
    write_summary_csv(summary_rows, summary_csv)

    summary_json = base_output / "sweep_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    print(f"\nSweep complete. Summary written to {summary_csv} and {summary_json}")


if __name__ == "__main__":
    main()
