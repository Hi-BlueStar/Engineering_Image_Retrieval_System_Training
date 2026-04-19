"""模型訓練管線入口 (Model Training Pipeline Entry Point)。

============================================================
Pipeline 2: 載入設定 → 建立 DataLoader → 多 Run 訓練迴圈 → 彙整結果

假設資料已由 ``prepare_data.py`` (Pipeline 1) 準備完成。

使用方式::

    python v2/train.py --config v2/configs/default.yaml
    python v2/train.py --config v2/configs/default.yaml training.lr=3e-4 training.epochs=100
============================================================
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# --- 確保專案根目錄在 Python Path ---
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AppConfig
from src.dataset.dataloader import create_dataloaders
from src.experiment.reporter import generate_run_reports
from src.experiment.tracker import ExperimentTracker
from src.logger import get_logger, setup_logging
from src.model.simsiam import SimSiam
from src.training.checkpoint import CheckpointManager
from src.training.timer import TimerCollection
from src.training.trainer import Trainer

logger = get_logger(__name__)


def _run_single_training(
    cfg: AppConfig,
    run_idx: int,
    seed: int,
    tracker: ExperimentTracker,
    timers: TimerCollection,
) -> dict:
    """執行單個 Run 的完整訓練流程。

    Args:
        cfg: 應用程式設定。
        run_idx: Run 索引（0-indexed）。
        seed: 此 Run 的隨機種子。
        tracker: 實驗追蹤器。
        timers: 計時器集合。

    Returns:
        dict: 該 Run 的結果摘要。
    """
    run_name = f"Run_{run_idx + 1:02d}_Seed_{seed}"
    run_timer = timers.create(f"run_{run_idx + 1:02d}_total")
    run_timer.start()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        "啟動訓練: %s (device=%s, seed=%d)", run_name, device, seed
    )

    d = cfg.data
    m = cfg.model
    t = cfg.training
    e = cfg.experiment

    try:
        # --- DataLoader ---
        run_dir_name = f"Run_{run_idx + 1:02d}_Seed_{seed}"
        train_path = Path(d.dataset_dir) / run_dir_name / d.train_subpath
        val_path = Path(d.dataset_dir) / run_dir_name / d.val_subpath

        train_loader, val_loader, n_train, n_val = create_dataloaders(
            train_path, val_path, t, in_channels=m.in_channels
        )

        # --- Model (每 Run 重新初始化) ---
        torch.manual_seed(seed)
        model = SimSiam(
            backbone=m.backbone,
            proj_dim=m.proj_dim,
            pred_hidden=m.pred_hidden,
            pretrained=m.pretrained,
            in_channels=m.in_channels,
        ).to(device)

        # --- Optimizer & Scheduler ---
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=t.lr, weight_decay=t.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t.epochs
        )
        scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

        # --- Checkpoint Manager ---
        run_tracker = tracker.create_run(run_name)
        ckpt_mgr = CheckpointManager(
            ckpt_dir=run_tracker.run_dir / "checkpoints",
            save_freq=e.save_freq,
            config_dict=cfg.to_dict(),
        )

        # --- Trainer ---
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            checkpoint_mgr=ckpt_mgr,
            device=device,
        )

        # --- 訓練 ---
        result = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=t.epochs,
            epoch_callback=lambda m: run_tracker.log_epoch(m),
        )

        # --- 報表 ---
        df = run_tracker.get_logs_dataframe()
        generate_run_reports(df, run_tracker.run_dir, run_name)

        run_timer.stop()

        return {
            "run_name": run_name,
            "seed": seed,
            "best_val_loss": result["best_val_loss"],
            "n_train": n_train,
            "n_val": n_val,
            "status": "success",
            "log_dir": str(run_tracker.run_dir),
        }

    except Exception as e:
        run_timer.stop()
        logger.error("Run %s 失敗: %s", run_name, e, exc_info=True)
        return {
            "run_name": run_name,
            "seed": seed,
            "best_val_loss": None,
            "status": "failed",
            "error": str(e),
        }


def main() -> None:
    """SimSiam 訓練管線主入口。

    流程：
        1. 載入與驗證設定。
        2. 初始化實驗追蹤器。
        3. 多 Run 訓練迴圈。
        4. 彙整結果與報告。
    """
    # --- CLI ---
    parser = argparse.ArgumentParser(
        description="SimSiam 模型訓練管線",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="v2/configs/default.yaml",
        help="YAML 設定檔路徑",
    )
    args, overrides = parser.parse_known_args()

    # --- 設定 ---
    cfg = AppConfig.from_yaml(args.config, cli_overrides=overrides)
    cfg.validate()

    # --- 日誌 ---
    setup_logging(
        level="INFO",
        log_file=str(
            Path(cfg.experiment.output_dir) / cfg.experiment.log_file
        ),
        use_rich=True,
    )

    # --- 計時器 ---
    timers = TimerCollection()
    total_timer = timers.create("total_training")
    total_timer.start()

    # --- 實驗追蹤 ---
    tracker = ExperimentTracker(config=cfg, timers=timers)

    # --- 多 Run 訓練 ---
    logger.info(
        "排程: %d 個 Run, base_seed=%d",
        cfg.data.n_runs,
        cfg.data.base_seed,
    )

    run_results: list[dict] = []
    for run_idx in range(cfg.data.n_runs):
        seed = cfg.data.base_seed + run_idx
        result = _run_single_training(cfg, run_idx, seed, tracker, timers)
        run_results.append(result)

    # --- 總結 ---
    total_elapsed = total_timer.stop()

    # 儲存報告
    tracker.save_timing_report()
    tracker.save_summary(run_results)

    # 輸出結果摘要
    logger.info("=" * 60)
    logger.info("訓練完成 — 總結報告")
    logger.info("=" * 60)

    for r in run_results:
        loss_str = (
            f"{r['best_val_loss']:.4f}"
            if isinstance(r.get("best_val_loss"), float)
            else str(r.get("best_val_loss", "N/A"))
        )
        status = "✔" if r.get("status") == "success" else "✗"
        logger.info(
            "  %s %s — Best Val Loss: %s",
            status,
            r.get("run_name", ""),
            loss_str,
        )

    logger.info("實驗目錄: %s", tracker.experiment_dir)
    logger.info("總執行時間: %.1f 分鐘", total_elapsed / 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("使用者中斷訓練。已完成的紀錄已保存。")
    except Exception:
        logger.exception("訓練管線異常終止")
        sys.exit(1)
