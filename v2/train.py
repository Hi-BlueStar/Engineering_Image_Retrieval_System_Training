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
import torch.backends.cudnn as cudnn
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


# --- 確保專案根目錄在 Python Path ---
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AppConfig
from src.dataset.dataloader import create_dataloaders
from src.dataset.gpu_transforms import GPUAugmentation
from src.dataset.labeled_dataset import LabeledImageDataset
from src.evaluation.evaluator import evaluate_model, save_metrics
from src.experiment.reporter import generate_run_reports
from src.experiment.tracker import ExperimentTracker
from src.logger import get_logger, setup_logging
from src.model.simsiam import SimSiam
from src.training.checkpoint import CheckpointManager
from src.training.timer import TimerCollection
from src.training.trainer import Trainer

logger = get_logger(__name__)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    t: "TrainingConfig",  # noqa: F821
) -> torch.optim.lr_scheduler.LRScheduler:
    """依設定建立學習率排程器。

    Args:
        optimizer: 優化器。
        t: TrainingConfig，讀取 ``scheduler`` 與 ``epochs``。

    Returns:
        LRScheduler 實例。

    Raises:
        ValueError: 當 ``scheduler`` 值不被支援時。
    """
    name = t.scheduler.lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t.epochs
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, t.epochs // 3), gamma=0.1
        )
    if name == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    raise ValueError(
        f"不支援的 scheduler: '{t.scheduler}'。可用: cosine, step, constant"
    )


def _run_single_training(
    cfg: AppConfig,
    run_idx: int,
    seed: int,
    tracker: ExperimentTracker,
    timers: TimerCollection,
    labeled_ds: Optional[LabeledImageDataset] = None,
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
        val_path = Path(d.dataset_dir) / run_dir_name / d.test_subpath

        use_gpu_aug = t.use_gpu_augmentation and device == "cuda"
        train_loader, val_loader, n_train, n_val = create_dataloaders(
            train_path, val_path, t,
            in_channels=m.in_channels,
            seed=seed,
            use_gpu_augmentation=use_gpu_aug,
        )

        # --- Model (每 Run 重新初始化) ---
        torch.manual_seed(seed)
        model = SimSiam(
            backbone=m.backbone,
            proj_dim=m.proj_dim,
            proj_hidden=m.proj_hidden,
            pred_hidden=m.pred_hidden,
            pretrained=m.pretrained,
            in_channels=m.in_channels,
        ).to(device)

        # --- GPU Augmentation ---
        gpu_aug = None
        if use_gpu_aug:
            gpu_aug = GPUAugmentation(
                img_size=t.img_size,
                use_augmentation=t.use_augmentation,
                in_channels=m.in_channels,
            ).to(device)

        # --- Optimizer & Scheduler ---
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=t.lr, weight_decay=t.weight_decay
        )
        scheduler = _build_scheduler(optimizer, t)
        scaler = torch.amp.GradScaler(device="cuda", enabled=(device == "cuda" and t.use_amp))

        # --- Checkpoint Manager ---
        run_tracker = tracker.create_run(run_name)
        ckpt_mgr = CheckpointManager(
            ckpt_dir=run_tracker.run_dir / "checkpoints",
            save_freq=e.save_freq,
            config_dict=cfg.to_dict(),
        )

        # --- 斷點恢復 ---
        start_epoch = 1
        resume_best_val_loss = float("inf")
        if t.resume:
            latest_ckpt = ckpt_mgr.find_latest_checkpoint()
            if latest_ckpt is not None:
                state = ckpt_mgr.load(
                    latest_ckpt, model, optimizer, scheduler, scaler
                )
                start_epoch = state.get("epoch", 0) + 1
                resume_best_val_loss = state.get("val_loss", float("inf"))
                logger.info(
                    "Resume 成功: %s → 從 epoch %d 繼續訓練",
                    latest_ckpt.name,
                    start_epoch,
                )

        # --- torch.compile (PyTorch 2.0+) ---
        if hasattr(torch, "compile") and device == "cuda":
            try:
                logger.info("使用 torch.compile 優化模型...")
                model = torch.compile(model)
            except Exception as e:
                logger.warning("torch.compile 失敗: %s", e)

        # --- Trainer ---
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            checkpoint_mgr=ckpt_mgr,
            device=device,
            grad_clip=t.grad_clip,
            gpu_aug=gpu_aug,
            max_batches=t.max_batches,
        )

        # --- 訓練 ---
        result = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=t.epochs,
            start_epoch=start_epoch,
            best_val_loss=resume_best_val_loss,
            epoch_callback=lambda metrics: run_tracker.log_epoch(metrics),
        )

        # --- 報表 ---
        df = run_tracker.get_logs_dataframe()
        generate_run_reports(df, run_tracker.run_dir, run_name)

        # --- 評估 ---
        eval_metrics = {}
        if labeled_ds is not None:
            try:
                eval_model = getattr(model, "_orig_mod", model)
                eval_metrics = evaluate_model(
                    model=eval_model,
                    labeled_dataset=labeled_ds,
                    device=device,
                    top_k_values=d.eval_top_k_values,
                    batch_size=t.batch_size,
                    num_workers=t.num_workers,
                )
                save_metrics(
                    eval_metrics,
                    run_tracker.run_dir / "retrieval_metrics.json",
                    extra_info={"run_name": run_name},
                )
            except Exception as eval_err:
                logger.warning("評估失敗: %s", eval_err)

        run_timer.stop()

        return {
            "run_name": run_name,
            "seed": seed,
            "best_val_loss": result["best_val_loss"],
            "n_train": n_train,
            "n_val": n_val,
            "status": "success",
            "log_dir": str(run_tracker.run_dir),
            **eval_metrics,
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
        level=cfg.logging.level,
        log_file=(
            str(Path(cfg.experiment.output_dir) / cfg.experiment.log_file)
            if cfg.logging.log_to_file
            else None
        ),
        use_rich=cfg.logging.use_rich,
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

    # --- 預先載入評估資料集 (避免多 Run 重複掃描) ---
    labeled_ds = None
    labeled_path = Path(cfg.data.labeled_data_path)
    if labeled_path.exists():
        logger.info("正在初始化評估資料集: %s", labeled_path)
        labeled_ds = LabeledImageDataset(
            root=labeled_path,
            img_size=cfg.training.img_size,
            in_channels=cfg.model.in_channels,
        )

    # --- GPU 核心優化 ---
    if torch.cuda.is_available():
        cudnn.benchmark = True
        logger.info("已啟用 cudnn.benchmark")

    run_results: list[dict] = []
    for run_idx in range(cfg.data.n_runs):
        seed = cfg.data.base_seed + run_idx
        result = _run_single_training(cfg, run_idx, seed, tracker, timers, labeled_ds)
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
