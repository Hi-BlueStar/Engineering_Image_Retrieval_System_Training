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
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """建立具備 Warmup 與 Cosine Annealing 的 Scheduler。

    Args:
        optimizer: 優化器。
        t: TrainingConfig，讀取 ``epochs``。
        steps_per_epoch: 每個 epoch 的 batch 數。

    Returns:
        LRScheduler 實例。
    """
    total_iters = steps_per_epoch * t.epochs
    warmup_epochs = min(10, max(1, t.epochs // 10))
    warmup_iters = warmup_epochs * steps_per_epoch

    scheduler1 = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_iters
    )
    
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_iters - warmup_iters)
    )
    
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[scheduler1, scheduler2],
        milestones=[warmup_iters]
    )


def _get_param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """分離需套用 Weight Decay 的參數。
    
    將 biases 和 BatchNorm 參數 (ndim <= 1) 從 weight decay 中排除。
    """
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


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

        use_channels_last = bool(getattr(t, "channels_last", False)) and device == "cuda"
        if use_channels_last:
            model = model.to(memory_format=torch.channels_last)
            logger.info("Model 已轉為 channels_last 記憶體格式")

        # --- GPU Augmentation ---
        gpu_aug = None
        if use_gpu_aug:
            gpu_aug = GPUAugmentation(
                img_size=t.img_size,
                use_augmentation=t.use_augmentation,
                in_channels=m.in_channels,
            ).to(device)

        # --- Optimizer & Scheduler ---
        scaled_lr = t.lr * t.batch_size / 256.0
        logger.info("Scaling learning rate: base_lr=%.4f, batch_size=%d -> scaled_lr=%.4f", t.lr, t.batch_size, scaled_lr)
        param_groups = _get_param_groups(model, t.weight_decay)
        optimizer = torch.optim.SGD(param_groups, lr=scaled_lr, momentum=0.9)
        steps_per_epoch = len(train_loader) if t.max_batches is None else min(len(train_loader), t.max_batches)
        scheduler = _build_scheduler(optimizer, t, steps_per_epoch)
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
        resume_best_score = -float("inf")
        if t.resume:
            latest_ckpt = ckpt_mgr.find_latest_checkpoint()
            if latest_ckpt is not None:
                state = ckpt_mgr.load(
                    latest_ckpt, model, optimizer, scheduler, scaler
                )
                start_epoch = state.get("epoch", 0) + 1
                # 向下相容處理
                if "best_score" in state:
                    resume_best_score = state["best_score"]
                elif "val_loss" in state:
                    resume_best_score = -state["val_loss"]
                logger.info(
                    "Resume 成功: %s → 從 epoch %d 繼續訓練",
                    latest_ckpt.name,
                    start_epoch,
                )

        # --- torch.compile (PyTorch 2.0+) ---
        if hasattr(torch, "compile") and device == "cuda":
            try:
                logger.info("使用 torch.compile 優化模型 (mode=reduce-overhead, dynamic=False)...")
                model = torch.compile(model, mode="reduce-overhead", dynamic=False)
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
            channels_last=use_channels_last,
        )

        # --- Eval Callback 閉包 ---
        from typing import Optional
        def epoch_callback(metrics: dict, current_model: torch.nn.Module) -> Optional[float]:
            # 先寫入基本訓練與 SSL loss 紀錄
            run_tracker.log_epoch(metrics)
            
            epoch = metrics.get("epoch", 0)
            eval_freq = getattr(e, "eval_freq", 10)
            should_eval = (epoch % eval_freq == 0) or (epoch == t.epochs)
            
            # 若提供 labeled_ds，我們計算 KNN / top-1 作為 Early Stopping 依據
            if labeled_ds is not None:
                if should_eval:
                    try:
                        eval_metrics = evaluate_model(
                            model=current_model,
                            labeled_dataset=labeled_ds,
                            device=device,
                            top_k_values=d.eval_top_k_values,
                            batch_size=t.batch_size,
                            num_workers=t.num_workers,
                        )
                        metrics.update(eval_metrics)
                        run_tracker.log_epoch(metrics) # 補寫入 evaluation 紀錄
                        return eval_metrics.get("top_1_precision", None)
                    except Exception as eval_err:
                        logger.warning("Epoch %d 檢索評估失敗: %s", epoch, eval_err)
                else:
                    # 若不進行評估，且我們本來是用 top-1 當指標，則回傳 -inf 確保不覆寫 best_score
                    return -float('inf')
            return None

        # --- 訓練 ---
        result = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=t.epochs,
            start_epoch=start_epoch,
            best_score=resume_best_score,
            epoch_callback=epoch_callback,
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
            "best_score": result.get("best_score", None),
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
            "best_score": None,
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
        # TF32 matmul：在 Ampere+ 上對 FP32 路徑加速；AMP 的 FP16 路徑不受影響但 fallback 也更快
        torch.set_float32_matmul_precision("high")
        logger.info("已啟用 cudnn.benchmark, float32_matmul_precision=high")

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
        score_str = (
            f"{r['best_score']:.4f}"
            if isinstance(r.get("best_score"), float)
            else str(r.get("best_score", "N/A"))
        )
        status = "✔" if r.get("status") == "success" else "✗"
        logger.info(
            "  %s %s — Best Score: %s",
            status,
            r.get("run_name", ""),
            score_str,
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
