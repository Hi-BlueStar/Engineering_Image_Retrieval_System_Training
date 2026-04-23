"""消融實驗訓練管線 (Ablation Study Training Pipeline)。

============================================================
依序執行 6 個消融實驗，每個實驗：
    1. 依設定前處理資料（連通元件、拓撲、Logo 移除）
    2. 分層抽樣切割 train/test
    3. 訓練 SimSiam 模型（GPU 增強）
    4. 評估檢索指標（IACS, Inter, Margin, Top-K）
    5. 將指標結果彙整至 ablation_summary.json

使用方式::

    # 執行全部 6 個消融實驗
    python v2/ablation_train.py

    # 執行指定實驗
    python v2/ablation_train.py --configs v2/configs/ablation/01_baseline.yaml v2/configs/ablation/06_all_preprocessing.yaml

    # 跳過前處理（已完成）
    python v2/ablation_train.py --skip-prep
============================================================
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AppConfig
from src.data.preprocessing import PreprocessConfig, preprocess_images
from src.data.splitter import split_dataset
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

_DEFAULT_CONFIGS = [
    "v2/configs/ablation/01_baseline.yaml",
    "v2/configs/ablation/02_no_topology.yaml",
    "v2/configs/ablation/03_no_cc.yaml",
    "v2/configs/ablation/04_no_augmentation.yaml",
    "v2/configs/ablation/05_no_logo.yaml",
    "v2/configs/ablation/06_all_preprocessing.yaml",
]


# ============================================================
# 主流程
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="SimSiam 消融實驗管線")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=_DEFAULT_CONFIGS,
        help="消融實驗 YAML 設定檔路徑列表",
    )
    parser.add_argument(
        "--skip-prep",
        action="store_true",
        help="跳過資料前處理（假設已完成）",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="跳過訓練（僅做評估）",
    )
    args = parser.parse_args()

    setup_logging(level="INFO", use_rich=True)
    logger.info("=" * 70)
    logger.info("SimSiam 消融實驗開始: %d 個設定", len(args.configs))
    logger.info("=" * 70)

    all_results: list[dict] = []

    for config_path in args.configs:
        logger.info("\n" + "─" * 70)
        logger.info("處理設定: %s", config_path)
        logger.info("─" * 70)

        try:
            result = run_ablation_experiment(
                config_path,
                skip_prep=args.skip_prep,
                skip_train=args.skip_train,
            )
            all_results.append(result)
        except Exception:
            logger.exception("實驗失敗: %s", config_path)
            all_results.append({
                "config": config_path,
                "status": "failed",
            })

    # ── 彙整結果 ──
    summary_path = Path("outputs/ablation/ablation_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    logger.info("\n" + "=" * 70)
    logger.info("消融實驗完成！結果彙整: %s", summary_path)
    logger.info("=" * 70)
    _print_summary_table(all_results)


# ============================================================
# 單一消融實驗
# ============================================================


def run_ablation_experiment(
    config_path: str,
    skip_prep: bool = False,
    skip_train: bool = False,
) -> dict:
    """執行單一消融實驗的完整流程。

    Returns:
        dict: 包含 exp_name, metrics, best_val_loss 的結果字典。
    """
    cfg = AppConfig.from_yaml(config_path)
    cfg.validate()

    exp_name = cfg.experiment.exp_name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = cfg.data.base_seed
    d = cfg.data
    m = cfg.model
    t = cfg.training
    e = cfg.experiment

    logger.info("實驗: %s | device=%s | seed=%d", exp_name, device, seed)

    # ── Step 1: 前處理 ──
    if not skip_prep:
        _run_preprocessing(cfg)

    # ── Step 2: 資料集分割 ──
    run_name = f"Run_01_Seed_{seed}"
    if not skip_prep:
        n_train, n_test = split_dataset(
            source_root=d.preprocessed_image_dir,
            output_root=d.dataset_dir,
            run_name=run_name,
            split_ratio=d.split_ratio,
            seed=seed,
        )
        logger.info("分割完成: train_stems=%d, test_stems=%d", n_train, n_test)

    # ── Step 3: 建立 DataLoader ──
    train_path = Path(d.dataset_dir) / run_name / d.train_subpath
    val_path = Path(d.dataset_dir) / run_name / d.test_subpath

    use_gpu_aug = t.use_gpu_augmentation and device == "cuda"
    train_loader, val_loader, n_train_imgs, n_val_imgs = create_dataloaders(
        train_path,
        val_path,
        t,
        in_channels=m.in_channels,
        seed=seed,
        use_gpu_augmentation=use_gpu_aug,
    )

    # ── Step 4: 訓練 ──
    best_val_loss = None
    timers = TimerCollection()

    if not skip_train:
        torch.manual_seed(seed)
        model = SimSiam(
            backbone=m.backbone,
            proj_dim=m.proj_dim,
            proj_hidden=m.proj_hidden,
            pred_hidden=m.pred_hidden,
            pretrained=m.pretrained,
            in_channels=m.in_channels,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=t.lr, weight_decay=t.weight_decay
        )
        scheduler = _build_scheduler(optimizer, t)
        scaler = torch.amp.GradScaler(enabled=(device == "cuda" and t.use_amp))

        gpu_aug = None
        if use_gpu_aug:
            gpu_aug = GPUAugmentation(
                img_size=t.img_size,
                use_augmentation=t.use_augmentation,
                in_channels=m.in_channels,
            ).to(device)

        tracker = ExperimentTracker(config=cfg, timers=timers)
        run_tracker = tracker.create_run(run_name)

        ckpt_mgr = CheckpointManager(
            ckpt_dir=run_tracker.run_dir / "checkpoints",
            save_freq=e.save_freq,
            config_dict=cfg.to_dict(),
        )

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            checkpoint_mgr=ckpt_mgr,
            device=device,
            grad_clip=t.grad_clip,
            gpu_aug=gpu_aug,
        )

        train_timer = timers.create("training")
        train_timer.start()

        train_result = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=t.epochs,
            epoch_callback=lambda metrics: run_tracker.log_epoch(metrics),
        )
        train_timer.stop()

        best_val_loss = train_result["best_val_loss"]

        # 報表
        df = run_tracker.get_logs_dataframe()
        generate_run_reports(df, run_tracker.run_dir, run_name)

        # ── Step 5: 評估 ──
        eval_metrics = _run_evaluation(cfg, model, device)

        # 儲存評估結果
        save_metrics(
            eval_metrics,
            run_tracker.run_dir / "retrieval_metrics.json",
            extra_info={"exp_name": exp_name, "best_val_loss": best_val_loss},
        )
    else:
        logger.info("跳過訓練 (--skip-train)")
        eval_metrics = {}
        model = None

    return {
        "config": str(config_path),
        "exp_name": exp_name,
        "best_val_loss": best_val_loss,
        "n_train_imgs": n_train_imgs,
        "n_val_imgs": n_val_imgs,
        "status": "success",
        **eval_metrics,
    }


def _run_preprocessing(cfg: AppConfig) -> None:
    d = cfg.data
    prep_cfg = PreprocessConfig(
        input_dir=d.converted_image_dir,
        output_root=d.preprocessed_image_dir,
        max_workers=d.preprocess_max_workers,
        top_n=d.preprocess_top_n,
        max_bbox_ratio=d.preprocess_max_bbox_ratio,
        padding=d.preprocess_padding,

        use_connected_components=d.use_connected_components,
        use_topology_analysis=d.use_topology_analysis,
        use_topology_pruning=d.use_topology_pruning,
        topology_pruning_iters=d.topology_pruning_iters,
        topology_pruning_ksize=d.topology_pruning_ksize,
        min_simple_area=d.min_simple_area,
        remove_gifu_logo=d.remove_gifu_logo,
        logo_template_path=d.logo_template_path,
        logo_mask_region=d.logo_mask_region,
    )
    preprocess_images(prep_cfg, skip=d.skip_preprocessing)


def _run_evaluation(cfg: AppConfig, model: torch.nn.Module, device: str) -> dict:
    d = cfg.data
    t = cfg.training
    m = cfg.model

    labeled_path = Path(d.labeled_data_path)
    if not labeled_path.exists():
        logger.warning("評估資料路徑不存在: %s，跳過評估", labeled_path)
        return {}

    try:
        labeled_ds = LabeledImageDataset(
            root=labeled_path,
            img_size=t.img_size,
            in_channels=m.in_channels,
        )
        return evaluate_model(
            model=model,
            labeled_dataset=labeled_ds,
            device=device,
            top_k_values=d.eval_top_k_values,
        )
    except Exception as exc:
        logger.error("評估失敗: %s", exc, exc_info=True)
        return {}


def _build_scheduler(optimizer, t):
    name = t.scheduler.lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t.epochs)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, t.epochs // 3), gamma=0.1
        )
    return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)


def _print_summary_table(results: list[dict]) -> None:
    """列印消融實驗彙整表格。"""
    header = f"{'實驗名稱':<30} {'IACS':>8} {'Inter':>8} {'Margin':>8} {'Top1':>8} {'Top5':>8} {'Top10':>8} {'狀態':>8}"
    logger.info(header)
    logger.info("─" * len(header))
    for r in results:
        if r.get("status") == "failed":
            logger.info("%-30s  %-8s", r.get("exp_name", r.get("config", "?")), "FAILED")
            continue
        logger.info(
            "%-30s  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f  %8s",
            r.get("exp_name", ""),
            r.get("IACS", 0.0),
            r.get("inter_class_avg_sim", 0.0),
            r.get("contrastive_margin", 0.0),
            r.get("top1_precision", 0.0),
            r.get("top5_precision", 0.0),
            r.get("top10_precision", 0.0),
            r.get("status", ""),
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("使用者中斷實驗")
    except Exception:
        logger.exception("消融實驗管線異常終止")
        sys.exit(1)
