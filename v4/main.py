"""SimSiam v4 訓練管線啟動器 (Main Executor CLI)。

============================================================
這是自監督學習 SimSiam v4 的 CLI 入口點。

執行流程：
    1. 載入與組合命令列參數（覆寫預設設定檔）。
    2. 配置高亮日誌。
    3. **前處理防呆檢測**：
       - 若設定直接加載快取且 `.npz` 檔存在，則秒級加載資料。
       - 否則，檢測是否需要解壓縮、PDF轉圖與連通域前處理裁切，隨後對子圖執行 CPU Letterbox 等比例縮放至 $512 \times 512$，並寫入 `.npz` 壓縮快取。
    4. 初始化非同步預取資料載入器 `GPUPrefetchDataLoader` 與 GPU 增強模組。
    5. 初始化模型解耦結構：`SimSiamEncoder` 與 `SimSiamPredictor`。
    6. 呼叫 `SimSiamTrainer` 進行 BF16 混合精度訓練。
    7. 生成 Plotly HTML 互動式報告並匯整指標。
============================================================
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# 將 v4 目錄優先加入至 Python 模組搜索路徑，避免與專案根目錄的 src 模組衝突
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import numpy as np

from src.config import AppConfig
from src.logger import setup_logging, get_logger
from src.data_pipeline import (
    extract_raw_zip,
    convert_pdfs,
    run_preprocessing_pipeline,
    build_and_cache_dataset,
    NPZDataset,
    GPUPrefetchDataLoader,
    GPUAugmentationModule
)
from src.models import SimSiamEncoder, SimSiamPredictor
from src.criterion import SimSiamLossCriterion
from src.trainer import SimSiamTrainer
from src.mlops import RunTracker, generate_plotly_report

logger = get_logger("v4.main")


# 取得預設的設定檔絕對路徑，確保不論在何處啟動，皆能精確讀取 v4/v4_config.yaml
DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().parent / "v4_config.yaml")


def parse_args() -> argparse.Namespace:
    """解析 CLI 入口參數"""
    parser = argparse.ArgumentParser(description="SimSiam v4 工程圖自監督學習訓練入口 CLI")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="預設 YAML 設定檔路徑")
    parser.add_argument("--load_cached_npz", action="store_true", help="是否強制直接讀取已壓縮的 .npz 快取檔以進行訓練")
    parser.add_argument("--epochs", type=int, default=None, help="設定訓練 Epochs 數量")
    parser.add_argument("--batch_size", type=int, default=None, help="設定 Batch 大小")
    parser.add_argument("--lr", type=float, default=None, help="設定學習率")
    parser.add_argument("--backbone", type=str, default=None, choices=["resnet18", "resnet50"], help="骨幹網路名稱")
    parser.add_argument("--use_bf16", type=str, default=None, choices=["true", "false"], help="是否啟用 AMP BF16 訓練")
    parser.add_argument("--compile_model", type=str, default=None, choices=["true", "false"], help="是否對模型啟用編譯加速")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. 建立 CLI 覆寫列表 (Dotlist 格式)
    overrides = []
    if args.load_cached_npz:
        overrides.append("data.load_cached_npz=True")
    if args.epochs is not None:
        overrides.append(f"training.epochs={args.epochs}")
    if args.batch_size is not None:
        overrides.append(f"training.batch_size={args.batch_size}")
    if args.lr is not None:
        overrides.append(f"training.lr={args.lr}")
    if args.backbone is not None:
        overrides.append(f"model.backbone={args.backbone}")
    if args.use_bf16 is not None:
        val = "True" if args.use_bf16 == "true" else "False"
        overrides.append(f"training.use_bf16={val}")
    if args.compile_model is not None:
        val = "True" if args.compile_model == "true" else "False"
        overrides.append(f"training.compile_model={val}")

    # 2. 載入與組合強型別設定，並執行驗證
    cfg = AppConfig.from_yaml(args.config, cli_overrides=overrides)
    cfg.validate()

    # 3. 初始化日誌系統 (日誌檔案保存在輸出目錄)
    log_dir = Path(cfg.experiment.output_dir) / cfg.experiment.exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(
        level="INFO",
        log_file=str(log_dir / cfg.experiment.log_file),
        use_rich=True,
        force=True
    )

    logger.info("=========================================")
    logger.info("   SimSiam v4 系統前處理與訓練管線啟動")
    logger.info("=========================================")

    # 4. 前處理防呆檢測與快取加載機制
    cache_path = Path(cfg.data.cache_npz_path)
    loaded_from_cache = False
    
    if cfg.data.load_cached_npz and cache_path.is_file():
        logger.info("偵測到指定載入已存檔快取，正在讀取: %s", cache_path)
        try:
            with np.load(cache_path, allow_pickle=True) as data:
                train_x = data["train_images"]
                train_y = data["train_labels"]
                val_x = data["val_images"]
                val_y = data["val_labels"]
                class_names = list(data["class_names"])
            loaded_from_cache = True
            logger.info("極速快取讀取成功: Train=%d 張, Val=%d 張 | 類別數=%d", len(train_x), len(val_x), len(class_names))
        except Exception as e:
            logger.error("讀取 .npz 快取失敗，將回退到常規前處理防呆流程。錯誤: %s", e)

    if not loaded_from_cache:
        logger.info("快取未就緒，啟動防呆前處理與資料解構檢查...")
        
        # A. 檢查前處理目錄是否已有裁切圖，若無則啟動上游管線
        preprocessed_dir = Path(cfg.data.preprocessed_image_dir)
        has_preprocessed = preprocessed_dir.exists() and any(preprocessed_dir.glob("**/large_components/comp_*.png"))
        
        if not has_preprocessed:
            logger.info("未檢測到前處理零件影像，向上檢查原始 PNG 圖檔...")
            converted_dir = Path(cfg.data.converted_image_dir)
            has_converted = converted_dir.exists() and any(converted_dir.glob("**/*.png"))
            
            if not has_converted:
                logger.info("未檢測到已轉檔影像，向上檢查原始 PDF 目錄...")
                raw_pdfs = Path(cfg.data.raw_pdf_dir)
                has_pdfs = raw_pdfs.exists() and any(raw_pdfs.glob("**/*.pdf"))
                
                if not has_pdfs:
                    logger.info("未檢測到原始 PDF，將解壓原始壓縮檔 %s ...", cfg.data.raw_zip_path)
                    if cfg.data.raw_zip_path:
                        extract_raw_zip(cfg.data.raw_zip_path, cfg.data.raw_pdf_dir)
                    else:
                        raise FileNotFoundError("錯誤: 找不到 PDF，且 raw_zip_path 為 None，無法實作前處理。")
                
                # PDF 轉 PNG
                convert_pdfs(
                    pdf_dir=cfg.data.raw_pdf_dir,
                    output_dir=cfg.data.converted_image_dir,
                    dpi=cfg.data.pdf_dpi,
                    max_workers=cfg.data.pdf_max_workers
                )
            
            # 實作連通域分割前處理
            run_preprocessing_pipeline(
                input_dir=cfg.data.converted_image_dir,
                output_dir=cfg.data.preprocessed_image_dir,
                top_n=cfg.data.preprocess_top_n,
                max_bbox_ratio=cfg.data.preprocess_max_bbox_ratio,
                padding=cfg.data.preprocess_padding,
                remove_logo=cfg.data.remove_gifu_logo,
                logo_template=cfg.data.logo_template_path,
                logo_mask=cfg.data.logo_mask_region,
                max_workers=cfg.data.preprocess_max_workers
            )
            
        # B. 資料集切割、CPU Letterbox 縮放並寫入新快取
        train_x, train_y, val_x, val_y, class_names = build_and_cache_dataset(
            preprocessed_dir=cfg.data.preprocessed_image_dir,
            cache_path=str(cache_path),
            split_ratio=cfg.data.split_ratio,
            seed=cfg.data.base_seed,
            size=cfg.training.img_size
        )
        logger.info("已生成資料快取檔: %s", cache_path)

    # 5. 硬體配置初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("當前使用硬體設備: %s", device)
    
    # 固定隨機碼確保可重現性
    torch.manual_seed(cfg.data.base_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.data.base_seed)

    # 6. 初始化 PyTorch 高效預取 Dataloader
    train_ds = NPZDataset(train_x, train_y)
    val_ds = NPZDataset(val_x, val_y)
    
    train_loader = GPUPrefetchDataLoader(
        dataset=train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        device=device,
        use_bf16=cfg.training.use_bf16,
        mean=cfg.data.norm_mean,
        std=cfg.data.norm_std
    )
    
    val_loader = GPUPrefetchDataLoader(
        dataset=val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        device=device,
        use_bf16=cfg.training.use_bf16,
        mean=cfg.data.norm_mean,
        std=cfg.data.norm_std
    )

    # 7. 初始化模型拓撲與優化器
    encoder = SimSiamEncoder(
        backbone_name=cfg.model.backbone,
        proj_dim=cfg.model.proj_dim,
        in_channels=cfg.model.in_channels,
        pretrained=cfg.model.pretrained
    ).to(device)
    
    predictor = SimSiamPredictor(
        proj_dim=cfg.model.proj_dim,
        pred_hidden=cfg.model.pred_hidden
    ).to(device)
    
    # 幾何敏感隨機資料增強 (載入到 GPU)
    gpu_augmentor = GPUAugmentationModule(
        img_size=cfg.training.img_size,
        use_augmentation=cfg.training.use_augmentation
    ).to(device)
    
    # 損失與監控
    criterion = SimSiamLossCriterion().to(device)
    
    # 優化器與學習率排程
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.epochs
    )

    # 8. 初始化 MLOps 紀錄
    tracker = RunTracker(
        run_name=cfg.experiment.exp_name,
        run_dir=log_dir,
        config_dict=cfg.to_dict()
    )

    # 9. 初始化訓練器並啟動
    trainer = SimSiamTrainer(
        encoder=encoder,
        predictor=predictor,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        gpu_augmentor=gpu_augmentor,
        config=cfg,
        device=device
    )
    
    start_time = time.perf_counter()
    try:
        trainer.fit(train_loader, val_loader, tracker)
        logger.info("訓練成功完成！耗時: %.2f 分鐘", (time.perf_counter() - start_time) / 60)
    except Exception as e:
        logger.error("訓練中途崩潰，請檢查日誌資訊。錯誤: %s", e)
        sys.exit(1)

    # 10. 產出 Plotly HTML 互動式報告
    generate_plotly_report(
        log_csv_path=tracker.log_csv_path,
        output_report_path=log_dir / "training_report.html",
        exp_name=cfg.experiment.exp_name
    )
    logger.info("SimSiam v4 流程結束。")


if __name__ == "__main__":
    main()
