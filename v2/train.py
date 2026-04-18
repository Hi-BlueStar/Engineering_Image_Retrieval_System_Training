"""
SimSiam 模型訓練管線 (Training Pipeline)。

專司：透過 CSV 零拷貝技術載入 Dataloader，執行 AMP 訓練循環。
"""
import sys
import torch
import random
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core import ConfigManager, get_logger, ConfigError
from src.data.dataloader import setup_dataloaders
from src.models import SimSiamModel
from src.trainer import TrainerEngine, CheckpointCallback, MetricsTrackerCallback

def execute_run(config: ConfigManager, run_idx: int, seed: int, logger):
    run_name = f"Run_{run_idx+1:02d}_Seed_{seed}"
    logger.info(f"==== 開始訓練 {run_name} ====")

    # 建置專屬輸出目錄
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_out_dir = Path(config.training.output_dir) / f"{config.training.exp_name}_{timestamp}" / run_name
    
    # 零拷貝 CSV 指標讀取
    run_dataset_dir = Path(config.data.dataset_root) / run_name
    train_csv = run_dataset_dir / "train_index.csv"
    val_csv = run_dataset_dir / "val_index.csv"
    
    if config.data.use_csv_index and not train_csv.exists():
        logger.error(f"找不到 CSV 索引檔: {train_csv}。請確認您已正確執行 process_data.py！")
        return
        
    logger.info("正在從大數據 CSV 索引載入資料路徑至 DataLoader...")
    train_loader, val_loader, n_train, n_val = setup_dataloaders(config, train_csv, val_csv)
    
    logger.info(f"資料大小: Train={n_train}, Val={n_val}")

    model = SimSiamModel(
        backbone=config.model.backbone,
        proj_dim=config.model.proj_dim,
        pred_hidden=config.model.pred_hidden,
        pretrained=config.model.pretrained,
        in_channels=config.model.in_channels
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training.epochs)

    callbacks = [
        MetricsTrackerCallback(run_dir=run_out_dir),
        CheckpointCallback(save_dir=run_out_dir / "checkpoints", save_freq=config.training.save_freq)
    ]

    trainer = TrainerEngine(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        callbacks=callbacks
    )

    trainer.fit()

def main():
    try:
        config_path = PROJECT_ROOT / "configs" / "default_config.yaml"
        config = ConfigManager.load_from_yaml(config_path)
        
        # 設定全局日誌
        setup_log_path = Path(config.training.output_dir) / "training_pipeline.log"
        global_logger = get_logger("TrainPipeline", log_file=setup_log_path)
        
        for i in range(config.data.n_runs):
            seed = config.data.base_seed + i
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                
            execute_run(config, run_idx=i, seed=seed, logger=global_logger)

        global_logger.info(" 🎉 所有訓練任務完美結束！")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
