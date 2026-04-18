"""主訓練腳本 (Training Entry Point)。

執行 Phase 2 與 Phase 3：負責整合 DataLoader, Model, Trainer 與 Logger。
支援多個 Run 的連續訓練與報表彙整。

使用方式:
    python -m v2.scripts.train
    # 或覆寫配置
    python -m v2.scripts.train training.batch_size=32 model.backbone=resnet50
"""
import argparse
import sys
import time
from pathlib import Path

import torch

# 確保可使用 src module
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.structured import Config
from src.data.dataloader import build_dataloaders
from src.engine.trainer import SimSiamTrainer
from src.models.simsiam import SimSiam
from src.utils.experiment_logger import ExperimentLogger
from src.utils.timer import PrecisionTimer
from src.utils.seed import seed_everything

try:
    from rich.console import Console
    console = Console()
except ImportError:
    class Console:
        def print(self, msg, *args, **kwargs):
            print(msg)
    console = Console()


def run_single_session(cfg: Config, run_name: str, train_path: Path, val_path: Path, exp_logger: ExperimentLogger, seed: int) -> dict:
    # --- [MLOps Strictness] 確保此 Run 具有全局確定的隨機性 ---
    seed_everything(seed)
    
    timer = exp_logger.timers.create(f"train_{run_name}")
    timer.start()
    
    console.print(f"\n[bold blue]🚀 啟動訓練任務: {run_name}[/bold blue]")
    
    # 1. DataLoader
    train_dl, val_dl, n_train, n_val = build_dataloaders(
        train_dir=train_path,
        val_dir=val_path,
        img_size=cfg.training.img_size,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        img_exts=cfg.training.img_exts,
        in_channels=cfg.model.in_channels
    )
    console.print(f"   ✅ 資料準備完成: Train={n_train}, Val={n_val}")

    # 2. Model & Optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimSiam(
        backbone_name=cfg.model.backbone,
        pretrained=cfg.model.pretrained,
        in_channels=cfg.model.in_channels,
        proj_dim=cfg.model.proj_dim,
        pred_hidden=cfg.model.pred_hidden
    ).to(device)
    
    # --- [MLOps Strictness] Linear Learning Rate Scaling Rule ---
    # 自監督學習的 LR 應該與 batch_size 成正比，依照原論文標準處理：
    # 假設預設 cfg.training.lr 是對應到 batch_size=256 的 base_lr
    base_lr = cfg.training.lr
    scaled_lr = base_lr * (cfg.training.batch_size / 256.0)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=scaled_lr, weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)
    
    # 3. Trainer & Logger
    trainer = SimSiamTrainer(model=model, optimizer=optimizer, device=device)
    run_logger = exp_logger.create_run_logger(run_name)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, cfg.training.epochs + 1):
        ep_timer = PrecisionTimer(f"ep_{epoch}")
        ep_timer.start()
        
        train_loss, train_std = trainer.train_one_epoch(train_dl)
        val_loss, val_std = trainer.evaluate(val_dl) if len(val_dl) > 0 else (0.0, 0.0)
        
        # Pause for I/O
        ep_timer.pause()
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            
        run_logger.save_checkpoint(model, optimizer, epoch, val_loss, is_best)
        run_logger.log_epoch(epoch, train_loss, val_loss, train_std, val_std, optimizer.param_groups[0]['lr'], ep_timer.stop())
        
        scheduler.step()
        
        # Resume if anything else...
        if epoch % 5 == 0 or epoch == 1:
            console.print(f"   Epoch {epoch}/{cfg.training.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Std: {train_std:.4f}")
            
    run_logger.generate_report()
    timer.stop()
    
    return {
        "Run": run_name,
        "Best Val Loss": best_val_loss,
        "Log Dir": str(run_logger.run_dir)
    }


def main():
    parser = argparse.ArgumentParser(description="Run SimSiam Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    args, unknown = parser.parse_known_args()
    
    cfg = Config.load(args.config, unknown)
    exp_logger = ExperimentLogger(cfg)
    
    console.print(f"📁 實驗輸出目錄: {exp_logger.exp_dir}")

    # 尋找所有 Run
    root_path = Path(cfg.data.dataset_root)
    if not root_path.exists():
        console.print(f"[bold red]❌ 資料集目錄不存在: {root_path}[/bold red]")
        console.print("提示: 請先執行 python scripts/prepare_data.py")
        return

    run_folders = sorted([p for p in root_path.iterdir() if p.is_dir() and "Run_" in p.name])
    if not run_folders:
        console.print(f"[bold red]❌ 找不到任何 'Run_*' 資料夾在 {root_path}[/bold red]")
        return
        
    results = []
    for i, folder in enumerate(run_folders):
        train_p = folder / "train_manifest.csv"
        val_p = folder / "val_manifest.csv"
        # 自動從設定中取得 seed
        run_seed = cfg.data.base_seed + i
        
        try:
            res = run_single_session(cfg, folder.name, train_p, val_p, exp_logger, run_seed)
            results.append(res)
        except Exception as e:
            console.print(f"[bold red]❌ 任務 {folder.name} 失敗: {e}[/bold red]")
            results.append({"Run": folder.name, "Best Val Loss": "Failed", "Log Dir": str(e)})
            
    exp_logger.save_overall_summary(results)
    console.print("\n🎉 所有訓練完成！")


if __name__ == "__main__":
    main()
