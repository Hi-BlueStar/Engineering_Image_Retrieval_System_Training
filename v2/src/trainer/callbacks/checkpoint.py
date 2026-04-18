"""
Checkpoint 儲存回調 (Checkpoint Callback)。
解耦模型的儲存邏輯，獨立負責快照的 IO 行為。
"""

import torch
from pathlib import Path
from ..engine import Callback

class CheckpointCallback(Callback):
    """負責定期儲存快照，以及維護全局 Best Loss 的模型。"""
    
    def __init__(self, save_dir: Path, save_freq: int):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_freq = save_freq
        self.best_loss = float("inf")

    def on_epoch_end(self, trainer, epoch: int, metrics: dict):
        state = {
            "epoch": epoch,
            "state_dict": trainer.model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "scheduler": trainer.scheduler.state_dict(),
            "val_loss": metrics["val_loss"],
        }
        
        # 1. 每次 Epoch 儲存最新的模型快取 (Last)
        torch.save(state, self.save_dir / "checkpoint_last.pth")
        
        # 2. 依照頻率儲存定期快照 (Interval)
        if epoch % self.save_freq == 0:
            torch.save(state, self.save_dir / f"checkpoint_epoch_{epoch:04d}.pth")
            
        # 3. 如果監測到更好的 Val Loss，覆蓋 Best 模型
        if metrics["val_loss"] < self.best_loss:
            self.best_loss = metrics["val_loss"]
            torch.save(state, self.save_dir / "checkpoint_best.pth")
