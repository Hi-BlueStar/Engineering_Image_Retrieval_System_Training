"""模型訓練引擎模組 (Trainer Engine)。

處理模型的一個 Epoch 前向、反向傳播，與驗證集的評估邏輯。
解耦出 Trainer 控制訓練流。
"""
import contextlib
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.simsiam import simsiam_loss
from src.utils.metrics import calculate_collapse_std


class SimSiamTrainer:
    """SimSiam 的 Epoch 控制器。"""
    
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, device: str):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    def train_one_epoch(self, dataloader: DataLoader) -> tuple[float, float]:
        """執行一個 Epoch 的訓練。
        
        Returns:
            tuple[float, float]: (平均 Loss, 平均特徵標準差)
        """
        self.model.train()
        total_loss = 0.0
        total_std = 0.0
        num_batches = 0
        
        use_amp = str(self.device).startswith("cuda")
        
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16) if use_amp else contextlib.nullcontext()
        else:
            amp_ctx = torch.cuda.amp.autocast(enabled=use_amp)

        for v1, v2 in dataloader:
            v1 = v1.to(self.device, non_blocking=True)
            v2 = v2.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with amp_ctx:
                p1, p2, z1, z2 = self.model(v1, v2)
                loss = 0.5 * (simsiam_loss(p1, z2) + simsiam_loss(p2, z1))

            if use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            with torch.no_grad():
                batch_std = (calculate_collapse_std(z1) + calculate_collapse_std(z2)) / 2.0
                total_std += batch_std
            
            num_batches += 1

        return total_loss / max(num_batches, 1), total_std / max(num_batches, 1)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> tuple[float, float]:
        """驗證模型。
        
        Returns:
            tuple[float, float]: (平均 Loss, 平均特徵標準差)
        """
        self.model.eval()
        total_loss = 0.0
        total_std = 0.0
        num_batches = 0

        for v1, v2 in dataloader:
            v1 = v1.to(self.device, non_blocking=True)
            v2 = v2.to(self.device, non_blocking=True)
            
            p1, p2, z1, z2 = self.model(v1, v2)
            loss = 0.5 * (simsiam_loss(p1, z2) + simsiam_loss(p2, z1))
            
            total_loss += loss.item()
            batch_std = (calculate_collapse_std(z1) + calculate_collapse_std(z2)) / 2.0
            total_std += batch_std
            num_batches += 1

        return total_loss / max(num_batches, 1), total_std / max(num_batches, 1)
