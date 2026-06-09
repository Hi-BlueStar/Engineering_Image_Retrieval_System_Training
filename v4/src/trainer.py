"""訓練器引擎模組 (Trainer Engine Module)。

============================================================
整合模型的前向傳播、反向傳播、混合精度 (Autocast BF16)、
JIT 編編譯 (torch.compile) 優化與 MLOps 追蹤。

核心機制：
    1. **混合精度 (AMP BF16)**：在 Ampere 以上架構硬體上，BF16 擁有與 FP32 相同的動態範圍，
       故能直接避免對比學習因點積過大引發的數值溢出，且無須配置 GradScaler 即可穩定收斂。
    2. **JIT 編譯 (torch.compile)**：自動編譯優化模型的前向運算與 GPU 增強，減少 CUDA Launch 延遲。
    3. **崩塌安全斷路器 (Assertion Breaker)**：若特徵空間標準差 (std) 跌破 0.005，
       系統將主動拋出例外中斷訓練，以防資源浪費。
    4. **Rich 控制台進度條**：以優美的高亮表格展示每個 Epoch 的實時 Loss、Val Loss、特徵 Std。
============================================================
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from .config import AppConfig
from .data_pipeline import GPUPrefetchDataLoader, GPUAugmentationModule
from .criterion import SimSiamLossCriterion
from .logger import get_logger

logger = get_logger(__name__)


class SimSiamTrainer:
    """SimSiam 對比學習訓練器"""
    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        criterion: SimSiamLossCriterion,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        gpu_augmentor: GPUAugmentationModule,
        config: AppConfig,
        device: torch.device
    ) -> None:
        self.encoder = encoder
        self.predictor = predictor
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gpu_augmentor = gpu_augmentor
        self.cfg = config
        self.device = device
        
        # 決定是否使用 BF16 混合精度
        self.use_bf16 = self.cfg.training.use_bf16
        self.device_type = "cuda" if device.type == "cuda" else "cpu"
        
        if self.use_bf16:
            logger.info("啟用 PyTorch 自動混合精度 (AMP) 格式: bfloat16")
            
        # 實施模型 JIT 編譯優化 (torch.compile)
        if self.cfg.training.compile_model:
            # 判斷系統是否支援 PyTorch 2.x compile (Linux 通常相容良好)
            if hasattr(torch, "compile") and os.name != "nt":
                try:
                    logger.info("正在執行 torch.compile() 編譯優化...")
                    self.encoder = torch.compile(self.encoder)
                    self.predictor = torch.compile(self.predictor)
                    # 提示：不對 Kornia 資料增強模組 (gpu_augmentor) 進行編譯。
                    # 因為 Kornia 內部有隨機參數產生與 .item() 操作，編譯它會導致頻繁的 Graph break 與 Recompilation，反而降低效能。
                except Exception as e:
                    logger.warning("torch.compile() 編譯失敗，將回退到常規解釋執行。錯誤: %s", e)
            else:
                logger.warning("當前平台不支援或未安裝 torch.compile()，採用默認解釋模式執行。")

        # 建立保存目錄
        self.output_dir = Path(self.cfg.experiment.output_dir) / self.cfg.experiment.exp_name
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, loader: GPUPrefetchDataLoader) -> Tuple[float, float]:
        """執行一個 Epoch 的訓練"""
        self.encoder.train()
        self.predictor.train()
        
        total_loss = 0.0
        total_std = 0.0
        n_batches = 0
        
        # 設定 AMP autocast 上下文 (BF16 不需要 GradScaler)
        amp_ctx = torch.amp.autocast(
            device_type=self.device_type,
            dtype=torch.bfloat16,
            enabled=self.use_bf16
        )

        for x, _ in loader:
            self.optimizer.zero_grad(set_to_none=True)
            
            # 1. GPU 隨機雙視角增強 (於 FP32 下執行，避免 Kornia 因半精度報錯)
            v1, v2 = self.gpu_augmentor(x)
            
            # 使用 amp 進行混合精度計算
            with amp_ctx:
                # 2. 模型轉發
                z1 = self.encoder(v1)
                z2 = self.encoder(v2)
                p1 = self.predictor(z1)
                p2 = self.predictor(z2)
                
                # 3. 計算損失並實施 Stop-gradient
                loss, batch_std = self.criterion(p1, p2, z1, z2)
                
            loss.backward()
            
            # 梯度裁切
            if self.cfg.training.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.predictor.parameters()),
                    self.cfg.training.grad_clip
                )
                
            self.optimizer.step()
            
            total_loss += loss.item()
            total_std += batch_std
            n_batches += 1
            
            # 特徵坍塌安全斷路器：若 Batch 內平均維度標準差小於 0.005，說明已經徹底 collapsed
            if batch_std < 0.005:
                err_msg = f"嚴重警報: 檢測到特徵維度坍塌 (Batch Std: {batch_std:.6f} < 0.005)。自動觸發斷路器停止訓練！"
                logger.error(err_msg)
                raise AssertionError(err_msg)

        avg_loss = total_loss / max(n_batches, 1)
        avg_std = total_std / max(n_batches, 1)
        return avg_loss, avg_std

    @torch.no_grad()
    def evaluate(self, loader: GPUPrefetchDataLoader) -> Tuple[float, float]:
        """執行驗證集計算"""
        self.encoder.eval()
        self.predictor.eval()
        
        total_loss = 0.0
        total_std = 0.0
        n_batches = 0
        
        # 驗證階段依舊套用 BF16 以確保精確度一致
        amp_ctx = torch.amp.autocast(
            device_type=self.device_type,
            dtype=torch.bfloat16,
            enabled=self.use_bf16
        )

        for x, _ in loader:
            # 1. GPU 隨機雙視角增強 (於 FP32 下執行，避免 Kornia 因半精度報錯)
            v1, v2 = self.gpu_augmentor(x)
            
            with amp_ctx:
                z1 = self.encoder(v1)
                z2 = self.encoder(v2)
                p1 = self.predictor(z1)
                p2 = self.predictor(z2)
                
                loss, batch_std = self.criterion(p1, p2, z1, z2)
                
            total_loss += loss.item()
            total_std += batch_std
            n_batches += 1

        return total_loss / max(n_batches, 1), total_std / max(n_batches, 1)

    def fit(self, train_loader: GPUPrefetchDataLoader, val_loader: GPUPrefetchDataLoader, tracker: Any) -> None:
        """主要訓練排程控制迴圈"""
        best_val_loss = float("inf")
        epochs = self.cfg.training.epochs
        
        logger.info("啟動 SimSiam v4 訓練迴圈... 總 Epochs: %d | Batch Size: %d", epochs, self.cfg.training.batch_size)
        
        # 使用 Rich 進度條美化主控台
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=30, style="cyan"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("{task.fields[info]}"),
            refresh_per_second=2,
        ) as progress:
            task_id = progress.add_task("訓練進度", total=epochs, info="初始化中...")
            
            for epoch in range(1, epochs + 1):
                start_time = time.perf_counter()
                
                # 執行一輪訓練與驗證
                train_loss, train_std = self.train_epoch(train_loader)
                
                if len(val_loader) > 0:
                    val_loss, val_std = self.evaluate(val_loader)
                else:
                    val_loss, val_std = 0.0, 0.0
                    
                # 更新學習率
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]
                
                duration = time.perf_counter() - start_time
                
                # 記錄指標至 MLOps Tracker
                metrics = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_std": train_std,
                    "val_std": val_std,
                    "lr": current_lr,
                    "duration": duration
                }
                tracker.log_epoch(metrics)
                
                # 更新進度條顯示資訊
                info_text = f"[L] Tr:{train_loss:.4f}/Va:{val_loss:.4f} | [Std] Tr:{train_std:.4f}/Va:{val_std:.4f}"
                progress.update(task_id, advance=1, info=info_text)
                
                # 儲存與管理 Checkpoint
                is_best = val_loss < best_val_loss
                if is_best and len(val_loader) > 0:
                    best_val_loss = val_loss
                    
                self.save_checkpoint(epoch, val_loss, is_best)

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool) -> None:
        """自動儲存模型權重 (保留 latest、best 與特定週期 epoch)"""
        # 注意: 如果模型經過 torch.compile，直接儲存 state_dict 會包含 compile 的 prefix (_orig_mod.)，
        # 我們需要透過 _unwrap_model 剝離。
        def _get_state(module):
            if hasattr(module, "_orig_mod"):
                return module._orig_mod.state_dict()
            return module.state_dict()

        state = {
            "epoch": epoch,
            "encoder_state": _get_state(self.encoder),
            "predictor_state": _get_state(self.predictor),
            "optimizer": self.optimizer.state_dict(),
            "loss": val_loss,
            "config": self.cfg.to_dict(),
        }

        # 1. 儲存最新 Checkpoint (latest)
        torch.save(state, self.ckpt_dir / "checkpoint_latest.pth")

        # 2. 定期儲存 Checkpoint
        if epoch % self.cfg.experiment.save_freq == 0:
            torch.save(state, self.ckpt_dir / f"checkpoint_epoch_{epoch:04d}.pth")

        # 3. 儲存最佳 Checkpoint (best)
        if is_best:
            torch.save(state, self.ckpt_dir / "checkpoint_best.pth")
