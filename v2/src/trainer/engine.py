"""
訓練主引擎 (Training Engine)。

負責包含前向傳播、反向傳播、混合精度 (AMP)、學習率更新與資料加載的循環邏輯。
將基礎架構與模型切分開來。提供高擴展性的 callback 系統。
"""

import time
import contextlib
import torch
from typing import List, Optional
from torch.utils.data import DataLoader

from ..core.logger import get_logger
from ..core.config_manager import ConfigManager
from ..models.base import BaseModel
from ..models.loss import negative_cosine_similarity

logger = get_logger(__name__)

class Callback:
    """掛載點擴展介面。"""
    def on_train_start(self, trainer: "TrainerEngine"): pass
    def on_train_end(self, trainer: "TrainerEngine"): pass
    def on_epoch_start(self, trainer: "TrainerEngine", epoch: int): pass
    def on_epoch_end(self, trainer: "TrainerEngine", epoch: int, metrics: dict): pass

class TrainerEngine:
    """訓練核心引擎。"""

    def __init__(
        self,
        config: ConfigManager,
        model: BaseModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        callbacks: Optional[List[Callback]] = None
    ):
        """
        初始化訓練器，實作控制反轉 (IoC)。Trainer 負責運算流程的協調。

        Args:
            config: 全局配置。
            model: 實現了 forward(x1, x2) 等介面的抽象模型。
            train_loader: 訓練集 DataLoader。
            val_loader: 驗證集 DataLoader。
            optimizer: 優化器。
            scheduler: 學習率排程。
            callbacks: 要註冊的事件回調清單 (例如儲存 checkpoint 或 紀錄報表)。
        """
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.callbacks = callbacks or []
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # AMP 自動混合精度設定 - 降低 VRAM 消耗、加速運算，處理舊版 PyTorch 與 CPU 防呆
        self.use_amp = self.device.type == "cuda"
        if self.use_amp:
            try:
                # 嘗試使用最新版寫法
                self.scaler = torch.amp.GradScaler(enabled=True)
                self.amp_ctx = torch.amp.autocast(device_type=self.device.type, dtype=torch.float16)
            except AttributeError:
                # Fallback 給 PyTorch < 2.0
                self.scaler = torch.cuda.amp.GradScaler(enabled=True)
                self.amp_ctx = torch.cuda.amp.autocast(enabled=True)
        else:
            self.scaler = None
            self.amp_ctx = contextlib.nullcontext()

    def fire_callbacks(self, event_name: str, *args, **kwargs):
        """觸發已經註冊的所有 Callback。"""
        for cb in self.callbacks:
            func = getattr(cb, event_name, None)
            if callable(func):
                func(self, *args, **kwargs)

    def train_one_epoch(self) -> tuple[float, float]:
        """執行一整個 Epoch 的模型參數更新。"""
        self.model.train()
        total_loss, total_std, num_batches = 0.0, 0.0, 0

        for v1, v2 in self.train_loader:
            v1 = v1.to(self.device, non_blocking=True)
            v2 = v2.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with self.amp_ctx:
                p1, p2, z1, z2 = self.model(v1, v2)
                # Symmetrical Loss (對稱損失): Loss(p1, z2) 與 Loss(p2, z1) 的平均值
                loss = 0.5 * (negative_cosine_similarity(p1, z2) + negative_cosine_similarity(p2, z1))

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            with torch.no_grad():
                batch_std = (self.model.calculate_collapse_metric(z1) + self.model.calculate_collapse_metric(z2)) / 2.0
                total_std += batch_std
            
            num_batches += 1

        return total_loss / max(num_batches, 1), total_std / max(num_batches, 1)

    @torch.no_grad()
    def evaluate(self) -> tuple[float, float]:
        """驗證模型。不具更動參數的效果，單純監測。"""
        if not self.val_loader or len(self.val_loader) == 0:
            return 0.0, 0.0
            
        self.model.eval()
        total_loss, total_std, num_batches = 0.0, 0.0, 0

        for v1, v2 in self.val_loader:
            v1 = v1.to(self.device, non_blocking=True)
            v2 = v2.to(self.device, non_blocking=True)

            with self.amp_ctx:
                p1, p2, z1, z2 = self.model(v1, v2)
                loss = 0.5 * (negative_cosine_similarity(p1, z2) + negative_cosine_similarity(p2, z1))
            
            total_loss += loss.item()
            batch_std = (self.model.calculate_collapse_metric(z1) + self.model.calculate_collapse_metric(z2)) / 2.0
            total_std += batch_std
            num_batches += 1

        return total_loss / max(num_batches, 1), total_std / max(num_batches, 1)

    def load_checkpoint(self, path: str):
        """讀取斷點續傳。"""
        import os
        if not os.path.exists(path):
            logger.warning(f"找不到 Checkpoint {path}，使用初始設定。")
            return 0
        logger.info(f"正在載入 Checkpoint: {path}")
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["state_dict"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        return state.get("epoch", 0)

    def fit(self):
        """執行訓練迴圈的主方法。"""
        from ..utils.timer import PrecisionTimer
        from ..utils.system_monitor import SystemMonitor
        timer = PrecisionTimer()
        
        logger.info(f"開始訓練引擎... (裝置: {self.device})")
        self.fire_callbacks("on_train_start")

        start_epoch = 1
        if self.config.training.resume_from:
            start_epoch = self.load_checkpoint(self.config.training.resume_from) + 1

        for epoch in range(start_epoch, self.config.training.epochs + 1):
            self.fire_callbacks("on_epoch_start", epoch=epoch)
            
            # 使用我們精準的計時器，排除 callback (如存檔) 的 IO 耗時
            timer.reset()
            timer.start()
            
            train_loss, train_std = self.train_one_epoch()
            val_loss, val_std = self.evaluate()
            
            # Epoch 結束後步進 Scheduler
            self.scheduler.step()
            net_dur = timer.stop()
            
            # VRAM 防禦清理
            vram_use = SystemMonitor.get_vram_usage_mb()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            
            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_std": train_std,
                "val_std": val_std,
                "lr": self.optimizer.param_groups[0]["lr"],
                "duration": net_dur,
                "vram_mb": vram_use
            }
            
            self.fire_callbacks("on_epoch_end", epoch=epoch, metrics=metrics)

        self.fire_callbacks("on_train_end")
        logger.info("訓練引擎執行完畢。")
