"""Checkpoint 管理器模組 (Checkpoint Manager Module)。

============================================================
負責模型權重的儲存與載入，獨立於 Trainer 運作。

儲存策略：
    1. ``checkpoint_last.pth``：每 epoch 覆蓋。
    2. ``checkpoint_best.pth``：val_loss 最佳時額外儲存。
    3. ``checkpoint_epoch_XXXX.pth``：每 ``save_freq`` 個 epoch
       保留歷史快照。
============================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.logger import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """管理模型 Checkpoint 的儲存與載入。

    Args:
        ckpt_dir: Checkpoint 儲存目錄。
        save_freq: 歷史快照的儲存頻率（每 N 個 epoch）。
        config_dict: 設定字典，嵌入每個 checkpoint 中
            以確保可完全重現。
    """

    def __init__(
        self,
        ckpt_dir: str | Path,
        save_freq: int = 10,
        config_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.save_freq = save_freq
        self.config_dict = config_dict or {}

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_loss: float,
        is_best: bool = False,
    ) -> None:
        """儲存模型 Checkpoint。

        Args:
            model: 模型實例。
            optimizer: 優化器實例。
            epoch: 當前 epoch。
            val_loss: 當前驗證集 loss。
            is_best: 是否為目前最佳模型。
        """
        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss,
            "config": self.config_dict,
        }

        # 1. 最新 checkpoint（每 epoch 覆蓋）
        last_path = self.ckpt_dir / "checkpoint_last.pth"
        torch.save(state, last_path)

        # 2. 定期快照
        if epoch % self.save_freq == 0:
            epoch_path = self.ckpt_dir / f"checkpoint_epoch_{epoch:04d}.pth"
            torch.save(state, epoch_path)
            logger.debug("Checkpoint 快照: %s", epoch_path.name)

        # 3. 最佳模型
        if is_best:
            best_path = self.ckpt_dir / "checkpoint_best.pth"
            torch.save(state, best_path)
            logger.info(
                "最佳模型已儲存 (epoch=%d, val_loss=%.4f)", epoch, val_loss
            )

    def load(
        self,
        path: str | Path,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        """從 Checkpoint 載入模型權重與優化器狀態。

        Args:
            path: Checkpoint 檔案路徑。
            model: 目標模型（將被載入權重）。
            optimizer: 可選的優化器（將被載入狀態）。

        Returns:
            Dict[str, Any]: Checkpoint 的完整狀態字典。

        Raises:
            FileNotFoundError: 當 Checkpoint 檔案不存在時。
        """
        ckpt_path = Path(path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint 不存在: {ckpt_path}")

        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"])

        if optimizer is not None and "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])

        logger.info(
            "Checkpoint 載入完成: %s (epoch=%d, val_loss=%.4f)",
            ckpt_path.name,
            state.get("epoch", -1),
            state.get("val_loss", float("inf")),
        )
        return state
