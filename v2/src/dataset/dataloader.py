"""DataLoader 工廠模組 (DataLoader Factory)。

建立訓練與驗證 DataLoader，整合增強器與 seed 管理。
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import TrainingConfig
from src.dataset.dataset import UnlabeledImageDataset
from src.dataset.transforms import EngineeringDrawingAugmentation
from src.logger import get_logger

logger = get_logger(__name__)


def create_dataloaders(
    train_path: Path,
    val_path: Path,
    t: TrainingConfig,
    in_channels: int = 1,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, int, int]:
    """建立訓練與驗證 DataLoader。

    Args:
        train_path: 訓練集目錄。
        val_path: 驗證集目錄。
        t: 訓練超參數設定（含 img_size, batch_size 等）。
        in_channels: 輸入通道數。
        seed: 隨機種子；``None`` 表示不固定。

    Returns:
        Tuple: ``(train_loader, val_loader, n_train, n_val)``。
    """
    mean = tuple(0.5 for _ in range(in_channels))
    std = tuple(0.5 for _ in range(in_channels))
    aug = EngineeringDrawingAugmentation(img_size=t.img_size, mean=mean, std=std)

    train_ds = UnlabeledImageDataset(train_path, t.img_exts, aug, in_channels)
    val_ds = UnlabeledImageDataset(val_path, t.img_exts, aug, in_channels)

    generator = None
    worker_init = None

    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

        def worker_init(worker_id: int) -> None:
            worker_seed = seed + worker_id
            random.seed(worker_seed)
            np.random.seed(worker_seed)

    # prefetch_factor 僅在 num_workers > 0 時有效
    pf = t.prefetch_factor if t.num_workers > 0 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=t.batch_size,
        shuffle=True,
        num_workers=t.num_workers,
        pin_memory=True,
        prefetch_factor=pf,
        drop_last=len(train_ds) >= t.batch_size,
        generator=generator,
        worker_init_fn=worker_init,
        persistent_workers=t.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=t.batch_size,
        shuffle=False,
        num_workers=t.num_workers,
        pin_memory=True,
        prefetch_factor=pf,
        worker_init_fn=worker_init,
        persistent_workers=t.num_workers > 0,
    )

    logger.info(
        "DataLoader 建立完成: n_train=%d, n_val=%d, batch=%d, workers=%d, seed=%s",
        len(train_ds),
        len(val_ds),
        t.batch_size,
        t.num_workers,
        seed,
    )

    return train_loader, val_loader, len(train_ds), len(val_ds)
