"""DataLoader 工廠模組 (DataLoader Factory)。

============================================================
建立訓練與驗證 DataLoader，支援兩種增強模式：

1. **GPU 增強模式**（use_gpu_augmentation=True，預設）：
   - Dataset 回傳 raw tensors（SingleViewDataset）
   - Trainer 在 GPU 上做增強（GPUAugmentation）
   - Worker 負擔輕，解決 CPU 瓶頸

2. **CPU 增強模式**（use_gpu_augmentation=False）：
   - Dataset 回傳已增強的 (view1, view2) 雙視角（UnlabeledImageDataset）
   - 保留原有行為，向下相容
============================================================
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader


from src.config import TrainingConfig
from src.dataset.dataset import SingleViewDataset, UnlabeledImageDataset
from src.dataset.transforms import EngineeringDrawingAugmentation
from src.logger import get_logger

logger = get_logger(__name__)


def create_dataloaders(
    train_path: Path,
    val_path: Path,
    t: TrainingConfig,
    in_channels: int = 1,
    seed: Optional[int] = None,
    use_gpu_augmentation: bool = True,
) -> Tuple[DataLoader, DataLoader, int, int]:
    """建立訓練與驗證 DataLoader。

    Args:
        train_path: 訓練集目錄。
        val_path: 驗證集目錄（使用 test 分割路徑）。
        t: 訓練超參數設定。
        in_channels: 輸入通道數。
        seed: 隨機種子。
        use_gpu_augmentation: True → GPU 增強模式；False → CPU 增強模式。

    Returns:
        Tuple: ``(train_loader, val_loader, n_train, n_val)``。
    """
    generator, worker_init = _make_seed_utils(seed)
    pf = t.prefetch_factor if t.num_workers > 0 else None

    if use_gpu_augmentation:
        # GPU 模式：Dataset 回傳 raw tensor，Trainer 做增強
        cache = getattr(t, "cache_in_memory", False)
        train_ds = SingleViewDataset(
            train_path, t.img_size, t.img_exts, in_channels, cache_in_memory=cache,
        )
        val_ds = SingleViewDataset(
            val_path, t.img_size, t.img_exts, in_channels, cache_in_memory=cache,
        )
    else:
        # CPU 模式：Dataset 回傳 (v1, v2) 增強雙視角
        mean = tuple(0.5 for _ in range(in_channels))
        std = tuple(0.5 for _ in range(in_channels))
        aug = EngineeringDrawingAugmentation(
            img_size=t.img_size,
            mean=mean,
            std=std,
            use_augmentation=t.use_augmentation,
        )
        train_ds = UnlabeledImageDataset(train_path, t.img_exts, aug, in_channels)
        val_ds = UnlabeledImageDataset(val_path, t.img_exts, aug, in_channels)

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
        "DataLoader 建立完成: n_train=%d, n_val=%d, batch=%d, workers=%d, "
        "gpu_aug=%s, seed=%s",
        len(train_ds),
        len(val_ds),
        t.batch_size,
        t.num_workers,
        use_gpu_augmentation,
        seed,
    )

    return train_loader, val_loader, len(train_ds), len(val_ds)


def _make_seed_utils(seed: Optional[int]):
    """建立 Generator 與 worker_init_fn。"""
    if seed is None:
        return None, None

    generator = torch.Generator()
    generator.manual_seed(seed)

    def worker_init(worker_id: int) -> None:
        # 確保 worker 進程也關閉 PIL 像素限制
        Image.MAX_IMAGE_PIXELS = None
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return generator, worker_init
