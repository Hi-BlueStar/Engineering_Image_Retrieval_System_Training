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

分區快取模式（cache_mode="partitioned"）：
   - train_loader 自動包裝為 PartitionedDataLoaderWrapper
   - 每個 epoch 依序輪換所有分區，對 Trainer 完全透明
   - val_loader 強制 cache_mode="none"，避免重複佔用 RAM
============================================================
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler


from src.config import TrainingConfig
from src.dataset.dataset import SingleViewDataset, UnlabeledImageDataset
from src.dataset.transforms import EngineeringDrawingAugmentation
from src.logger import get_logger

logger = get_logger(__name__)


class PartitionedDataLoaderWrapper:
    """透明封裝分區輪換的訓練 DataLoader。

    對外介面與 DataLoader 相同（``__len__`` + ``__iter__``），
    內部依序對每個分區建立子集 DataLoader，並在切換分區時
    觸發背景預載，以隱藏磁碟讀取延遲。

    Args:
        dataset: SingleViewDataset（cache_mode="partitioned"）。
        t: 訓練超參數設定。
        generator: 用於 shuffle 的 torch.Generator（可為 None）。
        worker_init: DataLoader 的 worker_init_fn（可為 None）。
    """

    def __init__(
        self,
        dataset: SingleViewDataset,
        t: TrainingConfig,
        generator,
        worker_init,
    ) -> None:
        self.dataset = dataset
        self.t = t
        self.generator = generator
        self.worker_init = worker_init
        self._total_len = self._compute_len()

    def _compute_len(self) -> int:
        """計算所有分區的總 batch 數（與 __iter__ 的 drop_last 邏輯一致）。"""
        total = 0
        for p_idx in range(self.dataset.n_partitions):
            p_n = len(self.dataset.get_partition_indices(p_idx))
            drop_last = p_n >= self.t.batch_size
            if drop_last:
                total += p_n // self.t.batch_size
            else:
                total += math.ceil(p_n / self.t.batch_size)
        return total

    def __len__(self) -> int:
        return self._total_len

    def __iter__(self):
        t = self.t
        pf = t.prefetch_factor if t.num_workers > 0 else None

        for p_idx in range(self.dataset.n_partitions):
            # 切換至此分區（若背景預載完成則直接 swap，否則同步載入）
            self.dataset.activate_partition(p_idx)

            # 觸發下一分區背景預載
            next_p = p_idx + 1
            if next_p < self.dataset.n_partitions:
                self.dataset.preload_partition_async(next_p)

            indices = self.dataset.get_partition_indices(p_idx)
            subset = Subset(self.dataset, indices)

            loader = DataLoader(
                subset,
                batch_size=t.batch_size,
                shuffle=True,
                num_workers=t.num_workers,
                pin_memory=True,
                prefetch_factor=pf,
                drop_last=len(indices) >= t.batch_size,
                generator=self.generator,
                worker_init_fn=self.worker_init,
                persistent_workers=False,  # 分區切換時不保留 workers，避免 stale cache ref
            )

            logger.info(
                "PartitionedDataLoaderWrapper: 開始分區 %d/%d (n=%d, batches=%d)",
                p_idx, self.dataset.n_partitions - 1, len(indices), len(loader),
            )

            yield from loader


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
        train_loader 可能為 PartitionedDataLoaderWrapper（分區模式）。
    """
    generator, worker_init = _make_seed_utils(seed)
    pf = t.prefetch_factor if t.num_workers > 0 else None

    # 解析 cache_mode（向下相容：cache_in_memory 優先覆寫）
    cache_mode = getattr(t, "cache_mode", "auto")
    memory_fraction = getattr(t, "cache_memory_fraction", 0.6)
    cache_in_memory = getattr(t, "cache_in_memory", None)
    if cache_in_memory is True:
        cache_mode = "full"
    elif cache_in_memory is False and cache_mode == "auto":
        # 僅在 cache_mode 未明確設定時才讓舊 False 覆寫
        cache_mode = "none"

    if use_gpu_augmentation:
        train_ds = SingleViewDataset(
            train_path, t.img_size, t.img_exts, in_channels,
            cache_mode=cache_mode, memory_fraction=memory_fraction,
        )
        # val 強制不快取：val 每 epoch 只跑一次，且常與 train 共享同一目錄
        val_ds = SingleViewDataset(
            val_path, t.img_size, t.img_exts, in_channels,
            cache_mode="none",
        )
    else:
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

    # 訓練 DataLoader：分區模式用 Wrapper，否則用標準 DataLoader
    if (
        use_gpu_augmentation
        and isinstance(train_ds, SingleViewDataset)
        and train_ds.n_partitions > 1
    ):
        train_loader = PartitionedDataLoaderWrapper(train_ds, t, generator, worker_init)
        logger.info(
            "train_loader: PartitionedDataLoaderWrapper (partitions=%d, total_batches=%d)",
            train_ds.n_partitions, len(train_loader),
        )
    else:
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
        "gpu_aug=%s, cache_mode=%s, seed=%s",
        len(train_ds),
        len(val_ds),
        t.batch_size,
        t.num_workers,
        use_gpu_augmentation,
        cache_mode,
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
        Image.MAX_IMAGE_PIXELS = None
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return generator, worker_init
