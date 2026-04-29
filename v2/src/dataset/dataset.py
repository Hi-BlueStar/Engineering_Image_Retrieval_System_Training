"""影像資料集模組 (Image Dataset Module)。

============================================================
提供兩種資料集類別：

1. **SingleViewDataset**（推薦，GPU 增強模式）：
   - 每次 __getitem__ 回傳單張 resize 後的 raw tensor [C, H, W]
   - Trainer 在 GPU 上呼叫 GPUAugmentation.create_views() 生成雙視角
   - 大幅降低 CPU worker 負擔，解決 CPU 瓶頸

2. **UnlabeledImageDataset**（向下相容，CPU 增強模式）：
   - 每次 __getitem__ 回傳 (view1, view2) CPU 增強後的雙視角
   - 保留原有行為，供 use_gpu_augmentation=False 時使用
============================================================
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from src.logger import get_logger

logger = get_logger(__name__)

_IMG_EXTS = {".jpg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class Letterbox:
    """等比例縮放並填充 (Letterboxing)，確保影像不變形。

    Args:
        size: 目標正方形邊長。
        fill: 填充顏色（預設 255 為白色）。
    """

    def __init__(self, size: int, fill: int = 255) -> None:
        self.size = size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img.resize((self.size, self.size), Image.Resampling.BILINEAR)

        scale = self.size / max(w, h)
        nw, nh = int(w * scale), int(h * scale)

        img = img.resize((nw, nh), Image.Resampling.BILINEAR)
        new_img = Image.new(img.mode, (self.size, self.size), self.fill)
        new_img.paste(img, ((self.size - nw) // 2, (self.size - nh) // 2))
        return new_img


class SingleViewDataset(Dataset):
    """GPU 增強模式資料集：回傳 resize 後的單張 raw tensor。

    Worker 只做最輕量的工作（I/O + resize + ToTensor），
    增強在 GPU 上由 GPUAugmentation 完成。

    Args:
        root: 影像根目錄（遞迴搜尋）。
        img_size: 輸出影像邊長（正方形，僅做 resize）。
        img_exts: 支援的影像副檔名列表。
        in_channels: 輸入通道數；``1`` 載入灰階，``3`` 載入 RGB。
        cache_in_memory: 是否在 ``__init__`` 一次性把所有影像解碼為 uint8
            numpy 陣列保留在 RAM。開啟可消除每個 epoch 的 disk I/O 與
            PNG 解碼成本，是高階硬體上消除 CPU 瓶頸的關鍵手段。
            Linux fork 的 worker 會以 copy-on-write 共享此 cache。
    """

    def __init__(
        self,
        root: Path,
        img_size: int,
        img_exts: List[str],
        in_channels: int = 1,
        cache_in_memory: bool = False,
    ) -> None:
        self.root = root
        self._mode = "L" if in_channels == 1 else "RGB"
        self.images = self._scan(img_exts)
        # 使用 Letterbox 取代 T.Resize，保留原始比例並填充為正方形，避免工程圖形狀失真
        self._transform = T.Compose([
            Letterbox(img_size, fill=255),
            T.ToTensor(),
        ])
        self._cache: Optional[List[np.ndarray]] = (
            self._build_cache() if cache_in_memory else None
        )
        logger.info(
            "SingleViewDataset: root=%s, n=%d, mode=%s, cached=%s",
            root, len(self.images), self._mode, self._cache is not None,
        )

    def _scan(self, img_exts: List[str]) -> List[Path]:
        ext_set = {e.lower() for e in img_exts}
        return sorted(p for p in self.root.rglob("*") if p.suffix.lower() in ext_set)

    def _build_cache(self) -> List[np.ndarray]:
        n = len(self.images)
        logger.info("SingleViewDataset: 開始 in-RAM 解碼快取 (n=%d, root=%s)", n, self.root)
        t0 = time.perf_counter()
        cache: List[np.ndarray] = []
        total_bytes = 0
        log_every = max(1, n // 10)
        for i, path in enumerate(self.images):
            arr = np.asarray(Image.open(path).convert(self._mode), dtype=np.uint8)
            cache.append(arr)
            total_bytes += arr.nbytes
            if (i + 1) % log_every == 0 or (i + 1) == n:
                logger.info(
                    "  快取進度: %d/%d (%.0f%%, 累計 %.2f GB)",
                    i + 1, n, 100.0 * (i + 1) / n, total_bytes / 1e9,
                )
        elapsed = time.perf_counter() - t0
        logger.info(
            "SingleViewDataset: 快取完成 — %d 張, %.2f GB, 耗時 %.1fs",
            n, total_bytes / 1e9, elapsed,
        )
        return cache

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self._cache is not None:
            img = Image.fromarray(self._cache[idx], mode=self._mode)
        else:
            img = Image.open(self.images[idx]).convert(self._mode)
        return self._transform(img)


class UnlabeledImageDataset(Dataset):
    """CPU 增強模式資料集（向下相容）：回傳 (view1, view2) 增強雙視角。

    Args:
        root: 影像根目錄（遞迴搜尋）。
        img_exts: 支援的影像副檔名列表（含點，例如 ``[".png"]``）。
        transform: 接受 PIL 影像並回傳 ``(view1, view2)`` 的可呼叫物件。
        in_channels: 輸入通道數；``1`` 載入灰階，其他值載入 RGB。
    """

    def __init__(
        self,
        root: Path,
        img_exts: List[str],
        transform,
        in_channels: int = 1,
    ) -> None:
        self.root = root
        self.transform = transform
        self._mode = "L" if in_channels == 1 else "RGB"
        self.images = self._scan(img_exts)

        logger.info(
            "UnlabeledImageDataset: root=%s, n=%d, mode=%s",
            root, len(self.images), self._mode,
        )

    def _scan(self, img_exts: List[str]) -> List[Path]:
        ext_set = {e.lower() for e in img_exts}
        return sorted(p for p in self.root.rglob("*") if p.suffix.lower() in ext_set)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.images[idx]).convert(self._mode)
        return self.transform(img)
