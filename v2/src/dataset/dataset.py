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
Image.MAX_IMAGE_PIXELS = None

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
        cache_in_memory: 是否在 ``__init__`` 一次性把所有影像「letterbox 後」
            存成單一連續 uint8 tensor ``[N, C, H, W]`` 並 ``share_memory_()``。
            開啟可消除每個 epoch 的 disk I/O 與 PNG 解碼成本。
            記憶體用量精準等於 ``N * C * img_size² bytes``，且因為是單一
            連續 tensor + share_memory_，DataLoader workers 真正共享，
            不會發生 list-of-ndarray 在 CPython refcount 寫入時的
            copy-on-write 失效（先前實作會放大數倍～數十倍）。
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
        self.img_size = img_size
        self.in_channels = in_channels
        self._mode = "L" if in_channels == 1 else "RGB"
        self._letterbox = Letterbox(img_size, fill=255)
        self.images = self._scan(img_exts)
        # 非 cache 路徑：保留原本的 PIL → Letterbox → ToTensor 管線
        self._transform = T.Compose([
            self._letterbox,
            T.ToTensor(),
        ])
        self._cache: Optional[torch.Tensor] = (
            self._build_cache() if cache_in_memory else None
        )
        logger.info(
            "SingleViewDataset: root=%s, n=%d, mode=%s, cached=%s",
            root, len(self.images), self._mode, self._cache is not None,
        )

    def _scan(self, img_exts: List[str]) -> List[Path]:
        ext_set = {e.lower() for e in img_exts}
        return sorted(p for p in self.root.rglob("*") if p.suffix.lower() in ext_set)

    def _build_cache(self) -> torch.Tensor:
        """一次性解碼 + letterbox 為單一連續 uint8 tensor。

        - 在 build 時就套用 Letterbox（deterministic），避免 cache 儲存
          原圖尺寸（在 400 DPI 工程圖上原圖可能 3–15 MB / 張，被 256² 縮放
          後僅 64 KB / 張，差距 50–200 倍）。
        - 單一連續 tensor + ``share_memory_()`` → fork 的 workers 直接
          共享同一塊記憶體，不會被 Python refcount 寫入觸發 CoW 複製。
        - 回傳形狀 ``[N, C, img_size, img_size]`` 的 ``torch.uint8`` tensor。
        """
        n = len(self.images)
        c = self.in_channels
        s = self.img_size
        expected_gb = n * c * s * s / 1e9
        logger.info(
            "SingleViewDataset: 開始 in-RAM 解碼快取 (n=%d, shape=[N,%d,%d,%d], "
            "預期 %.2f GB, root=%s)",
            n, c, s, s, expected_gb, self.root,
        )
        t0 = time.perf_counter()

        cache = torch.empty((n, c, s, s), dtype=torch.uint8)
        log_every = max(1, n // 10)

        for i, path in enumerate(self.images):
            img = Image.open(path).convert(self._mode)
            img = self._letterbox(img)  # PIL → letterboxed PIL (s × s)
            # 用 np.array（非 np.asarray）強制取得可寫副本：PIL 的緩衝區是唯讀，
            # np.asarray 會共享之並讓 torch.from_numpy 噴 "non-writable tensor"
            # UserWarning。我們之後會 copy 進 cache[i]，所以提前複製是零額外成本。
            arr = np.array(img, dtype=np.uint8)  # [H, W] or [H, W, C]
            if arr.ndim == 2:  # grayscale
                cache[i, 0] = torch.from_numpy(arr)
            else:  # RGB
                # arr [H, W, C] → tensor [C, H, W]
                cache[i] = torch.from_numpy(arr).permute(2, 0, 1)

            if (i + 1) % log_every == 0 or (i + 1) == n:
                logger.info("  快取進度: %d/%d (%.0f%%)", i + 1, n, 100.0 * (i + 1) / n)

        cache.share_memory_()  # 跨 worker 真正共享，避免 CoW 倍增
        elapsed = time.perf_counter() - t0
        logger.info(
            "SingleViewDataset: 快取完成 — %d 張, %.2f GB（單一 share_memory tensor）, 耗時 %.1fs",
            n, cache.numel() / 1e9, elapsed,
        )
        return cache

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self._cache is not None:
            # cache hit fast path：直接從共享 uint8 tensor 取 → float [0,1]，
            # 跳過 PIL.fromarray 與 Letterbox（cache 已經是 letterboxed）
            return self._cache[idx].to(dtype=torch.float32).div_(255.0)
        # 非 cache：原 PIL 路徑
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
