"""有標籤影像資料集模組 (Labeled Image Dataset)。

============================================================
用於訓練後的檢索評估：
    - 以類別子目錄自動發現標籤（folder = class）
    - 回傳 (image_tensor, label_idx) 供特徵提取與指標計算
    - 不做 SimSiam 雙視角增強，僅做 resize + normalize
============================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchvision.transforms as T
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from torch.utils.data import Dataset

from src.dataset.dataset import Letterbox
from src.logger import get_logger

logger = get_logger(__name__)

_IMG_EXTS = {".jpg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class LabeledImageDataset(Dataset):
    """以類別子目錄結構讀取影像，回傳帶標籤的張量。

    目錄結構::

        root/
          class_A/
            image1.png
            ...
          class_B/
            ...

    Args:
        root: 資料根目錄（含類別子目錄）。
        img_size: 影像縮放尺寸（正方形邊長）。
        img_exts: 支援的副檔名集合。
        in_channels: 輸入通道數（1=灰階，3=RGB）。
    """

    def __init__(
        self,
        root: Path,
        img_size: int = 512,
        img_exts: Optional[List[str]] = None,
        in_channels: int = 1,
        use_preprocessing: bool = True,
        use_invert: bool = True,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        cache_in_memory: bool = True,
    ) -> None:
        self.root = Path(root).resolve()
        ext_set = {e.lower() for e in (img_exts or list(_IMG_EXTS))}
        self._mode = "L" if in_channels == 1 else "RGB"

        # 掃描所有影像
        image_paths: List[Path] = []
        for p in self.root.rglob("*"):
            if p.is_file() and p.suffix.lower() in ext_set:
                image_paths.append(p)

        if not image_paths:
            raise ValueError(f"LabeledImageDataset: 根目錄下找不到支援的影像: {root}")

        # 以影像的父目錄名稱作為類別名稱（自動適應巢狀結構）
        self.classes: List[str] = sorted(list(set(p.parent.name for p in image_paths)))
        self.class_to_idx: Dict[str, int] = {
            c: i for i, c in enumerate(self.classes)
        }

        # 建立 samples
        self.samples: List[Tuple[Path, int]] = []
        for p in image_paths:
            label = self.class_to_idx[p.parent.name]
            self.samples.append((p, label))

        # 排序以確保重現性
        self.samples.sort(key=lambda x: (x[1], x[0].name))

        if mean is None:
            mean = [0.0394] * in_channels
        if std is None:
            std = [0.1752] * in_channels

        transform_ops = []
        if use_preprocessing:
            transform_ops.append(Letterbox(img_size, fill=255))
        else:
            transform_ops.append(T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR))
        
        if use_invert:
            transform_ops.append(T.RandomInvert(p=1.0))

        transform_ops.extend([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self._transform = T.Compose(transform_ops)

        self._cache_images = None
        self._cache_labels = None
        if cache_in_memory:
            self._build_cache(img_size)

        logger.info(
            "LabeledImageDataset: root=%s, classes=%d, samples=%d, cached=%s",
            root,
            len(self.classes),
            len(self.samples),
            cache_in_memory,
        )

    def _build_cache(self, img_size: int) -> None:
        """並行讀取並預處理所有影像，存入記憶體快取。"""
        import concurrent.futures
        import os
        n = len(self.samples)
        c = 1 if self._mode == "L" else 3
        s = img_size

        logger.info("LabeledImageDataset: 開始快取評估影像至記憶體 (n=%d)...", n)
        self._cache_images = torch.empty((n, c, s, s), dtype=torch.float32)
        self._cache_labels = torch.empty(n, dtype=torch.long)

        def _load_one(i: int) -> Tuple[int, torch.Tensor, int]:
            path, label = self.samples[i]
            img = Image.open(path).convert(self._mode)
            tensor = self._transform(img)
            return i, tensor, label

        max_workers = min(16, os.cpu_count() or 4)
        completed = 0
        log_every = max(1, n // 10)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_load_one, i) for i in range(n)]
            for future in concurrent.futures.as_completed(futures):
                i, tensor, label = future.result()
                self._cache_images[i] = tensor
                self._cache_labels[i] = label
                completed += 1
                if completed % log_every == 0 or completed == n:
                    logger.info("  快取進度: %d/%d (%.0f%%)", completed, n, 100.0 * completed / n)

        self._cache_images.share_memory_()
        self._cache_labels.share_memory_()
        logger.info("LabeledImageDataset: 快取完成！")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self._cache_images is not None and self._cache_labels is not None:
            return self._cache_images[idx], self._cache_labels[idx].item()
        path, label = self.samples[idx]
        img = Image.open(path).convert(self._mode)
        return self._transform(img), label

    def get_class_name(self, idx: int) -> str:
        return self.classes[idx]
