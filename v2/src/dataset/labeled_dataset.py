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
from torch.utils.data import Dataset

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
    ) -> None:
        self.root = Path(root)
        ext_set = {e.lower() for e in (img_exts or list(_IMG_EXTS))}
        self._mode = "L" if in_channels == 1 else "RGB"

        # 發現類別
        class_dirs = sorted(d for d in self.root.iterdir() if d.is_dir())
        if not class_dirs:
            raise ValueError(f"LabeledImageDataset: 根目錄無類別子目錄: {root}")

        self.classes: List[str] = [d.name for d in class_dirs]
        self.class_to_idx: Dict[str, int] = {
            c: i for i, c in enumerate(self.classes)
        }

        # 掃描所有影像
        self.samples: List[Tuple[Path, int]] = []
        for class_dir in class_dirs:
            label = self.class_to_idx[class_dir.name]
            for p in sorted(class_dir.rglob("*")):
                if p.suffix.lower() in ext_set:
                    self.samples.append((p, label))

        mean = [0.5] * in_channels
        std = [0.5] * in_channels
        self._transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

        logger.info(
            "LabeledImageDataset: root=%s, classes=%d, samples=%d",
            root,
            len(self.classes),
            len(self.samples),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert(self._mode)
        return self._transform(img), label

    def get_class_name(self, idx: int) -> str:
        return self.classes[idx]
