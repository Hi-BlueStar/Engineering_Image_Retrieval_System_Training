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

from pathlib import Path
from typing import List, Tuple, Union

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
    """

    def __init__(
        self,
        root: Path,
        img_size: int,
        img_exts: List[str],
        in_channels: int = 1,
    ) -> None:
        self.root = root
        self._mode = "L" if in_channels == 1 else "RGB"
        self.images = self._scan(img_exts)
        # 使用 Letterbox 取代 T.Resize，保留原始比例並填充為正方形，避免工程圖形狀失真
        self._transform = T.Compose([
            Letterbox(img_size, fill=255),
            T.ToTensor(),
        ])
        logger.info(
            "SingleViewDataset: root=%s, n=%d, mode=%s",
            root, len(self.images), self._mode,
        )

    def _scan(self, img_exts: List[str]) -> List[Path]:
        ext_set = {e.lower() for e in img_exts}
        return sorted(p for p in self.root.rglob("*") if p.suffix.lower() in ext_set)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
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
