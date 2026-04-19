"""無標籤影像資料集模組 (Unlabeled Image Dataset)。

用於 SimSiam 自監督學習：每次 __getitem__ 對同一影像
透過增強器生成兩個視角，作為正樣本對。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.logger import get_logger

logger = get_logger(__name__)


class UnlabeledImageDataset(Dataset):
    """遞迴掃描目錄下所有影像，回傳雙視角增強結果。

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
            root,
            len(self.images),
            self._mode,
        )

    def _scan(self, img_exts: List[str]) -> List[Path]:
        ext_set = {e.lower() for e in img_exts}
        paths = [
            p for p in self.root.rglob("*")
            if p.suffix.lower() in ext_set
        ]
        return sorted(paths)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.images[idx]).convert(self._mode)
        return self.transform(img)
