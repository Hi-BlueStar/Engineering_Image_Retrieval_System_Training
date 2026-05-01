"""SimSiam 資料增強模組 (Data Augmentation Module)。

============================================================
針對工程圖設計的雙視角增強管線。每次呼叫對同一影像
隨機生成兩個不同視角（view1, view2），作為 SimSiam 的
正樣本對。

增強策略（工程圖考量）：
    - RandomResizedCrop：學習局部特徵
    - RandomHorizontalFlip / RandomVerticalFlip：圖紙方向不定
    - RandomRotation：旋轉不變性
    - Normalize：均值 0.5，標準差 0.5
============================================================
"""

from __future__ import annotations

from typing import Tuple, Union

import torchvision.transforms as T
from PIL import Image

from src.dataset.dataset import Letterbox


class EngineeringDrawingAugmentation:
    """工程圖 SimSiam 雙視角增強器。

    Args:
        img_size: 輸出影像尺寸（正方形邊長）。
        mean: 正規化均值，灰階用 ``(0.5,)``，RGB 用 ``(0.5, 0.5, 0.5)``。
        std: 正規化標準差。
    """

    def __init__(
        self,
        img_size: int = 512,
        mean: Tuple[float, ...] = (0.5,),
        std: Tuple[float, ...] = (0.5,),
        use_augmentation: bool = True,
    ) -> None:
        if use_augmentation:
            self._transform = T.Compose([
                T.RandomResizedCrop(img_size, scale=(0.2, 1.0), ratio=(0.75, 1.333)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomApply([T.RandomRotation(degrees=30, fill=255)], p=0.5),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])
        else:
            # 消融實驗：無增強版（僅 Letterbox + normalize）
            self._transform = T.Compose([
                Letterbox(img_size, fill=255),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])

    def __call__(
        self, img: Image.Image
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:  # noqa: F821
        """生成兩個獨立隨機增強視角。

        Args:
            img: PIL 影像（'L' 或 'RGB'）。

        Returns:
            Tuple: ``(view1, view2)``，各自形狀為 ``[C, H, W]``。
        """
        return self._transform(img), self._transform(img)
