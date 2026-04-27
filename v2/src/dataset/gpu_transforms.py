"""GPU 加速增強模組 (GPU Augmentation Module)。

============================================================
將增強管線從 CPU DataLoader workers 移至 GPU，消除訓練時的
CPU 瓶頸。

設計原理：
    - DataLoader workers 只做 I/O（PIL 讀取 + Resize + ToTensor）
    - Trainer 在每個 batch 移至 GPU 後立即呼叫 create_views()
    - kornia 在 GPU 上平行處理整個 batch，速度遠快於 CPU 逐張處理

增強策略（與 CPU 版 EngineeringDrawingAugmentation 等效）：
    - RandomResizedCrop(0.2~1.0 scale)：學習局部特徵
    - RandomHorizontalFlip / RandomVerticalFlip：圖紙方向不定
    - RandomRotation(±30°)：旋轉不變性（工程圖常旋轉擺放）
    - ColorJitter(brightness, contrast)：模擬掃描品質差異
    - Normalize(0.5, 0.5)：標準化

Dependencies:
    kornia >= 0.7.0  (安裝: pip install kornia)
    若未安裝 kornia，自動退化為 torchvision CPU 增強（效能較低）。
============================================================
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.logger import get_logger

logger = get_logger(__name__)

try:
    import kornia.augmentation as K
    import kornia.morphology as KM
    _KORNIA_AVAILABLE = True
except ImportError:
    _KORNIA_AVAILABLE = False
    logger.warning(
        "kornia 未安裝，GPU 增強將退化為 resize+normalize。"
        "建議安裝: pip install kornia"
    )


class RandomGPUMorphology(nn.Module):
    """GPU 隨機形態學增強 (膨脹/腐蝕)。"""

    def __init__(self, p: float = 0.5, kernel_size: int = 3) -> None:
        super().__init__()
        self.p = p
        self.kernel_size = kernel_size
        self.register_buffer("kernel", torch.ones(kernel_size, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or torch.rand(1).item() > self.p:
            return x

        # 隨機選擇膨脹或腐蝕
        if torch.rand(1).item() > 0.5:
            return KM.dilation(x, self.kernel)
        else:
            return KM.erosion(x, self.kernel)


class GPUAugmentation(nn.Module):
    """GPU 批次增強器，呼叫 create_views() 生成 SimSiam 雙視角。

    輸入：raw image batch [B, C, H, W]，值域 [0, 1]（ToTensor 後）
    輸出：(view1, view2) 各 [B, C, H, W]，值域已正規化

    Args:
        img_size: 輸出影像邊長（正方形）。
        use_augmentation: 是否套用完整增強；False 時僅 resize + normalize。
        in_channels: 輸入通道數（1=灰階，3=RGB）。
    """

    def __init__(
        self,
        img_size: int = 512,
        use_augmentation: bool = True,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.use_augmentation = use_augmentation
        self.in_channels = in_channels

        # 正規化參數
        self._mean = torch.tensor([0.5] * in_channels)
        self._std = torch.tensor([0.5] * in_channels)

        self._aug = self._build_aug() if _KORNIA_AVAILABLE else None

    def _build_aug(self) -> nn.Module:
        mean = self._mean
        std = self._std

        if self.use_augmentation:
            return K.AugmentationSequential(
                # 1. 基礎幾何與縮放
                K.RandomResizedCrop(
                    (self.img_size, self.img_size),
                    scale=(0.2, 1.0),
                    ratio=(0.75, 1.333),
                    same_on_batch=False,
                ),
                K.RandomHorizontalFlip(p=0.5, same_on_batch=False),
                K.RandomVerticalFlip(p=0.5, same_on_batch=False),
                
                # 2. 結構變換 (對應 Pipeline 1 的兩次 RandomAffine)
                K.RandomAffine(
                    degrees=45.0,
                    p=0.3,
                    same_on_batch=False,
                ),
                K.RandomAffine(
                    degrees=45.0,
                    p=0.3,
                    same_on_batch=False,
                ),

                # 3. 幾何變型 (彈性變換)
                K.RandomElasticTransform(
                    alpha=(50.0, 50.0),
                    sigma=(5.0, 5.0),
                    p=0.3,
                    same_on_batch=False,
                ),

                # 4. 形態學擾動 (線條粗細)
                RandomGPUMorphology(p=0.3, kernel_size=3),

                # 5. 雜訊
                K.RandomSaltAndPepperNoise(
                    amount=(0.0, 0.02),
                    p=0.2,
                    same_on_batch=False,
                ),

                # 6. 顏色擾動
                K.ColorJitter(
                    brightness=0.4, contrast=0.4,
                    p=0.8, same_on_batch=False,
                ),

                # 7. 正規化
                K.Normalize(mean=mean, std=std),

                # 8. 遮擋 (Cutout)
                K.RandomErasing(
                    scale=(0.02, 0.15),
                    ratio=(0.3, 3.3),
                    p=0.3,
                    same_on_batch=False,
                ),
                data_keys=["input"],
            )
        else:
            return K.AugmentationSequential(
                K.Resize((self.img_size, self.img_size)),
                K.Normalize(mean=mean, std=std),
                data_keys=["input"],
            )

    def create_views(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """從輸入 batch 生成兩個獨立增強視角。

        兩次呼叫增強管線各自使用不同的隨機參數，
        等效於 SimSiam 的雙正樣本生成。

        Args:
            x: [B, C, H, W] float tensor，值域 [0, 1]，已在 GPU 上。

        Returns:
            Tuple[Tensor, Tensor]: (view1, view2)，各 [B, C, H, W]。
        """
        if self._aug is not None:
            # 兩次獨立前向 → 不同隨機增強
            v1 = self._aug(x)
            v2 = self._aug(x)
        else:
            # Fallback: resize + normalize on GPU（無 kornia）
            v1 = self._manual_normalize(x)
            v2 = v1  # 無增強時兩視角相同；SimSiam 可能退化
            logger.debug(
                "GPUAugmentation: kornia 不可用，使用無增強模式（可能導致 collapse）"
            )
        return v1, v2

    def _manual_normalize(self, x: torch.Tensor) -> torch.Tensor:
        x_resized = F.interpolate(
            x, size=(self.img_size, self.img_size),
            mode="bilinear", align_corners=False,
        )
        mean = self._mean.to(x.device).view(1, -1, 1, 1)
        std = self._std.to(x.device).view(1, -1, 1, 1)
        return (x_resized - mean) / std

    def to(self, *args, **kwargs) -> "GPUAugmentation":
        super().to(*args, **kwargs)
        self._mean = self._mean.to(*args, **kwargs)
        self._std = self._std.to(*args, **kwargs)
        if self._aug is not None:
            self._aug = self._aug.to(*args, **kwargs)
        return self
