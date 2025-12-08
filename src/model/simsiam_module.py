# src/models/simsiam_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image


# -----------------------------------------------------------------------------
# 1. Geometry-Aware Augmentation (針對工程圖優化)
# -----------------------------------------------------------------------------


class GeometryAwareTransform:
    """
    針對工程圖 (CAD/Line Art) 設計的幾何敏感資料增強。
    移除對線條有害的高斯模糊與顏色抖動，專注於幾何不變性。
    """

    def __init__(
        self, img_size: int = 224, mean: list[float] = [0.5], std: list[float] = [0.5]
    ):
        self.transform = T.Compose(
            [
                # 隨機裁切：保留局部與全域的關聯性，scale 下限設為 0.4 避免破壞結構
                T.RandomResizedCrop(
                    img_size,
                    scale=(0.4, 1.0),
                    interpolation=T.InterpolationMode.BICUBIC,
                ),
                # 翻轉不變性
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                # 關鍵：仿射變換 (模擬圖紙旋轉、平移、比例不一)
                T.RandomApply(
                    [
                        T.RandomAffine(
                            degrees=90,
                            translate=(0.1, 0.1),
                            scale=(0.8, 1.2),
                            shear=10,
                            interpolation=T.InterpolationMode.BILINEAR,
                        )
                    ],
                    p=0.5,
                ),
                # 關鍵：透視變換 (模擬拍攝角度偏差)
                T.RandomPerspective(distortion_scale=0.2, p=0.3),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, x: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        """回傳同一張圖片的兩個不同增強視角 (Two Crops)"""
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2


# -----------------------------------------------------------------------------
# 2. SimSiam Model Architecture
# -----------------------------------------------------------------------------


class SimSiam(nn.Module):
    """
    SimSiam 模型主體 (Backbone + Projector + Predictor)。
    支援單通道 (Grayscale) 輸入。
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        proj_dim: int = 2048,
        pred_hidden: int = 512,
        in_channels: int = 1,  # Default to 1 for engineering drawings
        pretrained: bool = False,
    ):
        super().__init__()

        # 建立 Backbone
        if backbone == "resnet18":
            self.backbone = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
            feat_dim = 512
        elif backbone == "resnet50":
            self.backbone = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            )
            feat_dim = 2048
        else:
            raise NotImplementedError(f"Backbone {backbone} not supported")

        # 修改第一層卷積以適配輸入通道 (如灰階)
        if in_channels != 3:
            old_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias,
            )
            # 若載入預訓練權重，將 RGB 權重平均化
            if pretrained:
                with torch.no_grad():
                    self.backbone.conv1.weight[:] = (
                        old_conv.weight.sum(dim=1, keepdim=True) / 3.0
                    )

        # 移除 FC 層
        self.backbone.fc = nn.Identity()

        # Projector (3-layer MLP recommended by paper, typically)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim, affine=True),
        )

        # Predictor (2-layer MLP)
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_hidden, bias=False),
            nn.BatchNorm1d(pred_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(pred_hidden, proj_dim),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """前向傳播。

        參數:
            x1: 第一個視角的圖片 Batch [B, C, H, W]。
            x2: 第二個視角的圖片 Batch [B, C, H, W]。

        返回:
            p1, p2: Predictor 的預測向量。
            z1, z2: Projector 的目標投影向量 (已 detach，用於作為 Target)。
        """
        # 共享 Backbone 與 Projector
        f1 = self.backbone(x1)
        f2 = self.backbone(x2)

        z1 = self.projector(f1)
        z2 = self.projector(f2)

        # Predictor 轉換
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # 關鍵：回傳 detach 的 z，在 Loss 計算時作為常數目標 (Stop-Gradient)
        return p1, p2, z1.detach(), z2.detach()


def negative_cosine_similarity_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    SimSiam Loss function: - (p_norm * z_norm)
    """
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return -(p * z).sum(dim=1).mean()
