"""
SimSiam 網路模型實作。

實現論文 "Exploring Simple Siamese Representation Learning" 所提架構，
並針對使用者專案做特殊處理 (如：支援工程圖預設的一維灰階單通道輸入、適配參數初始化等)。
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple

from .base import BaseModel


def build_mlp(in_dim: int, hidden_dim: int, out_dim: int, bn_last: bool = True) -> nn.Sequential:
    """構造多層感知機 (MLP) 區塊。"""
    layers = [
        nn.Linear(in_dim, hidden_dim, bias=False),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim, bias=False)
    ]
    if bn_last:
        layers.append(nn.BatchNorm1d(out_dim, affine=True))
    return nn.Sequential(*layers)


class SimSiamModel(BaseModel):
    """SimSiam 主模型區塊。"""

    def __init__(self, backbone: str, proj_dim: int, pred_hidden: int, pretrained: bool, in_channels: int):
        """
        Args:
            backbone (str): 骨幹網路類型 (如 'resnet18' 或 'resnet50')。
            proj_dim (int): Projector 輸出維度 (即 Representation Dimension)。
            pred_hidden (int): Predictor 的隱藏瓶頸層 (Bottleneck) 維度。
            pretrained (bool): 是否使用 ImageNet 預訓練權重。
            in_channels (int): 輸入頻道 (1 for 灰階, 3 for RGB)。
        
        Raises:
            NotImplementedError: 遇到尚未支援的骨幹網路時丟出。
        """
        super().__init__()

        # 1. 初始化骨幹網路
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.resnet18(weights=weights)
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.resnet50(weights=weights)
        else:
            raise NotImplementedError(f"未支援的 backbone: {backbone}")

        # 適應通道數量 (例如灰階為 1)
        if in_channels != 3:
            old_conv = net.conv1
            new_conv = nn.Conv2d(
                in_channels, old_conv.out_channels, 
                kernel_size=old_conv.kernel_size, stride=old_conv.stride, 
                padding=old_conv.padding, bias=old_conv.bias is not None
            )
            # 將 RGB 預訓練權重藉由跨頻道加總並平均，對齊至單通道
            if pretrained and in_channels == 1:
                with torch.no_grad():
                    new_conv.weight[:] = old_conv.weight.sum(dim=1, keepdim=True) / 3.0
            net.conv1 = new_conv

        feat_dim = net.fc.in_features
        net.fc = nn.Identity()  # 移除原始分類頭
        self.backbone = net

        # 2. 投影頭 Projector (論文預設為 3-layer MLP，許多實作採用 2-layer)
        self.projector = build_mlp(feat_dim, 2048, proj_dim, bn_last=True)
        
        # 3. 預測頭 Predictor (2-layer MLP bottleneck)
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_hidden, bias=False),
            nn.BatchNorm1d(pred_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(pred_hidden, proj_dim)
        )

        # 4. 初始化
        self._init_weights()

    def _init_weights(self):
        """對新增的 MLP Classifier Head 初始化權重。"""
        for m in list(self.projector.modules()) + list(self.predictor.modules()):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向傳播。

        回傳目標投影矩陣 (z1, z2) 時，同時調用 .detach() 進行梯度截斷 (Stop-Gradient)，
        避免特徵學習時發生 Collapse，這也是 SimSiam 能擺脫 Negative Samples 運作的神奇之處。

        Args:
            x1 (torch.Tensor): 視角 1 (增強圖片 1)
            x2 (torch.Tensor): 視角 2 (增強圖片 2)

        Returns:
            Tuple: 包含 p1, p2, z1.detach(), z2.detach()
        """
        f1 = self.backbone(x1)
        f2 = self.backbone(x2)

        z1 = self.projector(f1)
        z2 = self.projector(f2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return p1, p2, z1.detach(), z2.detach()
