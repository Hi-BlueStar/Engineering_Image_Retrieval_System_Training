"""SimSiam 模型架構 (Architecture)。

實作 Projector、Predictor 以及負餘弦相似度損失函數 (Loss)。
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbones import build_backbone


def _build_mlp(in_dim: int, hidden_dim: int, out_dim: int, bn_last: bool = True, dropout: float = 0.0) -> nn.Sequential:
    """輔助函數：建立多層感知機 (MLP) 區塊。"""
    layers = [
        nn.Linear(in_dim, hidden_dim, bias=False),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, out_dim, bias=False))
    if bn_last:
        layers.append(nn.BatchNorm1d(out_dim, affine=True))
    return nn.Sequential(*layers)


class SimSiam(nn.Module):
    """SimSiam 模型主體。
    
    機制: Loss = D(p1, stop_gradient(z2)) + D(p2, stop_gradient(z1))
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        in_channels: int = 1,
        proj_dim: int = 2048,
        pred_hidden: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()

        # 1. Backbone
        self.backbone, feat_dim = build_backbone(backbone_name, pretrained, in_channels)

        # 2. Projector
        self.projector = _build_mlp(feat_dim, 2048, proj_dim, bn_last=True, dropout=dropout)

        # 3. Predictor
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_hidden, bias=False),
            nn.BatchNorm1d(pred_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(pred_hidden, proj_dim)
        )

        # 4. 初始化 MLP 權重 ( Backbone 可能是 pretrained 的即不覆寫 )
        for m in list(self.projector.modules()) + list(self.predictor.modules()):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向傳播。
        
        回傳預測 p 與 目標 z (這點很重要，z 被 detach 出來阻止梯度流動)。
        """
        f1 = self.backbone(x1)
        f2 = self.backbone(x2)

        z1 = self.projector(f1)
        z2 = self.projector(f2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # 核心：回傳 detached 的 z
        return p1, p2, z1.detach(), z2.detach()


def simsiam_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """計算負餘弦相似度。"""
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return -(p * z).sum(dim=1).mean()
