"""SimSiam 模型架構模組 (SimSiam Model Module)。

============================================================
實作 SimSiam (Simple Siamese Representation Learning) 的
核心 ``nn.Module``。

結構::

    x → Backbone → f (features)
    f → Projector → z (embeddings)
    z → Predictor → p (predictions)

損失計算::

    Loss = D(p1, stop_gradient(z2)) + D(p2, stop_gradient(z1))

此模組僅包含模型定義與前向傳播邏輯，
不包含任何訓練迴圈、損失函數或 I/O 操作。

References:
    Chen & He. "Exploring Simple Siamese Representation Learning".
    CVPR 2021.
============================================================
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.model.backbone import create_backbone
from src.logger import get_logger

logger = get_logger(__name__)


def _mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    *,
    bn_last: bool = True,
    dropout: float = 0.0,
) -> nn.Sequential:
    """建立 MLP 區塊（Linear → BN → ReLU → [Dropout] → Linear [→ BN]）。

    Args:
        in_dim: 輸入維度。
        hidden_dim: 隱藏層維度。
        out_dim: 輸出維度。
        bn_last: 最後一層是否加 BatchNorm。
        dropout: 第一層後的 Dropout 比率。

    Returns:
        nn.Sequential: MLP 模組。
    """
    layers: list[nn.Module] = [
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

    組成：
        - **Backbone**：透過 ``create_backbone`` 工廠建立。
        - **Projector**：2 層 MLP，將 backbone 特徵投影至
          ``proj_dim`` 維度空間。
        - **Predictor**：非對稱 MLP（瓶頸結構），
          是 SimSiam 不需要負樣本的關鍵設計。

    Args:
        backbone: Backbone 名稱（``"resnet18"`` / ``"resnet50"``）。
        proj_dim: Projector 輸出維度。
        pred_hidden: Predictor 隱藏層維度。
        dropout: Projector Dropout 比率。
        pretrained: 是否載入 ImageNet 預訓練權重。
        in_channels: 輸入影像通道數。
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        proj_dim: int = 2048,
        pred_hidden: int = 512,
        dropout: float = 0.0,
        pretrained: bool = False,
        in_channels: int = 1,
    ) -> None:
        super().__init__()

        # 1. Backbone
        self.backbone, feat_dim = create_backbone(
            name=backbone, pretrained=pretrained, in_channels=in_channels
        )

        # 2. Projector
        self.projector = _mlp(
            feat_dim, 2048, proj_dim, bn_last=True, dropout=dropout
        )

        # 3. Predictor
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_hidden, bias=False),
            nn.BatchNorm1d(pred_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(pred_hidden, proj_dim),
        )

        # 4. 權重初始化（僅 MLP 部分）
        self._init_mlp_weights()

        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "SimSiam 初始化完成: total_params=%s, trainable=%s",
            f"{total_params:,}",
            f"{trainable:,}",
        )

    def _init_mlp_weights(self) -> None:
        """初始化 Projector 與 Predictor 的權重。

        使用 Truncated Normal (std=0.02) 初始化線性層，
        偏差項歸零。
        """
        for module in list(self.projector.modules()) + list(
            self.predictor.modules()
        ):
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向傳播：雙視角輸入，回傳預測與目標投影。

        Args:
            x1: 第一個視角 ``[B, C, H, W]``。
            x2: 第二個視角 ``[B, C, H, W]``。

        Returns:
            tuple: ``(p1, p2, z1_detached, z2_detached)``。
                - ``p1, p2``：Predictor 的預測向量。
                - ``z1, z2``：Projector 的投影向量（已 detach，
                  作為 stop-gradient 目標）。
        """
        f1 = self.backbone(x1)
        f2 = self.backbone(x2)

        z1 = self.projector(f1)
        z2 = self.projector(f2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return p1, p2, z1.detach(), z2.detach()
