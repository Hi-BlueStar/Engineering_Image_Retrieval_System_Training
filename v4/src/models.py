"""模型架構與拓撲模組 (Model Architecture & Topology Module)。

============================================================
實作 SimSiam (Simple Siamese) 對比學習的模型拓撲。

為了極大化模組的內聚力與下游評估（如特徵檢索）的解耦，模型被拆分為：
    1. **SimSiamEncoder**：負責提取影像特徵並進行非線性投影，產出目標向量 $z$。
    2. **SimSiamPredictor**：非對稱預測頭，負責將 $z$ 轉換為預測向量 $p$。
    3. **Backbone Factory**：自動適配單通道輸入，並將 ImageNet 預訓練權重平均化。
============================================================
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models

from .logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 1. 骨幹網路建立與單通道適配 (Backbone Factory & Channel Adapt)
# ============================================================

def create_backbone(
    name: str,
    pretrained: bool = True,
    in_channels: int = 1,
) -> Tuple[nn.Module, int]:
    """建立 ResNet 骨幹網路特徵提取器。

    自動處理輸入通道適配：當 `in_channels != 3` 時，
    將 `conv1` 的 RGB 權重沿著通道軸平均化，以確保特徵活化值量級的一致性。

    Args:
        name: Backbone 名稱 ("resnet18" 或 "resnet50")。
        pretrained: 是否載入 ImageNet 預訓練權重。
        in_channels: 輸入影像的通道數（灰階圖為 1）。

    Returns:
        Tuple[nn.Module, int]: (特徵提取器, 輸出特徵維度)。
            特徵提取器的 fc 層已替換為 Identity。
    """
    # 建立 ResNet 模型
    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet18(weights=weights)
        feat_dim = net.fc.in_features
    elif name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet50(weights=weights)
        feat_dim = net.fc.in_features
    else:
        raise NotImplementedError(f"不支援的 backbone: {name}")

    # 移除最後用於分類的全連接層
    net.fc = nn.Identity()

    # 若為單通道 (灰階圖) 輸入，修改 conv1 的結構與權重
    if in_channels != 3:
        old_conv: nn.Conv2d = net.conv1
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        if pretrained:
            with torch.no_grad():
                # 平均 RGB 三通道的權重 [out_ch, 1, kH, kW]
                avg_weight = old_conv.weight.mean(dim=1, keepdim=True)
                # 擴展到目標通道數，並除以通道數以保持常態活化值
                new_conv.weight.copy_(avg_weight.expand(-1, in_channels, -1, -1) / in_channels)
        else:
            logger.warning("提示: 未加載預訓練權重，conv1 通道將採用隨機初始化。")

        net.conv1 = new_conv

    logger.info("骨幹網路建立完成: %s (pretrained=%s, in_channels=%d, feat_dim=%d)",
                name, pretrained, in_channels, feat_dim)
    return net, feat_dim


# ============================================================
# 2. 輔助函數：多層感知機 (MLP Builder)
# ============================================================

def _build_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    num_layers: int = 3,
    bn_last: bool = True,
) -> nn.Sequential:
    """建立多層感知機 (MLP)，常用於 Projector"""
    layers: list[nn.Module] = []
    current_dim = in_dim
    for _ in range(num_layers - 1):
        layers.extend([
            nn.Linear(current_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        ])
        current_dim = hidden_dim

    layers.append(nn.Linear(current_dim, out_dim, bias=False))
    if bn_last:
        layers.append(nn.BatchNorm1d(out_dim, affine=True))
    return nn.Sequential(*layers)


# ============================================================
# 3. 解耦核心模型類別 (Decoupled SimSiam Modules)
# ============================================================

class SimSiamEncoder(nn.Module):
    """SimSiam 編碼器模組。

    功能：輸入影像，提取特徵，並映射到低維度的投影特徵空間。
    下游特徵檢索（LOO 評估）僅需獨立使用此模組。
    """
    def __init__(
        self,
        backbone_name: str = "resnet18",
        proj_dim: int = 2048,
        in_channels: int = 1,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        # 1. 建立骨幹網路
        self.backbone, feat_dim = create_backbone(
            name=backbone_name, pretrained=pretrained, in_channels=in_channels
        )
        
        # 2. 建立三層投影頭 (Projector)，隱藏層維度固定為 512 或 feat_dim
        hidden_dim = 512 if feat_dim <= 512 else feat_dim
        self.projector = _build_mlp(
            in_dim=feat_dim,
            hidden_dim=hidden_dim,
            out_dim=proj_dim,
            num_layers=3,
            bn_last=True
        )

        # 3. 初始化權重
        self._init_weights()

    def _init_weights(self) -> None:
        """投影器線性層 Truncated Normal 初始化"""
        for m in self.projector.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """只通過 Backbone 提取特徵，用於下游檢索"""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """編碼器前向傳播：輸入影像 x，輸出投影向量 z"""
        f = self.backbone(x)
        z = self.projector(f)
        return z


class SimSiamPredictor(nn.Module):
    """SimSiam 預測器頭部模組。

    功能：對投影向量進行不對稱變換，生成預測向量 p。
    採用非對稱瓶頸結構 (Bottleneck MLP)，最後一層無 BatchNorm 與 ReLU。
    """
    def __init__(self, proj_dim: int = 2048, pred_hidden: int = 512) -> None:
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_hidden, bias=False),
            nn.BatchNorm1d(pred_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(pred_hidden, proj_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """預測器線性層 Truncated Normal 初始化"""
        for m in self.predictor.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """預測器前向傳播：輸入投影向量 z，輸出預測向量 p"""
        return self.predictor(z)
