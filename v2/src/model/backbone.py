"""Backbone 工廠模組 (Backbone Factory Module)。

============================================================
提供 ``create_backbone`` 工廠函式，負責：

1. 依名稱建立 ResNet backbone。
2. 自動處理非標準輸入通道（如灰階 ``in_channels=1``）
   的 ``conv1`` 適配。
3. 移除原始分類頭 (``fc``)，回傳純特徵抽取器。

設計原則（開放封閉）：
    新增 backbone 類型時，只需在 ``_BACKBONE_REGISTRY``
    中註冊新的建構函式，不需修改消費端程式碼。
============================================================
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn
import torchvision.models as models

from src.logger import get_logger

logger = get_logger(__name__)

# ============================================================
# Backbone 註冊表
# ============================================================

# 每個建構函式接受 (pretrained: bool)，回傳 (nn.Module, feat_dim)
BackboneConstructor = Callable[[bool], Tuple[nn.Module, int]]

_BACKBONE_REGISTRY: Dict[str, BackboneConstructor] = {}


def _register(name: str) -> Callable:
    """裝飾器：將 backbone 建構函式註冊到全域表中。

    Args:
        name: Backbone 名稱（例如 ``"resnet18"``）。

    Returns:
        Callable: 原始函式（不修改）。
    """

    def decorator(fn: BackboneConstructor) -> BackboneConstructor:
        _BACKBONE_REGISTRY[name] = fn
        return fn

    return decorator


@_register("resnet18")
def _resnet18(pretrained: bool) -> Tuple[nn.Module, int]:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.resnet18(weights=weights)
    feat_dim = net.fc.in_features
    net.fc = nn.Identity()
    return net, feat_dim


@_register("resnet50")
def _resnet50(pretrained: bool) -> Tuple[nn.Module, int]:
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.resnet50(weights=weights)
    feat_dim = net.fc.in_features
    net.fc = nn.Identity()
    return net, feat_dim


# ============================================================
# 公開介面
# ============================================================


def create_backbone(
    name: str,
    pretrained: bool = False,
    in_channels: int = 3,
) -> Tuple[nn.Module, int]:
    """建立指定的 backbone 特徵抽取器。

    自動處理輸入通道適配：當 ``in_channels != 3`` 時，
    將 ``conv1`` 的 RGB 權重平均化至目標通道數。

    Args:
        name: Backbone 名稱，須為已註冊的名稱。
            目前支援 ``"resnet18"`` 與 ``"resnet50"``。
        pretrained: 是否載入 ImageNet 預訓練權重。
        in_channels: 輸入影像通道數（``1`` = 灰階，``3`` = RGB）。

    Returns:
        Tuple[nn.Module, int]: ``(backbone, feature_dim)``。
            backbone 的 ``fc`` 已被替換為 ``Identity()``。

    Raises:
        ValueError: 當 ``name`` 未在註冊表中時。
    """
    if name not in _BACKBONE_REGISTRY:
        available = ", ".join(sorted(_BACKBONE_REGISTRY.keys()))
        raise ValueError(
            f"不支援的 backbone: '{name}'。可用選項: {available}"
        )

    constructor = _BACKBONE_REGISTRY[name]
    backbone, feat_dim = constructor(pretrained)

    # --- 通道適配 ---
    if in_channels != 3:
        backbone = _adapt_input_channels(
            backbone, in_channels, pretrained
        )

    logger.info(
        "Backbone 建立完成: %s (pretrained=%s, in_channels=%d, feat_dim=%d)",
        name,
        pretrained,
        in_channels,
        feat_dim,
    )
    return backbone, feat_dim


def _adapt_input_channels(
    backbone: nn.Module,
    in_channels: int,
    pretrained: bool,
) -> nn.Module:
    """適配 backbone 的 conv1 以接受非 RGB 輸入。

    策略：建立新的 Conv2d 並將 RGB 權重沿通道維度平均化。

    Args:
        backbone: 原始 backbone 模組。
        in_channels: 目標通道數。
        pretrained: 是否有預訓練權重（決定是否平均化）。

    Returns:
        nn.Module: 已適配的 backbone。
    """
    old_conv: nn.Conv2d = backbone.conv1  # type: ignore[attr-defined]
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )

    if pretrained and in_channels != 3:
        with torch.no_grad():
            # 平均 RGB 三通道 → [out_ch, 1, kH, kW]
            avg = old_conv.weight.mean(dim=1, keepdim=True)
            # 每個 in_channels 分配等比例權重，保持激活值量級一致
            new_conv.weight.copy_(avg.expand(-1, in_channels, -1, -1) / in_channels)
    elif not pretrained and in_channels != 3:
        logger.warning(
            "in_channels=%d 且 pretrained=False，conv1 以隨機權重初始化",
            in_channels,
        )

    backbone.conv1 = new_conv  # type: ignore[attr-defined]
    logger.debug(
        "conv1 通道已適配: 3 → %d (pretrained weight avg=%s)",
        in_channels,
        pretrained,
    )
    return backbone
