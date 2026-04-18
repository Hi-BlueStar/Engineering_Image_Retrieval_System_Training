"""Backbone 模型模組 (Backbone Architectures)。

負責實例化 ResNet 等骨幹網路，並針對自訂的輸入通道 (如灰階單通道) 進行權重轉換與調整。
"""
import torch
import torch.nn as nn
import torchvision.models as models


def build_backbone(name: str = "resnet18", pretrained: bool = True, in_channels: int = 1) -> Tuple[nn.Module, int]:
    """建構特徵萃取骨幹網路 (Backbone)。
    
    Args:
        name: 模型名稱 ('resnet18' 或 'resnet50')。
        pretrained: 是否載入 ImageNet 預訓練權重。
        in_channels: 輸入影像的通道數 (1 代表灰階)。
        
    Returns:
        Tuple[nn.Module, int]:
            - backbone_net: 處理後的骨幹網路。
            - out_dim: 特徵層輸出的維度長度。
    """
    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet18(weights=weights)
    elif name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet50(weights=weights)
    else:
        raise NotImplementedError(f"未支援的 backbone: {name}")

    # 針對輸入通道進行修改 (例如 3 RGB -> 1 Grayscale)
    if in_channels != 3:
        old_conv = net.conv1
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        # 若載入了 ImageNet pretrained 且目標為單通道，通常將 RGB 權重取平均
        if pretrained and in_channels == 1:
            with torch.no_grad():
                # [64, 3, 7, 7] -> [64, 1, 7, 7]
                new_conv.weight[:] = old_conv.weight.sum(dim=1, keepdim=True) / 3.0

        net.conv1 = new_conv

    # 取得 Backbone 在進入 FC 分類層前的特徵輸出維度
    feat_dim = net.fc.in_features
    # 把原本的全連接分類器移除，替換為恆等映射 (Identity)
    net.fc = nn.Identity()

    return net, feat_dim
