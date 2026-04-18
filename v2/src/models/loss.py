"""
損失函數模組 (Loss Function)。

獨立定義損失計算方式，符合單一職責原則。
"""

import torch
import torch.nn.functional as F

def negative_cosine_similarity(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """計算負餘弦相似度損失 (Negative Cosine Similarity)。

    公式: Loss = -(p / ||p||2) dot (z / ||z||2)

    Args:
        p (torch.Tensor): Predictor 的輸出預測向量 [B, Dim]。
        z (torch.Tensor): Projector 輸出的目標向量 (常數目標) [B, Dim]。

    Returns:
        torch.Tensor: 平均損失純量。
    """
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return -(p * z).sum(dim=1).mean()
