"""輔助指標模組 (Metrics)。

計算特徵向量標準差等輔助指標，用於觀察模型是否產生 Collapse (坍塌)。
"""
import torch
import torch.nn.functional as F

def calculate_collapse_std(z: torch.Tensor) -> float:
    """計算 L2 正規化後特徵在各維度的標準差平均值。
    
    用於監控維度坍塌 (Dimensional Collapse)。
    理論上，一個完美均勻分佈的特徵空間，其維度標準差應趨近於 1/sqrt(d)。
    若此數值趨近於 0，表示模型僅在少數維度上產生變化。
    
    Args:
        z: Projector 產生的 embeddings，形狀為 [Batch, Dim]
        
    Returns:
        float: 維度標準差的平均值
    """
    if z.size(0) <= 1:
        return 0.0  # 無法計算單一樣本的 batch_std
        
    z_norm = F.normalize(z, dim=1)
    # 沿著 Batch 維度計算各維度的 std，再對維度取平均
    return z_norm.std(dim=0).mean().item()
