"""檢索評估指標模組 (Retrieval Evaluation Metrics)。

============================================================
實作實驗計劃中定義的三類指標，所有計算純向量化，
在 GPU 上執行（支援大資料集）。

指標定義：
    IACS  — 類別內平均餘弦相似度（越接近 1 越好，代表 Alignment）
    Inter — 類別間平均餘弦相似度（越接近 0 或負越好，代表 Uniformity）
    Margin — IACS - Inter（越大越好，代表對比能力）
    Top-K — 前 K 名中同類別影像的比例（Precision@K）
============================================================
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F


def compute_retrieval_metrics(
    features: torch.Tensor,
    labels: torch.Tensor,
    top_k_values: List[int] = (1, 5, 10),
) -> Dict[str, float]:
    """計算完整的檢索評估指標組。

    Args:
        features: 特徵矩陣 [N, D]，可未正規化（函式內自動 L2 normalize）。
        labels: 整數類別標籤 [N]。
        top_k_values: 需計算的 Top-K 值列表。

    Returns:
        dict: 包含以下鍵的結果字典：
            - ``IACS``: 類別內平均餘弦相似度
            - ``inter_class_avg_sim``: 類別間平均餘弦相似度
            - ``contrastive_margin``: Margin = IACS - Inter
            - ``top{k}_precision``: 各 k 值的 Precision@K
    """
    device = features.device
    N = features.shape[0]

    # L2 normalize → 餘弦相似度 = 內積
    feat_norm = F.normalize(features, dim=1)

    # 相似度矩陣 [N, N]
    sim_matrix = feat_norm @ feat_norm.T

    # 同類別遮罩 (含自身)
    labels = labels.to(device)
    same_class = labels.unsqueeze(0) == labels.unsqueeze(1)  # [N, N]
    self_mask = torch.eye(N, dtype=torch.bool, device=device)

    intra_mask = same_class & ~self_mask   # 同類、不含自身
    inter_mask = ~same_class               # 不同類

    # ---- IACS ----
    iacs = _masked_mean_per_row(sim_matrix, intra_mask).mean().item()

    # ---- Inter-Class ----
    inter = _masked_mean_per_row(sim_matrix, inter_mask).mean().item()

    # ---- Contrastive Margin ----
    margin = iacs - inter

    # ---- Top-K Precision ----
    # 排除自身（設為 -inf）
    sim_no_self = sim_matrix.clone()
    sim_no_self.fill_diagonal_(-float("inf"))

    topk_metrics = {}
    max_k = min(max(top_k_values), N - 1)
    # 一次取最大 k，避免重複排序
    _, top_indices = sim_no_self.topk(max_k, dim=1)  # [N, max_k]

    for k in top_k_values:
        k_clamped = min(k, N - 1)
        topk_labels = labels[top_indices[:, :k_clamped]]  # [N, k]
        is_same = (topk_labels == labels.unsqueeze(1)).float()
        precision_at_k = is_same.sum(dim=1) / k_clamped
        topk_metrics[f"top{k}_precision"] = precision_at_k.mean().item()

    return {
        "IACS": iacs,
        "inter_class_avg_sim": inter,
        "contrastive_margin": margin,
        **topk_metrics,
    }


def _masked_mean_per_row(
    matrix: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """對每行依遮罩計算平均值，回傳 [N] 向量。

    對於遮罩全為 False 的行（例如單一樣本類別），回傳 0.0。
    """
    masked = matrix * mask.float()
    counts = mask.float().sum(dim=1).clamp(min=1.0)
    return masked.sum(dim=1) / counts
