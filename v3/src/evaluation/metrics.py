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
    """計算完整的檢索評估指標組（Leave-One-Out 策略，防 OOM 分批實作）。

    Args:
        features: 特徵矩陣 [N, D]。
        labels: 整數類別標籤 [N]。
        top_k_values: 需計算的 Top-K 值列表。

    Returns:
        dict: 包含以下鍵的結果字典：
            - ``IACS``: 類別內平均餘弦相似度
            - ``inter_class_avg_sim``: 類別間平均餘弦相似度
            - ``contrastive_margin``: Margin = IACS - Inter
            - ``macro_mAP``: 宏觀平均平均精度 (Macro-mAP)
            - ``top{k}_precision``: 各 k 值的 Precision@K
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = features.to(device)
    labels = labels.to(device)
    N = features.shape[0]

    # L2 normalize → 餘弦相似度 = 內積
    feat_norm = F.normalize(features, dim=1)

    ap_all = torch.zeros(N, device=device)
    intra_sim_sum = torch.zeros(N, device=device)
    inter_sim_sum = torch.zeros(N, device=device)
    intra_counts = torch.zeros(N, device=device)
    inter_counts = torch.zeros(N, device=device)

    # 預先統計每個類別的樣本數
    unique_labels, labels_count = torch.unique(labels, return_counts=True)
    label_to_count = {l.item(): c.item() for l, c in zip(unique_labels, labels_count)}

    topk_precisions = {f"top{k}_precision": torch.zeros(N, device=device) for k in top_k_values}

    # 批次大小設為 512，防記憶體 OOM
    eval_batch_size = min(512, N)

    for i in range(0, N, eval_batch_size):
        end_idx = min(i + eval_batch_size, N)
        batch_feats = feat_norm[i:end_idx]  # [B, D]
        batch_labels = labels[i:end_idx]  # [B]

        # 計算當前 Batch 對所有 Gallery 的相似度 -> [B, N]
        sim_b = batch_feats @ feat_norm.T  # [B, N]

        # 類別比對遮罩 -> [B, N]
        same_class_b = batch_labels.unsqueeze(1) == labels.unsqueeze(0)  # [B, N]

        # 建立 Self-mask (排除自身)
        self_mask_b = torch.zeros_like(same_class_b, dtype=torch.bool)
        for j in range(end_idx - i):
            self_mask_b[j, i + j] = True

        intra_mask_b = same_class_b & ~self_mask_b
        inter_mask_b = ~same_class_b

        # 累積統計數量與相似度和
        intra_counts[i:end_idx] = intra_mask_b.sum(dim=1).float()
        inter_counts[i:end_idx] = inter_mask_b.sum(dim=1).float()

        intra_sim_sum[i:end_idx] = (sim_b * intra_mask_b.float()).sum(dim=1)
        inter_sim_sum[i:end_idx] = (sim_b * inter_mask_b.float()).sum(dim=1)

        # ---- Leave-One-Out AP 與 Top-K 計算 ----
        sim_b_no_self = sim_b.clone()
        sim_b_no_self[self_mask_b] = -float("inf")

        # 排序
        sorted_sim, sorted_indices = torch.sort(sim_b_no_self, dim=1, descending=True)
        sorted_indices = sorted_indices[:, :-1]  # 移除最後一項 (即被設為 -inf 的自身)

        # 取得排序後的標籤
        sorted_labels = labels[sorted_indices]  # [B, N-1]
        is_match = (sorted_labels == batch_labels.unsqueeze(1)).float()  # [B, N-1]

        # 計算累積匹配數與各 rank 上的 Precision
        cum_matches = torch.cumsum(is_match, dim=1)
        ranks = torch.arange(1, N, device=device).float().unsqueeze(0)  # [1, N-1]
        precisions = cum_matches / ranks  # [B, N-1]

        # 計算每個 Query 的 AP
        for j in range(end_idx - i):
            q_label = batch_labels[j].item()
            R_q = label_to_count[q_label] - 1  # 扣除自身後的 Gallery 同類總數
            if R_q > 0:
                ap_all[i + j] = (precisions[j] * is_match[j]).sum() / R_q
            else:
                ap_all[i + j] = 0.0

        # 計算每個 Query 的 Top-K Precision
        for k in top_k_values:
            k_clamped = min(k, N - 1)
            topk_precisions[f"top{k}_precision"][i:end_idx] = cum_matches[:, k_clamped - 1] / k_clamped

    # ---- 指標彙整 ----
    # 避免除以 0
    intra_counts_clamped = intra_counts.clamp(min=1.0)
    inter_counts_clamped = inter_counts.clamp(min=1.0)

    iacs_per_img = intra_sim_sum / intra_counts_clamped
    inter_per_img = inter_sim_sum / inter_counts_clamped

    # 若該樣本無同類則設為 0
    iacs_per_img[intra_counts == 0] = 0.0
    inter_per_img[inter_counts == 0] = 0.0

    iacs = iacs_per_img.mean().item()
    inter = inter_per_img.mean().item()
    margin = iacs - inter

    # 計算 Macro-mAP (先算類別內平均 mAP_g，再跨類別平均)
    mAP_groups = []
    for label_val in unique_labels:
        group_mask = labels == label_val
        group_aps = ap_all[group_mask]
        mAP_groups.append(group_aps.mean().item())
    macro_map = sum(mAP_groups) / len(mAP_groups) if mAP_groups else 0.0

    topk_results = {}
    for k in top_k_values:
        topk_results[f"top{k}_precision"] = topk_precisions[f"top{k}_precision"].mean().item()

    return {
        "IACS": iacs,
        "inter_class_avg_sim": inter,
        "contrastive_margin": margin,
        "macro_mAP": macro_map,
        **topk_results,
    }
