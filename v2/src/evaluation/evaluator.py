"""檢索評估器模組 (Retrieval Evaluator)。

============================================================
訓練完成後，從訓練好的 SimSiam 模型提取特徵，
並對有標籤資料集計算所有檢索指標。

評估流程：
    1. 載入 Best Checkpoint（Backbone + Projector）
    2. 遍歷有標籤資料集，提取每張影像的 z（Projector 輸出）
    3. 呼叫 metrics.compute_retrieval_metrics() 計算指標
    4. 將結果儲存至 JSON
============================================================
"""

from __future__ import annotations

import json
import csv
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_retrieval_metrics
from src.logger import get_logger

logger = get_logger(__name__)


def evaluate_model(
    model: torch.nn.Module,
    labeled_dataset,
    device: str,
    top_k_values: List[int] = (1, 5, 10),
    batch_size: int = 64,
    num_workers: int = 4,
    output_csv_path: Optional[str] = None,
) -> Dict[str, float]:
    """從模型提取特徵並計算全套檢索指標。

    使用 Projector 輸出（z）作為特徵向量，與 SimSiam 訓練目標一致。

    Args:
        model: 已訓練的 SimSiam 模型。
        labeled_dataset: LabeledImageDataset，回傳 (tensor, label_idx)。
        device: 運算裝置。
        top_k_values: 計算 Precision@K 的 K 值列表。
        batch_size: 特徵提取的 batch 大小。
        num_workers: DataLoader workers 數量。
        output_csv_path: 若提供，將配對的 similarity_score 與 is_tp 寫入 CSV，供繪圖使用。

    Returns:
        dict: 指標結果（IACS, Inter, Margin, Top-K）。
    """
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod

    loader = DataLoader(
        labeled_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    model.eval()
    all_features: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            # 使用 backbone → projector 的特徵（SimSiam z 空間）
            features = model.backbone(images)
            features = model.projector(features)
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

    features_cat = torch.cat(all_features, dim=0)  # [N, D]
    labels_cat = torch.cat(all_labels, dim=0)       # [N]

    logger.info(
        "特徵提取完成: N=%d, D=%d, classes=%d",
        features_cat.shape[0],
        features_cat.shape[1],
        labels_cat.max().item() + 1,
    )

    metrics = compute_retrieval_metrics(
        features_cat,
        labels_cat,
        top_k_values=list(top_k_values),
    )

    if output_csv_path:
        _export_similarity_scores(features_cat, labels_cat, Path(output_csv_path))

    _log_metrics(metrics)
    return metrics


def save_metrics(
    metrics: Dict[str, float],
    output_path: Path,
    extra_info: Optional[dict] = None,
) -> None:
    """將指標結果儲存至 JSON 檔案。

    Args:
        metrics: 指標字典。
        output_path: 輸出 JSON 路徑。
        extra_info: 額外資訊（如 run_name, epoch）。
    """
    payload = {**(extra_info or {}), **metrics}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("評估結果已儲存: %s", output_path)

def _export_similarity_scores(
    features: torch.Tensor,
    labels: torch.Tensor,
    output_csv: Path,
    max_fp_samples: int = 100000,
) -> None:
    """計算配對的 similarity score 並導出到 CSV。

    為了避免 CSV 過大，False Positive 會隨機抽樣。
    """
    logger.info("開始計算配對相似度以供圖表使用...")
    device = features.device
    N = features.shape[0]

    # L2 normalize -> cosine similarity = dot product
    feat_norm = torch.nn.functional.normalize(features, dim=1)
    sim_matrix = feat_norm @ feat_norm.T

    # 找出 TP 與 FP
    labels = labels.to(device)
    same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
    self_mask = torch.eye(N, dtype=torch.bool, device=device)

    tp_mask = same_class & ~self_mask
    fp_mask = ~same_class

    # 取得 TP 分數 (全取，因為通常數量較少)
    tp_scores = sim_matrix[tp_mask].cpu().numpy().tolist()
    
    # 取得 FP 分數，若超過 max_fp_samples 則隨機抽樣
    fp_scores_full = sim_matrix[fp_mask].cpu().numpy()
    if len(fp_scores_full) > max_fp_samples:
        import numpy as np
        fp_scores = np.random.choice(fp_scores_full, max_fp_samples, replace=False).tolist()
    else:
        fp_scores = fp_scores_full.tolist()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["similarity_score", "is_tp"])
        for score in tp_scores:
            writer.writerow([score, "True Positive"])
        for score in fp_scores:
            writer.writerow([score, "False Positive"])
            
    logger.info("相似度 CSV 匯出完成: %s (TP: %d, FP: %d)", output_csv, len(tp_scores), len(fp_scores))


def _log_metrics(metrics: Dict[str, float]) -> None:
    logger.info("=" * 50)
    logger.info("檢索評估結果")
    logger.info("=" * 50)
    logger.info("  IACS  (類別內平均相似度): %.4f", metrics.get("IACS", 0.0))
    logger.info("  Inter (類別間平均相似度): %.4f", metrics.get("inter_class_avg_sim", 0.0))
    logger.info("  Margin (對比度差距):       %.4f", metrics.get("contrastive_margin", 0.0))
    for k, v in metrics.items():
        if k.startswith("top"):
            logger.info("  Precision@%-4s          %.4f", k.replace("_precision", ""), v)
    logger.info("=" * 50)
