"""損失函數與坍塌監控模組 (Criterion & Collapse Monitor Module)。

============================================================
負責實作 SimSiam 的核心對稱負餘弦相似度損失函數，並確保自監督對比學習中的
「梯度阻斷 (Stop-gradient)」機制被嚴格且安全地執行，以徹底防止特徵空間坍塌。

設計功能：
    1. **對稱損失計算**：計算 D(p1, stop_gradient(z2)) 與 D(p2, stop_gradient(z1)) 的平均值。
    2. **梯度阻斷 (Stop-gradient)**：顯式在計算圖中調用 `.detach()` 截斷目標分支，
       確保優化只發生在預測頭與另一個視角的編碼路徑上。
    3. **維度坍塌指標 (Collapse Metric)**：監控 L2 正規化後的嵌入特徵 z 在 Batch 維度上的標準差均值。
============================================================
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimSiamLossCriterion(nn.Module):
    """SimSiam 對稱負餘弦相似度損失與坍塌監控器"""
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        p1: torch.Tensor,
        p2: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """計算 SimSiam 損失函數並監控坍塌狀態。

        本功能嚴格對 target 表徵 z1 與 z2 進行 `.detach()` 操作，以防止特徵表徵坍塌。

        Args:
            p1: 第一視角經過 Predictor 的預測向量，形狀 [B, D]
            p2: 第二視角經過 Predictor 的預測向量，形狀 [B, D]
            z1: 第一視角經過 Projector 的投影向量，形狀 [B, D]
            z2: 第二視角經過 Projector 的投影向量，形狀 [B, D]

        Returns:
            Tuple[torch.Tensor, float]: (損失純量, 平均維度標準差)。
                - 損失純量：用於反向傳播。
                - 平均維度標準差：若此數值趨近於 0，表示發生特徵空間坍塌；
                  均勻分佈之特徵空間標準差理論上應接近 1/sqrt(d)。
        """
        # --- 核心機制：嚴格執行 Stop-gradient (梯度阻斷) ---
        z1_target = z1.detach()
        z2_target = z2.detach()

        # 計算對稱負餘弦相似度損失： 0.5 * D(p1, z2) + 0.5 * D(p2, z1)
        loss1 = self._negative_cosine_similarity(p1, z2_target)
        loss2 = self._negative_cosine_similarity(p2, z1_target)
        loss = 0.5 * (loss1 + loss2)

        # 計算特徵坍塌監控指標 (取 z1_target 與 z2_target 的平均標準差)
        with torch.no_grad():
            std_z1 = self.calculate_collapse_std(z1_target)
            std_z2 = self.calculate_collapse_std(z2_target)
            avg_std = (std_z1 + std_z2) / 2.0

        return loss, avg_std

    def _negative_cosine_similarity(self, p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """計算兩個向量集合之間的負餘弦相似度平均值。

        公式: D(p, z) = - (p / ||p||_2) * (z / ||z||_2)
        """
        # L2 正規化
        p_norm = F.normalize(p, p=2.0, dim=1)
        z_norm = F.normalize(z, p=2.0, dim=1)
        
        # 點積後取 Mean 並取負號
        # 內積後形狀為 [B]，再求 Batch 的平均
        return -(p_norm * z_norm).sum(dim=1).mean()

    def calculate_collapse_std(self, z: torch.Tensor) -> float:
        """計算 L2 正規化特徵在 Batch 維度上的平均標準差。

        正常特徵空間的標準差應接近 1 / sqrt(dim)。
        若跌破臨界值 (如 0.01)，則代表發生特徵坍塌 (Collapse)，特徵退化為常數。
        """
        # 先對特徵進行 L2 正規化
        z_norm = F.normalize(z, p=2.0, dim=1)
        
        # 沿著 Batch 維度 (dim=0) 計算 std，再對所有的特徵維度 (dim=1) 取平均值
        # eps 避免微小精度誤差造成的 NaN 或是零除錯誤
        std_metric = z_norm.std(dim=0).mean().item()
        return std_metric
