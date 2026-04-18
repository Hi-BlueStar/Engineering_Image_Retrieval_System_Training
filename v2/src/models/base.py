"""
抽象模型基底 (Abstract Base Model)。

定義模型必須實作的介面，確保擴展性 (如未來替換為 MoCo 或 BYOL 時，
Trainer 的呼叫介面不需要有任何改變)。
"""

import abc
import torch
import torch.nn as nn
from typing import Tuple

class BaseModel(nn.Module, abc.ABC):
    """類神經網路抽象基底。"""

    @abc.abstractmethod
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """模型前向傳播。

        Args:
            x1 (torch.Tensor): 第一視角圖片 Batch。
            x2 (torch.Tensor): 第二視角圖片 Batch。

        Returns:
            Tuple: (p1, p2, z1, z2) 分別對應預測與目標特徵，
                   以供損失函數計算。
        """
        pass

    def get_trainable_parameters_count(self) -> int:
        """計算可訓練的參數總數。

        Returns:
            int: 參數數量。
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def calculate_collapse_metric(self, z: torch.Tensor) -> float:
        """計算內部特徵在給定批次的維度標準差，監控是否發生坍塌。

        對於自監督學習，若能完美均勻展開，維度標準差會接近 1/sqrt(d)。
        若偏向 0，則象徵模型已經塌陷。

        Args:
            z (torch.Tensor): 網路的中繼隱藏特徵向量。

        Returns:
            float: 平均維度標準差。
        """
        z_norm = torch.nn.functional.normalize(z, dim=1)
        return z_norm.std(dim=0).mean().item()
