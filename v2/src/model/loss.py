"""損失函數模組 (Loss Functions Module)。

============================================================
實作 SimSiam 的損失函數與坍塌監控指標。

所有函式均為純向量化操作，無 Python 層級迴圈，
確保在 GPU 上高效執行。

包含：
    - ``simsiam_loss``：對稱負餘弦相似度損失。
    - ``negative_cosine_similarity``：單方向餘弦損失。
    - ``calculate_collapse_std``：維度坍塌監控指標。
============================================================
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def negative_cosine_similarity(
    p: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor:
    """負餘弦相似度損失。

    .. math::

        \\mathcal{L} = - \\frac{p}{\\|p\\|_2} \\cdot \\frac{z}{\\|z\\|_2}

    Args:
        p: Predictor 輸出 ``[B, D]``。
        z: Projector 目標投影 ``[B, D]``（已 stop-gradient）。

    Returns:
        torch.Tensor: 標量損失（batch 平均值）。
    """
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return -(p * z).sum(dim=1).mean()


def simsiam_loss(
    p1: torch.Tensor,
    p2: torch.Tensor,
    z1: torch.Tensor,
    z2: torch.Tensor,
) -> torch.Tensor:
    """SimSiam 對稱損失函數。

    .. math::

        \\mathcal{L} = \\frac{1}{2} \\left[
            D(p_1, \\text{sg}(z_2)) + D(p_2, \\text{sg}(z_1))
        \\right]

    其中 :math:`D` 為負餘弦相似度，:math:`\\text{sg}` 為
    stop-gradient（已在模型 forward 中透過 ``.detach()`` 實現）。

    Args:
        p1: 視角 1 的 Predictor 輸出 ``[B, D]``。
        p2: 視角 2 的 Predictor 輸出 ``[B, D]``。
        z1: 視角 1 的 Projector 投影（已 detach）``[B, D]``。
        z2: 視角 2 的 Projector 投影（已 detach）``[B, D]``。

    Returns:
        torch.Tensor: 標量對稱損失。
    """
    return 0.5 * (
        negative_cosine_similarity(p1, z2)
        + negative_cosine_similarity(p2, z1)
    )


def calculate_collapse_std(z: torch.Tensor) -> float:
    """計算 L2 正規化後特徵的維度標準差平均值。

    用於監控維度坍塌 (Dimensional Collapse)：
    - 理想值趨近 ``1 / sqrt(d)``（均勻分佈）。
    - 若趨近 0，表示模型僅在極少數維度產生變化。

    計算方式：沿 batch 維度計算每個特徵維度的 std，
    再對所有特徵維度取平均。

    Args:
        z: 特徵向量 ``[B, D]``。

    Returns:
        float: 維度標準差平均值。
    """
    z_norm = F.normalize(z, dim=1)
    return z_norm.std(dim=0).mean().item()
