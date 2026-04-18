"""隨機種子管理模組 (Seed Utilities)。

負責設定所有亂數生成器的 Seed，以保證實驗結果 100% 可重現 (Reproducible)。
"""
import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """設定全域隨機種子。
    
    涵蓋 Python 原生 random、NumPy 以及 PyTorch (CPU/GPU)。
    
    Args:
        seed: 隨機種子數值。
    """
    # 1. 內建 Python 隨機模組
    random.seed(seed)
    # 2. 作業系統環境變數 (針對 hash)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 3. NumPy 隨機模組
    np.random.seed(seed)
    # 4. PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多 GPU
    
    # 強制使用確定性演算法 (會稍微降低效能，但保證結果完全一致)
    # 若有效能考量可以把 benchmark 設為 True，deterministic 設為 False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
