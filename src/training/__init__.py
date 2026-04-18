"""SimSiam 訓練管線支援模組。

提供計時、實驗紀錄、設定管理等訓練基礎設施，
供 main_training.py 一站式訓練入口使用。

Note:
    此模組採用延遲匯入 (lazy import)，避免在不需要全部功能時
    載入 pandas、torch 等重型依賴。使用者應直接從子模組匯入：
        from src.training.timer import PrecisionTimer
        from src.training.config import TrainingConfig
"""

__all__ = [
    "TrainingConfig",
    "ExperimentLogger",
    "PrecisionTimer",
    "TimerCollection",
]
