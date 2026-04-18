"""
系統資源監控模組 (System Monitor)。

用於檢測 RAM, VRAM 使用量，防止 OOM (Out Of Memory) 發生的防禦性設計。
"""

import psutil
import torch

class SystemMonitor:
    """提供系統資源檢測與警告機制。"""
    
    @staticmethod
    def get_ram_usage_mb() -> float:
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

    @staticmethod
    def get_vram_usage_mb(device_id: int = 0) -> float:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(device_id) / (1024 * 1024)
        return 0.0
        
    @staticmethod
    def get_vram_reserved_mb(device_id: int = 0) -> float:
        if torch.cuda.is_available():
             return torch.cuda.memory_reserved(device_id) / (1024 * 1024)
        return 0.0
