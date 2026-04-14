"""
TTA 實驗通用工具模組 (Ablation Utilities)
------------------------------------------------------------------------------
本模組提供 TTA (Test-Time Augmentation) 實驗所需的通用工具函數與裝飾器，包含：
1. 資源監控與時間測量裝飾器。
2. 錯誤重試與輸入驗證機制。
3. 模型載入與檢查點處理。
4. 結果快取與日誌記錄。

Design Patterns:
- Decorator Pattern: 用於橫切關切點 (Logging, Monitoring, Retry).
- Factory Pattern (Implicit): 透過 load_simsiam_model 建立模型實例.
"""

import functools
import time
import psutil
import logging
import gc
import json
import hashlib
from pathlib import Path
from typing import Callable, Any, Dict, Optional, Union, List

import torch
import torch.nn as nn
import numpy as np

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Decorators (裝飾器)
# -----------------------------------------------------------------------------

def monitor_resource(func: Callable) -> Callable:
    """
    監控函數執行的記憶體與 CPU 使用量的裝飾器。
    
    參數:
    func (Callable): 目標函數。
    
    返回:
    wrapper (Callable): 包裝後的函數。
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        cpu_before = process.cpu_percent()
        
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Function {func.__name__} failed: {e}")
            raise e
        finally:
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            cpu_after = process.cpu_percent()
            logger.debug(
                f"Resource usage for {func.__name__}: "
                f"Mem: {mem_before:.2f} -> {mem_after:.2f} MB, "
                f"CPU: {cpu_before}% -> {cpu_after}%"
            )
        return result
    return wrapper

def monitor_time(func: Callable) -> Callable:
    """
    計算函數執行時間的裝飾器。
    
    參數:
    func (Callable): 目標函數。
    
    返回:
    wrapper (Callable): 包裝後的函數。
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time
        logger.info(f"Function {func.__name__} took {duration:.4f} seconds")
        return result
    return wrapper

def validate_input(func: Callable) -> Callable:
    """
    驗證輸入參數非空的簡易裝飾器。
    
    參數:
    func (Callable): 目標函數。
    
    返回:
    wrapper (Callable): 包裝後的函數。
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not args and not kwargs:
            logger.warning(f"Function {func.__name__} called with no arguments.")
        for i, arg in enumerate(args):
            if arg is None:
                raise ValueError(f"Argument {i} in {func.__name__} is None")
        for k, v in kwargs.items():
            if v is None:
                raise ValueError(f"Argument {k} in {func.__name__} is None")
        return func(*args, **kwargs)
    return wrapper

def cache_result(cache_dir: str = ".cache") -> Callable:
    """
    基於檔案的結果快取裝飾器。
    
    參數:
    cache_dir (str): 快取目錄路徑。
    
    返回:
    decorator (Callable): 裝飾器。
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 建立快取鍵值
            # 注意：這裡簡單使用函數名和參數做 hash，複雜物件可能需要更嚴謹的序列化
            key_str = f"{func.__name__}_{args}_{kwargs}"
            key_hash = hashlib.md5(key_str.encode()).hexdigest()
            
            cache_path = Path(cache_dir) / f"{key_hash}.json"
            
            if cache_path.exists():
                logger.info(f"Loading cached result for {func.__name__} from {cache_path}")
                try:
                    with open(cache_path, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load cache: {e}, re-computing...")
            
            result = func(*args, **kwargs)
            
            # 嘗試儲存結果 (僅支援可 JSON 序列化的結果)
            try:
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                with open(cache_path, 'w') as f:
                    json.dump(result, f)
            except Exception as e:
                logger.debug(f"Result not JSON serializable, skipping cache: {e}")
            
            return result
        return wrapper
    return decorator

def retry(max_retries: int = 3, delay: float = 1.0) -> Callable:
    """
    失敗重試裝飾器。
    
    參數:
    max_retries (int): 最大重試次數。
    delay (float): 重試間隔秒數。
    
    返回:
    decorator (Callable): 裝飾器。
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}: {e}")
                    time.sleep(delay)
            logger.error(f"Function {func.__name__} failed after {max_retries} retries.")
            raise last_exception
        return wrapper
    return decorator

def log_debug(func: Callable) -> Callable:
    """
    紀錄 Debug 訊息的裝飾器 (Entry/Exit)。
    
    參數:
    func (Callable): 目標函數。
    
    返回:
    wrapper (Callable): 包裝後的函數。
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Entering {func.__name__}")
        try:
            return func(*args, **kwargs)
        finally:
            logger.debug(f"Exiting {func.__name__}")
    return wrapper

# -----------------------------------------------------------------------------
# Model Helpers (模型工具)
# -----------------------------------------------------------------------------

@monitor_resource
@retry(max_retries=2)
def load_simsiam_model(
    checkpoint_path: Union[str, Path],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    arch_config: Dict[str, Any] = None
) -> nn.Module:
    """
    載入 SimSiam 模型權重。
    
    參數:
    checkpoint_path (str | Path): 檢查點檔案路徑 (.pth)。
    device (str): 運算裝置 ('cuda' 或 'cpu')。
    arch_config (Dict): 模型架構設定，若為 None 則使用預設值。
    
    返回:
    model (nn.Module): 已載入權重並設定為 eval 模式的模型。
    """
    from src.model.simsiam2 import SimSiam # Lazy import to avoid circular dependency
    
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    # 預設架構參數
    config = {
        "backbone": "resnet18", # Default to ResNet18 as per project config
        "proj_dim": 2048,
        "pred_hidden": 512,
        "in_channels": 1 # 工程圖預設為灰階
    }
    if arch_config:
        config.update(arch_config)
        
    logger.info(f"Initializing SimSiam model with config: {config}")
    model = SimSiam(**config)
    
    logger.info(f"Loading weights from {path}")
    checkpoint = torch.load(path, map_location=device)
    
    # 處理可能的 key 差異 (例如有無 'module.' 前綴)
    state_dict = checkpoint.get('state_dict', checkpoint)
    # 如果是 DDP 訓練出來的，會有 module. 前綴，需移除
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    msg = model.load_state_dict(new_state_dict, strict=True)
    logger.info(f"Model loaded with message: {msg}")
    
    model.to(device)
    model.eval()
    return model

# -----------------------------------------------------------------------------
# Vector Math (向量運算)
# -----------------------------------------------------------------------------

def l2_normalize(v: torch.Tensor) -> torch.Tensor:
    """
    對向量進行 L2 正規化。
    
    參數:
    v (torch.Tensor): 輸入向量 shape [N, D] 或 [D]。
    
    返回:
    normalized_v (torch.Tensor): 正規化後的向量。
    """
    return torch.nn.functional.normalize(v, p=2, dim=-1)

def compute_centroid(vectors: torch.Tensor) -> torch.Tensor:
    """
    計算幾何中心 (Geometric Centroid) 並進行 L2 正規化。
    注意：在 SimSiam 的 Cosine 空間中，平均後的向量必須重新投影到單位球面上。
    
    參數:
    vectors (torch.Tensor): 一組向量 shape [N, D]。
    
    返回:
    centroid (torch.Tensor): 正規化後的中心向量 shape [1, D]。
    """
    # 1. 算術平均
    mean_vec = torch.mean(vectors, dim=0, keepdim=True)
    # 2. L2 正規化 (關鍵步驟)
    centroid = l2_normalize(mean_vec)
    return centroid

def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """
    計算兩組向量的 Cosine Similarity。
    假設輸入已做過 L2 Normalize，則只需計算 Dot Part。
    
    參數:
    v1 (torch.Tensor): shape [N, D] 或 [1, D]
    v2 (torch.Tensor): shape [M, D] 或 [1, D]
    
    返回:
    sim (torch.Tensor): 相似度矩陣。
    """
    # 確保已經正規化 (雖然呼叫端應該做，但再次確保較安全)
    v1 = l2_normalize(v1)
    v2 = l2_normalize(v2)
    return torch.mm(v1, v2.t())
