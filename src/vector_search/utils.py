"""
Vector Search 共用工具函式庫。
包含 ROI 切割、前處理等共用邏輯。
"""
import logging
import numpy as np
from typing import List, Tuple, Dict, Any
from pathlib import Path

# 引入影像前處理工具
# 允許從專案根目錄或相對路徑引用
try:
    from src.image_preprocessing3 import auto_binarize, analyze_components, select_large_small
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.image_preprocessing3 import auto_binarize, analyze_components, select_large_small

logger = logging.getLogger(__name__)

def extract_rois_from_image(img: np.ndarray, top_n: int = 5) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
    """
    將影像二值化並分割為多個元件 (ROI)。
    此函式由 Indexer 與 Engine 共用，確保特徵提取的來源一致。

    Args:
        img: 原始影像 (BGR)。
        top_n: 保留前 N 大的元件。

    Returns:
        List[tuple]: [(cropped_img, metadata_dict), ...]
        metadata_dict 包含 'component_index', 'bbox', 'area'。
    """
    if img is None:
        return []

    # 1. 自動二值化
    # auto_binarize 回傳 (bw01, bg_mode)
    bw01, bg_mode = auto_binarize(img)
    
    # 2. 連通元件分析
    comps = analyze_components(bw01)
    
    # 3. 篩選前 N 大元件
    # large_comps: 主要 ROI
    large_comps, _ = select_large_small(comps, top_n=top_n, remove_largest=False)
    
    results = []
    for i, comp in enumerate(large_comps):
        x, y, w, h = comp.bbox
        
        # 從原圖裁切 ROI
        # 增加一點 Padding 避免太貼 (與 Indexer 邏輯一致)
        pad = 2
        h_img, w_img = img.shape[:2]
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_img, x + w + pad)
        y2 = min(h_img, y + h + pad)
        
        roi_img = img[y1:y2, x1:x2]
        
        # 若裁切失敗 (例如面積為0)，跳過
        if roi_img.size == 0:
            continue
            
        # 元件資訊
        info = {
            "component_index": i,  # 0是最大
            "bbox": [x, y, w, h],
            "area": comp.area
        }
        results.append((roi_img, info))
        
    return results
