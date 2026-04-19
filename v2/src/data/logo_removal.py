"""Gifu Logo 移除模組 (Logo Removal Module)。

============================================================
移除工程圖中的 Gifu（岐阜）品牌標誌，避免模型學習到與
圖紙內容無關的標誌特徵，干擾相似度量。

支援兩種移除策略（依優先順序）：
    1. 模板匹配 (Template Matching)：提供 logo_template_path 時使用。
    2. 區域遮罩 (Region Masking)：提供 logo_mask_region 時使用。
       格式：[x1_ratio, y1_ratio, x2_ratio, y2_ratio]（相對影像尺寸）。

若兩者皆未提供，嘗試自動偵測：在影像角落尋找高密度小區域。
============================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from src.logger import get_logger

logger = get_logger(__name__)


def remove_logo(
    image: np.ndarray,
    template_path: Optional[str] = None,
    mask_region: Optional[List[float]] = None,
    match_threshold: float = 0.75,
) -> np.ndarray:
    """移除影像中的 Gifu Logo。

    Args:
        image: 灰階影像 [H, W]，dtype=uint8。
        template_path: Logo 模板圖像路徑（灰階 PNG）。
        mask_region: 遮罩區域比例 [x1_r, y1_r, x2_r, y2_r]。
        match_threshold: 模板匹配接受閾值（0~1）。

    Returns:
        np.ndarray: 移除 Logo 後的影像（Logo 區域填白）。
    """
    image = image.copy()

    if template_path and Path(template_path).is_file():
        image = _remove_by_template(image, template_path, match_threshold)
    elif mask_region and len(mask_region) == 4:
        image = _remove_by_mask(image, mask_region)
    else:
        image = _remove_by_corner_detection(image)

    return image


# ============================================================
# 內部實作
# ============================================================


def _remove_by_template(
    image: np.ndarray,
    template_path: str,
    threshold: float,
) -> np.ndarray:
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        logger.warning("無法讀取 Logo 模板: %s", template_path)
        return image

    th, tw = template.shape
    if th > image.shape[0] or tw > image.shape[1]:
        logger.warning("Logo 模板尺寸大於影像，跳過移除")
        return image

    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)
    for pt in zip(*locations[::-1]):
        image[pt[1]: pt[1] + th, pt[0]: pt[0] + tw] = 255

    if len(locations[0]) > 0:
        logger.debug("模板匹配移除 %d 個 Logo 實例", len(locations[0]))

    return image


def _remove_by_mask(
    image: np.ndarray,
    mask_region: List[float],
) -> np.ndarray:
    H, W = image.shape[:2]
    x1 = max(0, int(mask_region[0] * W))
    y1 = max(0, int(mask_region[1] * H))
    x2 = min(W, int(mask_region[2] * W))
    y2 = min(H, int(mask_region[3] * H))
    image[y1:y2, x1:x2] = 255
    return image


def _remove_by_corner_detection(
    image: np.ndarray,
    corner_ratio: float = 0.15,
    density_threshold: float = 0.3,
) -> np.ndarray:
    """在四個角落尋找高密度小區域（疑似 Logo）並清除。

    策略：計算每個角落的前景像素密度；若高於閾值則視為 Logo。
    Gifu Logo 通常在圖框外的右下角或右上角。

    Args:
        image: 灰階影像。
        corner_ratio: 偵測角落大小（佔影像比例）。
        density_threshold: 前景密度閾值（超過即視為 Logo）。
    """
    H, W = image.shape[:2]
    ch = int(H * corner_ratio)
    cw = int(W * corner_ratio)

    _, binary = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    corners = [
        (0, 0, cw, ch),           # 左上
        (W - cw, 0, W, ch),       # 右上
        (0, H - ch, cw, H),       # 左下
        (W - cw, H - ch, W, H),   # 右下 ← Gifu Logo 常見位置
    ]

    for x1, y1, x2, y2 in corners:
        region = binary[y1:y2, x1:x2]
        density = region.mean() / 255.0
        if density > density_threshold:
            image[y1:y2, x1:x2] = 255

    return image
