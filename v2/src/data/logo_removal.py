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
    fill_value: int = 255,
) -> np.ndarray:
    """移除影像中的 Gifu Logo。

    Args:
        image: 灰階或二值化影像 [H, W]，dtype=uint8。
        template_path: Logo 模板圖像路徑（灰階 PNG）。
        mask_region: 遮罩區域比例 [x1_r, y1_r, x2_r, y2_r]。
        match_threshold: 模板匹配接受閾值（0~1）。
        fill_value: 填滿 Logo 區域的數值（255=白色, 0=黑色）。

    Returns:
        np.ndarray: 移除 Logo 後的影像。
    """
    image_c = image.copy()
    regions = find_logo_regions(
        image_c, template_path, mask_region, match_threshold
    )
    for x1, y1, x2, y2 in regions:
        image_c[y1:y2, x1:x2] = fill_value
    return image_c


def find_logo_regions(
    image: np.ndarray,
    template_path: Optional[str] = None,
    mask_region: Optional[List[float]] = None,
    match_threshold: float = 0.75,
) -> List[Tuple[int, int, int, int]]:
    """偵測影像中的 Logo 區域。

    Returns:
        List[Tuple[int, int, int, int]]: Logo 區域座標 (x1, y1, x2, y2) 列表。
    """
    if mask_region and len(mask_region) == 4:
        return [_get_mask_bbox(image, mask_region)]
    elif template_path and Path(template_path).is_file():
        return _get_template_bboxes(image, template_path, match_threshold)
    else:
        return _get_corner_bboxes(image)


# ============================================================
# 內部實作
# ============================================================


def _get_template_bboxes(
    image: np.ndarray,
    template_path: str,
    threshold: float,
) -> List[Tuple[int, int, int, int]]:
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        logger.warning("無法讀取 Logo 模板: %s", template_path)
        return []

    th, tw = template.shape
    if th > image.shape[0] or tw > image.shape[1]:
        logger.warning("Logo 模板尺寸大於影像，跳過偵測")
        return []

    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)
    bboxes = []
    for pt in zip(*locations[::-1]):
        bboxes.append((pt[0], pt[1], pt[0] + tw, pt[1] + th))

    if bboxes:
        logger.info("模板匹配發現 %d 個 Logo 區域", len(bboxes))
    return bboxes


def _get_mask_bbox(
    image: np.ndarray,
    mask_region: List[float],
) -> Tuple[int, int, int, int]:
    H, W = image.shape[:2]
    x1 = max(0, int(mask_region[0] * W))
    y1 = max(0, int(mask_region[1] * H))
    x2 = min(W, int(mask_region[2] * W))
    y2 = min(H, int(mask_region[3] * H))
    return (x1, y1, x2, y2)


def _get_corner_bboxes(
    image: np.ndarray,
    corner_ratio: float = 0.10,
    density_threshold: float = 0.3,
) -> List[Tuple[int, int, int, int]]:
    H, W = image.shape[:2]
    ch = int(H * corner_ratio)
    cw = int(W * corner_ratio)

    # 判斷輸入是否為二值化圖像，若否則進行臨時二值化
    if len(np.unique(image)) > 2:
        _, binary = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    else:
        # 假設 255 是前景，0 是背景 (Otsu Inverted 慣例)
        binary = image

    corners = [
        (0, 0, cw, ch),           # 左上
        (W - cw, 0, W, ch),       # 右上
        (0, H - ch, cw, H),       # 左下
        (W - cw, H - ch, W, H),   # 右下
    ]

    logo_bboxes = []
    for x1, y1, x2, y2 in corners:
        region = binary[y1:y2, x1:x2]
        density = region.mean() / 255.0
        if density > density_threshold:
            logo_bboxes.append((x1, y1, x2, y2))

    return logo_bboxes
