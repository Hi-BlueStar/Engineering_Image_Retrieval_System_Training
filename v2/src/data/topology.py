"""拓撲分析模組 (Topology Analysis Module)。

============================================================
對連通元件進行拓撲特徵分析，提取 Euler 數（孔洞數量），
用於消融實驗中拓撲感知的元件排序與選擇。

主要功能：
    1. 計算單一元件的孔洞數 (n_holes = β₁)
    2. 依拓撲複雜度排序元件（有孔洞的元件優先）
    3. 對完整影像進行拓撲分析（無 CC 前處理時使用）

拓撲複雜度直觀說明：
    - 孔洞 = 工程圖中的封閉迴路（孔、槽、圓弧）
    - 拓撲複雜度高的元件通常攜帶更多辨識資訊
============================================================
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


def count_holes(binary_component: np.ndarray) -> int:
    """計算二值化元件中的孔洞數 (β₁ Betti 數)。

    使用 RETR_CCOMP 輪廓樹：有父輪廓的輪廓即為孔洞。

    Args:
        binary_component: 二值化元件影像（白色=前景）。

    Returns:
        int: 孔洞數量；讀取失敗時回傳 0。
    """
    contours, hierarchy = cv2.findContours(
        binary_component, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None or len(contours) == 0:
        return 0
    return int((hierarchy[0, :, 3] != -1).sum())


def sort_crops_by_topology(
    crops: List[np.ndarray],
) -> List[np.ndarray]:
    """依拓撲複雜度（孔洞數 desc，面積 desc）排序裁切圖。

    孔洞數相同時，面積較大的元件排前。

    Args:
        crops: 連通元件裁切影像列表（白色=前景）。

    Returns:
        List[np.ndarray]: 排序後的裁切圖列表。
    """
    scored: List[Tuple[int, int, np.ndarray]] = []
    for crop in crops:
        n_holes = count_holes(crop)
        area = int(crop.sum())
        scored.append((n_holes, area, crop))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [c for _, _, c in scored]


def topology_guided_mask(gray: np.ndarray) -> np.ndarray:
    """對完整影像（未做 CC 分解）進行拓撲感知遮罩。

    用於「無 CC 但有拓撲分析」的消融實驗。
    策略：保留有孔洞的輪廓區域（以孔洞為中心的 bounding box），
    其餘區域填白（背景）。若無孔洞，回傳原始影像。

    Args:
        gray: 灰階影像 [H, W]。

    Returns:
        np.ndarray: 保留拓撲顯著區域的灰階影像 [H, W]。
    """
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None or len(contours) == 0:
        return gray

    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    for idx, hier in enumerate(hierarchy[0]):
        parent = hier[3]
        if parent == -1:
            # 外層輪廓：計算其孔洞數
            child_idx = hier[2]
            n_children = 0
            while child_idx != -1:
                n_children += 1
                child_idx = hierarchy[0, child_idx, 0]

            if n_children > 0:
                # 外層有孔洞 → 填入此輪廓的 bounding box
                x, y, cw, ch = cv2.boundingRect(contours[idx])
                mask[y:y + ch, x:x + cw] = 255

    if mask.sum() == 0:
        return gray

    # 保留遮罩區域，其餘填白
    result = np.full_like(gray, 255)
    result[mask > 0] = gray[mask > 0]
    return result
