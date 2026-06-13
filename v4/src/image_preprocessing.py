"""工程圖連通元分析與二值化模組 (Image Preprocessing Utilities)。

============================================================
從二值化影像中尋找大小組件，並將落在大元件內部的微小細節（如線條、標註）
歸併回主體元件中，保留高保真度的工件輪廓。

此模組直接實作於 v4 內部，以確保 v4 資料管線完全獨立自主，不依賴外部檔案。
============================================================
"""

from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class Component:
    """連通元件結構體"""
    label: int
    area: int
    bbox: tuple[int, int, int, int]  # x, y, w, h
    roi_mask: np.ndarray | None = None  # 裁切後的 ROI 遮罩 (uint8 0/1)


def auto_binarize(img_bgr: np.ndarray, bin_thresh: int = 0) -> tuple[np.ndarray, str]:
    """自適應二值化影像 (前景=1, 背景=0)

    自動計算黑/白底版本，並選取前景像素數較少的一種作為前景（線條通常佔比低）。
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if bin_thresh and 0 < bin_thresh < 255:
        _, bin_white = cv2.threshold(gray, bin_thresh, 255, cv2.THRESH_BINARY)
    else:
        _, bin_white = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cnt_w = int(np.count_nonzero(bin_white))
    h, w = gray.shape
    total = h * w
    cnt_b = total - cnt_w

    w_valid = 0 < cnt_w < total * 0.95
    b_valid = 0 < cnt_b < total * 0.95

    if w_valid and b_valid:
        if cnt_w < cnt_b:
            bw = (bin_white > 0).astype(np.uint8)
            bg_mode = "black"
        else:
            bw = (bin_white == 0).astype(np.uint8)
            bg_mode = "white"
    elif w_valid:
        bw = (bin_white > 0).astype(np.uint8)
        bg_mode = "black"
    else:
        bw = (bin_white == 0).astype(np.uint8)
        bg_mode = "white"

    return bw, bg_mode


def analyze_components(bw01: np.ndarray, return_labels: bool = False) -> list[Component] | tuple[list[Component], np.ndarray]:
    """連通元件分析，回傳按外接矩形面積 (w*h) 遞減排序的組件列表"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bw01, connectivity=8
    )

    comps: list[Component] = []

    for lbl in range(1, num_labels):
        x, y, w, h, _ = stats[lbl]
        bbox_area = w * h

        if not return_labels:
            roi_labels = labels[y : y + h, x : x + w]
            roi_mask = (roi_labels == lbl).astype(np.uint8)
        else:
            roi_mask = None

        comps.append(Component(lbl, bbox_area, (x, y, w, h), roi_mask))

    comps.sort(key=lambda c: c.area, reverse=True)
    
    if return_labels:
        return comps, labels
    return comps


def select_large_small(
    comps: list[Component],
    top_n: int,
    remove_largest: bool = True,
    max_bbox_ratio: float | None = None,
    img_shape: tuple[int, int] | None = None
) -> tuple[list[Component], list[Component]]:
    """篩選出大組件與小組件 (可選擇剔除大於整張圖一定比例的外接矩形元件，通常為圖框)"""
    ordered = comps.copy()
    
    # 優先使用更精確的 max_bbox_ratio 來過濾圖紙框架
    if max_bbox_ratio is not None and img_shape is not None and ordered:
        total_area = img_shape[0] * img_shape[1]
        filtered_ordered = []
        for c in ordered:
            ratio = c.area / total_area
            if ratio > max_bbox_ratio:
                # 排除符合框架特徵的超大元件
                continue
            filtered_ordered.append(c)
        ordered = filtered_ordered
    else:
        # 退化回原本的 remove_largest 機制 (盲目剔除排序第一的元件)
        if remove_largest and ordered:
            ordered = ordered[1:]
            
    large = ordered[:top_n]
    small = ordered[top_n:]
    return large, small


def filled_region_from_component(comp: Component) -> np.ndarray:
    """尋找大元件外輪廓並將其填滿，回傳 ROI 大小的填滿遮罩"""
    mask255 = (comp.roi_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filled_roi = np.zeros_like(comp.roi_mask)
    if contours:
        cv2.drawContours(filled_roi, contours, -1, color=1, thickness=-1)

    return filled_roi


def assign_small_to_large(
    large: list[Component], small: list[Component], labels: np.ndarray | None = None
) -> dict[int, list[Component]]:
    """判斷小元件是否『完全落在』某一個大元件的外輪廓填滿區域內"""
    if not large or not small:
        return {L.label: [] for L in large}

    filled_map = {L.label: filled_region_from_component(L) for L in large}
    assignment: dict[int, list[Component]] = {L.label: [] for L in large}

    for s in small:
        sx, sy, sw, sh = s.bbox
        
        # BBox 快速篩選包含關係
        passed_bbox = False
        for L in large:
            Lx, Ly, Lw, Lh = L.bbox
            if Lx <= sx and Ly <= sy and (Lx + Lw) >= (sx + sw) and (Ly + Lh) >= (sy + sh):
                passed_bbox = True
                break
        
        if not passed_bbox:
            continue

        if s.roi_mask is None and labels is not None:
            roi_labels = labels[sy : sy + sh, sx : sx + sw]
            s.roi_mask = (roi_labels == s.label).astype(np.uint8)

        if s.roi_mask is None:
            continue

        s_ys, s_xs = np.where(s.roi_mask > 0)
        if len(s_ys) == 0:
            continue

        best_label = None
        best_cover = -1

        s_abs_y = s_ys + sy
        s_abs_x = s_xs + sx

        for L in large:
            Lx, Ly, Lw, Lh = L.bbox

            if not (Lx <= sx and Ly <= sy and (Lx + Lw) >= (sx + sw) and (Ly + Lh) >= (sy + sh)):
                continue

            rel_y = s_abs_y - Ly
            rel_x = s_abs_x - Lx

            # 雙重安全邊界檢查
            valid_idx = (rel_y >= 0) & (rel_y < Lh) & (rel_x >= 0) & (rel_x < Lw)
            if not np.all(valid_idx):
                continue

            f_roi = filled_map[L.label]
            cover_vals = f_roi[rel_y, rel_x]

            if np.all(cover_vals == 1):
                cover = int(np.sum(cover_vals))
                if cover > best_cover:
                    best_cover = cover
                    best_label = L.label

        if best_label is not None:
            assignment[best_label].append(s)

    return assignment


def merge_small_into_large(
    large: list[Component], assignment: dict[int, list[Component]]
) -> dict[int, np.ndarray]:
    """將指派給大元件的小元件遮罩併入大元件 ROI 遮罩"""
    merged: dict[int, np.ndarray] = {}
    for L in large:
        Lx, Ly, Lw, Lh = L.bbox
        merged_mask = L.roi_mask.copy()

        for s in assignment.get(L.label, []):
            sx, sy, sw, sh = s.bbox
            offset_x = sx - Lx
            offset_y = sy - Ly

            roi_slice = merged_mask[offset_y : offset_y + sh, offset_x : offset_x + sw]
            merged_mask[offset_y : offset_y + sh, offset_x : offset_x + sw] = (
                np.maximum(roi_slice, s.roi_mask)
            )

        merged[L.label] = merged_mask
    return merged
