"""拓撲分析模組 (Topology Analysis Module)。

============================================================
對連通元件進行拓撲特徵分析，提取 Euler 數（孔洞數量），
用於消融實驗中拓撲感知的元件排序與選擇。

主要功能：
    1. 計算單一元件的孔洞數 (n_holes = β₁)
    2. 依拓撲複雜度排序元件（有孔洞的元件優先）
    3. 對完整影像進行拓撲分析（無 CC 前處理時使用）

GPU 加速支援（v2.1）：
    - 使用 Kornia（已在 requirements.txt）進行 GPU 形態學運算
    - 自動偵測 CUDA 可用性，不可用時回退至 CPU
    - count_holes_euler() 提供 GPU-friendly 的孔洞計算

拓撲複雜度直觀說明：
    - 孔洞 = 工程圖中的封閉迴路（孔、槽、圓弧）
    - 拓撲複雜度高的元件通常攜帶更多辨識資訊
============================================================
"""

from __future__ import annotations

import logging
import time
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ============================================================
# GPU 加速基礎設施
# ============================================================

_GPU_AVAILABLE = False
_TORCH_DEVICE = None
_kornia_morph = None  # lazy import reference

# 最小像素數門檻：低於此值使用 CPU（避免 GPU 傳輸 overhead）
_GPU_MIN_PIXELS = 10_000  # ~100×100


def _init_gpu() -> None:
    """延遲初始化 GPU 支援（首次呼叫時觸發）。"""
    global _GPU_AVAILABLE, _TORCH_DEVICE, _kornia_morph
    if _TORCH_DEVICE is not None:
        return  # 已初始化
    try:
        import torch
        import kornia.morphology as km

        _kornia_morph = km
        if torch.cuda.is_available():
            _GPU_AVAILABLE = True
            _TORCH_DEVICE = torch.device("cuda")
            logger.debug("topology: GPU 加速已啟用 (device=%s)", torch.cuda.get_device_name())
        else:
            _TORCH_DEVICE = torch.device("cpu")
            logger.debug("topology: CUDA 不可用，使用 CPU")
    except ImportError:
        _TORCH_DEVICE = "cpu"  # sentinel: 標記已嘗試初始化
        logger.debug("topology: torch/kornia 未安裝，使用 CPU")


# ============================================================
# 孔洞計算
# ============================================================


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


def count_holes_euler(binary_component: np.ndarray) -> int:
    """使用 Euler 數像素公式計算孔洞數（GPU-friendly，無需 findContours）。

    使用 2×2 quad-tree 方法計算 8-connectivity Euler number：
        χ₈ = (n_Q1 - n_Q3 - 2·n_QD) / 4
        孔洞數 H = C - χ₈

    假設輸入為單一連通元件（C=1），此假設在
    topology_preserving_pruning 的使用情境下成立。

    Args:
        binary_component: 二值化元件影像（白色=前景）。

    Returns:
        int: 孔洞數量。
    """
    if binary_component.size == 0 or binary_component.sum() == 0:
        return 0

    b = (binary_component > 0).astype(np.int32)

    # 取 2×2 鄰域的四個角
    q_tl = b[:-1, :-1]
    q_tr = b[:-1, 1:]
    q_bl = b[1:, :-1]
    q_br = b[1:, 1:]

    s = q_tl + q_tr + q_bl + q_br

    n_q1 = int(np.sum(s == 1))
    n_q3 = int(np.sum(s == 3))
    # QD: 恰好 2 個前景像素且為對角位置
    n_qd = int(np.sum(
        (s == 2)
        & (
            ((q_tl == 1) & (q_br == 1) & (q_tr == 0) & (q_bl == 0))
            | ((q_tr == 1) & (q_bl == 1) & (q_tl == 0) & (q_br == 0))
        )
    ))

    euler_8 = (n_q1 - n_q3 - 2 * n_qd) // 4
    holes = 1 - euler_8  # 假設單一連通元件 (C=1)
    return max(0, holes)


def analyze_topology(binary_component: np.ndarray) -> dict:
    """分析元件拓撲特徵。

    Returns:
        dict: {
            "n_holes": int,
            "is_complex": bool  # 是否具有孔洞
        }
    """
    n_holes = count_holes(binary_component)
    return {
        "n_holes": n_holes,
        "is_complex": n_holes > 0,
    }


# ============================================================
# 形態學重建（測地線膨脹）
# ============================================================


def _morphological_reconstruction(marker: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """形態學重建（測地線膨脹重建）— 自動分派 GPU/CPU。

    以 marker 為起始，在 mask 的約束下進行迭代膨脹，
    直到結果不再變化為止。這是 Pruning with Reconstruction
    方法的核心步驟，可恢復被過度剪枝移除的結構。

    原理：
        - 每次迭代對 marker 進行 3×3 膨脹
        - 膨脹結果與 mask 取 min（逐像素），確保不超出 mask 範圍
        - 重複直到收斂（結果不再改變）

    效果：
        - 被開運算移除的「孤立雜訊」不會被恢復（因為不與 marker 相連）
        - 被開運算「過度移除」的結構（與存活部分相連者）會被恢復

    Args:
        marker: 標記影像（必須逐像素 <= mask），通常為開運算後的結果。
        mask: 約束影像（開運算前的影像），重建不會超出此範圍。

    Returns:
        重建後的影像。
    """
    _init_gpu()

    n_pixels = marker.shape[0] * marker.shape[1]
    if _GPU_AVAILABLE and n_pixels >= _GPU_MIN_PIXELS:
        try:
            return _morphological_reconstruction_gpu(marker, mask)
        except Exception as exc:
            logger.debug("GPU 重建失敗，回退 CPU: %s", exc)

    return _morphological_reconstruction_cpu(marker, mask)


def _morphological_reconstruction_cpu(marker: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """CPU 版形態學重建（原始實作）。"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    result = marker.copy()
    while True:
        dilated = cv2.dilate(result, kernel)
        result_new = cv2.min(dilated, mask)
        if np.array_equal(result_new, result):
            break
        result = result_new
    return result


def _morphological_reconstruction_gpu(marker: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """GPU 加速形態學重建（Kornia 實作）。

    使用 kornia.morphology.dilation 在 GPU 上執行測地線膨脹，
    並以週期性收斂檢查減少 GPU-CPU 同步次數。

    Note:
        二值影像 (0/255 uint8) 轉換為 float32 (0.0/1.0) 以供 Kornia 使用，
        最終轉回 uint8。使用 > 0.5 閾值避免浮點精度問題。
    """
    import torch

    device = _TORCH_DEVICE

    # NumPy uint8 → Torch float32 [B=1, C=1, H, W]，歸一化到 {0.0, 1.0}
    marker_t = (
        torch.from_numpy(marker.astype(np.float32))
        .unsqueeze(0).unsqueeze(0).to(device) / 255.0
    )
    mask_t = (
        torch.from_numpy(mask.astype(np.float32))
        .unsqueeze(0).unsqueeze(0).to(device) / 255.0
    )
    kernel = torch.ones(3, 3, device=device)

    result = marker_t
    max_iter = max(marker.shape)
    # 減少收斂檢查頻率以降低 GPU 同步開銷
    check_interval = max(5, max_iter // 20)

    for i in range(max_iter):
        dilated = _kornia_morph.dilation(result, kernel)
        result_new = torch.min(dilated, mask_t)

        # 週期性收斂檢查（避免每次迭代都同步 GPU）
        if (i + 1) % check_interval == 0 or i == max_iter - 1:
            if torch.equal(result_new, result):
                logger.debug("GPU 重建收斂: %d 次迭代", i + 1)
                break
        result = result_new

    # Torch float32 → NumPy uint8（用 > 0.5 閾值避免浮點精度問題）
    return ((result.squeeze().cpu().numpy() > 0.5) * 255).astype(np.uint8)


# ============================================================
# 拓撲保留剪枝
# ============================================================


def topology_preserving_pruning(
    binary_component: np.ndarray, max_iters: int = 3, start_ksize: int = 2
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """結構級遞進式剪枝（帶重建）：執行微小清理，但確保不改變拓撲（孔洞數）。

    使用 Pruning with Reconstruction（帶重建的修剪）方法：
      1. 對元件執行形態學開運算（剪枝 — 去除邊緣雜訊與微小凸起）
      2. 以開運算結果為 marker，以當前影像為 mask，
         進行測地線膨脹重建（Geodesic Dilation Reconstruction）
      3. 重建步驟可恢復被開運算過度移除的結構，
         只保留真正的雜訊移除效果
      4. 驗證拓撲不變性（孔洞數 β₁ 不變）

    相較於無重建的版本，此方法的優勢在於：
      - 開運算可能過度移除結構（尤其在 kernel 較大時）
      - 重建步驟將「與存活結構相連」的被移除部分恢復
      - 最終結果僅去除真正的孤立雜訊，保留主要結構完整性

    GPU 加速說明：
      - 形態學重建自動分派至 GPU（若可用且影像夠大）
      - 拓撲驗證使用 count_holes_euler()（純像素運算，無 findContours）

    Args:
        binary_component: 二值化元件影像。
        max_iters: 最大剪枝迭代次數。
        start_ksize: 起始結構元素大小。

    Returns:
        (最終結果, 歷史紀錄列表)
    """
    t_start = time.perf_counter()
    original_holes = count_holes_euler(binary_component)
    current = binary_component.copy()
    history = []

    for i in range(max_iters):
        k_size = start_ksize + i
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))

        # Step 1: 執行形態學開運算 (剪枝 — 去除邊緣雜訊)
        pruned = cv2.morphologyEx(current, cv2.MORPH_OPEN, kernel)

        # Step 2: Pruning with Reconstruction（帶重建的修剪）
        # 以開運算結果為 marker，以當前影像為 mask，
        # 進行測地線膨脹重建。重建可恢復被過度移除的結構，
        # 只保留真正的雜訊移除效果。
        if pruned.sum() > 0:
            reconstructed = _morphological_reconstruction(pruned, current)
        else:
            reconstructed = pruned

        # Step 3: 驗證拓撲不變性（不允許孔洞消失或形狀破碎成多個）
        # 同時要確保面積不為 0 (除非原本就是 0)
        if reconstructed.sum() > 0 or current.sum() == 0:
            if count_holes_euler(reconstructed) == original_holes:
                current = reconstructed
                history.append(current.copy())
            else:
                break  # 拓撲改變，終止迭代並保留上一次結果
        else:
            break  # 面積歸零，終止迭代

    elapsed = time.perf_counter() - t_start
    logger.debug(
        "topology_preserving_pruning: %.4fs | shape=%s | iters=%d | gpu=%s",
        elapsed, binary_component.shape, len(history),
        _GPU_AVAILABLE and binary_component.size >= _GPU_MIN_PIXELS,
    )
    return current, history


# ============================================================
# 排序與遮罩
# ============================================================


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
    # 確保輸入為灰階（處理 preview.py 傳入 RGB 視覺化圖的情況）
    if len(gray.shape) == 3:
        gray_input = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    else:
        gray_input = gray

    _, binary = cv2.threshold(
        gray_input, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
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
