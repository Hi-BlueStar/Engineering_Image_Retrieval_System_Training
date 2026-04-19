"""影像前處理模組 (Image Preprocessing Module)。

============================================================
針對工程圖 PNG 進行連通元件分析，提取關鍵元件並生成隨機排列變體。

流程：
    1. 載入灰階影像，Otsu 二值化（反轉：元件為白色）
    2. 連通元件分析，依面積排序
    3. 可選移除最大元件（通常為圖框）
    4. 保留 top_n 個最大元件並裁切（含邊距）
    5. 將裁切結果隨機排列至空白畫布，重複 random_count 次
    6. 儲存為 PNG（白底黑圖，與原圖格式相同）
============================================================
"""

from __future__ import annotations

import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.logger import get_logger

logger = get_logger(__name__)

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class PreprocessConfig:
    """前處理管線設定。

    Attributes:
        input_dir: 輸入影像目錄（PDF 轉換輸出）。
        output_root: 前處理結果根目錄。
        max_workers: 並行程序數。
        top_n: 保留的最大元件數。
        remove_largest: 是否移除面積最大的元件（圖框）。
        padding: 裁切邊距（像素）。
        max_attempts: 隨機放置嘗試次數上限。
        random_count: 每張影像生成的排列變體數。
    """

    input_dir: str
    output_root: str
    max_workers: int = 12
    top_n: int = 5
    remove_largest: bool = True
    padding: int = 2
    max_attempts: int = 400
    random_count: int = 10


def preprocess_images(cfg: PreprocessConfig, skip: bool = False) -> None:
    """批次前處理所有影像。

    Args:
        cfg: 前處理設定。
        skip: ``True`` 強制跳過此步驟。
    """
    if skip:
        logger.info("跳過影像前處理 (skip=True)")
        return

    src_dir = Path(cfg.input_dir)
    dst_root = Path(cfg.output_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    images = [
        p for p in src_dir.rglob("*")
        if p.suffix.lower() in _IMG_EXTS
    ]

    if not images:
        logger.warning("前處理輸入目錄無影像: %s", src_dir)
        return

    logger.info(
        "開始影像前處理: %d 張，top_n=%d，remove_largest=%s，random_count=%d",
        len(images),
        cfg.top_n,
        cfg.remove_largest,
        cfg.random_count,
    )

    cfg_dict = {
        "top_n": cfg.top_n,
        "remove_largest": cfg.remove_largest,
        "padding": cfg.padding,
        "max_attempts": cfg.max_attempts,
        "random_count": cfg.random_count,
    }
    args_list = [(str(p), str(dst_root), cfg_dict) for p in images]

    success = 0
    with ProcessPoolExecutor(max_workers=cfg.max_workers) as pool:
        futures = {pool.submit(_process_one, *a): a[0] for a in args_list}
        for fut in as_completed(futures):
            img_path = futures[fut]
            try:
                fut.result()
                success += 1
            except Exception as exc:
                logger.error("前處理失敗: %s — %s", Path(img_path).name, exc)

    logger.info("影像前處理完成: %d/%d 成功", success, len(images))


# ============================================================
# 內部實作（可 pickle，供 ProcessPoolExecutor 使用）
# ============================================================


def _process_one(img_path: str, dst_root: str, cfg: dict) -> None:
    path = Path(img_path)
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"無法讀取影像: {img_path}")

    binary = _binarize(gray)
    crops = _extract_crops(binary, cfg["top_n"], cfg["remove_largest"], cfg["padding"])
    if not crops:
        return

    h, w = gray.shape
    out_dir = Path(dst_root) / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(hash(img_path))
    for i in range(cfg["random_count"]):
        canvas = _arrange(crops, h, w, cfg["max_attempts"], rng)
        # 反轉回白底黑圖
        result = 255 - canvas
        cv2.imwrite(str(out_dir / f"arr_{i:03d}.png"), result)


def _binarize(gray: np.ndarray) -> np.ndarray:
    """Otsu 二值化；輸出白色=元件、黑色=背景。"""
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binary


def _extract_crops(
    binary: np.ndarray,
    top_n: int,
    remove_largest: bool,
    padding: int,
) -> List[np.ndarray]:
    """提取連通元件裁切圖（白色=元件像素）。"""
    h, w = binary.shape
    num_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    if num_labels <= 1:
        return []

    # 跳過背景 label=0，依面積排序
    components = [
        (i, int(stats[i, cv2.CC_STAT_AREA]))
        for i in range(1, num_labels)
    ]
    components.sort(key=lambda x: x[1], reverse=True)

    if remove_largest and len(components) > 1:
        components = components[1:]

    components = components[:top_n]

    crops: List[np.ndarray] = []
    for label_idx, _ in components:
        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        cw = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        ch = int(stats[label_idx, cv2.CC_STAT_HEIGHT])

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + cw + padding)
        y2 = min(h, y + ch + padding)

        if x2 - x1 < 2 or y2 - y1 < 2:
            continue

        crops.append(binary[y1:y2, x1:x2].copy())

    return crops


def _arrange(
    crops: List[np.ndarray],
    canvas_h: int,
    canvas_w: int,
    max_attempts: int,
    rng: random.Random,
) -> np.ndarray:
    """將裁切圖隨機排列到黑色畫布（白色=元件）。"""
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    placed: List[Tuple[int, int, int, int]] = []  # (x1, y1, x2, y2)

    order = list(range(len(crops)))
    rng.shuffle(order)

    for idx in order:
        crop = crops[idx]
        ch, cw = crop.shape
        if ch > canvas_h or cw > canvas_w:
            continue

        x, y = _find_position(canvas_h, canvas_w, ch, cw, placed, max_attempts, rng)
        canvas[y:y + ch, x:x + cw] = np.maximum(canvas[y:y + ch, x:x + cw], crop)
        placed.append((x, y, x + cw, y + ch))

    return canvas


def _find_position(
    canvas_h: int,
    canvas_w: int,
    crop_h: int,
    crop_w: int,
    placed: List[Tuple[int, int, int, int]],
    max_attempts: int,
    rng: random.Random,
) -> Tuple[int, int]:
    """找到不重疊的放置位置；失敗則隨機回傳。"""
    max_y = canvas_h - crop_h
    max_x = canvas_w - crop_w

    for _ in range(max_attempts):
        y = rng.randint(0, max_y)
        x = rng.randint(0, max_x)
        box = (x, y, x + crop_w, y + crop_h)

        if not any(_overlaps(box, p) for p in placed):
            return x, y

    return rng.randint(0, max_y), rng.randint(0, max_x)


def _overlaps(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
) -> bool:
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])
