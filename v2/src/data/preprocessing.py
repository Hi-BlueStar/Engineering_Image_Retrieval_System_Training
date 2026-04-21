"""影像前處理模組 (Image Preprocessing Module)。

============================================================
針對工程圖 PNG 進行前處理，支援消融研究的四個可開關模組：

    1. remove_gifu_logo: 移除 Gifu 標誌（Otsu 反轉前套用）
    2. use_connected_components: 連通元件分析 + 隨機排列
    3. use_topology_analysis:
        - 若有 CC：依拓撲複雜度排序元件（孔洞數 desc）
        - 若無 CC：對整張影像進行拓撲感知遮罩

輸出結構（保留類別子目錄，供分層抽樣使用）::

    <output_root>/<class_name>/<image_stem>/arr_000.png ... arr_N.png

若輸入為無類別平坦目錄，<class_name> 省略，直接在
<output_root>/<image_stem>/ 下輸出。
============================================================
"""

from __future__ import annotations

import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from src.logger import get_logger

logger = get_logger(__name__)

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_CHUNK_SIZE = 64


@dataclass
class PreprocessConfig:
    """前處理管線設定。

    Attributes:
        input_dir: 輸入影像目錄（支援類別子目錄結構）。
        output_root: 前處理結果根目錄。
        max_workers: 並行程序數。
        top_n: 保留的最大元件數。
        remove_largest: 是否移除面積最大的元件（通常為圖框）。
        padding: 裁切邊距（像素）。
        max_attempts: 隨機放置嘗試次數上限。
        random_count: 每張影像生成的排列變體數。
        use_connected_components: 是否啟用連通元件分析。
        use_topology_analysis: 是否啟用拓撲感知排序/遮罩。
        remove_gifu_logo: 是否移除 Gifu Logo。
        logo_template_path: Logo 模板路徑（可選）。
        logo_mask_region: Logo 遮罩區域比例（可選）。
    """

    input_dir: str
    output_root: str
    max_workers: int = 12
    top_n: int = 5
    remove_largest: bool = True
    padding: int = 2
    max_attempts: int = 400
    random_count: int = 10
    use_connected_components: bool = True
    use_topology_analysis: bool = True
    remove_gifu_logo: bool = True
    logo_template_path: Optional[str] = None
    logo_mask_region: Optional[List[float]] = None


def preprocess_images(cfg: PreprocessConfig, skip: bool = False) -> None:
    """批次前處理所有影像，保留類別子目錄結構。

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

    # 支援類別子目錄與平坦目錄
    images = sorted(
        Path(dp) / fn
        for dp, _, fns in os.walk(src_dir)
        for fn in fns
        if Path(fn).suffix.lower() in _IMG_EXTS
    )

    if not images:
        logger.warning("前處理輸入目錄無影像: %s", src_dir)
        return

    logger.info(
        "開始影像前處理: %d 張 | CC=%s | Topology=%s | LogoRemoval=%s | variants=%d",
        len(images),
        cfg.use_connected_components,
        cfg.use_topology_analysis,
        cfg.remove_gifu_logo,
        cfg.random_count,
    )

    cfg_dict = {
        "top_n": cfg.top_n,
        "remove_largest": cfg.remove_largest,
        "padding": cfg.padding,
        "max_attempts": cfg.max_attempts,
        "random_count": cfg.random_count,
        "use_connected_components": cfg.use_connected_components,
        "use_topology_analysis": cfg.use_topology_analysis,
        "remove_gifu_logo": cfg.remove_gifu_logo,
        "logo_template_path": cfg.logo_template_path,
        "logo_mask_region": cfg.logo_mask_region,
    }
    args_list = [
        (str(p), str(src_dir), str(dst_root), cfg_dict) for p in images
    ]
    chunks = [args_list[i:i + _CHUNK_SIZE] for i in range(0, len(args_list), _CHUNK_SIZE)]

    success = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        refresh_per_second=4,
    ) as progress:
        task = progress.add_task("影像前處理", total=len(args_list))
        with ProcessPoolExecutor(max_workers=cfg.max_workers) as pool:
            futures = {pool.submit(_process_chunk, chunk): chunk for chunk in chunks}
            for fut in as_completed(futures):
                chunk = futures[fut]
                try:
                    for img_path, err_msg in fut.result():
                        if err_msg is None:
                            success += 1
                        else:
                            logger.error("前處理失敗: %s — %s", Path(img_path).name, err_msg)
                        progress.advance(task)
                except Exception as exc:
                    logger.error("Chunk 執行失敗 (%d 張): %s", len(chunk), exc)
                    progress.advance(task, len(chunk))

    logger.info("影像前處理完成: %d/%d 成功", success, len(images))


# ============================================================
# 內部實作（可 pickle，供 ProcessPoolExecutor 使用）
# ============================================================


def _process_chunk(chunk: list) -> list:
    """批次處理一組影像，回傳 (path, error_msg|None) 清單。"""
    results = []
    for args in chunk:
        img_path = args[0]
        try:
            _process_one(*args)
            results.append((img_path, None))
        except Exception as exc:
            results.append((img_path, str(exc)))
    return results


def _process_one(
    img_path: str,
    src_root: str,
    dst_root: str,
    cfg: dict,
) -> None:
    """處理單張影像：前處理 → 生成 random_count 個變體。"""
    cv2.setNumThreads(0)
    from src.data.logo_removal import remove_logo
    from src.data.topology import sort_crops_by_topology, topology_guided_mask

    path = Path(img_path)

    # --- 保留類別相對路徑 ---
    try:
        rel = path.relative_to(src_root)
        # rel = class_name/image.png OR image.png (flat)
        class_part = rel.parent   # Path(".") if flat, else Path("class_name")
    except ValueError:
        class_part = Path(".")

    # --- 斷點恢復：輸出已齊全則跳過（避免重複讀圖與運算）---
    out_dir = Path(dst_root) / class_part / path.stem
    if out_dir.exists() and len(list(out_dir.glob("arr_*.png"))) >= cfg["random_count"]:
        return None

    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"無法讀取影像: {img_path}")

    # --- Step 1: Logo 移除 ---
    if cfg["remove_gifu_logo"]:
        gray = remove_logo(
            gray,
            template_path=cfg.get("logo_template_path"),
            mask_region=cfg.get("logo_mask_region"),
        )

    # --- Step 2: 連通元件 or 全圖拓撲 ---
    if cfg["use_connected_components"]:
        binary = binarize(gray)
        crops = extract_crops(
            binary,
            cfg["top_n"],
            cfg["remove_largest"],
            cfg["padding"],
        )
        if not crops:
            return

        if cfg["use_topology_analysis"]:
            crops = sort_crops_by_topology(crops)

        h, w = gray.shape
        out_dir.mkdir(parents=True, exist_ok=True)

        rng = random.Random(hash(img_path))
        for i in range(cfg["random_count"]):
            canvas = arrange_crops(crops, h, w, cfg["max_attempts"], rng)
            result = 255 - canvas
            cv2.imwrite(str(out_dir / f"arr_{i:03d}.png"), result)

    else:
        # 無 CC：全圖處理
        if cfg["use_topology_analysis"]:
            gray = topology_guided_mask(gray)

        out_dir.mkdir(parents=True, exist_ok=True)

        # 無 CC 時直接複製/儲存影像（仍生成 random_count 張以保持一致性）
        rng = random.Random(hash(img_path))
        for i in range(cfg["random_count"]):
            cv2.imwrite(str(out_dir / f"arr_{i:03d}.png"), gray)


def binarize(gray: np.ndarray) -> np.ndarray:
    """Otsu 二值化；輸出白色=元件、黑色=背景。"""
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binary


def discover_components(
    binary: np.ndarray,
    top_n: int,
    remove_largest: bool,
    padding: int,
) -> List[dict]:
    """偵測並提取連通元件及其詮釋資料。

    Args:
        binary: 二值化影像（白色為元件）。
        top_n: 保留的大元件數。
        remove_largest: 是否移除最大元件。
        padding: 裁切邊界填充。

    Returns:
        List[dict]: 包含 'crop', 'bbox' (x1, y1, x2, y2), 'area' 的字典列表。
    """
    h, w = binary.shape
    num_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    if num_labels <= 1:
        return []

    components = [
        (i, int(stats[i, cv2.CC_STAT_AREA]))
        for i in range(1, num_labels)
    ]
    components.sort(key=lambda x: x[1], reverse=True)

    if remove_largest and len(components) > 1:
        components = components[1:]

    components = components[:top_n]

    results: List[dict] = []
    for label_idx, area in components:
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

        results.append({
            "crop": binary[y1:y2, x1:x2].copy(),
            "bbox": (x1, y1, x2, y2),
            "area": area,
        })

    return results


def extract_crops(
    binary: np.ndarray,
    top_n: int,
    remove_largest: bool,
    padding: int,
) -> List[np.ndarray]:
    """提取連通元件裁切圖（白色=元件像素）。"""
    comps = discover_components(binary, top_n, remove_largest, padding)
    return [c["crop"] for c in comps]


def arrange_crops(
    crops: List[np.ndarray],
    canvas_h: int,
    canvas_w: int,
    max_attempts: int,
    rng: random.Random,
) -> np.ndarray:
    """將裁切圖隨機排列到黑色畫布（白色=元件）。"""
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    placed: List[Tuple[int, int, int, int]] = []

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
    """找到不重疊的放置位置；超過嘗試上限則隨機回傳。"""
    max_y = max(0, canvas_h - crop_h)
    max_x = max(0, canvas_w - crop_w)

    for _ in range(max_attempts):
        y = rng.randint(0, max_y)
        x = rng.randint(0, max_x)
        box = (x, y, x + crop_w, y + crop_h)

        if not any(_overlaps(box, p) for p in placed):
            return x, y

    # 備案：若無法找到完全不重疊位置，隨機選取一個位置（保持回傳順序為 x, y）
    return rng.randint(0, max_x), rng.randint(0, max_y)


def _overlaps(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
) -> bool:
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])
