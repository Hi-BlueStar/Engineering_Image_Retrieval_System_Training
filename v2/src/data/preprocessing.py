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
        max_bbox_ratio: 排除大於整張圖一定比例的外階矩形元件（通常為圖框）。
        padding: 裁切邊距（像素）。

        use_connected_components: 是否啟用連通元件分析。
        use_topology_analysis: 是否啟用拓撲感知排序/遮罩。
        use_topology_pruning: 是否啟用拓撲分類與剪枝。
        topology_pruning_iters: 結構級剪枝最大迭代次數。
        topology_pruning_ksize: 結構級剪枝起始 Kernel 尺寸。
        min_simple_area: 無孔洞元件的最小面積門檻（剪枝用）。
        remove_gifu_logo: 是否移除 Gifu Logo。
        logo_template_path: Logo 模板路徑（可選）。
        logo_mask_region: Logo 遮罩區域比例（可選）。
    """

    input_dir: str
    output_root: str
    max_workers: int = 12
    top_n: int = 5
    max_bbox_ratio: float = 0.9
    padding: int = 2

    use_connected_components: bool = True
    use_topology_analysis: bool = True
    use_topology_pruning: bool = True
    topology_pruning_iters: int = 3
    topology_pruning_ksize: int = 2
    min_simple_area: int = 40
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
        "開始影像前處理: %d 張 | CC=%s | Topology=%s | LogoRemoval=%s",
        len(images),
        cfg.use_connected_components,
        cfg.use_topology_analysis,
        cfg.remove_gifu_logo,
    )

    cfg_dict = {
        "top_n": cfg.top_n,
        "max_bbox_ratio": cfg.max_bbox_ratio,
        "padding": cfg.padding,

        "use_connected_components": cfg.use_connected_components,
        "use_topology_analysis": cfg.use_topology_analysis,
        "use_topology_pruning": cfg.use_topology_pruning,
        "topology_pruning_iters": cfg.topology_pruning_iters,
        "topology_pruning_ksize": cfg.topology_pruning_ksize,
        "min_simple_area": cfg.min_simple_area,
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
    """處理單張影像：前處理 → 生成獨立元件圖。"""
    cv2.setNumThreads(0)
    from src.data.logo_removal import remove_logo
    from src.data.topology import (
        analyze_topology,
        sort_crops_by_topology,
        topology_guided_mask,
        topology_preserving_pruning,
    )


    path = Path(img_path)

    # --- 保留類別相對路徑 ---
    try:
        rel = path.relative_to(src_root)
        # rel = class_name/image.png OR image.png (flat)
        class_part = rel.parent   # Path(".") if flat, else Path("class_name")
    except ValueError:
        class_part = Path(".")

    # --- 斷點恢復：輸出目錄已有圖則跳過（避免重複讀圖與運算）---
    out_dir = Path(dst_root) / class_part / path.stem
    if out_dir.exists() and (list(out_dir.glob("comp_*.png")) or list(out_dir.glob("full_*.png"))):
        return None

    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"無法讀取影像: {img_path}")

    # --- 連通元件 or 全圖拓撲 ---
    if cfg["use_connected_components"]:
        binary = binarize(gray)
        crops = extract_crops(
            binary,
            cfg["top_n"],
            cfg["max_bbox_ratio"],
            cfg["padding"],
            remove_logo_cfg=cfg["remove_gifu_logo"],
            logo_template_path=cfg.get("logo_template_path"),
            logo_mask_region=cfg.get("logo_mask_region"),
            use_topology_pruning=cfg.get("use_topology_pruning", True),
            topology_pruning_iters=cfg.get("topology_pruning_iters", 3),
            topology_pruning_ksize=cfg.get("topology_pruning_ksize", 2),
            min_simple_area=cfg.get("min_simple_area", 40),
        )
        if not crops:
            return

        if cfg["use_topology_analysis"]:
            crops = sort_crops_by_topology(crops)

        h, w = gray.shape
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, crop in enumerate(crops):
            # crop 預設是 255 為特徵，0 為背景。我們反轉為白底黑線
            inv_crop = 255 - crop
            # 加上純白邊框 padding
            pad = cfg["padding"]
            padded_crop = cv2.copyMakeBorder(
                inv_crop, pad, pad, pad, pad, 
                cv2.BORDER_CONSTANT, value=255
            )
            cv2.imwrite(str(out_dir / f"comp_{i:03d}.png"), padded_crop)

    else:
        # 無 CC：全圖處理
        if cfg["remove_gifu_logo"]:
            gray = remove_logo(
                gray,
                template_path=cfg.get("logo_template_path"),
                mask_region=cfg.get("logo_mask_region"),
            )

        if cfg["use_topology_analysis"]:
            gray = topology_guided_mask(gray)

        out_dir.mkdir(parents=True, exist_ok=True)

        # 無 CC 時直接複製/儲存影像加上 padding
        pad = cfg["padding"]
        padded_gray = cv2.copyMakeBorder(
            gray, pad, pad, pad, pad, 
            cv2.BORDER_CONSTANT, value=255
        )
        cv2.imwrite(str(out_dir / f"full_000.png"), padded_gray)


def binarize(gray: np.ndarray) -> np.ndarray:
    """Otsu 二值化；輸出白色=元件、黑色=背景。"""
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binary


def discover_components(
    binary: np.ndarray,
    top_n: int,
    max_bbox_ratio: float,
    padding: int,
    remove_logo_cfg: bool = False,
    logo_template_path: Optional[str] = None,
    logo_mask_region: Optional[List[float]] = None,
    use_topology_pruning: bool = True,
    topology_pruning_iters: int = 3,
    topology_pruning_ksize: int = 2,
    min_simple_area: int = 40,
) -> List[dict]:
    """偵測並提取連通元件及其詮釋資料。

    Args:
        binary: 二值化影像（白色為元件）。
        top_n: 保留的大元件數。
        max_bbox_ratio: 排除大於整張圖一定比例的元件。
        padding: 裁切邊界填充。
        remove_logo_cfg: 是否在此階段移除 Logo。
        logo_template_path: Logo 模板路徑。
        logo_mask_region: Logo 遮罩區域。

    Returns:
    """
    h, w = binary.shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    if num_labels <= 1:
        return []

    # 1. 取得所有 Candidate 元件 (不含背景)
    all_components = []
    for i in range(1, num_labels):
        pixel_area = int(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        bbox_area = cw * ch
        all_components.append({
            "idx": i,
            "pixel_area": pixel_area,
            "bbox_area": bbox_area,
            "bbox": (x, y, x + cw, y + ch)
        })

    # 2. Logo 偵測與過濾 (延後到此處理)
    filtered_indices = set()
    if remove_logo_cfg:
        from src.data.logo_removal import find_logo_regions
        logo_boxes = find_logo_regions(
            binary,
            template_path=logo_template_path,
            mask_region=logo_mask_region
        )
        
        for comp in all_components:
            cx1, cy1, cx2, cy2 = comp["bbox"]
            # 判斷元件是否在 Logo 區域內 (IoU 或 包含關係)
            for lx1, ly1, lx2, ly2 in logo_boxes:
                # 簡單判定：元件 bounding box 中心點在 Logo 區域內
                center_x, center_y = (cx1 + cx2) / 2, (cy1 + cy2) / 2
                if lx1 <= center_x <= lx2 and ly1 <= center_y <= ly2:
                    filtered_indices.add(comp["idx"])
                    break
        
        if filtered_indices:
            logger.debug("過濾掉 %d 個 Logo 元件", len(filtered_indices))

    # 3. 拓撲分析與剪枝 (先分析拓撲，再做剪枝)
    from src.data.topology import analyze_topology, topology_preserving_pruning

    components = []
    for comp in all_components:
        if comp["idx"] in filtered_indices:
            continue

        # 計算拓撲 (針對該元件所在的 label 區域)
        x1, y1, x2, y2 = comp["bbox"]
        comp_crop = (labels[y1:y2, x1:x2] == comp["idx"]).astype(np.uint8) * 255
        topo = analyze_topology(comp_crop)
        comp.update(topo)

        # 剪枝判斷：保留具有拓撲特徵者，或面積達標的簡單元件
        if use_topology_pruning:
            if not comp["is_complex"] and comp["pixel_area"] < min_simple_area:
                continue

        components.append(comp)

    # 4. 排序與篩選 (依照 bbox 面積排序，並排除大於比例者)
    components.sort(key=lambda x: x["bbox_area"], reverse=True)

    # 排除大於比例的元件 (通常是圖框)
    total_area = h * w
    filtered_components = []
    for comp in components:
        ratio = comp["bbox_area"] / total_area
        if ratio > max_bbox_ratio:
            logger.debug("排除超大元件: idx=%d, ratio=%.4f", comp["idx"], ratio)
            continue
        filtered_components.append(comp)
    
    components = filtered_components[:top_n]

    results: List[dict] = []
    for comp in components:
        x1, y1, x2, y2 = comp["bbox"]
        # 在輸出的 results 中，'area' 指的是用於排序的 bbox_area
        area = comp["bbox_area"]

        px1 = max(0, x1 - padding)
        py1 = max(0, y1 - padding)
        px2 = min(w, x2 + padding)
        py2 = min(h, y2 + padding)

        if px2 - px1 < 2 or py2 - py1 < 2:
            continue

        crop_labels = labels[py1:py2, px1:px2]
        crop = (crop_labels == comp["idx"]).astype(np.uint8) * 255

        # 結構級剪枝 (確保不改動拓撲結構)
        history = []
        if use_topology_pruning:
            crop, history = topology_preserving_pruning(
                crop, 
                max_iters=topology_pruning_iters, 
                start_ksize=topology_pruning_ksize
            )

        results.append({
            "crop": crop,
            "crop_history": history,
            "bbox": (px1, py1, px2, py2),
            "area": area,
            "pixel_area": comp["pixel_area"],
            "is_complex": comp.get("is_complex", False),
            "n_holes": comp.get("n_holes", 0),
        })

    return results


def extract_crops(
    binary: np.ndarray,
    top_n: int,
    max_bbox_ratio: float,
    padding: int,
    remove_logo_cfg: bool = False,
    logo_template_path: Optional[str] = None,
    logo_mask_region: Optional[List[float]] = None,
    use_topology_pruning: bool = True,
    topology_pruning_iters: int = 3,
    topology_pruning_ksize: int = 2,
    min_simple_area: int = 40,
) -> List[np.ndarray]:
    """提取連通元件裁切圖（白色=元件像素）。"""
    comps = discover_components(
        binary,
        top_n,
        max_bbox_ratio,
        padding,
        remove_logo_cfg=remove_logo_cfg,
        logo_template_path=logo_template_path,
        logo_mask_region=logo_mask_region,
        use_topology_pruning=use_topology_pruning,
        topology_pruning_iters=topology_pruning_iters,
        topology_pruning_ksize=topology_pruning_ksize,
        min_simple_area=min_simple_area,
    )
    return [c["crop"] for c in comps]



