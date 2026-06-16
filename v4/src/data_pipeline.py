"""資料管線與前處理模組 (Data Pipeline and Preprocessing Module)。

============================================================
負責處理資料集的解壓、PDF 轉影像、Logo 移除、連通元件分割、
CPU 端 Letterbox 等比例縮放填充，以及高吞吐的 GPU 預取資料載入。

核心機制：
    1. **快取防呆流程**：偵測到 `.npz` 快取檔時，直接讀取快取（1秒載入）；
       若無，則自動向上檢查並實作連通元件前處理、資料集劃分，並生成快取。
    2. **CPU Letterbox**：在 CPU 端將裁剪組件等比例縮放並 Padding 至恆定 $512 \times 512 \times 1$。
    3. **GPUPrefetcher**：透過 Pinned Memory 與獨立 CUDA Stream，
       在 GPU 計算當前 Batch 的同時，非同步傳輸下一個 Batch，消除 PCIe 傳輸瓶頸。
    4. **GPU 雙視角增強**：將 Batch 移至 GPU 後，使用 Kornia 進行向量化雙視角增強。
============================================================
"""

from __future__ import annotations

import concurrent.futures
import os
import random
import sys
import zipfile
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .logger import get_logger

logger = get_logger(__name__)

# 偵測 Kornia 是否可用以決定 GPU 增強機制
try:
    import kornia.augmentation as K
    from kornia.constants import Resample
    _KORNIA_AVAILABLE = True
except ImportError:
    _KORNIA_AVAILABLE = False
    logger.warning("未偵測到 Kornia 庫，GPU 增強將自動退化為僅包含縮放與標準化。")

# 引入 v4 內置的連通元件前處理演算法，確保完全自主解耦
from .image_preprocessing import (
    auto_binarize,
    analyze_components,
    select_large_small,
    assign_small_to_large,
    merge_small_into_large
)


# ============================================================
# 1. 檔案解壓與 PDF 轉換子模組 (Extraction & PDF Converter)
# ============================================================

def extract_raw_zip(zip_path: str, output_dir: str) -> None:
    """平行解壓縮 ZIP 檔案至指定路徑 (CPU 多執行緒加速)"""
    src = Path(zip_path)
    dst = Path(output_dir)
    if not src.is_file():
        raise FileNotFoundError(f"找不到 ZIP 壓縮檔: {src.resolve()}")
    
    dst.mkdir(parents=True, exist_ok=True)
    logger.info("開始解壓縮 ZIP 檔案: %s", src.name)
    
    with zipfile.ZipFile(src, "r") as z_file:
        members = z_file.namelist()
        if not members:
            logger.warning("ZIP 檔案為空。")
            return
        
        # 預先在單執行緒建立所有父目錄，防範多執行緒建立目錄時發生競爭衝突
        for member in members:
            target = dst / member
            target.parent.mkdir(parents=True, exist_ok=True)
            
        max_workers = min(32, (os.cpu_count() or 4) * 2)
        chunk_size = max(1, len(members) // (max_workers * 2))
        chunks = [members[i:i + chunk_size] for i in range(0, len(members), chunk_size)]
        
        def _extract_chunk(chunk_members: list[str]) -> None:
            with zipfile.ZipFile(src, "r") as z_in:
                for member in chunk_members:
                    z_in.extract(member, dst)
                    
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(_extract_chunk, chunks))
            
    logger.info("解壓完成: %s -> %s (共 %d 個檔案)", src.name, dst, len(members))


def _pdf_to_png_worker(pdf_path: Path, root_dir: Path, out_dir: Path, dpi: int) -> dict:
    """單個 PDF 轉檔工作執行緒"""
    import fitz  # PyMuPDF
    scale = dpi / 72.0
    try:
        class_label = pdf_path.parent.name
        rel_dir = pdf_path.parent.relative_to(root_dir)
        fname = f"{pdf_path.stem.replace(' ', '_')}.png"
        dest_abs = out_dir / rel_dir / fname
        
        # 支援斷點續傳：檔案已存在則直接跳過
        if dest_abs.exists():
            return {"status": "skipped", "source_pdf": str(pdf_path), "class_label": class_label, "image_path": dest_abs.as_posix()}
            
        dest_abs.parent.mkdir(parents=True, exist_ok=True)
        
        with fitz.open(str(pdf_path)) as doc:
            page = doc.load_page(0)  # 僅轉印第一頁
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            pix.save(dest_abs.as_posix())
            
        return {"status": "success", "source_pdf": str(pdf_path), "class_label": class_label, "image_path": dest_abs.as_posix()}
    except Exception as e:
        return {"status": "error", "file": str(pdf_path), "error": str(e)}


def convert_pdfs(pdf_dir: str, output_dir: str, dpi: int = 100, max_workers: int = 16) -> None:
    """將 PDF 目錄內的所有檔案轉換為 PNG (多行程加速)"""
    src_dir = Path(pdf_dir)
    dst_dir = Path(output_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_files = sorted(Path(dp) / fn for dp, _, fns in os.walk(src_dir) for fn in fns if fn.lower().endswith(".pdf"))
    if not pdf_files:
        logger.warning("PDF 來源目錄無 PDF 檔案: %s", src_dir)
        return
        
    logger.info("開始轉換 PDF: 共 %d 個檔案，DPI=%d", len(pdf_files), dpi)
    rows_out = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_pdf_to_png_worker, p, src_dir, dst_dir, dpi): p for p in pdf_files}
        for fut in concurrent.futures.as_completed(futures):
            p = futures[fut]
            try:
                res = fut.result()
                if res["status"] in ("success", "skipped"):
                    rows_out.append((res["source_pdf"], res["class_label"], res["image_path"]))
                else:
                    logger.error("PDF 轉換錯誤 (%s): %s", p.name, res.get("error"))
            except Exception as e:
                logger.error("PDF 轉檔程序崩潰 (%s): %s", p.name, e)
                
    df = pd.DataFrame(rows_out, columns=["source_pdf", "class_label", "image_path"])
    df.to_csv(dst_dir / "manifest.csv", index=False)
    logger.info("PDF 轉換結束，已產出 manifest.csv")


# ============================================================
# 2. Gifu Logo 擦除與影像前處理子模組 (Logo Removal & Processing)
# ============================================================

def remove_gifu_logo(
    image: np.ndarray,
    template_path: Optional[str] = None,
    mask_region: Optional[List[float]] = None,
    fill_value: int = 255
) -> np.ndarray:
    """移除影像中的吉輔 Logo，若未指定模板或遮罩則自動在四個角落作像素密度探測與擦除"""
    h, w = image.shape[:2]
    img_c = image.copy()
    
    # 策略 1: 比例遮罩
    if mask_region and len(mask_region) == 4:
        x1 = max(0, int(mask_region[0] * w))
        y1 = max(0, int(mask_region[1] * h))
        x2 = min(w, int(mask_region[2] * w))
        y2 = min(h, int(mask_region[3] * h))
        img_c[y1:y2, x1:x2] = fill_value
        return img_c
        
    # 策略 2: 模板匹配
    if template_path and Path(template_path).is_file():
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is not None:
            th, tw = template.shape[:2]
            if th <= h and tw <= w:
                res = cv2.matchTemplate(img_c, template, cv2.TM_CCOEFF_NORMED)
                locs = np.where(res >= 0.75)
                for pt in zip(*locs[::-1]):
                    img_c[pt[1]:pt[1]+th, pt[0]:pt[0]+tw] = fill_value
                return img_c

    # 策略 3: 自動角落密度擦除 (在四個角落寬高 10% 範圍內，若前景密度 > 0.3 則判定為 Logo 並擦除)
    ch, cw = int(h * 0.10), int(w * 0.10)
    corners = [
        (0, 0, cw, ch),
        (w - cw, 0, w, ch),
        (0, h - ch, cw, h),
        (w - cw, h - ch, w, h)
    ]
    # 二值化以計算密度
    _, binary = cv2.threshold(img_c, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    for x1, y1, x2, y2 in corners:
        roi = binary[y1:y2, x1:x2]
        density = roi.mean() / 255.0
        if density > 0.3:
            img_c[y1:y2, x1:x2] = fill_value
            
    return img_c


def _preprocess_single_image_worker(
    img_path: Path, src_root: Path, dst_root: Path, cfg: dict
) -> tuple[str, bool]:
    """單一影像連通域分析與裁切工件子圖任務"""
    cv2.setNumThreads(0)
    try:
        rel = img_path.relative_to(src_root)
        class_part = rel.parent
        out_dir = dst_root / class_part / img_path.stem
        
        # 支援斷點續傳：已有輸出則跳過
        if out_dir.exists() and list(out_dir.glob("comp_*.png")):
            return str(img_path), True
            
        # 讀取影像 (支援中文路徑)
        arr = np.fromfile(str(img_path), dtype=np.uint8)
        src = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if src is None:
            return str(img_path), False
            
        # 二值化
        bw01, bg_mode = auto_binarize(src)
        comps = analyze_components(bw01)
        
        # 篩選大組件 (排除大於比例的圖框，保留 Top-N)
        large, small = select_large_small(
            comps,
            top_n=cfg["top_n"],
            max_bbox_ratio=cfg.get("max_bbox_ratio", 0.9),
            img_shape=bw01.shape
        )
        assignment = assign_small_to_large(large, small)
        merged_masks = merge_small_into_large(large, assignment)
        
        # 備用降級機制：若無大組件，直接回退為整張圖加邊框
        if not large:
            gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            if cfg["remove_logo"]:
                gray = remove_logo_fn(gray, cfg["logo_template"], cfg["logo_mask"])
            
            large_comp_dir = out_dir / "large_components"
            large_comp_dir.mkdir(parents=True, exist_ok=True)
            padded = cv2.copyMakeBorder(gray, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=255)
            _, encoded = cv2.imencode(".png", padded)
            encoded.tofile(str(large_comp_dir / "comp_000.png"))
            return str(img_path), True

        # 寫入各獨立組件圖 (白底黑線)
        h_src, w_src, _ = src.shape
        large_comp_dir = out_dir / "large_components"
        large_comp_dir.mkdir(parents=True, exist_ok=True)
        bg_val = 255 if bg_mode == "white" else 0
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        
        pad = cfg["padding"]
        for i, L in enumerate(large):
            roi_mask = merged_masks[L.label]
            x, y, cw, ch = L.bbox
            
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(w_src, x + cw + pad), min(h_src, y + ch + pad)
            
            canvas_h, canvas_w = y1 - y0, x1 - x0
            comp_img = np.full((canvas_h, canvas_w), bg_val, dtype=np.uint8)
            
            off_x = x - x0
            off_y = y - y0
            
            roi_src = src_gray[y:y+ch, x:x+cw]
            idx = roi_mask.astype(bool)
            comp_img[off_y:off_y+ch, off_x:off_x+cw][idx] = roi_src[idx]
            
            # 反轉為白底黑線
            if bg_mode == "black":
                comp_img = 255 - comp_img
            else:
                # 若原本就是白底，二值前景對應的是黑線，故直接反轉非遮罩部分以保持白底
                comp_img = comp_img
                
            out_path = large_comp_dir / f"comp_{i:03d}.png"
            _, encoded = cv2.imencode(".png", comp_img)
            encoded.tofile(str(out_path))
            
        return str(img_path), True
    except Exception as e:
        logger.error("處理 %s 失敗: %s", img_path.name, e)
        return str(img_path), False


def remove_logo_fn(img, template, mask):
    """移除 Logo 的包裝"""
    return remove_gifu_logo(img, template_path=template, mask_region=mask)


def run_preprocessing_pipeline(
    input_dir: str,
    output_dir: str,
    top_n: int = 5,
    max_bbox_ratio: float = 0.9,
    padding: int = 2,
    remove_logo: bool = True,
    logo_template: Optional[str] = None,
    logo_mask: Optional[List[float]] = None,
    max_workers: int = 12
) -> None:
    """前處理排程控制中心，將所有轉檔影像分割為工件組件"""
    src_dir = Path(input_dir)
    dst_dir = Path(output_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    images = sorted(Path(dp) / fn for dp, _, fns in os.walk(src_dir) for fn in fns if Path(fn).suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not images:
        logger.warning("沒有可供前處理的影像，來源目錄: %s", src_dir)
        return
        
    logger.info("開始影像前處理連通域裁切: 共 %d 張圖", len(images))
    cfg = {
        "top_n": top_n,
        "max_bbox_ratio": max_bbox_ratio,
        "padding": padding,
        "remove_logo": remove_logo,
        "logo_template": logo_template,
        "logo_mask": logo_mask
    }
    
    success = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_preprocess_single_image_worker, p, src_dir, dst_dir, cfg): p for p in images}
        for fut in concurrent.futures.as_completed(futures):
            p = futures[fut]
            try:
                _, ok = fut.result()
                if ok:
                    success += 1
            except Exception as e:
                logger.error("影像前處理執行異常 (%s): %s", p.name, e)
                
    logger.info("影像前處理完畢: %d/%d 成功", success, len(images))


# ============================================================
# 3. CPU Letterbox 縮放與 Padding 子模組 (Letterbox & npz Caching)
# ============================================================

def letterbox_image(image: np.ndarray, size: int = 512, pad_value: int = 255) -> np.ndarray:
    """等比例縮放工程圖，長邊縮至指定 size，短邊對稱填充 pad_value (白底)，確保拓撲幾何比例不失真"""
    h, w = image.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    
    # 使用 CUBIC 插值以保留精細線條的連貫性
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    
    # 建立目標尺寸的白底畫布
    canvas = np.full((size, size), pad_value, dtype=np.uint8)
    
    # 對稱置中填充
    dy = (size - nh) // 2
    dx = (size - nw) // 2
    canvas[dy:dy+nh, dx:dx+nw] = resized
    
    return canvas


def build_and_cache_dataset(
    preprocessed_dir: str,
    cache_path: str,
    split_ratio: float = 0.8,
    seed: int = 42,
    size: int = 512
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """資料集分割、CPU Letterbox 縮放與封裝為單一 `.npz` 快取"""
    src_dir = Path(preprocessed_dir)
    logger.info("開始對前處理組件進行資料集分層劃分與 CPU Letterbox ($%dx%d)...", size, size)
    
    # 搜尋所有裁切組件，格式為: src_dir / class_name / image_stem / large_components / comp_*.png
    all_comp_paths = sorted(src_dir.glob("**/large_components/comp_*.png"))
    if not all_comp_paths:
        raise FileNotFoundError(f"在前處理路徑 {src_dir.resolve()} 中找不到組件檔案。")
        
    # 分類映射表
    class_map = {}
    class_names = []
    
    # 按類別組織檔案 (分層劃分)
    class_to_paths = {}
    for p in all_comp_paths:
        # 類別資料夾名稱
        class_name = p.relative_to(src_dir).parts[0]
        if class_name not in class_to_paths:
            class_to_paths[class_name] = []
        class_to_paths[class_name].append(p)
        
    class_names = sorted(list(class_to_paths.keys()))
    class_map = {name: idx for idx, name in enumerate(class_names)}
    
    train_paths = []
    val_paths = []
    
    # 分層抽樣拆分
    random.seed(seed)
    for c_name, paths in class_to_paths.items():
        shuffled = sorted(paths)  # 保證跨平台初始排序一致
        random.shuffle(shuffled)
        idx = int(len(shuffled) * split_ratio)
        train_paths.extend(shuffled[:idx])
        val_paths.extend(shuffled[idx:])
        
    # 讀取並執行 CPU Letterbox 縮放
    def _load_and_process(path_list: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
        def _single_task(p: Path) -> Tuple[np.ndarray, int] | None:
            c_name = p.relative_to(src_dir).parts[0]
            label_id = class_map[c_name]
            # 讀取為灰階圖
            arr = np.fromfile(str(p), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_padded = letterbox_image(img, size=size, pad_value=255)
                return img_padded, label_id
            return None

        # 使用執行緒池加速 (多核心 CPU 平行化讀取與解碼)
        # 由於主要瓶頸在 I/O 與 OpenCV 的 C++ 影像解碼/縮放，執行緒池能有效釋放 GIL 並平行處理
        max_workers = min(32, (os.cpu_count() or 4) * 2)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 透過 executor.map 確保結果順序與輸入 path_list 完全相同，保證結果不變
            results = list(executor.map(_single_task, path_list))
            
        images = []
        labels = []
        for res in results:
            if res is not None:
                img_padded, label_id = res
                images.append(img_padded)
                labels.append(label_id)
                
        return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.int64)
        
    logger.info("正在載入並 Letterbox 縮放 Train 集 (共 %d 張)...", len(train_paths))
    train_x, train_y = _load_and_process(train_paths)
    
    logger.info("正在載入並 Letterbox 縮放 Val 集 (共 %d 張)...", len(val_paths))
    val_x, val_y = _load_and_process(val_paths)
    
    # 確保快取檔的父目錄存在
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)

    # 儲存快取檔
    np.savez_compressed(
        cache_path,
        train_images=train_x,
        train_labels=train_y,
        val_images=val_x,
        val_labels=val_y,
        class_names=class_names
    )
    logger.info("快取歸檔完成，快取路徑: %s (檔案大小: %.2f MB)", cache_path, Path(cache_path).stat().st_size / (1024*1024))
    return train_x, train_y, val_x, val_y, class_names


def build_and_cache_dataset_json(
    preprocessed_dir: str,
    cache_path: str,
    split_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[str], List[int], List[str], List[int], List[str]]:
    """資料集路徑劃分與輕量級 JSON 快取儲存"""
    src_dir = Path(preprocessed_dir)
    logger.info("開始對前處理組件進行資料集分層劃分...")
    
    # 搜尋所有裁切組件
    all_comp_paths = sorted(src_dir.glob("**/large_components/comp_*.png"))
    if not all_comp_paths:
        raise FileNotFoundError(f"在前處理路徑 {src_dir.resolve()} 中找不到組件檔案。")
        
    # 分類映射表
    class_to_paths = {}
    for p in all_comp_paths:
        class_name = p.relative_to(src_dir).parts[0]
        if class_name not in class_to_paths:
            class_to_paths[class_name] = []
        class_to_paths[class_name].append(p)
        
    class_names = sorted(list(class_to_paths.keys()))
    class_map = {name: idx for idx, name in enumerate(class_names)}
    
    train_paths = []
    train_labels = []
    val_paths = []
    val_labels = []
    
    # 分層抽樣拆分
    random.seed(seed)
    for c_name, paths in class_to_paths.items():
        shuffled = sorted(paths)  # 保證跨平台初始排序一致
        random.shuffle(shuffled)
        idx = int(len(shuffled) * split_ratio)
        
        # 轉換為字串路徑以利 JSON 序列化
        c_train = [str(p) for p in shuffled[:idx]]
        c_val = [str(p) for p in shuffled[idx:]]
        
        train_paths.extend(c_train)
        train_labels.extend([class_map[c_name]] * len(c_train))
        val_paths.extend(c_val)
        val_labels.extend([class_map[c_name]] * len(c_val))
        
    # 確保快取檔的父目錄存在
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)

    # 儲存 JSON 快取檔
    cache_data = {
        "train_paths": train_paths,
        "train_labels": train_labels,
        "val_paths": val_paths,
        "val_labels": val_labels,
        "class_names": class_names
    }
    
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
    logger.info("快取歸檔完成，快取路徑: %s (檔案大小: %.2f MB)", cache_path, Path(cache_path).stat().st_size / (1024*1024))
    return train_paths, train_labels, val_paths, val_labels, class_names


# ============================================================
# 4. 記憶體載入與 GPU 預取子模組 (PrefetchDataLoader & GPU Aug)
# ============================================================

class NPZDataset(torch.utils.data.Dataset):
    """唯讀記憶體快取之 NumPy Tensor 資料集"""
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.images = torch.from_numpy(x)  # 形狀: [N, 512, 512] uint8
        self.labels = torch.from_numpy(y)  # 形狀: [N] int64

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx]


class DiskDataset(torch.utils.data.Dataset):
    """動態從硬碟讀取與 Letterbox 縮放的 PyTorch 資料集"""
    def __init__(self, image_paths: List[str] | List[Path], labels: List[int], img_size: int = 512) -> None:
        self.image_paths = [str(p) for p in image_paths]
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.image_paths[idx]
        
        # 讀取為灰階圖 (支援中文路徑)
        try:
            arr = np.fromfile(p, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        except Exception:
            img = None
            
        if img is None:
            # 防呆：若讀取失敗則返回一個空白白底圖
            img_padded = np.full((self.img_size, self.img_size), 255, dtype=np.uint8)
        else:
            img_padded = letterbox_image(img, size=self.img_size, pad_value=255)
            
        # 轉為 Tensor 格式，形狀: [512, 512]，資料型態為 uint8，與 NPZDataset 對齊
        x_tensor = torch.from_numpy(img_padded)
        return x_tensor, self.labels[idx]


class GPUPrefetcher:
    """非同步預取資料搬運器 (H2D Overlapping)

    在 GPU 上計算當前 Batch 時，使用獨立 Stream 與 non_blocking 非同步將下一個 Batch 複製到 GPU，
    並在此處執行型別轉換 (uint8 -> bfloat16/float32) 與背景正規化。
    """
    def __init__(
        self,
        loader: torch.utils.data.DataLoader,
        device: torch.device,
        use_bf16: bool = True,
        mean: float = 0.0394,
        std: float = 0.1752
    ) -> None:
        self.loader = loader
        self.loader_iter = iter(loader)
        self.device = device
        
        # 僅在裝置為 CUDA 時才初始化 CUDA Stream，避免在 CPU 運作時崩潰
        self.use_cuda = device.type == "cuda"
        if self.use_cuda:
            self.stream = torch.cuda.Stream()
        else:
            self.stream = None
        
        # 影像載入均採用 float32 格式，以維持高精度的資料增強運算並相容 Kornia (進入模型時 autocast 會自動轉為 bfloat16)
        self.dtype = torch.float32
        
        # 正規化參數 (CAD白底線條：反轉後背景為0，線條為非0)
        self.mean = torch.tensor([mean], device=device, dtype=self.dtype).view(1, 1, 1, 1)
        self.std = torch.tensor([std], device=device, dtype=self.dtype).view(1, 1, 1, 1)
        
        self.next_x: Optional[torch.Tensor] = None
        self.next_y: Optional[torch.Tensor] = None
        self.preload()

    def preload(self) -> None:
        try:
            self.next_x, self.next_y = next(self.loader_iter)
        except StopIteration:
            self.next_x = None
            self.next_y = None
            return
            
        if self.use_cuda:
            with torch.cuda.stream(self.stream):
                # 非同步 H2D 傳輸 (需要配合 DataLoader pin_memory=True)
                self.next_x = self.next_x.to(self.device, non_blocking=True)
                self.next_y = self.next_y.to(self.device, non_blocking=True)
        else:
            self.next_x = self.next_x.to(self.device)
            self.next_y = self.next_y.to(self.device)

    def __iter__(self) -> GPUPrefetcher:
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_cuda:
            # 同步搬運 Stream，確保數據可供使用
            torch.cuda.current_stream().wait_stream(self.stream)
            
        x = self.next_x
        y = self.next_y
        
        if x is None:
            raise StopIteration
            
        # 於 GPU/CPU 上進行型別轉換與標準化
        # [B, 512, 512] -> [B, 1, 512, 512]
        x_processed = x.unsqueeze(1).to(dtype=self.dtype) / 255.0
        
        # 標準化
        x_norm = (x_processed - self.mean) / self.std
        
        self.preload()
        return x_norm, y


class GPUPrefetchDataLoader:
    """封裝 GPUPrefetcher 的迭代裝飾器"""
    def __init__(
        self,
        dataset: NPZDataset,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        pin_memory: bool,
        device: torch.device,
        use_bf16: bool = True,
        mean: float = 0.0394,
        std: float = 0.1752
    ) -> None:
        self.loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=shuffle,  # 訓練集 shuffle 時 drop_last=True，以防 BN 層 Batch 不足崩潰
        )
        self.device = device
        self.use_bf16 = use_bf16
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self) -> GPUPrefetcher:
        return GPUPrefetcher(
            self.loader,
            self.device,
            use_bf16=self.use_bf16,
            mean=self.mean,
            std=self.std
        )


# ============================================================
# 5. GPU 隨機資料增強 (Kornia-based GPU Transformation)
# ============================================================

class GPUAugmentationModule(nn.Module):
    """使用 Kornia 在 GPU 端平行生成雙視角隨機增強

    為了減少 Kernel Launch 延遲，將 Batch x 拼接成 2B，一次完成隨機變形再拆分。
    """
    def __init__(self, img_size: int = 512, use_augmentation: bool = True) -> None:
        super().__init__()
        self.img_size = img_size
        self.use_augmentation = use_augmentation
        
        if _KORNIA_AVAILABLE and use_augmentation:
            # 建立幾何敏感的自監督對比增強管線 (移除顏色抖動與高斯模糊，關注旋轉、翻轉與仿射幾何不變性)
            self.aug = K.AugmentationSequential(
                # 影像反轉 (50% 機率將白底黑線反轉為黑底白線，迫使模型建立通道亮度獨立不變性)
                K.RandomInvert(p=0.5, same_on_batch=False),
                
                # 翻轉與旋轉
                K.RandomHorizontalFlip(p=0.5, same_on_batch=False),
                K.RandomVerticalFlip(p=0.5, same_on_batch=False),
                K.RandomAffine(
                    degrees=15.0,
                    translate=(0.1, 0.1),
                    resample=Resample.BILINEAR.name,
                    padding_mode="zeros",
                    p=0.7,
                    same_on_batch=False
                ),
                
                # 局部裁剪與縮放 (0.4 ~ 1.0 迫使建立局部與全域結構表徵關聯)
                K.RandomResizedCrop(
                    size=(img_size, img_size),
                    scale=(0.4, 1.0),
                    resample=Resample.BILINEAR.name,
                    p=0.7,
                    same_on_batch=False
                ),
                data_keys=["input"]
            )
        else:
            self.aug = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """產生同一 Batch x 的雙正樣本增強視角"""
        # x 形狀為 [B, 1, 512, 512]，已經在 GPU 上並完成 normalized
        if self.aug is not None:
            b = x.shape[0]
            # 拼接 2B batch
            x2 = torch.cat([x, x], dim=0)
            out = self.aug(x2)
            # 拆分回 v1 與 v2
            v1, v2 = out[:b], out[b:]
            return v1, v2
        else:
            # 退化降級：無隨機增強 (僅複製)
            return x, x
