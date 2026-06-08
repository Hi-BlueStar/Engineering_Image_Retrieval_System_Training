
# ==========================================
# CELL 0 (MARKDOWN)
# ==========================================
# # SimSiam 自監督學習工程圖檢索訓練與評估 (v3)
# 
# 本 Jupyter Notebook 根據《實驗計畫書.pdf》中的理論基礎與實驗設計進行實作。核心目標為建構並驗證專用於「工程圖」特徵提取與檢索之深度學習模型。
# 
# ## 核心設計要點與學理依據
# 1. **防止特徵崩塌 (Representation Collapse)**：
#    - **模型架構**：骨幹網路為 **ResNet18**。投影層 (Projector) 採用 3 層 MLP，隱藏維度為 512，輸出維度 $d=2048$，所有全連接層皆套用批次正規化 (Batch Normalization, BN)，且輸出層不包含 ReLU。預測層 (Predictor) 採用 2 層 MLP 的瓶頸 (Bottleneck) 設計，輸入/輸出維度為 2048，隱藏維度為 128，隱藏層含 BN+ReLU，輸出層無 BN/ReLU。
#    - **Symmetric Loss & Stop-Gradient**：對稱負餘弦相似度損失函數：
#      $$ L = \frac{1}{2} \mathcal{D}(p_1, \text{stop\_gradient}(z_2)) + \frac{1}{2} \mathcal{D}(p_2, \text{stop\_gradient}(z_1)) $$
#      `z.detach()` 用於截斷梯度（Stop-Gradient），這在數學上被證實是防止所有樣本投影至單一常數點（Collapse）的關鍵機制。
# 2. **防止資料洩漏 (Data Leakage)**：
#    - 驗證集 $V$（包含 50 組由專家篩選之幾何相似圖集與 800 張干擾項背景圖）必須從訓練集 $T_{small}$ 與 $T_{large}$ 中完全剔除。訓練資料直接加載已完成清洗之 `dataset_v2` 目錄。
# 3. **資料預處理 (Preprocessing) 與資料擴增 (Augmentation)**：
#    - **資料預處理**：在實驗開始前執行。包含「移除 Logo」與「連通元件提取、裁切與導出子圖」。
#      - 移除 Gifu Logo：採用角落高密度小區域偵測或區域遮罩，將其填充為純白色（255）。
#      - 連通元件提取：Otsu反轉二值化後，提取 8-連通元件，排除與 Logo 區域重合及大於圖像比例 0.8 的外框元件（如圖紙框架），裁剪子圖並還原為白底黑線格式。
#    - **資料擴增**：
#      - **Baseline**：無預處理，無擴增。僅做縮放，複製產生兩個完全相同的視角。
#      - **實驗 A (Exp A)**：啟用前處理（載入裁切子圖），啟用 GPU 批次隨機擴增，產生兩個不同的隨機增強視角。
#      - **實驗 B (Exp B)**：無前處理，啟用 GPU 批次隨機擴增，產生兩個不同的隨機增強視角。
# 4. **計算與 I/O 瓶頸優化 (Compute & I/O Speedup)**：
#    - **GPU 批次擴增**：利用 **Kornia** 函式庫，在 GPU 上平行處理整個批次的影像增強，消除 CPU 逐張解碼與處理瓶頸。
#    - **RAM 快取驗證集**：評估時需頻繁讀取驗證集 $V$，我們在初始化時一次性將所有驗證集圖像載入 RAM，徹底消除硬碟 I/O 瓶頸。
# 5. **檢索與驗證指標**：
#    - 使用 **Leave-One-Out (LOO)** 檢索策略，將驗證集 $V$ 中的每一張影像輪流作為 Query，其餘 $|V|-1$ 張作為 Gallery。
#    - 以 **Macro-mAP** 作為核心驗證指標，確保 50 個組別對最終指標的權重相等，消除數量不一造成的統計偏差。
# 6. **優化器與學習率**：
#    - **SGD 最佳化器**：動量 0.9，權重衰減 $1.0 \times 10^{-4}$，Cosine Decay Schedule。
#    - **Linear Scaling Rule**：實際學習率 $lr = 0.05 \times \frac{\text{BatchSize}}{256}$。

# ==========================================
# CELL 1 (CODE)
# ==========================================
import os
import sys
import glob
import time
import random
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import cv2
import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.amp as amp
import torchvision.models as models
import torchvision.transforms as T

import plotly.graph_objects as go

# 檢查 CUDA 設備加速狀態
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[環境檢查] 使用設備: {device}")
print(f"[環境檢查] PyTorch 版本: {torch.__version__}")

try:
    import kornia
    import kornia.augmentation as K
    print(f"[環境檢查] Kornia 版本: {kornia.__version__} (GPU 批次擴增已就緒)")
except ImportError:
    print("[環境檢查] 警告：未偵測到 Kornia 庫，GPU 批次增強將退化為 Resize+Normalize。")

# 設定亂數種子，確保實驗可重複性 (Reproducibility)
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)
print("[環境檢查] 隨機種子已設定為: 42")

# --- 確保專案根目錄與 v3 目錄在 sys.path 中 ---
PROJECT_ROOT = Path.cwd().resolve()
sys.path = [str(PROJECT_ROOT / "v3"), str(PROJECT_ROOT)] + [p for p in sys.path if p not in (str(PROJECT_ROOT), str(PROJECT_ROOT / "v3"))]

from v3.src.data.extraction import extract_archive
from v3.src.data.pdf_converter import convert_pdfs_to_images
from v3.src.data.preprocessing import preprocess_images, PreprocessConfig
from v3.src.data.splitter import split_dataset
from v3.src.data.dataset_builder import (
    sample_query_seeds,
    extract_features,
    find_candidate_pools,
    programmatic_select_gts,
    build_dataset_v2,
    interactive_expert_filter_notebook
)
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ==========================================
# CELL 2 (CODE)
# ==========================================
def get_dynamic_workspace() -> str:
    curr = Path.cwd().resolve()
    for parent in [curr] + list(curr.parents):
        if (parent / "dataset_v2").exists() or (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return str(parent)
    if os.path.exists("/workspace"):
        return "/workspace"
    return "/home/master-user/Desktop/Engineering_Image_Retrieval_System_Training"

class Config:
    # --- 資料路徑設定 ---
    # 動態偵測工作目錄以相容於 Docker 容器與本機主機環境
    workspace_dir = get_dynamic_workspace()
    dataset_root = f"{workspace_dir}/dataset_v2"
    dataset_type = "T_small"      # 可選: "T_small" 或 "T_large"
    experiment_type = "Exp_A"     # 可選: "Baseline", "Exp_A", "Exp_B"
    img_size = 256                # 圖片尺寸 (256/512)
    in_channels = 1               # 輸入通道數 (1=灰階)
    
    # --- 離線資料增強設定 ---
    offline_aug = True            # 是否在訓練前執行離線資料增強
    num_augmented_versions = 20   # 每張原始影像預先生成的增強版本數量
    
    # --- 模型網路架構 ---
    backbone = "resnet18"
    pretrained = True             # 是否載入 ImageNet 預訓練權重
    proj_hidden = 512             # 投影層隱藏維度
    proj_dim = 2048               # 投影層輸出維度 d
    pred_hidden = 128             # 預測層瓶頸隱藏維度 (Bottleneck)
    
    # --- SGD 優化器超參數 ---
    epochs = 100
    base_lr = 0.05                # 用於線性縮放規則的基礎學習率
    weight_decay = 1e-4           # 權重衰減
    seed = 42
    
    # --- 特徵崩塌警告閾值 ---
    collapse_warning_threshold = 0.01  # 特徵維度標準差警告閾值
    
    # --- 硬體加速與 I/O 優化 ---
    num_workers = 8               # DataLoader 線程數
    pin_memory = True             # 鎖定記憶體加速 Tensor 傳輸
    eval_freq = 10                # 評估 Macro-mAP 的 Epoch 頻率
    save_dir = f"{workspace_dir}/outputs_v3_local"  # Checkpoint 輸出目錄

    # --- 前處理超參數 ---
    padding = 2                   # 裁切邊距填充
    max_bbox_ratio = 0.8          # 排除大於整張圖一定比例的外接矩形元件 (圖框過濾)
    min_bbox_area = 20            # 元件外接矩形最小面積
    top_n = 5                     # 最多保留的前 n 個連通元件
    
    # --- Gifu Logo 擦除參數 ---
    remove_gifu_logo = True
    logo_template_path = f"{workspace_dir}/data/Gifu_logo.png"
    logo_mask_region = [0.0, 0.9, 0.2, 1.0] # 預設遮罩比率 (例如左下角)

    # --- 資料預處理管線 (PDF/解壓/轉換) 設定 ---
    raw_zip_path = f"{workspace_dir}/data/PDF.zip"
    labeled_zip_path = f"{workspace_dir}/data/吉輔提供資料.zip"
    skip_extraction = True
    skip_labeled_extraction = True
    
    raw_pdf_dir = f"{workspace_dir}/data/raw_pdfs"
    raw_labeled_pdf_dir = f"{workspace_dir}/data/raw_labeled_pdfs"
    converted_image_dir = f"{workspace_dir}/data/converted_images"
    converted_labeled_image_dir = f"{workspace_dir}/data/converted_labeled_images"
    
    pdf_dpi = 400
    pdf_max_workers = 16
    skip_pdf_conversion = True
    skip_labeled_pdf_conversion = True
    
    preprocessed_image_dir = f"{workspace_dir}/data/preprocessed_images"
    preprocessed_labeled_image_dir = f"{workspace_dir}/data/preprocessed_labeled_images"
    preprocess_max_workers = 24
    skip_preprocessing = False
    skip_labeled_preprocessing = False
    
    split_ratio = 0.8
    n_runs = 1
    base_seed = 42
    
    n_seeds = 50
    n_distractors = 500

    def to_dict(self):
        return {k: getattr(self, k) for k in dir(self) if not k.startswith("__") and not callable(getattr(self, k))}

cfg = Config()
print("[參數設定] Config 實例已建立。當前子實驗設定:", cfg.experiment_type, "在", cfg.dataset_type)


# ==========================================
# CELL 3 (CODE)
# ==========================================
def find_logo_regions(
    image: np.ndarray,
    template_path: Optional[str] = None,
    mask_region: Optional[List[float]] = None,
    match_threshold: float = 0.75,
) -> List[Tuple[int, int, int, int]]:
    """
    對工程圖進行 Logo 位置定位。
    1. 優先使用區域遮罩比率 (Region Masking)
    2. 其次使用模板匹配 (Template Matching)
    3. 最後退化為角落高密度二值像素區自動偵測
    """
    H, W = image.shape[:2]
    
    # 1. 區域遮罩
    if mask_region and len(mask_region) == 4:
        x1 = max(0, int(mask_region[0] * W))
        y1 = max(0, int(mask_region[1] * H))
        x2 = min(W, int(mask_region[2] * W))
        y2 = min(H, int(mask_region[3] * H))
        return [(x1, y1, x2, y2)]
        
    # 2. 模板匹配
    if template_path and os.path.exists(template_path):
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is not None:
            th, tw = template.shape
            if th <= H and tw <= W:
                res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
                locs = np.where(res >= match_threshold)
                bboxes = []
                for pt in zip(*locs[::-1]):
                    bboxes.append((pt[0], pt[1], pt[0] + tw, pt[1] + th))
                if bboxes:
                    return bboxes
                    
    # 3. 角落高密度小區域偵測 (通常 Gifu Logo 在邊界 Corners)
    ch = int(H * 0.10)
    cw = int(W * 0.10)
    
    if len(np.unique(image)) > 2:
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        binary = image
        
    corners = [
        (0, 0, cw, ch),           # 左上
        (W - cw, 0, W, ch),       # 右上
        (0, H - ch, cw, H),       # 左下
        (W - cw, H - ch, W, H),   # 右下
    ]
    
    logo_bboxes = []
    for x1, y1, x2, y2 in corners:
        reg = binary[y1:y2, x1:x2]
        density = reg.mean() / 255.0
        if density > 0.3:  # 高密度前景點判定為 Logo
            logo_bboxes.append((x1, y1, x2, y2))
            
    return logo_bboxes

def remove_logo(
    image: np.ndarray,
    template_path: Optional[str] = None,
    mask_region: Optional[List[float]] = None,
    match_threshold: float = 0.75,
    fill_value: int = 255,
) -> np.ndarray:
    """
    將定位出的 Logo 區域填充為純白色 (255)，避免自監督學到無關之商標特徵。
    """
    img_clean = image.copy()
    regions = find_logo_regions(img_clean, template_path, mask_region, match_threshold)
    for x1, y1, x2, y2 in regions:
        img_clean[y1:y2, x1:x2] = fill_value
    return img_clean

def discover_components(
    binary: np.ndarray,
    top_n: int,
    max_bbox_ratio: float,
    min_bbox_area: int,
    padding: int,
    remove_logo_cfg: bool = False,
    logo_template_path: Optional[str] = None,
    logo_mask_region: Optional[List[float]] = None,
) -> List[Dict]:
    """
    利用 8-連通域分析將工程圖拆分成獨立的子圖形元件。
    同時過濾掉:
      - 位於 Logo 區域內的元件
      - 面積大於 max_bbox_ratio 的超大元件 (排除圖紙外部框架線)
    """
    H, W = binary.shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    if num_labels <= 1:
        return []
        
    # 收集連通元件 (排除背景 label 0)
    candidates = []
    for i in range(1, num_labels):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        candidates.append({
            "idx": i,
            "bbox": (x, y, x + w, y + h),
            "area": w * h,
            "pixel_area": area
        })
        
    # Logo 元件過濾
    filtered_idx = set()
    if remove_logo_cfg:
        logo_boxes = find_logo_regions(binary, template_path=logo_template_path, mask_region=logo_mask_region)
        for cand in candidates:
            cx1, cy1, cx2, cy2 = cand["bbox"]
            center_x = (cx1 + cx2) / 2
            center_y = (cy1 + cy2) / 2
            for lx1, ly1, lx2, ly2 in logo_boxes:
                if lx1 <= center_x <= lx2 and ly1 <= center_y <= ly2:
                    filtered_idx.add(cand["idx"])
                    break
                    
    # 尺寸與框架線過濾
    total_area = H * W
    valid_candidates = []
    for cand in candidates:
        if cand["idx"] in filtered_idx:
            continue
        ratio = cand["area"] / total_area
        if ratio > max_bbox_ratio:
            continue
        if cand["area"] < min_bbox_area:
            continue
        valid_candidates.append(cand)
        
    # 依 bounding box 面積降序排序，取前 top_n 個
    valid_candidates.sort(key=lambda x: x["area"], reverse=True)
    if top_n > 0:
        valid_candidates = valid_candidates[:top_n]
        
    # 裁剪與還原白底黑線子圖
    results = []
    for cand in valid_candidates:
        x1, y1, x2, y2 = cand["bbox"]
        px1 = max(0, x1 - padding)
        py1 = max(0, y1 - padding)
        px2 = min(W, x2 + padding)
        py2 = min(H, y2 + padding)
        
        # 提取二值掩膜對應的像素
        crop_mask = (labels[py1:py2, px1:px2] == cand["idx"]).astype(np.uint8) * 255
        
        # 還原為白底黑線格式 (255 為底, 元件像素為 0)
        crop_inverted = 255 - crop_mask
        
        # 外圍添加純白邊框
        padded_crop = cv2.copyMakeBorder(
            crop_inverted, padding, padding, padding, padding,
            cv2.BORDER_CONSTANT, value=255
        )
        
        results.append({
            "crop": padded_crop,
            "bbox": (px1, py1, px2, py2),
            "area": cand["area"]
        })
        
    return results

# ==========================================
# CELL 4 (CODE)
# ==========================================
def preprocess_dataset(
    input_dir: str,
    output_dir: str,
    use_connected_components: bool,
    remove_gifu_logo: bool,
    config: Config
):
    """
    批次預處理資料集入口：
    - 遍歷 input_dir 中的所有工程圖圖片。
    - 若啟用連通元件，則導出元件子圖 comp_*.png。
    - 若未啟用，則僅做 Gifu Logo 擦除，導出整圖 full_000.png。
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    img_exts = {".png", "*.jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    image_files = []
    for dp, _, fns in os.walk(input_path):
        for fn in fns:
            if Path(fn).suffix.lower() in img_exts:
                image_files.append(Path(dp) / fn)
                
    print(f"[前處理] 開始處理目錄: {input_path.name} | 共 {len(image_files)} 張圖片")
    
    success_count = 0
    for img_path in image_files:
        try:
            # 保留原本相對目錄結構
            rel_path = img_path.relative_to(input_path)
            out_subdir = output_path / rel_path.parent / img_path.stem
            out_subdir.mkdir(parents=True, exist_ok=True)
            
            gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
                
            if use_connected_components:
                # Otsu 二值化反轉
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                crops = discover_components(
                    binary,
                    top_n=config.top_n,
                    max_bbox_ratio=config.max_bbox_ratio,
                    min_bbox_area=config.min_bbox_area,
                    padding=config.padding,
                    remove_logo_cfg=remove_gifu_logo,
                    logo_template_path=config.logo_template_path,
                    logo_mask_region=config.logo_mask_region
                )
                
                if not crops:
                    # 備用機制：若連通元件分析未找到子圖，保存整張擦除 Logo 後的圖
                    clean_gray = remove_logo(gray, config.logo_template_path, config.logo_mask_region) if remove_gifu_logo else gray
                    padded_gray = cv2.copyMakeBorder(clean_gray, config.padding, config.padding, config.padding, config.padding, cv2.BORDER_CONSTANT, value=255)
                    cv2.imwrite(str(out_subdir / "comp_000.png"), padded_gray)
                else:
                    for i, crop_dict in enumerate(crops):
                        cv2.imwrite(str(out_subdir / f"comp_{i:03d}.png"), crop_dict["crop"])
            else:
                # 整圖模式：去 Logo 並 padding 輸出
                clean_gray = remove_logo(gray, config.logo_template_path, config.logo_mask_region) if remove_gifu_logo else gray
                padded_gray = cv2.copyMakeBorder(clean_gray, config.padding, config.padding, config.padding, config.padding, cv2.BORDER_CONSTANT, value=255)
                cv2.imwrite(str(out_subdir / "full_000.png"), padded_gray)
                
            success_count += 1
        except Exception as e:
            print(f"   [失敗] 處理 {img_path.name} 錯誤: {e}")
            
    print(f"[前處理] 完成。成功處理 {success_count}/{len(image_files)} 張原始影像。")

# 學理備註: 為了避免大資料集 T_large 重複前處理的開銷，我們一般在實驗正式開始前一鍵跑完並快取至 dataset_v2 下的分割區。
# 以下指令可以對小資料集種子目錄進行預覽前處理測試:
preprocess_dataset("data/吉輔提供資料", "temp_outputs/test_prep", use_connected_components=True, remove_gifu_logo=True, config=cfg)

# ==========================================
# CELL 5 (MARKDOWN)
# ==========================================
# ## 🚀 資料預處理與建置管線 (Data Preparation Pipeline)
# 
# 本節執行從原始 PDF 壓縮檔解壓縮、PDF 轉灰階 PNG、影像前處理與人工篩選並建置資料集分割的完整管線。

# ==========================================
# CELL 6 (CODE)
# ==========================================
# ========================================================
# Step 1: ZIP 解壓縮
# ========================================================
print("="*60)
print("Step 1: 壓縮檔解壓縮")
print("="*60)

# 1.1 解壓無標籤訓練資料
print("解壓無標籤訓練資料...")
extract_archive(
    archive_path=cfg.raw_zip_path,
    output_dir=cfg.raw_pdf_dir,
    skip=cfg.skip_extraction,
)

# 1.2 解壓標註評估資料
if cfg.labeled_zip_path:
    print("解壓標註評估資料...")
    extract_archive(
        archive_path=cfg.labeled_zip_path,
        output_dir=cfg.raw_labeled_pdf_dir,
        skip=cfg.skip_labeled_extraction,
)


# ==========================================
# CELL 7 (CODE)
# ==========================================
# ========================================================
# Step 2: PDF → Image 轉換
# ========================================================
print("="*60)
print("Step 2: PDF → Image 轉換")
print("="*60)

# 2.1 轉換無標籤 PDF
if not cfg.skip_pdf_conversion:
    print("轉換無標籤 PDF...")
    convert_pdfs_to_images(
        pdf_dir=cfg.raw_pdf_dir,
        output_dir=cfg.converted_image_dir,
        dpi=cfg.pdf_dpi,
        max_workers=cfg.pdf_max_workers,
        skip=cfg.skip_pdf_conversion,
        preserve_structure=False,
    )
else:
    print("跳過無標籤 PDF 轉換 (skip_pdf_conversion=True)")

# 2.2 轉換標註 PDF
if Path(cfg.raw_labeled_pdf_dir).exists():
    if not cfg.skip_labeled_pdf_conversion:
        print("開始轉換標註 PDF (保留結構)...")
        convert_pdfs_to_images(
            pdf_dir=cfg.raw_labeled_pdf_dir,
            output_dir=cfg.converted_labeled_image_dir,
            dpi=cfg.pdf_dpi,
            max_workers=cfg.pdf_max_workers,
            skip=cfg.skip_labeled_pdf_conversion,
            preserve_structure=True,
        )
    else:
        print("跳過標註 PDF 轉換 (skip_labeled_pdf_conversion=True)")
else:
    print(f"標註 PDF 目錄不存在，跳過: {cfg.raw_labeled_pdf_dir}")


# ==========================================
# CELL 8 (CODE)
# ==========================================
# ========================================================
# Step 3: 影像前處理（連通元件分析與 Logo 移除）
# ========================================================
print("="*60)
print("Step 3: 影像前處理")
print("="*60)

# 3.1 訓練集影像前處理 (無標籤)
if Path(cfg.converted_image_dir).exists():
    prep_cfg = PreprocessConfig(
        input_dir=cfg.converted_image_dir,
        output_root=cfg.preprocessed_image_dir,
        max_workers=cfg.preprocess_max_workers,
        top_n=cfg.top_n,
        max_bbox_ratio=cfg.max_bbox_ratio,
        min_bbox_area=cfg.min_bbox_area,
        padding=cfg.padding,
        remove_gifu_logo=cfg.remove_gifu_logo,
        logo_template_path=cfg.logo_template_path,
        logo_mask_region=cfg.logo_mask_region,
    )
    preprocess_images(prep_cfg, skip=cfg.skip_preprocessing)
else:
    print(f"警告: 無標籤影像目錄不存在 {cfg.converted_image_dir}")

# 3.2 評估集影像前處理 (標註)
if Path(cfg.converted_labeled_image_dir).exists():
    labeled_prep_cfg = PreprocessConfig(
        input_dir=cfg.converted_labeled_image_dir,
        output_root=cfg.preprocessed_labeled_image_dir,
        max_workers=cfg.preprocess_max_workers,
        top_n=cfg.top_n,
        max_bbox_ratio=cfg.max_bbox_ratio,
        min_bbox_area=cfg.min_bbox_area,
        padding=cfg.padding,
        remove_gifu_logo=cfg.remove_gifu_logo,
        logo_template_path=cfg.logo_template_path,
        logo_mask_region=cfg.logo_mask_region,
    )
    preprocess_images(labeled_prep_cfg, skip=cfg.skip_labeled_preprocessing)
else:
    print(f"警告: 標籤影像目錄不存在 {cfg.converted_labeled_image_dir}")


# ==========================================
# CELL 9 (CODE)
# ==========================================
# ========================================================
# Step 4: 專家篩選與資料集分割建置
# ========================================================
print("="*60)
print("Step 4: 專家篩選與資料集分割建置")
print("="*60)

da_dir = Path(cfg.converted_labeled_image_dir)
db_dir = Path(cfg.converted_image_dir)
out_root = Path(cfg.dataset_root)

# 掃描有標籤與無標籤影像
all_da = sorted([p for p in da_dir.rglob("*") if p.is_file() and p.suffix.lower() in _IMG_EXTS])
all_db = sorted([p for p in db_dir.rglob("*") if p.is_file() and p.suffix.lower() in _IMG_EXTS])

if not all_da:
    print(f"錯誤: 找不到任何有標籤影像，請確認路徑 {cfg.converted_labeled_image_dir}")
else:
    seeds = sample_query_seeds(cfg.converted_labeled_image_dir, n_seeds=cfg.n_seeds, seed=cfg.base_seed, unlabeled_dir=cfg.converted_image_dir)
    
    # 2. 自動偵測與建立類別對照表以進行自動備用篩選 (Programmatic Fallback)
    class_to_images = {}
    for p in all_da:
        class_name = p.parent.name
        class_to_images.setdefault(class_name, []).append(p)
    
    programmatic_gts = programmatic_select_gts(seeds, class_to_images, max_gts=15, seed=cfg.base_seed)
    
    # 3. 提取特徵並建置相似特徵候選池 (使用 ImageNet 預訓練骨幹)
    checkpoint_to_use = None
    try:
        feats, paths = extract_features(
            all_da + all_db, 
            checkpoint_path=checkpoint_to_use, 
            img_size=cfg.img_size, 
            device=device
        )
        candidate_pools = find_candidate_pools(seeds, feats, paths, top_k=30)
    except Exception as e:
        print(f"特徵提取失敗，退化為自動目錄匹配: {e}")
        candidate_pools = {str(s): [] for s in seeds}

    # 4. 啟動互動式專家篩選介面
    interactive_expert_filter_notebook(
        seeds=seeds,
        candidate_pools=candidate_pools,
        programmatic_gts=programmatic_gts,
        output_root=cfg.dataset_root,
        all_da=all_da,
        all_db=all_db,
        n_distractors=cfg.n_distractors,
        split_ratio=cfg.split_ratio,
        seed=cfg.base_seed
    )


# ==========================================
# CELL 10 (CODE)
# ==========================================
def get_workspace_dir() -> Path:
    curr = Path.cwd().resolve()
    for parent in [curr] + list(curr.parents):
        if (parent / "dataset_v2").exists() or (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    if os.path.exists("/workspace"):
        return Path("/workspace")
    return Path("/home/master-user/Desktop/Engineering_Image_Retrieval_System_Training")


def resolve_path(f: Path) -> Path:
    """
    遞迴解析符號連結，並動態映射本機與容器之間的工作目錄，避免 FileNotFoundError。
    """
    workspace_dir = get_workspace_dir()
    host_workspace = "/home/master-user/Desktop/Engineering_Image_Retrieval_System_Training"
    container_workspace = "/workspace"
    if f.is_symlink():
        target_str = os.readlink(f)
        if target_str.startswith(container_workspace):
            target_str = target_str.replace(container_workspace, str(workspace_dir), 1)
        elif target_str.startswith(host_workspace):
            target_str = target_str.replace(host_workspace, str(workspace_dir), 1)
        target = Path(target_str)
        if not target.is_absolute():
            target = Path(os.path.normpath(f.parent / target))
        return resolve_path(target)
    else:
        f_str = str(f)
        if f_str.startswith(container_workspace):
            f = Path(f_str.replace(container_workspace, str(workspace_dir), 1))
        elif f_str.startswith(host_workspace):
            f = Path(f_str.replace(host_workspace, str(workspace_dir), 1))
        return f

class Letterbox:
    """
    保持圖片長寬比例 (Aspect Ratio) 的等比例縮放。
    在空白處填充白色 (255) 構成正方形 (Square Image)，避免扭曲形狀損害特徵表達。
    """
    def __init__(self, size: int, fill: int = 255) -> None:
        self.size = size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img.resize((self.size, self.size), Image.Resampling.BILINEAR)

        scale = self.size / max(w, h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))

        img = img.resize((nw, nh), Image.Resampling.BILINEAR)
        new_img = Image.new(img.mode, (self.size, self.size), self.fill)
        new_img.paste(img, ((self.size - nw) // 2, (self.size - nh) // 2))
        return new_img


class TrainingDataset(Dataset):
    """
    訓練集資料載入器。
    - Baseline / 實驗 B：加載去 Logo 後的整張原始影像 (T.Resize)
    - 實驗 A：加載已前處理好的連通域子圖 (Letterbox)
    """
    def __init__(self, dataset_path: str, experiment_type: str = "Exp_A", img_size: int = 256, in_channels: int = 1, offline_aug: bool = False, num_versions: int = 20):
        self.dataset_path = Path(dataset_path).resolve()
        self.experiment_type = experiment_type
        self.img_size = img_size
        self.in_channels = in_channels
        self.mode = "L" if in_channels == 1 else "RGB"
        self.offline_aug = offline_aug
        self.num_versions = num_versions
        
        # 定義離線資料增強的根目錄
        workspace_dir = get_workspace_dir()
        self.augmented_root = workspace_dir / "data" / "augmented_images" / f"{self.experiment_type}_{getattr(cfg, 'dataset_type', 'T_small')}"
        
        # 預先定義帶有歸一化的 transform 用於離線模式
        transform_ops_with_norm = []
        if self.experiment_type == "Exp_A":
            transform_ops_with_norm.append(Letterbox(self.img_size))
        else:
            transform_ops_with_norm.append(T.Resize((self.img_size, self.img_size), interpolation=T.InterpolationMode.BILINEAR))
        transform_ops_with_norm.extend([
            T.ToTensor(),
            T.Normalize(mean=[0.0394] * self.in_channels, std=[0.1752] * self.in_channels)
        ])
        self.transform_with_norm = T.Compose(transform_ops_with_norm)
        
        # 1. 掃描符號連結 (Symlinks)
        symlink_files = sorted(list(self.dataset_path.glob("*.png")))
        if not symlink_files:
            symlink_files = sorted(list(self.dataset_path.rglob("*.png")))
            
        # 2. 定義路徑，排除非本機映射造成的 FileNotFoundError
        self.image_paths = []
        workspace_dir = get_workspace_dir()
        preprocessed_root = workspace_dir / "data" / "preprocessed_images"
        
        for f in symlink_files:
            target = resolve_path(f)
            if self.experiment_type in ["Baseline", "Exp_B"]:
                self.image_paths.append(target)
            elif self.experiment_type == "Exp_A":
                stem = f.stem
                comp_dir = preprocessed_root / stem
                if comp_dir.exists():
                    comp_images = sorted(list(comp_dir.glob("comp_*.png")))
                    if comp_images:
                        self.image_paths.extend(comp_images)
                    else:
                        self.image_paths.append(target)
                else:
                    self.image_paths.append(target)
                    
        print(f"   [Dataloader] TrainingDataset 初始化完成: 總張數 = {len(self.image_paths)}")
        
        # 3. CPU 輕量級前處理 (縮放與轉張量)，複雜增強延後在 GPU 完成
        transform_ops = []
        if self.experiment_type == "Exp_A":
            transform_ops.append(Letterbox(self.img_size))
        else:
            transform_ops.append(T.Resize((self.img_size, self.img_size), interpolation=T.InterpolationMode.BILINEAR))
        transform_ops.append(T.ToTensor())
        self.transform = T.Compose(transform_ops)
            
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        stem = img_path.stem
        
        if self.offline_aug:
            # 隨機選擇兩個不同的增強版本載入
            comp_dir = self.augmented_root / stem
            if comp_dir.exists():
                v_ids = random.sample(range(self.num_versions), 2)
                path_1 = comp_dir / f"aug_{v_ids[0]:03d}.png"
                path_2 = comp_dir / f"aug_{v_ids[1]:03d}.png"
                
                if path_1.exists() and path_2.exists():
                    img1 = Image.open(path_1).convert(self.mode)
                    img2 = Image.open(path_2).convert(self.mode)
                    t = T.Compose([
                        T.ToTensor(),
                        T.Normalize(mean=[0.0394] * self.in_channels, std=[0.1752] * self.in_channels)
                    ])
                    return t(img1), t(img2)
            
            # 若沒找到離線增強檔案，則退化為複製同一張圖 (Baseline)
            img = Image.open(img_path).convert(self.mode)
            v = self.transform_with_norm(img)
            return v, v
        else:
            img = Image.open(img_path).convert(self.mode)
            return self.transform(img)


class ValidationDataset(Dataset):
    """
    驗證集資料載入器。
    - 一次性載入 RAM 快取：徹底解決測試時多次讀取磁碟產生的 I/O 瓶頸，提升 Macro-mAP 評估速度。
    - 類別劃分：group_000..049 對應標籤 0..49；distractor_* 對應背景干擾項 -1。
    """
    def __init__(self, dataset_path: str, img_size: int = 256, in_channels: int = 1, use_preprocessing: bool = True):
        self.dataset_path = Path(dataset_path).resolve()
        self.img_size = img_size
        self.in_channels = in_channels
        self.mode = "L" if in_channels == 1 else "RGB"
        
        # 評估採用標記後的全圖前處理標準 (歸一化使用整體統計值)
        transform_ops = []
        if use_preprocessing:
            transform_ops.append(Letterbox(self.img_size))
        else:
            transform_ops.append(T.Resize((self.img_size, self.img_size), interpolation=T.InterpolationMode.BILINEAR))
            
        mean = [0.0394] if in_channels == 1 else [0.0394] * 3
        std = [0.1752] if in_channels == 1 else [0.1752] * 3
        
        transform_ops.extend([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
        self.transform = T.Compose(transform_ops)
        
        self.image_paths = []
        self.labels = []
        
        # 遍歷資料庫劃分標籤
        for folder in sorted(list(self.dataset_path.iterdir())):
            if not folder.is_dir():
                continue
            name = folder.name
            if name.startswith("group_"):
                label = int(name.split("_")[1])
            elif name.startswith("distractor_"):
                label = -1
            else:
                continue
                
            for img_file in folder.glob("*.png"):
                self.image_paths.append(resolve_path(img_file))
                self.labels.append(label)
                
        print(f"   [Dataloader] ValidationDataset 載入影像數量: {len(self.image_paths)}")
        
        # 多線程加載圖片至記憶體
        import concurrent.futures
        self.cached_tensors = [None] * len(self.image_paths)
        print("   [I/O 優化] 正在快取驗證集圖片至記憶體 (RAM Caching)... ")
        max_workers = min(16, os.cpu_count() or 4)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._load_and_transform, idx): idx
                for idx in range(len(self.image_paths))
            }
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                self.cached_tensors[idx] = future.result()
        print("   [I/O 優化] 記憶體快取完成！評估讀取延遲降至接近 0s。")
        
    def _load_and_transform(self, idx: int) -> torch.Tensor:
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert(self.mode)
        return self.transform(img)
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        return self.cached_tensors[idx], self.labels[idx]


# ==========================================
# CELL 11 (CODE)
# ==========================================
class GPUAugmentation(nn.Module):
    """
    GPU 批次資料擴增模組 (GPU Batch Augmentation)。
    學理設計:
        - 把隨機幾何變換與翻轉放於 GPU 完成，降低 CPU workload 並加速多倍。
        - 二合一拼裝 (same_on_batch=False) 來處理整個 batch，減少 kernel launch 耗時。
        - `Baseline`: 僅進行歸一化，輸出兩個完全一致的 View。
        - `Exp A / B`: 兩視角隨機進行幾何隨機裁切與旋轉，符合自監督正樣本建置原則。
    """
    def __init__(
        self, 
        img_size: int = 256, 
        use_augmentation: bool = True, 
        in_channels: int = 1
    ):
        super().__init__()
        self.img_size = img_size
        self.use_augmentation = use_augmentation
        self.in_channels = in_channels
        
        # 資料集整體反轉後之特徵分布統計
        self._mean = torch.tensor([0.0394] * in_channels)
        self._std = torch.tensor([0.1752] * in_channels)
        
        self._has_kornia = "kornia" in sys.modules
        if self._has_kornia:
            self._aug = self._build_aug() 
        else:
            self._aug = None
            
    def _build_aug(self) -> nn.Module:
        from kornia.constants import Resample
        
        mean = self._mean
        std = self._std
        
        if self.use_augmentation:
            return K.AugmentationSequential(
                # 1. 工程圖反轉 (使背景為 0，線條為 1，加速收斂)
                K.RandomInvert(p=0.5, same_on_batch=False),
                # 2. 幾何旋轉/翻轉
                K.RandomHorizontalFlip(p=0.5, same_on_batch=False),
                K.RandomVerticalFlip(p=0.5, same_on_batch=False),
                K.RandomAffine(
                    degrees=15.0,
                    translate=(0.1, 0.1),
                    resample=Resample.BILINEAR.name,
                    padding_mode='zeros',
                    p=0.7,
                    same_on_batch=False
                ),
                # 3. 仿照官方 SimSiam 文獻的隨機裁剪
                K.RandomResizedCrop(
                    size=(self.img_size, self.img_size),
                    resample=Resample.BILINEAR.name,
                    align_corners=None,
                    p=0.5,
                    same_on_batch=False
                ),
                # 4. 正規化
                K.Normalize(mean=mean, std=std),
                data_keys=["input"],
            )
        else:
            # Baseline 僅做 Resize + Normalize
            return K.AugmentationSequential(
                K.Resize((self.img_size, self.img_size), resample=Resample.BILINEAR.name),
                K.Normalize(mean=mean, std=std),
                data_keys=["input"],
            )
            
    def create_views(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        一鍵輸出雙重視角。
        若為 Baseline (use_augmentation=False)，兩視角內容完全相同。
        """
        if self._aug is not None:
            # 批次串接 2B 操作降低 Kernel launch overhead
            b = x.shape[0]
            x2 = torch.cat([x, x], dim=0)
            out = self._aug(x2)
            v1, v2 = out[:b], out[b:]
        else:
            v1 = self._manual_normalize(x)
            v2 = v1
        return v1, v2
        
    def _manual_normalize(self, x: torch.Tensor) -> torch.Tensor:
        x_resized = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        mean = self._mean.to(x.device).view(1, -1, 1, 1)
        std = self._std.to(x.device).view(1, -1, 1, 1)
        return (x_resized - mean) / std
        
    def to(self, *args, **kwargs) -> "GPUAugmentation":
        super().to(*args, **kwargs)
        self._mean = self._mean.to(*args, **kwargs)
        self._std = self._std.to(*args, **kwargs)
        if self._aug is not None:
            self._aug = self._aug.to(*args, **kwargs)
        return self

# ==========================================
# CELL 12 (CODE)
# ==========================================
class SimSiam(nn.Module):
    """
    工程圖檢索專用 SimSiam 孿生架構。
    設計原理完全遵循官方文獻最佳實踐：
      - Backbone: ResNet18 (特徵維度 512)
      - Projector (3層 MLP): 512 -> 512 -> 512 -> 2048。所有層套用 BN，輸出層不使用 ReLU。
      - Predictor (2層 MLP): 2048 -> 128 -> 2048 (Bottleneck)。隱藏層套用 BN+ReLU，輸出層無 BN/ReLU。
    """
    def __init__(
        self,
        backbone: str = "resnet18",
        proj_dim: int = 2048,
        proj_hidden: int = 512,
        pred_hidden: int = 128,
        pretrained: bool = True,
        in_channels: int = 1
    ):
        super().__init__()
        
        # 1. 載入骨幹網絡
        if backbone == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        else: 
            raise NotImplementedError("僅支援 ResNet18 Backbone")
            
        # 修改第一層卷積層以適應單通道灰階輸入
        if in_channels != 3:
            old_conv = net.conv1
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            # 將預訓練的 3 通道權重平均分配到單通道，以保存遷移學習特徵
            if pretrained and in_channels == 1:
                with torch.no_grad():
                    new_conv.weight[:] = old_conv.weight.sum(dim=1, keepdim=True) / 3.0
            net.conv1 = new_conv
            
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()  # 移除最終全連接分類層
        self.backbone = net
        
        # 2. 投影層 (Projector) - 3層 MLP
        # 全層均有 BN；輸出層無 ReLU
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, proj_hidden, bias=False),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(inplace=True),
            # nn.Linear(proj_hidden, proj_hidden, bias=False),
            # nn.BatchNorm1d(proj_hidden),
            # nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim, affine=True)
        )
        
        # 3. 預測層 (Predictor) - 2層 MLP Bottleneck 結構
        # 隱藏層含 BN+ReLU；輸出層不含 BN 與 ReLU
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_hidden, bias=False),
            nn.BatchNorm1d(pred_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(pred_hidden, proj_dim)
        )
        
        # 4. 對新增的 MLP 線性層進行截斷常態分布權重初始化 (trunc_normal_)
        self._init_weights()
        
    def _init_weights(self):
        for m in list(self.projector.modules()) + list(self.predictor.modules()):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f1 = self.backbone(x1)
        f2 = self.backbone(x2)
        
        z1 = self.projector(f1)
        z2 = self.projector(f2)
        
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        
        # 輸出 detach 目標向量 (Stop-gradient 機制，避免崩塌)
        return p1, p2, z1.detach(), z2.detach()


def D(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    負餘弦相似度損失函數 (Negative Cosine Similarity Loss)。
    """
    p_norm = F.normalize(p, dim=1)
    z_norm = F.normalize(z, dim=1)
    return -(p_norm * z_norm).sum(dim=1).mean()

# ==========================================
# CELL 13 (CODE)
# ==========================================
def compute_retrieval_metrics(features: torch.Tensor, labels: torch.Tensor) -> dict:
    """
    使用全向量化計算加速 Leave-One-Out (LOO) 特徵相似檢索指標。
    - Macro-mAP: 排除各組樣本數量偏差之干擾項指標 (對 50 個 shape groups 各自求組內 mAP 後再求算術平均)。
    - AP(q): 單個 Query 精準度積分。
    - Margin: 組內餘弦相似度與組外餘弦相似度之差距。
    """
    # 強制轉為 float32 以防半精度 (float16) 矩陣相乘與索引溢出
    features = features.float()
    labels = labels.to(features.device)
    device = features.device
    N = features.shape[0]
    
    # 1. 歸一化與餘弦相似度矩陣 [N, N]
    feat_norm = F.normalize(features, dim=1)
    sim_matrix = feat_norm @ feat_norm.T
    
    # 2. 建置樣式遮罩
    labels_unsqueezed = labels.unsqueeze(0)
    labels_equal = (labels.unsqueeze(1) == labels_unsqueezed)
    self_mask = torch.eye(N, dtype=torch.bool, device=device)
    
    # 僅對非干擾項 (Label >= 0) 的查詢種子進行計算
    shape_group_mask = (labels >= 0)
    shape_group_indices = torch.where(shape_group_mask)[0]
    
    # 同組 (Intra-class)
    intra_mask = labels_equal & ~self_mask
    intra_mask[~shape_group_mask, :] = False
    intra_mask[:, ~shape_group_mask] = False
    
    # 異組 (Inter-class)
    inter_mask = ~labels_equal
    inter_mask[~shape_group_mask, :] = False
    inter_mask[:, ~shape_group_mask] = False
    
    # 計算相似度特徵差距
    iacs = sim_matrix[intra_mask].mean().item() if intra_mask.any() else 0.0
    inter = sim_matrix[inter_mask].mean().item() if inter_mask.any() else 0.0
    margin = iacs - inter
    
    # 3. GPU 向量化 LOO 檢索計算
    sim_matrix_masked = sim_matrix.clone()
    sim_matrix_masked.fill_diagonal_(-10000.0)  # 使用安全邊界值，防止 float16 轉換溢出
    
    # 對所有元素進行 GPU 排序並排除 self [N, N-1]
    sorted_indices = torch.argsort(sim_matrix_masked, dim=1, descending=True)[:, :-1]
    sorted_labels = labels[sorted_indices] # [N, N-1]
    
    # 判斷是否與 Query 同組
    rel = (sorted_labels == labels.unsqueeze(1)).float() # [N, N-1]
    
    # 篩選 Query 為非干擾項 (Label >= 0)
    rel_filtered = rel[shape_group_mask] # [N_queries, N-1]
    R_q_filtered = rel_filtered.sum(dim=1) # [N_queries]
    
    # 避免除以 0
    R_q_filtered_clamped = torch.clamp(R_q_filtered, min=1.0)
    
    # 計算 Top-1 與 Top-5 擊中
    top1_hits = rel_filtered[:, 0].sum().item()
    top5_hits = (rel_filtered[:, :5].sum(dim=1) > 0.0).float().sum().item()
    
    # 計算 AP
    cum_rel = torch.cumsum(rel_filtered, dim=1) # [N_queries, N-1]
    positions = torch.arange(1, N, device=device).float().unsqueeze(0) # [1, N-1]
    precision = cum_rel / positions # [N_queries, N-1]
    
    ap = (precision * rel_filtered).sum(dim=1) / R_q_filtered_clamped # [N_queries]
    ap = torch.where(R_q_filtered == 0.0, torch.zeros_like(ap), ap)
    
    ap_list = ap.cpu().tolist()
    q_indices_list = shape_group_indices.cpu().tolist()
    
    # 按組別將 AP 分組並取平均
    group_ap_sums = {}
    group_counts = {}
    for idx, q_idx in enumerate(q_indices_list):
        q_label = labels[q_idx].item()
        group_ap_sums[q_label] = group_ap_sums.get(q_label, 0.0) + ap_list[idx]
        group_counts[q_label] = group_counts.get(q_label, 0) + 1
        
    group_mAPs = []
    for g in range(50):
        if g in group_counts and group_counts[g] > 0:
            group_mAPs.append(group_ap_sums[g] / group_counts[g])
        else:
            group_mAPs.append(0.0)
            
    macro_map = sum(group_mAPs) / 50.0
    total_queries = len(q_indices_list)
    top1_accuracy = top1_hits / total_queries if total_queries > 0 else 0.0
    top5_accuracy = top5_hits / total_queries if total_queries > 0 else 0.0
    
    return {
        "macro_map": macro_map,
        "iacs": iacs,
        "inter": inter,
        "margin": margin,
        "top1_accuracy": top1_accuracy,
        "top5_accuracy": top5_accuracy
    }

# ==========================================
# CELL 14 (CODE)
# ==========================================
def calculate_collapse_std(z: torch.Tensor) -> float:
    """
    計算特徵維度標準差，用於監控是否特徵坍塌 (Collapse)。
    完美表徵空間下的標準差平均值應趨近於 1/sqrt(d)。
    """
    z_norm = F.normalize(z, dim=1)
    return z_norm.std(dim=0).mean().item()


def pregenerate_offline_augmentations(config: Config):
    """
    為所有訓練影像預先生成隨機增強影像，並儲存為壓縮 PNG。
    """
    import shutil
    workspace_dir = Path(config.workspace_dir)
    augmented_root = workspace_dir / "data" / "augmented_images" / f"{config.experiment_type}_{config.dataset_type}"
    
    # 建立輸出目錄，若存在則先清空
    if augmented_root.exists():
        shutil.rmtree(augmented_root)
    augmented_root.mkdir(parents=True, exist_ok=True)
    
    # 載入原始訓練資料集 (offline_aug=False)
    train_path = Path(config.dataset_root) / config.dataset_type / "Run_01_Seed_42" / "Component_Dataset" / "train"
    dataset = TrainingDataset(
        dataset_path=train_path,
        experiment_type=config.experiment_type,
        img_size=config.img_size,
        in_channels=config.in_channels,
        offline_aug=False
    )
    
    if len(dataset) == 0:
        print("[離線增強] 錯誤：找不到任何原始訓練影像！")
        return
        
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False
    )
    
    gpu_aug = GPUAugmentation(
        img_size=config.img_size,
        use_augmentation=(config.experiment_type != "Baseline"),
        in_channels=config.in_channels
    ).to(device)
    
    print(f"[離線增強] 開始為 {len(dataset)} 張原始影像生成 {config.num_augmented_versions} 個增強版本...")
    start_t = time.time()
    
    for version_idx in range(config.num_augmented_versions):
        img_idx = 0
        for x in loader:
            batch_size_curr = x.shape[0]
            x = x.to(device, non_blocking=True)
            
            with torch.no_grad():
                if gpu_aug.use_augmentation and gpu_aug._aug is not None:
                    # Kornia 單視角隨機增強
                    aug_x = gpu_aug._aug(x)
                else:
                    aug_x = gpu_aug._manual_normalize(x)
                
                # 反向歸一化還原為 [0, 255] uint8
                mean = gpu_aug._mean.to(device).view(1, -1, 1, 1)
                std = gpu_aug._std.to(device).view(1, -1, 1, 1)
                aug_x = aug_x * std + mean
                aug_x = torch.clamp(aug_x * 255.0, 0, 255).to(torch.uint8)
                
            aug_x_cpu = aug_x.cpu().numpy()
            for b in range(batch_size_curr):
                orig_path = dataset.image_paths[img_idx]
                stem = orig_path.stem
                
                img_dir = augmented_root / stem
                img_dir.mkdir(parents=True, exist_ok=True)
                
                # 儲存灰階圖
                img_np = aug_x_cpu[b, 0]
                cv2.imwrite(str(img_dir / f"aug_{version_idx:03d}.png"), img_np)
                
                img_idx += 1
                
    dur = time.time() - start_t
    print(f"[離線增強] 完成！共生成 {len(dataset) * config.num_augmented_versions} 張增強圖。總耗時: {dur:.2f} 秒。")


def run_training_session(config: Config):
    # 1. 線性學習率縮放
    batch_size = 32 if config.dataset_type == "T_small" else 128
    lr = config.base_lr * (batch_size / 256.0)
    
    print(f"========================================================")
    print(f" 啟動 SimSiam 自監督訓練: {config.experiment_type} ({config.dataset_type})")
    print(f"   Batch Size: {batch_size} | 縮放後學習率: {lr:.6f}")
    print(f"========================================================")
    
    # 2. 定義資料集路徑並加載
    train_path = Path(config.dataset_root) / config.dataset_type / "Run_01_Seed_42" / "Component_Dataset" / "train"
    val_path = Path(config.dataset_root) / "V"
    
    # 若啟用離線資料增強，則在訓練前執行預生成
    if config.offline_aug:
        pregenerate_offline_augmentations(config)

    train_dataset = TrainingDataset(
        dataset_path=train_path,
        experiment_type=config.experiment_type,
        img_size=config.img_size,
        in_channels=config.in_channels,
        offline_aug=config.offline_aug,
        num_versions=config.num_augmented_versions
    )
    
    val_dataset = ValidationDataset(
        dataset_path=val_path,
        img_size=config.img_size,
        in_channels=config.in_channels,
        use_preprocessing=(config.experiment_type == "Exp_A")
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=(config.num_workers > 0),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=(config.num_workers > 0)
    )
    
    # 3. 初始化模型、優化器、學習率排程與 GPU 增強
    model = SimSiam(in_channels=config.in_channels, pretrained=config.pretrained).to(device)
    
    gpu_aug = GPUAugmentation(
        img_size=config.img_size,
        use_augmentation=(config.experiment_type != "Baseline"),
        in_channels=config.in_channels
    ).to(device)
    
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or "bn" in name:
            no_decay.append(param)
        else:
            decay.append(param)
            
    optimizer = torch.optim.SGD([
        {'params': decay, 'weight_decay': config.weight_decay},
        {'params': no_decay, 'weight_decay': 0.0}
    ], lr=lr, momentum=0.9)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs
    )
    
    use_amp = (device == "cuda")
    scaler = amp.GradScaler("cuda") if use_amp else None
    
    # 建立輸出文件路徑
    out_dir = Path(config.save_dir) / f"{config.experiment_type}_{config.dataset_type}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    history = []
    best_macro_map = -1.0
    start_time = time.time()
    
    # 4. 開始訓練週期迴圈
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_std = 0.0
        batches = 0
        
        for batch in train_loader:
            if config.offline_aug:
                x1, x2 = batch
                x1 = x1.to(device, non_blocking=True)
                x2 = x2.to(device, non_blocking=True)
            else:
                x = batch.to(device, non_blocking=True)
                x1, x2 = gpu_aug.create_views(x)
            
            optimizer.zero_grad(set_to_none=True)
            
            # 混合精度 AMP 計算
            with amp.autocast(device_type=device, enabled=use_amp):
                p1, p2, z1, z2 = model(x1, x2)
                loss = 0.5 * D(p1, z2) + 0.5 * D(p2, z1)
                
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            epoch_loss += loss.item()
            with torch.no_grad():
                # 特徵標準差累計
                batch_std = (calculate_collapse_std(z1) + calculate_collapse_std(z2)) / 2.0
                epoch_std += batch_std
            batches += 1
            
        scheduler.step()
        
        train_loss = epoch_loss / batches
        train_std = epoch_std / batches
        if train_std < config.collapse_warning_threshold:
            print(f"   [⚠️ 警告] Epoch {epoch:03d}: 特徵標準差 ({train_std:.4f}) 低於臨界值 {config.collapse_warning_threshold}，模型可能正在發生特徵崩塌 (Representation Collapse)！")
        
        val_macro_map = None
        val_margin = None
        val_top1 = None
        
        # 驗證步驟
        if epoch % config.eval_freq == 0 or epoch == config.epochs:
            model.eval()
            all_features = []
            all_labels = []
            
            with torch.no_grad():
                for val_x, val_y in val_loader:
                    val_x = val_x.to(device, non_blocking=True)
                    with amp.autocast(device_type=device, enabled=use_amp):
                        # 依據計畫書，提取 Projector 輸出的 z 表徵進行檢索
                        features = model.projector(model.backbone(val_x))
                    all_features.append(features.cpu())
                    all_labels.append(val_y)
                    
            all_features = torch.cat(all_features, dim=0).to(device)
            all_labels = torch.cat(all_labels, dim=0).to(device)
            
            metrics = compute_retrieval_metrics(all_features, all_labels)
            
            val_macro_map = metrics["macro_map"]
            val_margin = metrics["margin"]
            val_top1 = metrics["top1_accuracy"]
            
            print(f"Epoch {epoch:03d}/{config.epochs} | Loss: {train_loss:.4f} | Std: {train_std:.4f} | Macro-mAP: {val_macro_map:.4f} | Top-1 Acc: {val_top1:.4f}")
            
            # 儲存 checkpoints
            chkpt_dir = out_dir / "checkpoints"
            chkpt_dir.mkdir(exist_ok=True)
            
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "macro_map": val_macro_map
            }
            torch.save(checkpoint, chkpt_dir / "latest.pth")
            
            if val_macro_map > best_macro_map:
                best_macro_map = val_macro_map
                torch.save(checkpoint, chkpt_dir / "best.pth")
                print(f"   --> [最優檢查點] 已更新最優權重，最高 Macro-mAP: {best_macro_map:.4f}")
        else:
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:03d}/{config.epochs} | Loss: {train_loss:.4f} | Std: {train_std:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
                
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_std": train_std,
            "val_macro_map": val_macro_map,
            "val_margin": val_margin,
            "val_top1": val_top1
        })
        
    total_time_min = (time.time() - start_time) / 60.0
    print(f"========================================================")
    print(f" 訓練完成。總耗時: {total_time_min:.2f} 分鐘 | 最佳 Macro-mAP: {best_macro_map:.4f}")
    print(f"========================================================")
    
    # 輸出歷史紀錄
    df = pd.DataFrame(history)
    df.to_csv(out_dir / "training_log.csv", index=False)
    return df, best_macro_map


# ==========================================
# CELL 15 (CODE)
# ==========================================
def plot_training_results(df: pd.DataFrame, experiment_name: str):
    """
    Plotly 互動式雙曲線圖，直觀顯示 Loss 衰減趨勢與 Macro-mAP 上升演進曲線。
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Loss 曲線 (主 y 軸)
    fig.add_trace(
        go.Scattergl(x=df["epoch"], y=df["train_loss"], mode="lines+markers", name="Train Loss", line=dict(color="blue")),
        secondary_y=False,
    )
    
    # 篩選非空評價資料
    eval_df = df[df["val_macro_map"].notnull()].copy()
    
    # Macro-mAP 曲線 (次 y 軸)
    fig.add_trace(
        go.Scattergl(x=eval_df["epoch"], y=eval_df["val_macro_map"].astype(float), mode="lines+markers", name="Macro-mAP", line=dict(color="green")),
        secondary_y=True,
    )
    
    # 特徵維度標準差 (次 y 軸 - 監控坍塌用)
    fig.add_trace(
        go.Scattergl(x=df["epoch"], y=df["train_std"], mode="lines", name="Feature Std (Target ~0.022)", line=dict(color="orange", dash="dash")),
        secondary_y=True,
    )
    
    fig.update_layout(
        title=f"SimSiam 自監督訓練趨勢 - {experiment_name}",
        xaxis_title="Epoch",
        template="plotly_white",
        hovermode="x unified"
    )
    
    fig.update_yaxes(title_text="Negative Cosine Similarity Loss", secondary_y=False)
    fig.update_yaxes(title_text="Metric Value", secondary_y=True)
    
    fig.show()


##########
#參數參考#
##########
"""
# --- 資料路徑設定 ---
# 動態偵測工作目錄以相容於 Docker 容器與本機主機環境
workspace_dir = get_dynamic_workspace()
dataset_root = f"{workspace_dir}/dataset_v2"
dataset_type = "T_small"      # 可選: "T_small" 或 "T_large"
experiment_type = "Exp_A"     # 可選: "Baseline", "Exp_A", "Exp_B"
img_size = 256                # 圖片尺寸 (256/512)
in_channels = 1               # 輸入通道數 (1=灰階)

# --- 模型網路架構 ---
backbone = "resnet18"
pretrained = True             # 是否載入 ImageNet 預訓練權重
proj_hidden = 512             # 投影層隱藏維度
proj_dim = 2048               # 投影層輸出維度 d
pred_hidden = 128             # 預測層瓶頸隱藏維度 (Bottleneck)

# --- SGD 優化器超參數 ---
epochs = 100
base_lr = 0.05                # 用於線性縮放規則的基礎學習率
weight_decay = 1e-4           # 權重衰減
seed = 42

# --- 特徵崩塌警告閾值 ---
collapse_warning_threshold = 0.01  # 特徵維度標準差警告閾值

# --- 硬體加速與 I/O 優化 ---
num_workers = 8               # DataLoader 線程數
pin_memory = True             # 鎖定記憶體加速 Tensor 傳輸
eval_freq = 10                # 評估 Macro-mAP 的 Epoch 頻率
save_dir = f"{workspace_dir}/outputs_v3_local"  # Checkpoint 輸出目錄

# --- 前處理超參數 ---
padding = 2                   # 裁切邊距填充
max_bbox_ratio = 0.8          # 排除大於整張圖一定比例的外接矩形元件 (圖框過濾)
min_bbox_area = 20            # 元件外接矩形最小面積
top_n = 5                     # 最多保留的前 n 個連通元件

# --- Gifu Logo 擦除參數 ---
remove_gifu_logo = True
logo_template_path = f"{workspace_dir}/data/Gifu_logo.png"
logo_mask_region = [0.0, 0.9, 0.2, 1.0] # 預設遮罩比率 (例如左下角)
"""



# ==========================================
# CELL 16 (CODE)
# ==========================================
# --- 模型網路架構 ---
backbone = "resnet18"
pretrained = True             # 是否載入 ImageNet 預訓練權重
proj_hidden = 512             # 投影層隱藏維度
proj_dim = 2048               # 投影層輸出維度 d
pred_hidden = 128             # 預測層瓶頸隱藏維度 (Bottleneck)

# batch_size = 32 if config.dataset_type == "T_small" else 128

from plotly.subplots import make_subplots
cfg.dataset_type = "T_small"
cfg.experiment_type = "Exp_B"
cfg.epochs = 100     # 100
cfg.eval_freq = 1  # 10
df_history, best_map = run_training_session(cfg)
plot_training_results(df_history, f"{cfg.experiment_type}_{cfg.dataset_type}")

# 持久化
experiment_small_B_2layerProjector = df_history, best_map
%store experiment_small_B_2layerProjector

# ==========================================
# CELL 17 (CODE)
# ==========================================
%store -r experiment_small_B_2layerProjector
df_history, best_map = experiment_small_B_2layerProjector
plot_training_results(df_history, f"{cfg.experiment_type}_{cfg.dataset_type}")

# ==========================================
# CELL 18 (CODE)
# ==========================================
# --- 執行示範 ---
# 下方程式碼將於背景執行。可手動修改 cfg 參數以進行其他 5 組對照子實驗。
from plotly.subplots import make_subplots
cfg.dataset_type = "T_small"
cfg.experiment_type = "Baseline"
cfg.epochs = 100     # 100     # 快速試跑 2 Epochs 驗證 Notebook 卡點狀況
cfg.eval_freq = 1  # 10
df_history, best_map = run_training_session(cfg)
plot_training_results(df_history, f"{cfg.experiment_type}_{cfg.dataset_type}")

# ==========================================
# CELL 19 (CODE)
# ==========================================
from plotly.subplots import make_subplots
cfg.dataset_type = "T_small"
cfg.experiment_type = "Exp_A"
cfg.epochs = 100     # 100     # 快速試跑 2 Epochs 驗證 Notebook 卡點狀況
cfg.eval_freq = 1  # 10
df_history, best_map = run_training_session(cfg)
plot_training_results(df_history, f"{cfg.experiment_type}_{cfg.dataset_type}")

# ==========================================
# CELL 20 (CODE)
# ==========================================
from plotly.subplots import make_subplots
cfg.dataset_type = "T_small"
cfg.experiment_type = "Exp_B"
cfg.epochs = 100     # 100
cfg.eval_freq = 1  # 10
df_history, best_map = run_training_session(cfg)
plot_training_results(df_history, f"{cfg.experiment_type}_{cfg.dataset_type}")

# ==========================================
# CELL 21 (CODE)
# ==========================================
from plotly.subplots import make_subplots
cfg.dataset_type = "T_large"
cfg.experiment_type = "Baseline"
cfg.epochs = 100     # 100
cfg.eval_freq = 1  # 10
df_history, best_map = run_training_session(cfg)
plot_training_results(df_history, f"{cfg.experiment_type}_{cfg.dataset_type}")

# ==========================================
# CELL 22 (CODE)
# ==========================================
from plotly.subplots import make_subplots
cfg.dataset_type = "T_large"
cfg.experiment_type = "Exp_A"
cfg.epochs = 100     # 100
cfg.eval_freq = 1  # 10
df_history, best_map = run_training_session(cfg)
plot_training_results(df_history, f"{cfg.experiment_type}_{cfg.dataset_type}")

# ==========================================
# CELL 23 (CODE)
# ==========================================
from plotly.subplots import make_subplots
cfg.dataset_type = "T_large"
cfg.experiment_type = "Exp_B"
cfg.epochs = 100     # 100
cfg.eval_freq = 1  # 10
df_history, best_map = run_training_session(cfg)
plot_training_results(df_history, f"{cfg.experiment_type}_{cfg.dataset_type}")
