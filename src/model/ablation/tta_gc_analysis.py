"""
TTA 效益評估與幾何中心 (GC) 分析實驗主程式
------------------------------------------------------------------------------
本腳本執行四個核心實驗以評估 Test-Time Augmentation (TTA) 在工程圖檢索中的效益，
並驗證「幾何中心查詢 (Query by GC)」策略的有效性。

實驗列表:
1. 總體漂移分析 (Global Drift Analysis): 測量 TTA 對 Embedding 的擾動分佈。
2. 幾何中心有效性 (Centroid Effectiveness): 驗證 GC 是否比單一 TTA 更接近原始圖 (Anchor)。
3. 個別 TTA 方法評估 (Ablation Study): 評估不同 TTA 策略的貢獻與風險。
4. 聚合力分析 (Cohesion Analysis): 偵測將 GC 拉偏的離群 TTA 策略。

Usage:
    python src/model/ablation/tta_gc_analysis.py --checkpoint lists/to/ckpt.pth --data_root data/source
"""

import sys
import os
import argparse
import random
import logging
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Any

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
import torchvision.transforms as T

# 設定專案路徑以匯入模組
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.model.simsiam2 import SimSiam
from src.model.augmentations import (
    make_inference_transform,
    TTAHorizontalFlip,
    TTAVerticalFlip, 
    TTAMultiScale,
    TTAMorphology,
    TTAGaussianNoise,
    TTARotation,    # New
    TTARotation90,  # New
    TTAColorJitter, # New
    TTAGaussianBlur,# New
    TTACLAHE,       # New
    TTAFiveCrop     # New
)
from src.model.ablation.ablation_utils import (
    monitor_resource,
    monitor_time,
    load_simsiam_model,
    l2_normalize,
    compute_centroid,
    cosine_similarity,
    validate_input
)

# 匯入 preprocessing 模組以進行資料準備
from src.pdf_to_image2 import run as pdf_to_image_run
from src.image_preprocessing3 import run_pipeline as roi_pipeline

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TTA_GC_Analysis")

# -----------------------------------------------------------------------------
# Data Preparation (資料準備)
# -----------------------------------------------------------------------------

@monitor_time
def prepare_component_images(
    source_pdf_dir: Path,
    temp_dir: Path,
    limit: int = 50  # 限制處理的PDF數量以加速實驗
) -> List[Path]:
    """
    準備實驗所需的「元件圖」。
    若 temp_dir 已存在且有資料，則直接使用；否則從 source_pdf_dir 執行萃取流程。
    
    流程: PDF -> PNG (Page) -> ROI Cleanup -> Component Crops
    
    參數:
    source_pdf_dir (Path): 原始 PDF 資料夾。
    temp_dir (Path): 存放萃取後元件圖的暫存資料夾。
    limit (int): 限制處理的 PDF 數量。
    
    返回:
    image_paths (List[Path]): 所有元件圖的路徑列表。
    """
    temp_dir = Path(temp_dir)
    images_dir = temp_dir / "components" # 最終元件圖存放處
    
    if images_dir.exists() and any(images_dir.iterdir()):
        logger.info(f"Found existing component images in {images_dir}. Skipping extraction.")
        return list(images_dir.glob("*.png"))
    
    logger.info("Starting data extraction pipeline...")
    
    # 1. PDF -> Page PNGs (Intermediate)
    pages_dir = temp_dir / "pages_png"
    if not pages_dir.exists() or not any(pages_dir.iterdir()):
        logger.info(f"Converting PDFs from {source_pdf_dir} to images...")
        try:
            pdf_to_image_run(
                root_dir=source_pdf_dir,
                output_dir=pages_dir,
                dpi=150, # 稍微高一點確保細節
                max_workers=os.cpu_count()
            )
        except Exception as e:
             logger.error(f"PDF conversion failed: {e}")
             # 如果失敗但有部分檔案，或許還能跑，先不 raise
             
    # 2. Page PNGs -> Component Crops
    images_dir.mkdir(parents=True, exist_ok=True)
    page_images = list(pages_dir.rglob("*.png"))
    
    # 取樣處理，避免太久
    if limit > 0 and len(page_images) > limit:
        random.shuffle(page_images)
        page_images = page_images[:limit]
        
    logger.info(f"Extracting components from {len(page_images)} pages...")
    
    valid_components = []
    
    for p_img in tqdm(page_images, desc="Extracting ROIs"):
        try:
            # 使用 image_preprocessing3 提取元件
            # 這裡我們稍微修改邏輯，直接調用 run_pipeline 並將結果複製到 components 資料夾
            # 為了避免路徑衝突，使用 unique name
            results = roi_pipeline(
                input_path=str(p_img),
                output_dir=str(temp_dir / "roi_temp" / p_img.stem),
                top_n=5,
                remove_largest=True, # 移除圖框
                random_count=0 # 不需要隨機排列圖
            )
            
            # results['large_dir'] 包含切出來的大元件
            large_dir = results['large_dir']
            if large_dir.exists():
                for comp_path in large_dir.glob("*.png"):
                    # 重新命名並移動
                    new_name = f"{p_img.stem}_{comp_path.name}"
                    dest = images_dir / new_name
                    shutil.move(str(comp_path), str(dest))
                    valid_components.append(dest)
                    
            # 清理 roi_temp 以節省空間
            shutil.rmtree(temp_dir / "roi_temp" / p_img.stem)
            
        except Exception as e:
            logger.warning(f"Failed to process {p_img}: {e}")
            continue
            
    logger.info(f"Data preparation complete. {len(valid_components)} components ready.")
    return valid_components

# -----------------------------------------------------------------------------
# Analysis Logic (分析邏輯)
# -----------------------------------------------------------------------------

class TTAAnalyzer:
    """TTA 實驗分析器"""
    
    def __init__(self, model: SimSiam, device: str):
        self.model = model
        self.device = device
        # 建立定錨轉換 (Resize + Normalize)
        self.anchor_transform = make_inference_transform()
        
        # 初始化 TTA 策略池
        # 每個 TTA 策略都接受 base_transform (anchor_transform)
        self.tta_strategies = {
            "s1_HorizontalFlip": TTAHorizontalFlip(self.anchor_transform),  # 水平翻轉
            "s2_VerticalFlip": TTAVerticalFlip(self.anchor_transform),  # 垂直翻轉
            # "s3_MultiScale_Standard": TTAMultiScale(self.anchor_transform, scales=[0.9, 1.0, 1.1]),
            # "s4_MultiScale_Wide": TTAMultiScale(self.anchor_transform, scales=[0.5, 1.0, 1.5]),
            # "s5_FiveCrop": TTAFiveCrop(img_size=512, mean=0.5, std=0.5),

            # 旋轉
            "s6_Rotation_Small": TTARotation(self.anchor_transform, degrees=[-5, 5]),
            "s7_Rotation_Large": TTARotation(self.anchor_transform, degrees=[-15, 15]),
            "s8_Rotation_Specific": TTARotation(self.anchor_transform, degrees=[30, 60]),
            
            "s9_Rotation90": TTARotation90(self.anchor_transform),
            
            # 顏色抖動
            # "s10_ColorJitter_Subtle": TTAColorJitter(self.anchor_transform, brightness=0.1, contrast=0.1),
            # "s11_ColorJitter_Strong": TTAColorJitter(self.anchor_transform, brightness=0.4, contrast=0.4),
            
            # 模糊化
            # "s12_GaussianBlur_Mild": TTAGaussianBlur(self.anchor_transform, kernel_size=3, sigma=1.0),
            # "s13_GaussianBlur_Strong": TTAGaussianBlur(self.anchor_transform, kernel_size=9, sigma=3.0),
            
            # 形態學擾動 (Morphological Perturbations)：針對線條粗細與拓撲結構的魯棒性。
            # "s14_Morphology_k3": TTAMorphology(self.anchor_transform, kernel_size=3),
            # "s15_Morphology_k5": TTAMorphology(self.anchor_transform, kernel_size=5),
            
            # 對比度增強
            "s16_CLAHE_Default": TTACLAHE(self.anchor_transform, clip_limit=2.0, tile_grid_size=(8, 8)),
            "s4_CLAHE_HighContrast": TTACLAHE(self.anchor_transform, clip_limit=4.0, tile_grid_size=(4, 4)),
            
            # 高斯雜訊
            # "s17_GaussianNoise_Low": TTAGaussianNoise(self.anchor_transform, sigma=0.02),
            # "s18_GaussianNoise_High": TTAGaussianNoise(self.anchor_transform, sigma=0.10),
        }
        
    @torch.no_grad()
    def get_embedding(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        取得單張圖的 Embedding。
        img_tensor: [C, H, W] -> unsqueeze -> [1, C, H, W]
        """
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        # SimSiam forward returns p1, p2, z1, z2. We use backbone output 'f' implies z?
        # 根據 simsiam2.py，forward 經過 backbone -> projector -> predictor
        # 但檢索通常使用 Projector output (z) 或 Backbone output (f)。
        # SimSiam 論文中，inference 通常使用 Encoder (Backbone) output。
        # 不過 simsiam2.py 的 implement，backbone 後直接接 projector。
        # 我們為了跟 training 一致，通常取 backbone output (f) 或 projector output (z)。
        # 但 SimSiam 的 forward 回傳的是 predictor output p 和 projector output z。
        # 一般來說，SSL inference 用 backbone feature。
        # 但此處 simsiam2.py 的 forward 比較複雜。
        # 讓我們直接呼叫 backbone。
        
        features = self.model.backbone(img_tensor) # [1, 2048]
        # features = self.model.projector(features) # Optional: Use projector embedding
        
        return l2_normalize(features).cpu()

    def run_analysis(self, image_paths: List[Path]) -> Dict[str, Any]:
        """
        執行所有實驗。
        """
        results = {
            "exp1_global_drift": [],
            "exp2_centroid_effectiveness": [],
            "exp3_ablation": {k: [] for k in self.tta_strategies.keys()},
            "exp4_cohesion": {k: [] for k in self.tta_strategies.keys()}
        }
        
        for img_path in tqdm(image_paths, desc="Analyzing Images"):
            try:
                img = Image.open(img_path).convert("L") # 確保灰階
                
                # 1. 產生 Anchor (Original) Embedding
                # Anchor 必須經過 inference transform
                # anchor_tensor = self.anchor_transform(img) # Error: anchor_transform is Compose
                # make_inference_transform returns Compose.
                # 但 Compose 直接吃 PIL return Tensor.
                anchor_tensor = self.anchor_transform(img)
                v_orig = self.get_embedding(anchor_tensor)
                
                # 2. 收集所有 TTA Embeddings
                all_aug_vectors = []
                tta_vectors_map = {} # {strategy_name: [v1, v2...]}
                
                for name, tta_func in self.tta_strategies.items():
                    # 各 TTA 回傳 list[Tensor] (包含 original + variations)
                    # 我們只取 variations? 大部分 TTA class definition 是 [orig, aug1...]
                    # 為了純粹比較「變化」，我們應該取其第二個以後的元素作為 v_aug
                    # 但像 MultiScale 回傳多個。
                    # 策略：將 TTA 產生的*所有*視角視為一個集合 (包含它產生的 orig 如果有的話)
                    # 但為了定義 "TTA Vector"，我們取該 TTA 產生的所有 vectors 的 GC? 
                    # 或是個別 vector?
                    # 根據實驗設計：v_aug_i 是 "經過第 i 種 TTA 方法後的 Embedding"。
                    # 簡單起見，我們將 TTA 產生的所有變體 (剔除完全未變的原始圖如果能識別) 都加入。
                    # 這裡假設 TTA class 如 TTAHorizontalFlip 回傳 [orig, flip]。
                    # 我們只取 flip (index 1)。 MultiScale [orig, s1, s2] -> 取 s1, s2。
                    
                    aug_tensors = tta_func(img)
                    
                    # 提取特徵
                    vecs = []
                    for t in aug_tensors:
                        v = self.get_embedding(t)
                        vecs.append(v)
                        
                    # 儲存
                    # 假設 index 0 總是類原始圖 (基於 transform 實作)，我們只拿 index 1:
                    # 但若 TTA 實作有變，需小心。暫且全拿計算 GC，但在比較單一 TTA 效益時需注意。
                    # 修正：Exp 3 個別 TTA 評估，應比較該 TTA 產生的變體與原始圖的距離。
                    # 為了精確，我們只取變體部分作為該策略的代表。
                    
                    strategy_vecs = vecs[1:] if len(vecs) > 1 else vecs # Skip index 0 (Original)
                    if not strategy_vecs: strategy_vecs = vecs # Fallback
                    
                    tta_vectors_map[name] = torch.cat(strategy_vecs, dim=0) # [M, D] (M=變體數)
                    all_aug_vectors.extend(strategy_vecs)

                if not all_aug_vectors:
                    continue

                # 彙整所有向量 (包含 v_orig) 計算全域 GC
                # v_orig (anchor) shape [1, D]
                # all_aug_vectors list of [1, D]
                # Combine: Anchor + All TTA Variations
                combined_vectors = torch.cat([v_orig] + all_aug_vectors, dim=0)
                v_gc = compute_centroid(combined_vectors) # [1, D]
                
                # --- Exp 1: Global Drift Analysis ---
                # 計算每個 v_aug (不分策略) 與 v_orig 的相似度
                # Flatten all TTA vecs
                all_variations = torch.cat(all_aug_vectors, dim=0)
                sims_global = cosine_similarity(v_orig, all_variations).squeeze().tolist()
                # 若只有一個值，tolist 回傳 float，需轉 list
                if isinstance(sims_global, float): sims_global = [sims_global]
                results["exp1_global_drift"].extend(sims_global)
                
                # --- Exp 2: Centroid Effectiveness ---
                # Sim(v_orig, v_gc) vs Avg(Sim(v_orig, v_aug_i))
                sim_gc = cosine_similarity(v_orig, v_gc).item()
                avg_sim_single = np.mean(sims_global) if sims_global else 0.0
                
                results["exp2_centroid_effectiveness"].append({
                    "sim_gc": sim_gc,
                    "avg_sim_single": avg_sim_single,
                    "gain": sim_gc - avg_sim_single
                })
                
                # --- Exp 3 & 4: Ablation & Cohesion ---
                for name, vecs_tensor in tta_vectors_map.items():
                    # vecs_tensor: [M, D] for this strategy
                    
                    # Exp 3: Ablation (Drift per strategy)
                    # Avg sim of these vars to v_orig
                    sims_strat = cosine_similarity(v_orig, vecs_tensor).squeeze()
                    avg_sim_strat = sims_strat.mean().item() if sims_strat.dim() > 0 else sims_strat.item()
                    results["exp3_ablation"][name].append(avg_sim_strat)
                    
                    # Exp 4: Cohesion (Distance to GC)
                    # Avg sim of these vars to v_gc
                    sims_gc_strat = cosine_similarity(v_gc, vecs_tensor).squeeze()
                    avg_sim_gc_strat = sims_gc_strat.mean().item() if sims_gc_strat.dim() > 0 else sims_gc_strat.item()
                    results["exp4_cohesion"][name].append(avg_sim_gc_strat)
                    
            except Exception as e:
                logger.error(f"Error analyzing image {img_path}: {e}")
                continue
                
        return results

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TTA & GC Analysis")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--data_root", type=str, required=True, help="Path to raw PDF dataset")
    parser.add_argument("--output_dir", type=str, default="outputs/ablation_results", help="Output directory")
    parser.add_argument("--limit_data", type=int, default=0, help="Limit number of PDFs processed (for speed)")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 載入模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # 嘗試自動載入 config.json
    ckpt_path = Path(args.checkpoint)
    config_path = ckpt_path.parent.parent / "config.json"
    arch_config = None
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                loaded_cfg = json.load(f)
                # 篩選出跟模型架構有關的參數
                arch_config = {
                    "backbone": loaded_cfg.get("backbone", "resnet18"),
                    "in_channels": loaded_cfg.get("in_channels", 1),
                    # config.json 可能沒有 proj_dim/pred_hidden，使用預設值
                }
                logger.info(f"Loaded architecture config from {config_path}: {arch_config}")
        except Exception as e:
            logger.warning(f"Failed to load config.json: {e}")

    try:
        model = load_simsiam_model(args.checkpoint, device=device, arch_config=arch_config)
    except Exception as e:
        logger.critical(f"Failed to load model: {e}")
        return

    # 2. 準備資料
    # 使用 outputs/temp_data 作為快取
    temp_data_dir = output_dir / "temp_data"
    temp_data_dir.mkdir(exist_ok=True)
    
    image_paths = prepare_component_images(
        source_pdf_dir=Path(args.data_root),
        temp_dir=temp_data_dir,
        limit=args.limit_data
    )
    
    if not image_paths:
        logger.critical("No images found for analysis.")
        return

    # 3. 執行分析
    analyzer = TTAAnalyzer(model, device)
    results = analyzer.run_analysis(image_paths)
    
    # 4. 儲存結果
    # Exp 1 list
    pd.DataFrame(results["exp1_global_drift"], columns=["similarity"]).to_csv(output_dir / "exp1_global_drift.csv", index=False)
    
    # Exp 2 dict list
    pd.DataFrame(results["exp2_centroid_effectiveness"]).to_csv(output_dir / "exp2_centroid.csv", index=False)
    
    # Exp 3 dict of lists -> DataFrame
    # 每個策略一欄
    pd.DataFrame(results["exp3_ablation"]).to_csv(output_dir / "exp3_ablation.csv", index=False)
    
    # Exp 4 dict of lists -> DataFrame
    pd.DataFrame(results["exp4_cohesion"]).to_csv(output_dir / "exp4_cohesion.csv", index=False)
    
    logger.info(f"Analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()


# uv run python src/model/ablation/tta_gc_analysis.py --checkpoint outputs/simsiam_exp_01_Run_01_Seed_42_20260130_105404/checkpoints/checkpoint_best.pth --data_root data/吉輔提供資料Clean --output_dir results/ablation
