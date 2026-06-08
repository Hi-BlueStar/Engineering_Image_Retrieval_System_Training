import os
import sys
import zipfile
from pathlib import Path
import torch

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Force v3 to be first in search path to avoid package name conflicts (e.g. src)
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
    build_dataset_v2
)

def run_pipeline():
    workspace_dir = PROJECT_ROOT
    
    # Check if directories exist and contain files with a specific extension recursively
    def has_files(directory, extension) -> bool:
        p = Path(directory)
        if not p.exists():
            return False
        # any() is lazy and returns True on first match
        return any(p.rglob(f"*{extension}"))

    # Resolve nested directories inside zip extractions or conversions
    def get_nested_dir(base_dir: str) -> str:
        path = Path(base_dir)
        if not path.exists():
            return base_dir
        # If there's exactly one subdirectory and no files at the top level
        subdirs = [p for p in path.iterdir() if p.is_dir()]
        files = [p for p in path.iterdir() if p.is_file()]
        if len(subdirs) == 1 and not files:
            return str(subdirs[0])
        return base_dir

    # 1. Config mockup (matches Config class in simsiam_training.ipynb)
    class cfg:
        raw_zip_path = f"{workspace_dir}/data/PDF.zip"
        labeled_zip_path = f"{workspace_dir}/data/吉輔提供資料.zip"
        
        raw_pdf_dir = f"{workspace_dir}/data/raw_pdfs"
        raw_labeled_pdf_dir = f"{workspace_dir}/data/raw_labeled_pdfs"
        converted_image_dir = f"{workspace_dir}/data/converted_images"
        converted_labeled_image_dir = f"{workspace_dir}/data/converted_labeled_images"
        
        pdf_dpi = 400  # 預設使用 400 DPI
        pdf_max_workers = 16
        
        preprocessed_image_dir = f"{workspace_dir}/data/preprocessed_images"
        preprocessed_labeled_image_dir = f"{workspace_dir}/data/preprocessed_labeled_images"
        preprocess_max_workers = 24
        
        # Preprocessing params
        padding = 2
        max_bbox_ratio = 0.8
        min_bbox_area = 20
        top_n = 5
        remove_gifu_logo = True
        logo_template_path = f"{workspace_dir}/data/Gifu_logo.png"
        logo_mask_region = [0.0, 0.9, 0.2, 1.0]
        
        dataset_root = f"{workspace_dir}/dataset_v2"
        dataset_type = "T_small"
        experiment_type = "Exp_A"
        img_size = 256
        in_channels = 1
        
        split_ratio = 0.8
        n_runs = 1
        base_seed = 42
        
        n_seeds = 50
        n_distractors = 500

    # Determine dynamic skip flags
    cfg.skip_extraction = has_files(cfg.raw_pdf_dir, ".pdf")
    cfg.skip_labeled_extraction = has_files(cfg.raw_labeled_pdf_dir, ".pdf")
    cfg.skip_pdf_conversion = has_files(cfg.converted_image_dir, ".png")
    cfg.skip_labeled_pdf_conversion = has_files(cfg.converted_labeled_image_dir, ".png")
    cfg.skip_preprocessing = has_files(cfg.preprocessed_image_dir, ".png")
    cfg.skip_labeled_preprocessing = has_files(cfg.preprocessed_labeled_image_dir, ".png")

    print("=" * 60)
    print("🚀 [prepare_data] 開始執行資料準備與預處理管線")
    print("=" * 60)

    # Step 1: ZIP 解壓縮
    print("\n--- Step 1: 解壓縮 ---")
    if not cfg.skip_extraction:
        print("正在解壓 PDF.zip...")
        extract_archive(cfg.raw_zip_path, cfg.raw_pdf_dir, skip=False)
    else:
        print(f"跳過無標籤 PDF 解壓 (skip_extraction=True, folder={cfg.raw_pdf_dir})")
        
    if not cfg.skip_labeled_extraction:
        print("正在解壓吉輔提供資料.zip...")
        extract_archive(cfg.labeled_zip_path, cfg.raw_labeled_pdf_dir, skip=False)
    else:
        print(f"跳過標註 PDF 解壓 (skip_labeled_extraction=True, folder={cfg.raw_labeled_pdf_dir})")

    # Resolve nested folders from ZIP extraction
    cfg.raw_pdf_dir = get_nested_dir(cfg.raw_pdf_dir)
    cfg.raw_labeled_pdf_dir = get_nested_dir(cfg.raw_labeled_pdf_dir)

    # Step 2: PDF 轉檔
    print("\n--- Step 2: PDF 轉影像 ---")
    if not cfg.skip_pdf_conversion:
        convert_pdfs_to_images(
            pdf_dir=cfg.raw_pdf_dir,
            output_dir=cfg.converted_image_dir,
            dpi=cfg.pdf_dpi,
            max_workers=cfg.pdf_max_workers,
            preserve_structure=False
        )
    else:
        print(f"跳過無標籤 PDF 轉換 (skip_pdf_conversion=True, folder={cfg.converted_image_dir})")
        
    if not cfg.skip_labeled_pdf_conversion:
        convert_pdfs_to_images(
            pdf_dir=cfg.raw_labeled_pdf_dir,
            output_dir=cfg.converted_labeled_image_dir,
            dpi=cfg.pdf_dpi,
            max_workers=cfg.pdf_max_workers,
            preserve_structure=True
        )
    else:
        print(f"跳過標註 PDF 轉換 (skip_labeled_pdf_conversion=True, folder={cfg.converted_labeled_image_dir})")

    # Resolve nested folders from PDF conversion
    cfg.converted_image_dir = get_nested_dir(cfg.converted_image_dir)
    cfg.converted_labeled_image_dir = get_nested_dir(cfg.converted_labeled_image_dir)

    # Step 3: 前處理
    print("\n--- Step 3: 影像前處理 (連通域 + Logo 移除) ---")
    if not cfg.skip_preprocessing:
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
        preprocess_images(prep_cfg, skip=False)
    else:
        print(f"跳過無標籤前處理 (skip_preprocessing=True, folder={cfg.preprocessed_image_dir})")

    if not cfg.skip_labeled_preprocessing:
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
        preprocess_images(labeled_prep_cfg, skip=False)
    else:
        print(f"跳過標記資料前處理 (skip_labeled_preprocessing=True, folder={cfg.preprocessed_labeled_image_dir})")

    # Resolve nested folders from preprocessing and conversion
    cfg.converted_image_dir = get_nested_dir(cfg.converted_image_dir)
    cfg.converted_labeled_image_dir = get_nested_dir(cfg.converted_labeled_image_dir)
    cfg.preprocessed_image_dir = get_nested_dir(cfg.preprocessed_image_dir)
    cfg.preprocessed_labeled_image_dir = get_nested_dir(cfg.preprocessed_labeled_image_dir)

    # Step 4: 專家篩選與資料集分割建置
    print("\n--- Step 4: 專家篩選與資料集建置 ---")
    da_dir = Path(cfg.converted_labeled_image_dir)
    db_dir = Path(cfg.converted_image_dir)
    
    if not da_dir.exists():
        print(f"錯誤: 有標籤影像目錄不存在: {da_dir}")
        return
        
    all_da = sorted([p for p in da_dir.rglob("*") if p.is_file() and p.suffix.lower() in _IMG_EXTS])
    all_db = sorted([p for p in db_dir.rglob("*") if p.is_file() and p.suffix.lower() in _IMG_EXTS])

    print(f"掃描影像: Da 有標籤={len(all_da)}張, Db 無標籤={len(all_db)}張")

    if not all_da:
        print("錯誤: 無有標籤影像！")
        return
    seeds = sample_query_seeds(cfg.converted_labeled_image_dir, n_seeds=cfg.n_seeds, seed=cfg.base_seed, unlabeled_dir=cfg.converted_image_dir)
    
    class_to_images = {}
    for p in all_da:
        class_name = p.parent.name
        class_to_images.setdefault(class_name, []).append(p)
        
    programmatic_gts = programmatic_select_gts(seeds, class_to_images, max_gts=15, seed=cfg.base_seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"特徵提取設備: {device}")
    try:
        feats, paths = extract_features(
            all_da + all_db, 
            checkpoint_path=None, 
            img_size=cfg.img_size, 
            batch_size=128,
            device=device
        )
        candidate_pools = find_candidate_pools(seeds, feats, paths, top_k=30)
        print("特徵檢索候選池成功建置！")
    except Exception as e:
        print(f"特徵提取失敗，退化為自動匹配: {e}")
        candidate_pools = {str(s): [] for s in seeds}

    # 執行資料集建置 (Programmatic 模式)
    build_dataset_v2(
        output_root=cfg.dataset_root,
        seeds=seeds,
        gt_selections=programmatic_gts,
        all_da=all_da,
        all_db=all_db,
        n_distractors=cfg.n_distractors,
        split_ratio=cfg.split_ratio,
        seed=cfg.base_seed
    )
    
    print("\n✅ [prepare_data] 資料預處理管線執行成功！")
    print("=" * 60)

if __name__ == "__main__":
    _IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    run_pipeline()
