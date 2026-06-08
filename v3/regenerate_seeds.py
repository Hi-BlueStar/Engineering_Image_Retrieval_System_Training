#!/usr/bin/env python3
import sys
import logging
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path = [str(PROJECT_ROOT / "v3"), str(PROJECT_ROOT)] + [p for p in sys.path if p not in (str(PROJECT_ROOT), str(PROJECT_ROOT / "v3"))]

from v3.src.data.dataset_builder import (
    sample_query_seeds,
    programmatic_select_gts,
    build_dataset_v2
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RegenerateSeeds")

def main():
    labeled_dir = PROJECT_ROOT / "data" / "converted_labeled_images"
    unlabeled_dir = PROJECT_ROOT / "data" / "converted_images"
    dataset_root = PROJECT_ROOT / "dataset_v2"
    
    n_seeds = 50
    n_distractors = 500
    split_ratio = 0.8
    seed = 42
    
    logger.info("Scanning image directories...")
    _IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    
    all_da = sorted([p for p in labeled_dir.rglob("*") if p.is_file() and p.suffix.lower() in _IMG_EXTS])
    all_db = sorted([p for p in unlabeled_dir.rglob("*") if p.is_file() and p.suffix.lower() in _IMG_EXTS])
    
    logger.info(f"Loaded {len(all_da)} labeled images and {len(all_db)} unlabeled images.")
    
    logger.info("Sampling seeds from both labeled and unlabeled pools...")
    seeds = sample_query_seeds(
        labeled_dir=str(labeled_dir),
        n_seeds=n_seeds,
        seed=seed,
        unlabeled_dir=str(unlabeled_dir)
    )
    
    # Stratified grouping of labeled images for programmatic GTs fallback
    class_to_images = {}
    for p in all_da:
        class_name = p.parent.name
        class_to_images.setdefault(class_name, []).append(p)
        
    logger.info("Building programmatic Ground Truths mappings...")
    programmatic_gts = programmatic_select_gts(seeds, class_to_images, max_gts=15, seed=seed)
    
    logger.info("Rebuilding dataset splits and directory symlinks...")
    build_dataset_v2(
        output_root=str(dataset_root),
        seeds=seeds,
        gt_selections=programmatic_gts,
        all_da=all_da,
        all_db=all_db,
        n_distractors=n_distractors,
        split_ratio=split_ratio,
        seed=seed
    )
    
    logger.info("🎉 Done! Successfully regenerated seeds and rebuilt dataset_v2!")

if __name__ == "__main__":
    main()
