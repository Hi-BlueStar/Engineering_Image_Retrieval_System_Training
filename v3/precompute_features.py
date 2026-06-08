#!/usr/bin/env python3
import os
import sys
import torch
import logging
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path = [str(PROJECT_ROOT / "v3"), str(PROJECT_ROOT)] + [p for p in sys.path if p not in (str(PROJECT_ROOT), str(PROJECT_ROOT / "v3"))]

from v3.src.data.dataset_builder import extract_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PrecomputeFeatures")

def main():
    labeled_dir = PROJECT_ROOT / "data" / "converted_labeled_images"
    unlabeled_dir = PROJECT_ROOT / "data" / "converted_images"
    dataset_root = PROJECT_ROOT / "dataset_v2"
    cache_path = dataset_root / "features_cache.pt"
    
    logger.info("Scanning image directories...")
    _IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    
    all_da = sorted([p for p in labeled_dir.rglob("*") if p.is_file() and p.suffix.lower() in _IMG_EXTS])
    all_db = sorted([p for p in unlabeled_dir.rglob("*") if p.is_file() and p.suffix.lower() in _IMG_EXTS])
    all_images = all_da + all_db
    
    logger.info(f"Loaded {len(all_da)} labeled images and {len(all_db)} unlabeled images. Total: {len(all_images)}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device} for feature extraction")
    
    dataset_root.mkdir(parents=True, exist_ok=True)
    
    features, paths = extract_features(
        all_images,
        checkpoint_path=None,
        img_size=256,
        batch_size=256,
        device=device
    )
    
    # Save to cache
    logger.info(f"Saving features cache to {cache_path}...")
    torch.save({
        "features": features,
        "paths": paths
    }, cache_path)
    logger.info("🎉 Feature precomputation complete and successfully cached!")

if __name__ == "__main__":
    main()
