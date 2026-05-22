#!/usr/bin/env python3
"""模擬人類專家篩選與混入干擾項，自動建立實驗計畫書要求的 dataset_v2 結構。"""

import json
import random
import sys
import shutil
from pathlib import Path

def build_symlinks(mapping: dict, target_dir: Path):
    """根據對照表建立符號連結以節省磁碟空間。"""
    target_dir.mkdir(parents=True, exist_ok=True)
    for category, paths in mapping.items():
        cat_dir = target_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        for p_str in paths:
            p = Path(p_str)
            link_path = cat_dir / p.name
            if link_path.is_symlink() or link_path.exists():
                try:
                    link_path.unlink()
                except Exception:
                    pass
            try:
                link_path.symlink_to(p.resolve())
            except Exception:
                # Windows 或不支援 symlink 環境下改用拷貝
                try:
                    shutil.copy(p, link_path)
                except Exception:
                    pass

def main():
    da_dir = Path("data/converted_labeled_images")
    db_dir = Path("data/converted_images")
    out_root = Path("dataset_v2")
    
    n_seeds = 50
    n_distractors = 500

    img_exts = {".jpg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    
    # 掃描所有影像
    all_da = sorted([p for p in da_dir.rglob("*") if p.is_file() and p.suffix.lower() in img_exts])
    all_db = sorted([p for p in db_dir.rglob("*") if p.is_file() and p.suffix.lower() in img_exts])
    
    print(f"掃描完成: Da 包含 {len(all_da)} 張標註影像, Db 包含 {len(all_db)} 張無標籤影像")
    
    if not all_da:
        print("錯誤: 找不到任何 Da 標註影像，請檢查 data/converted_labeled_images")
        sys.exit(1)
        
    # 建立每個影像與其同資料夾（即同類別）影像的對照表
    class_to_images = {}
    for p in all_da:
        class_name = p.parent.name
        if class_name not in class_to_images:
            class_to_images[class_name] = []
        class_to_images[class_name].append(p)
        
    # 1. 種子抽樣
    random.seed(42)  # 固定 Seed 確保重現性
    seeds = random.sample(all_da, min(n_seeds, len(all_da)))
    
    # 2. 模擬專家篩選：每個種子的 GT 就是同資料夾下的其他影像
    gt_selections = {}
    all_v_images = set(seeds)
    
    for seed in seeds:
        class_name = seed.parent.name
        # 該類別內除了 seed 以外的其他影像作為 GT
        gts = [p for p in class_to_images[class_name] if p != seed]
        # 限制每個 seed 的 GT 數量以防類別過大 (最多 15 張)
        gts = random.sample(gts, min(len(gts), 15))
        gt_selections[str(seed)] = [str(p) for p in gts]
        all_v_images.update(gts)
        
    # 3. 背景雜訊混入
    all_da_strs = {str(p) for p in all_da}
    all_db_strs = {str(p) for p in all_db}
    all_universe = all_da_strs.union(all_db_strs)
    
    all_v_images_strs = {str(p) for p in all_v_images}
    remaining_candidates = list(all_universe.difference(all_v_images_strs))
    
    chosen_distractors = random.sample(
        remaining_candidates,
        min(n_distractors, len(remaining_candidates))
    )
    
    # 驗證集 V 構成：Queries + GTs + Distractors
    all_v_images_strs.update(chosen_distractors)
    
    # 輸出劃分定義
    t_small_images = list(all_da_strs.difference(all_v_images_strs))
    t_large_images = list(all_universe.difference(all_v_images_strs))
    
    print(f"驗證集 V 大小: {len(all_v_images_strs)} 張 (含 {len(seeds)} Seeds, {len(all_v_images_strs) - len(seeds) - len(chosen_distractors)} GTs, {len(chosen_distractors)} Distractors)")
    print(f"T_small 大小: {len(t_small_images)} 張")
    print(f"T_large 大小: {len(t_large_images)} 張")
    
    # 建立並儲存 metadata.json
    meta = {
        "seeds": [str(p) for p in seeds],
        "gt_selections": gt_selections,
        "distractors": chosen_distractors,
        "V": list(all_v_images_strs),
        "T_small": t_small_images,
        "T_large": t_large_images
    }
    
    out_root.mkdir(parents=True, exist_ok=True)
    meta_path = out_root / "validation_split.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)
    print(f"對照表已寫入: {meta_path}")
    
    # 建立 T_small 劃分
    random.seed(42)
    random.shuffle(t_small_images)
    split_idx_s = int(len(t_small_images) * 0.8)
    t_small_train = t_small_images[:split_idx_s]
    t_small_test = t_small_images[split_idx_s:]
    build_symlinks({"Component_Dataset/train": t_small_train, "Component_Dataset/test": t_small_test}, out_root / "T_small" / "Run_01_Seed_42")
    
    # 建立 T_large 劃分
    random.shuffle(t_large_images)
    split_idx_l = int(len(t_large_images) * 0.8)
    t_large_train = t_large_images[:split_idx_l]
    t_large_test = t_large_images[split_idx_l:]
    build_symlinks({"Component_Dataset/train": t_large_train, "Component_Dataset/test": t_large_test}, out_root / "T_large" / "Run_01_Seed_42")
    
    # 建立 驗證集 V 結構
    v_mapping = {}
    for i, seed in enumerate(seeds):
        group_name = f"group_{i:03d}"
        group_images = [str(seed)] + gt_selections[str(seed)]
        group_images = list(dict.fromkeys(group_images))
        v_mapping[group_name] = group_images
        
    for j, dist in enumerate(chosen_distractors):
        v_mapping[f"distractor_{j:04d}"] = [dist]
        
    build_symlinks(v_mapping, out_root / "V")
    print("實驗資料集目錄結構與符號連結已成功建立！")

if __name__ == "__main__":
    main()
