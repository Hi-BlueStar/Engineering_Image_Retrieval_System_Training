import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

def calculate_stats(data_dir, invert=True, sample_n=1000):
    img_paths = list(Path(data_dir).rglob("*.png")) + list(Path(data_dir).rglob("*.jpg"))
    if sample_n > 0 and len(img_paths) > sample_n:
        import random
        img_paths = random.sample(img_paths, sample_n)
    
    print(f"Calculating stats for {len(img_paths)} images (Invert={invert})...")
    
    means = []
    stds = []
    
    for p in tqdm(img_paths):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        # 轉換為 float32 [0, 1]
        img = img.astype(np.float32) / 255.0
        
        if invert:
            # 假設原始背景是白色 (1.0)，反轉後背景變為黑色 (0.0)
            img = 1.0 - img
            
        means.append(np.mean(img))
        stds.append(np.std(img))
        
    global_mean = np.mean(means)
    global_std = np.mean(stds)
    
    print(f"\nResults (Invert={invert}):")
    print(f"Mean: {global_mean:.4f}")
    print(f"Std:  {global_std:.4f}")
    print(f"Suggested Normalization: T.Normalize(mean=({global_mean:.4f},), std=({global_std:.4f},))")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/converted_images")
    parser.add_argument("--invert", action="store_true", help="Invert images before calculation")
    parser.add_argument("--sample-n", type=int, default=1000)
    args = parser.parse_args()
    
    # 計算原始與反轉後的統計值
    print("--- Original (White Background) ---")
    calculate_stats(args.data_dir, invert=False, sample_n=args.sample_n)
    print("\n--- Inverted (Black Background) ---")
    calculate_stats(args.data_dir, invert=True, sample_n=args.sample_n)
