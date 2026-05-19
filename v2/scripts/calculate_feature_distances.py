"""計算 SimSiam 特徵空間中特徵向量的歐幾里得距離以檢測崩塌。

用法範例：
python v2/scripts/calculate_feature_distances.py --checkpoint v2/outputs/simsiam_exp/checkpoint_last.pth
"""

import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# 將 v2 目錄加入 sys.path 以便匯入 src 模組
v2_dir = Path(__file__).resolve().parent.parent
if str(v2_dir) not in sys.path:
    sys.path.append(str(v2_dir))

from src.dataset.dataset import SingleViewDataset
from src.model.simsiam import SimSiam

def calculate_distances(features: torch.Tensor, sample_size: int = 5000):
    """計算特徵向量之間的成對歐幾里得距離統計資訊。"""
    n = features.size(0)
    
    # 如果樣本數過多，進行隨機抽樣以避免記憶體不足 (O(N^2))
    if n > sample_size:
        indices = torch.randperm(n)[:sample_size]
        features = features[indices]
        n = sample_size
        print(f"  [註] 樣本數超過 {sample_size}，已隨機抽取 {sample_size} 筆特徵進行成對距離計算。")
        
    # 計算所有成對歐幾里得距離
    distances = torch.cdist(features, features)
    
    # 提取上三角部分（不包含對角線與重複計算的部分）
    triu_indices = torch.triu_indices(n, n, offset=1)
    pairwise_dist = distances[triu_indices[0], triu_indices[1]]
    
    return pairwise_dist.mean().item(), pairwise_dist.std().item(), pairwise_dist.min().item(), pairwise_dist.max().item()

def evaluate_collapse(features: torch.Tensor):
    """計算特徵的維度標準差以評估崩塌程度。
    
    將特徵做 L2 正規化後，計算每個維度的標準差。
    如果標準差趨近於 0，表示在該維度上所有樣本的值都一樣（特徵崩塌）。
    理想情況下，標準差應接近 1 / sqrt(d)。
    """
    features_norm = F.normalize(features, dim=1)
    std_per_dim = features_norm.std(dim=0)
    mean_std = std_per_dim.mean().item()
    ideal_std = 1.0 / (features.size(1) ** 0.5)
    return mean_std, ideal_std

def main():
    parser = argparse.ArgumentParser(description="計算 SimSiam 特徵向量的歐幾里得距離以檢測分佈")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路徑")
    parser.add_argument("--data_dir", type=str, default="data/preprocessed_labeled_images", help="測試資料集目錄")
    parser.add_argument("--batch_size", type=int, default=128, help="批次大小")
    parser.add_argument("--img_size", type=int, default=256, help="影像大小")
    parser.add_argument("--sample_size", type=int, default=5000, help="計算距離時的最大採樣數量")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    # 1. 載入 Checkpoint 與設定
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"找不到 Checkpoint: {ckpt_path}")
        
    print(f"載入 Checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = state.get("config", {})
    
    # 解析模型設定參數
    model_config = config.get("model", {})
    backbone_name = model_config.get("backbone", "resnet18")
    proj_dim = model_config.get("proj_dim", 2048)
    proj_hidden = model_config.get("proj_hidden", 2048)
    pred_hidden = model_config.get("pred_hidden", 512)
    in_channels = model_config.get("in_channels", 1)

    print(f"建構模型 (Backbone: {backbone_name}, in_channels: {in_channels})...")
    model = SimSiam(
        backbone=backbone_name,
        proj_dim=proj_dim,
        proj_hidden=proj_hidden,
        pred_hidden=pred_hidden,
        in_channels=in_channels,
    )
    # 移除 torch.compile 產生的 "_orig_mod." 前綴
    state_dict = state["state_dict"]
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    # 相容舊版 2-layer projector
    if "projector.6.weight" not in clean_state_dict:
        print("  [註] 檢測到 Checkpoint 使用舊版 2-layer Projector，自動調整模型架構以相容。")
        from src.model.simsiam import _mlp
        feat_dim = clean_state_dict["projector.0.weight"].shape[1]
        effective_hidden = clean_state_dict["projector.0.weight"].shape[0]
        actual_proj_dim = clean_state_dict["projector.3.weight"].shape[0]
        model.projector = _mlp(
            feat_dim, effective_hidden, actual_proj_dim, num_layers=2, bn_last=True, dropout=0.0
        )

    model.load_state_dict(clean_state_dict)
    model.to(device)
    model.eval()

    # 2. 準備資料集
    data_path = Path(args.data_dir)
    if not data_path.is_dir():
        raise NotADirectoryError(f"找不到資料集目錄: {data_path}")

    print(f"載入資料集: {data_path}")
    dataset = SingleViewDataset(
        root=data_path,
        img_size=args.img_size,
        img_exts=[".jpg", ".png", ".bmp", ".tif", ".webp"],
        in_channels=in_channels,
        cache_mode="none"  # 推論時不需快取，避免吃光記憶體
    )
    
    if len(dataset) == 0:
        print("資料集中沒有找到圖片，請確認目錄路徑與圖片格式。")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, max(1, torch.get_num_threads() // 2)),
        pin_memory=torch.cuda.is_available()
    )

    # 3. 抽取特徵
    print("開始抽取特徵...")
    all_f = []
    all_z = []
    
    with torch.no_grad():
        for imgs in tqdm(dataloader, desc="Extracting Features"):
            imgs = imgs.to(device)
            # 取得 backbone 特徵 (未經過 MLP)
            f = model.backbone(imgs)
            # 取得 projector 特徵 (經過 MLP)
            z = model.projector(f)
            
            all_f.append(f.cpu())
            all_z.append(z.cpu())
            
    features_f = torch.cat(all_f, dim=0)
    features_z = torch.cat(all_z, dim=0)
    
    print(f"\n共抽取了 {features_f.size(0)} 筆特徵。")

    # 4. 統計與分析
    print("\n" + "="*40)
    print("=== 1. 分析 Backbone 特徵 (f) ===")
    print("="*40)
    mean_dist, std_dist, min_dist, max_dist = calculate_distances(features_f, args.sample_size)
    print(f"成對歐幾里得距離 (Pairwise Euclidean Distance):")
    print(f"  - 平均值: {mean_dist:.4f}")
    print(f"  - 標準差: {std_dist:.4f}")
    print(f"  - 最小值: {min_dist:.4f}")
    print(f"  - 最大值: {max_dist:.4f}")
    
    mean_std, ideal_std = evaluate_collapse(features_f)
    print(f"\n維度標準差 (Dimensionality Std - 用於檢測崩塌):")
    print(f"  - 實際平均標準差: {mean_std:.4f}")
    print(f"  - 理想值 (無崩塌): {ideal_std:.4f}")
    print(f"  - 空間利用率指標: {mean_std / ideal_std * 100:.2f}% (越接近 0 越嚴重崩塌)")

    print("\n" + "="*40)
    print("=== 2. 分析 Projector 特徵 (z) ===")
    print("="*40)
    mean_dist, std_dist, min_dist, max_dist = calculate_distances(features_z, args.sample_size)
    print(f"成對歐幾里得距離 (Pairwise Euclidean Distance):")
    print(f"  - 平均值: {mean_dist:.4f}")
    print(f"  - 標準差: {std_dist:.4f}")
    print(f"  - 最小值: {min_dist:.4f}")
    print(f"  - 最大值: {max_dist:.4f}")
    
    mean_std, ideal_std = evaluate_collapse(features_z)
    print(f"\n維度標準差 (Dimensionality Std - 用於檢測崩塌):")
    print(f"  - 實際平均標準差: {mean_std:.4f}")
    print(f"  - 理想值 (無崩塌): {ideal_std:.4f}")
    print(f"  - 空間利用率指標: {mean_std / ideal_std * 100:.2f}% (越接近 0 越嚴重崩塌)")
    
    # 簡單的崩塌判斷邏輯
    print("\n" + "="*40)
    print("=== 結論 ===")
    if mean_std / ideal_std < 0.1 or mean_dist < 1e-2:
        print("[警告] 模型可能發生了嚴重的特徵崩塌 (Feature Collapse)！")
        print("所有影像被映射到了特徵空間中非常接近的點。")
    elif mean_std / ideal_std < 0.3:
        print("[注意] 模型的空間利用率偏低，可能發生了部分崩塌 (Partial Collapse)。")
        print("請檢查學習率、優化器或 Data Augmentation 的設定是否過弱。")
    else:
        print("[正常] 模型特徵分佈顯示沒有發生嚴重崩塌，特徵空間有被有效利用。")
    print("="*40)

if __name__ == "__main__":
    main()
