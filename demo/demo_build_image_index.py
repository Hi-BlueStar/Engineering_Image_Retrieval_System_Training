import uuid
import os
from pathlib import Path
from typing import Any, List

import chromadb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image, ImageFile
from chromadb.utils.data_loaders import ImageLoader
from chromadb import EmbeddingFunction, Embeddings 
from tqdm import tqdm

# 防止讀取損壞圖片時報錯
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- 設定參數 ---
ROOT_DIR = "results/batch2/engineering_images_Clean_100dpi"  # 圖片根目錄
COLLECTION_NAME = "engineering_components_simsiam_v1_demo"   # Collection 名稱建議區分模型
DB_PATH = "./chroma_db_store"                                # 資料庫儲存路徑
MODEL_CHECKPOINT_PATH = "./demo/checkpoint_last.pth"                # 訓練好的模型權重路徑
BATCH_SIZE = 200

# -----------------------------------------------------------------------------
# 1. SimSiam 模型架構 (必須與 Inference 階段完全一致)
# -----------------------------------------------------------------------------

def make_inference_transform(
    img_size: int = 224,
    mean: list[float] = [0.5],
    std: list[float] = [0.5]
) -> T.Compose:
    """建立推論用的確定性 Transform"""
    return T.Compose(
        [
            T.Resize(int(img_size * 1.14), interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )

def _mlp(in_dim: int, hidden_dim: int, out_dim: int, bn_last: bool = True, dropout: float = 0.0) -> nn.Sequential:
    layers = [
        nn.Linear(in_dim, hidden_dim, bias=False),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers += [nn.Linear(hidden_dim, out_dim, bias=False)]
    if bn_last:
        layers.append(nn.BatchNorm1d(out_dim, affine=True))
    return nn.Sequential(*layers)

class SimSiam(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet50",
        proj_dim: int = 2048,
        pred_hidden: int = 512,
        dropout: float = 0.0,
        pretrained: bool = False,
        in_channels: int = 1,
    ):
        super().__init__()
        # Backbone
        if backbone == "resnet18":
            net = models.resnet18(weights=None) 
        elif backbone == "resnet50":
            net = models.resnet50(weights=None)
        else:
            raise NotImplementedError(f"Unsupported backbone: {backbone}")

        # 修改第一層卷積以適應輸入通道
        if in_channels != 3:
            old_conv = net.conv1
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias,
            )
            net.conv1 = new_conv

        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net

        # Projector & Predictor
        self.projector = _mlp(feat_dim, 2048, proj_dim, bn_last=True, dropout=dropout)
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_hidden, bias=False),
            nn.BatchNorm1d(pred_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(pred_hidden, proj_dim),
        )

    def forward(self, x1, x2):
        f1, f2 = self.backbone(x1), self.backbone(x2)
        z1, z2 = self.projector(f1), self.projector(f2)
        p1, p2 = self.predictor(z1), self.predictor(z2)
        return p1, p2, z1.detach(), z2.detach()
    
    def get_embedding(self, x):
        return self.backbone(x)


# -----------------------------------------------------------------------------
# 2. ChromaDB 自定義 Embedding Function
# -----------------------------------------------------------------------------

class SimSiamEmbeddingFunction(EmbeddingFunction):
    """
    SimSiam Embedding Function for ChromaDB
    """
    def __init__(self, model_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   [SimSiam] Initializing on {self.device}...")
        
        # 初始化模型 (確保與訓練設定一致)
        self.model = SimSiam(backbone="resnet50", in_channels=1, pretrained=False)
        
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                state_dict = checkpoint.get("state_dict", checkpoint)
                # 處理可能的 key 差異
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                self.model.load_state_dict(state_dict, strict=False)
                print(f"   [SimSiam] 權重載入成功: {model_path}")
            except Exception as e:
                print(f"   [SimSiam] ❌ 權重載入失敗: {e}")
                raise e
        else:
            print(f"   [SimSiam] ❌ 找不到權重檔: {model_path}")
            raise FileNotFoundError(model_path)

        self.model.to(self.device)
        self.model.eval()
        self.transform = make_inference_transform(img_size=224)

    # 這裡將 input 的 type hint 從 Images 改為 list
    def __call__(self, input: list) -> Embeddings:
        """
        ChromaDB 傳入圖片列表 (通常是 numpy array)，回傳 Embeddings
        """
        tensor_batch = []
        for img_data in input:
            # ChromaDB ImageLoader 讀取後通常是 numpy array (RGB)
            if isinstance(img_data, np.ndarray):
                img = Image.fromarray(img_data.astype('uint8'))
            else:
                img = img_data # 若直接傳入 PIL 或路徑
            
            # 關鍵：SimSiam 訓練時使用灰階，這裡必須轉為 'L'
            img = img.convert("L")
            t = self.transform(img)
            tensor_batch.append(t)
            
        batch_input = torch.stack(tensor_batch).to(self.device)
        
        with torch.no_grad():
            features = self.model.get_embedding(batch_input)
            features = F.normalize(features, dim=1) # L2 Normalize
            embeddings = features.cpu().numpy().tolist()
            
        return embeddings


# -----------------------------------------------------------------------------
# 3. 輔助函式與主程式
# -----------------------------------------------------------------------------

def get_image_metadata(file_path: Path, root_path: Path) -> dict[str, Any]:
    """
    從檔案路徑解析 Metadata
    """
    relative_path = file_path.relative_to(root_path)
    parts = relative_path.parts

    metadata = {
        "filepath": str(file_path),
        "filename": file_path.name,
        "extension": file_path.suffix.lower(),
        "category": "unknown",
        "part_id": "unknown",
        "type": "standard",
        "variant": "unknown",
    }

    if len(parts) >= 2:
        metadata["category"] = parts[0]
        metadata["part_id"] = parts[1]

    if "large_components" in parts:
        metadata["type"] = "large_component"
        metadata["variant"] = "sub_component"
    else:
        fname = file_path.stem
        if "original" in fname:
            metadata["variant"] = "original"
        elif "merged" in fname:
            metadata["variant"] = "merged"
        elif "random" in fname:
            metadata["variant"] = "random_augmentation"
            try:
                metadata["random_seed"] = fname.split("_")[-1]
            except:
                pass

    return metadata


def main():
    print("--- 初始化 ChromaDB 與 SimSiam 模型 ---")

    if torch.cuda.is_available():
        device = "cuda"
        print("✅ 偵測到 NVIDIA GPU，將使用 CUDA 加速運算。")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("✅ 偵測到 Apple Silicon，將使用 MPS 加速運算。")
    else:
        device = "cpu"
        print("⚠️ 未偵測到 GPU，將使用 CPU 運算 (速度較慢)。")

    # 1. 初始化 ChromaDB 客戶端
    client = chromadb.PersistentClient(path=DB_PATH)

    # 2. 設定 Embedding 函數 (使用 SimSiam)
    try:
        embedding_func = SimSiamEmbeddingFunction(
            model_path=MODEL_CHECKPOINT_PATH, 
            device=device
        )
    except Exception as e:
        print("無法初始化 SimSiam 模型，請檢查權重路徑。程式終止。")
        return

    # 3. 建立或取得 Collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func,
        data_loader=ImageLoader(),  # 啟用圖片載入器
    )

    # 4. 遍歷目錄尋找圖片
    root_path = Path(ROOT_DIR)
    if not root_path.exists():
        print(f"錯誤: 找不到目錄 {ROOT_DIR}")
        return

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    image_files = [
        p for p in root_path.rglob("*") if p.suffix.lower() in image_extensions
    ]

    print(f"找到 {len(image_files)} 張圖片，準備處理...")

    # 5. 批次處理與寫入
    ids_batch = []
    uris_batch = []
    metadatas_batch = []

    for image_path in tqdm(image_files, desc="Processing Images"):
        try:
            # 產生唯一 ID
            img_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(image_path)))

            # 解析 Metadata
            meta = get_image_metadata(image_path, root_path)

            ids_batch.append(img_id)
            uris_batch.append(str(image_path))
            metadatas_batch.append(meta)

            # 當批次滿了，寫入資料庫
            if len(ids_batch) >= BATCH_SIZE:
                collection.add(
                    ids=ids_batch, uris=uris_batch, metadatas=metadatas_batch
                )
                ids_batch = []
                uris_batch = []
                metadatas_batch = []

        except Exception as e:
            print(f"處理 {image_path} 時發生錯誤: {e}")

    # 處理剩餘的批次
    if ids_batch:
        collection.add(ids=ids_batch, uris=uris_batch, metadatas=metadatas_batch)

    print(f"\n--- 完成! 已將 {len(image_files)} 張圖片 Embedding 並存入 ChromaDB ---")
    print(f"資料庫位置: {DB_PATH}")
    print(f"Collection 名稱: {COLLECTION_NAME}")
    print(f"使用模型權重: {MODEL_CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()