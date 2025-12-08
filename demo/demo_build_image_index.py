import os
import uuid
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from PIL import Image
from tqdm import tqdm  # 用於顯示進度條
# 先匯入 torch 來檢查 GPU
import torch

# 設定參數
ROOT_DIR = "results/batch2/engineering_images_100dpi_2_demo"  # 您的根目錄名稱
COLLECTION_NAME = "engineering_components_v1_demo"
BATCH_SIZE = 800  # 批次處理大小，避免記憶體溢出

def get_image_metadata(file_path: Path, root_path: Path) -> Dict[str, Any]:
    """
    從檔案路徑解析 Metadata，保留資料夾結構的語意。
    結構範例: root / Category / PartID / [Optional: SubDir] / filename
    """
    relative_path = file_path.relative_to(root_path)
    parts = relative_path.parts
    
    # 基本 Metadata
    metadata = {
        "filepath": str(file_path),
        "filename": file_path.name,
        "extension": file_path.suffix.lower(),
        "category": "unknown",  # 例如: ATC鈑金
        "part_id": "unknown",   # 例如: AK0OCVE8...
        "type": "standard",     # standard, large_component
        "variant": "unknown"    # original, merged, random, specific_component
    }

    if len(parts) >= 2:
        metadata["category"] = parts[0]
        metadata["part_id"] = parts[1]

    # 判斷是否為子組件 (large_components)
    if "large_components" in parts:
        metadata["type"] = "large_component"
        metadata["variant"] = "sub_component"
    else:
        # 根據檔名判斷變體類型
        fname = file_path.stem
        if "original" in fname:
            metadata["variant"] = "original"
        elif "merged" in fname:
            metadata["variant"] = "merged"
        elif "random" in fname:
            metadata["variant"] = "random_augmentation"
            # 提取 random 編號 (例如 random_05)
            try:
                metadata["random_seed"] = fname.split("_")[-1]
            except:
                pass
    
    return metadata

def main():
    print("--- 初始化 ChromaDB 與 Embedding 模型 ---")
    
    if torch.cuda.is_available():
        device = "cuda"
        print("✅ 偵測到 NVIDIA GPU，將使用 CUDA 加速運算。")
    elif torch.backends.mps.is_available():
        device = "mps" # 適用於 Mac M1/M2/M3 用戶
        print("✅ 偵測到 Apple Silicon，將使用 MPS 加速運算。")
    else:
        device = "cpu"
        print("⚠️ 未偵測到 GPU，將使用 CPU 運算 (速度較慢)。")
    
    # 1. 初始化 ChromaDB 客戶端 (保存於本地)
    client = chromadb.PersistentClient(path="./chroma_db_store")
    
    # 2. 設定多模態 Embedding 函數 (使用 OpenCLIP)
    # 這會將圖片轉換為高維向量，讓您可以透過文字搜尋圖片，或以圖搜圖
    embedding_func = OpenCLIPEmbeddingFunction(device=device)
    
    # 3. 建立或取得 Collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func,
        data_loader=ImageLoader() # 啟用圖片載入器
    )

    # 4. 遍歷目錄尋找圖片
    root_path = Path(ROOT_DIR)
    if not root_path.exists():
        print(f"錯誤: 找不到目錄 {ROOT_DIR}")
        return

    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_files = [
        p for p in root_path.rglob("*") 
        if p.suffix.lower() in image_extensions
    ]

    print(f"找到 {len(image_files)} 張圖片，準備處理...")

    # 5. 批次處理與寫入
    ids_batch = []
    uris_batch = [] # ChromaDB 使用 'uris' 欄位來指向圖片路徑
    metadatas_batch = []

    for image_path in tqdm(image_files, desc="Processing Images"):
        try:
            # 產生唯一 ID (使用 UUID 或路徑雜湊)
            img_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(image_path)))
            
            # 解析 Metadata
            meta = get_image_metadata(image_path, root_path)
            
            ids_batch.append(img_id)
            uris_batch.append(str(image_path)) # 儲存路徑，讓 Chroma 自動讀取並 Embed
            metadatas_batch.append(meta)

            # 當批次滿了，寫入資料庫
            if len(ids_batch) >= BATCH_SIZE:
                collection.add(
                    ids=ids_batch,
                    uris=uris_batch,
                    metadatas=metadatas_batch
                )
                # 清空批次
                ids_batch = []
                uris_batch = []
                metadatas_batch = []

        except Exception as e:
            print(f"處理 {image_path} 時發生錯誤: {e}")

    # 處理剩餘的批次
    if ids_batch:
        collection.add(
            ids=ids_batch,
            uris=uris_batch,
            metadatas=metadatas_batch
        )

    print(f"\n--- 完成! 已將 {len(image_files)} 張圖片存入 ChromaDB ---")
    print(f"資料庫位置: ./chroma_db_store")
    print(f"Collection 名稱: {COLLECTION_NAME}")

if __name__ == "__main__":
    main()