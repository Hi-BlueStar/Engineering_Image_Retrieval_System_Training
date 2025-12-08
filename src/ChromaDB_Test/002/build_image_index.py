import os
import chromadb
import torch
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
from tqdm import tqdm
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

# ==========================================
# 1. 系統參數設定 (Configuration)
# ==========================================
# 資料根目錄 (請依實際情況修改)
ROOT_DIR = "results/batch2/engineering_images_100dpi_2_demo"
# Collection 名稱 (建議使用 v3 以區隔舊版)
COLLECTION_NAME = "engineering_components_v3_integrated"
# 批次處理大小 (根據 VRAM/RAM 大小調整，800 為安全值)
BATCH_SIZE = 800
# ChromaDB 儲存路徑
PERSIST_PATH = "./chroma_db_store"

# ==========================================
# 2. 核心邏輯函式 (Core Logic)
# ==========================================

def get_compute_device() -> str:
    """偵測並回傳最佳的運算裝置 (CUDA > MPS > CPU)"""
    if torch.cuda.is_available():
        print("✅ 偵測到 NVIDIA GPU，將使用 CUDA 加速運算。")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("✅ 偵測到 Apple Silicon，將使用 MPS 加速運算。")
        return "mps"
    else:
        print("⚠️ 未偵測到 GPU，將使用 CPU 運算 (速度較慢)。")
        return "cpu"

def is_target_image(file_path: Path) -> bool:
    """
    [來自 Code B 的精華]
    嚴格過濾邏輯：判斷檔案是否為目標圖片。
    1. {名稱}_merged.png
    2. {名稱}_random_01.png ~ {名稱}_random_20.png
    3. large_{...}_pad2.png (子組件)
    """
    filename = file_path.name
    
    # 基本檢查：必須是 png
    if not filename.lower().endswith(".png"):
        return False

    # 1. 檢查 Merged (例如: AK0OCVE8..._merged.png)
    if filename.endswith("_merged.png"):
        return True
    
    # 2. 檢查 Large Components (針對 large_..._pad2.png)
    # 通常位於 large_components 子目錄下，或檔名包含特徵
    if filename.startswith("large_") and filename.endswith("_pad2.png"):
        return True
    
    # 3. 檢查 Random Augmentation (範圍 01 ~ 20)
    if "_random_" in filename:
        try:
            # 檔名範例: AK0OCVE8_random_05.png
            # 取出 .png 前的最後一部分
            parts = file_path.stem.split("_")
            if parts[-2] == "random":
                num = int(parts[-1])
                if 1 <= num <= 20:
                    return True
        except (ValueError, IndexError):
            return False
            
    return False

def extract_metadata_and_id(file_path: Path, root_path: Path) -> Tuple[str, Dict[str, Any]]:
    """
    [融合 Code A 與 Code B]
    解析路徑結構以產生 Metadata，並生成語意化 ID。
    """
    try:
        relative_path = file_path.relative_to(root_path)
        path_parts = relative_path.parts
    except ValueError:
        #若路徑不在 root 下，回退到檔名
        path_parts = []

    # 預設值
    category = "unknown_category"
    part_id = "unknown_part"
    
    # 解析目錄結構: root / Category / PartID / ...
    if len(path_parts) >= 2:
        category = path_parts[0]
        part_id = path_parts[1]

    # 判斷圖片類型 (Type Logic)
    fname = file_path.name
    img_type = "standard"
    variant = "original"
    
    if "merged" in fname:
        img_type = "merged_view"
        variant = "merged"
    elif filename_is_large_component(fname):
        img_type = "large_component"
        variant = "sub_component"
    elif "_random_" in fname:
        img_type = "augmented"
        variant = "random_sample"

    # 建構 Metadata (融合 Code B 的時間戳記與 Code A 的結構化資訊)
    metadata = {
        "category": category,
        "part_id": part_id,
        "filename": fname,
        "filepath": str(file_path), # 絕對路徑
        "type": img_type,
        "variant": variant,
        "added_date": datetime.now().isoformat(), # Code B 特性
        "source_layer": "sub_dir" if "large_components" in str(file_path) else "main_dir"
    }

    # 建構語意化 ID (Semantic ID) - [Code B 特性]
    # 格式: Category_PartID_Filename
    # 替換特殊字元以確保 ID 安全
    safe_id = f"{category}_{part_id}_{fname}".replace(" ", "_")
    
    return safe_id, metadata

def filename_is_large_component(filename: str) -> bool:
    """輔助函式：判斷是否為大型組件切圖"""
    return filename.startswith("large_") and filename.endswith("_pad2.png")

# ==========================================
# 3. 主程式 (Main Execution)
# ==========================================

def main():
    print(f"--- 啟動工程圖檔索引建置系統 (v3 Integrated) ---")
    print(f"目標目錄: {ROOT_DIR}")
    
    # 1. 硬體初始化
    device = get_compute_device()
    
    # 2. ChromaDB 初始化
    client = chromadb.PersistentClient(path=PERSIST_PATH)
    embedding_func = OpenCLIPEmbeddingFunction(device=device)
    
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func,
        data_loader=ImageLoader()
    )
    print(f"資料庫連線成功。Collection: {COLLECTION_NAME}")

    # 3. 掃描檔案 (使用 pathlib 的 rglob，效能與廣度兼具)
    root_path = Path(ROOT_DIR)
    if not root_path.exists():
        print(f"❌ 錯誤: 找不到目錄 {ROOT_DIR}")
        return

    # 先列出所有潛在圖片，顯示進度條總數用
    print("正在掃描目錄結構...")
    all_images = list(root_path.rglob("*.png"))
    print(f"目錄中共有 {len(all_images)} 個 PNG 檔案，開始進行篩選與索引...")

    # 4. 批次處理迴圈 (Code A 的 Streaming Batch 策略)
    ids_batch = []
    uris_batch = []
    metadatas_batch = []
    
    valid_count = 0
    skipped_count = 0

    for image_path in tqdm(all_images, desc="Indexing"):
        # [Code B 邏輯]：先過濾，不符合者直接跳過
        if not is_target_image(image_path):
            skipped_count += 1
            continue
            
        try:
            # 解析 ID 與 Metadata
            img_id, meta = extract_metadata_and_id(image_path, root_path)
            
            ids_batch.append(img_id)
            uris_batch.append(str(image_path))
            metadatas_batch.append(meta)
            valid_count += 1

            # 批次滿了就寫入
            if len(ids_batch) >= BATCH_SIZE:
                collection.add(
                    ids=ids_batch,
                    uris=uris_batch,
                    metadatas=metadatas_batch
                )
                ids_batch = []
                uris_batch = []
                metadatas_batch = []

        except Exception as e:
            print(f"⚠️ 處理 {image_path.name} 時發生錯誤: {e}")

    # 處理剩餘的尾數
    if ids_batch:
        collection.add(
            ids=ids_batch,
            uris=uris_batch,
            metadatas=metadatas_batch
        )

    # 5. 總結報告
    print(f"\n--- 索引建置完成 ---")
    print(f"📂 掃描總數: {len(all_images)}")
    print(f"✅ 成功入庫: {valid_count} (符合篩選條件)")
    print(f"⏩ 忽略檔案: {skipped_count} (非 Merged/Valid Random/Large)")
    print(f"💾 資料庫位置: {PERSIST_PATH}")
    print(f"📇 Collection: {COLLECTION_NAME}")
    
    # 簡單驗證
    count = collection.count()
    print(f"📊 目前資料庫內總筆數: {count}")

if __name__ == "__main__":
    main()