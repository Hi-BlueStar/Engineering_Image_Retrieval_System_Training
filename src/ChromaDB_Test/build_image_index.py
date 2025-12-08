import os
import re
import glob
import datetime
from typing import List, Dict, Any, Callable, Union
from PIL import Image
import chromadb
from chromadb.config import Settings
import open_clip
import torch
import numpy as np

# ==========================================
# 模組 1: Embedding 策略 (Strategy Pattern)
# ==========================================

class BaseEmbeddingFunction(chromadb.EmbeddingFunction):
    """
    嵌入函數的基礎介面，確保與 ChromaDB 相容。
    """
    def __call__(self, input: Any) -> Any:
        raise NotImplementedError

class OpenCLIPEmbedding(BaseEmbeddingFunction):
    """
    使用 OpenCLIP 進行影像嵌入的實作 (支援 GPU 加速)。
    """
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Info] Loading OpenCLIP model: {model_name} on {self.device}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, input: Any) -> List[List[float]]:
        """
        ChromaDB 會呼叫此函數。
        input: 應為圖片路徑列表 (URIs) 或 PIL Image 物件列表。
        """
        images = []
        # 預處理：將路徑轉為 PIL Image，若已是 Image 則直接使用
        for item in input:
            if isinstance(item, str):
                try:
                    img = Image.open(item).convert("RGB")
                    images.append(self.preprocess(img))
                except Exception as e:
                    print(f"[Error] Failed to load image {item}: {e}")
                    # 使用全黑圖作為 fallback，避免整個 batch 失敗
                    images.append(self.preprocess(Image.new('RGB', (224, 224))))
            elif isinstance(item, Image.Image):
                images.append(self.preprocess(item))
        
        if not images:
            return []

        image_tensor = torch.stack(images).to(self.device)
        
        with torch.no_grad():
            features = self.model.encode_image(image_tensor)
            features /= features.norm(dim=-1, keepdim=True) # 正規化
            
        return features.cpu().numpy().tolist()

# ==========================================
# 模組 2: 檔案系統爬蟲與過濾 (File Logic)
# ==========================================

class FileCrawler:
    """
    負責遍歷目錄並根據特定商業邏輯篩選檔案。
    """
    def __init__(self, root_dir: str):
        self.root_dir = root_dir

    def _is_valid_random_image(self, filename: str, item_name: str) -> bool:
        """檢查是否為 random_01 到 random_20"""
        # Regex 解釋:
        # ^... : 開頭
        # _random_ : 匹配字串
        # (?:0[1-9]|1[0-9]|20) : 非捕獲群組，匹配 01-09, 10-19, 或 20
        # \.png$ : 結尾是 .png
        pattern = rf"^{re.escape(item_name)}_random_(?:0[1-9]|1[0-9]|20)\.png$"
        return bool(re.match(pattern, filename))

    def _is_valid_large_component(self, filename: str) -> bool:
        """檢查是否為 large_components 規則"""
        # 匹配 large_L{數字}_area{數字}_pad2.png
        pattern = r"^large_L\d+_area\d+_pad2\.png$"
        return bool(re.match(pattern, filename))

    def scan(self) -> List[Dict[str, Any]]:
        """
        掃描並回傳符合條件的檔案資訊列表。
        """
        valid_files = []
        
        # 假設結構: root / Category / ItemName / ...
        # 使用 os.walk 進行深度優先搜尋，但我們需要控制層級
        
        # 獲取分類 (Category)
        categories = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        
        for category in categories:
            cat_path = os.path.join(self.root_dir, category)
            items = [d for d in os.listdir(cat_path) if os.path.isdir(os.path.join(cat_path, d))]
            
            for item_name in items:
                item_path = os.path.join(cat_path, item_name)
                
                # 1. 掃描 Item 層級的檔案
                try:
                    files_in_item = os.listdir(item_path)
                except OSError:
                    continue

                for f in files_in_item:
                    full_path = os.path.join(item_path, f)
                    if not os.path.isfile(full_path): continue
                    
                    # 規則 A: {名稱}_merged.png
                    if f == f"{item_name}_merged.png":
                        valid_files.append(self._create_meta(full_path, category, item_name, "merged"))
                    
                    # 規則 B: {名稱}_random_01~20.png
                    elif self._is_valid_random_image(f, item_name):
                        valid_files.append(self._create_meta(full_path, category, item_name, "random"))

                # 2. 掃描 large_components 子目錄
                large_comp_path = os.path.join(item_path, "large_components")
                if os.path.exists(large_comp_path) and os.path.isdir(large_comp_path):
                    for f in os.listdir(large_comp_path):
                        full_path = os.path.join(large_comp_path, f)
                        # 規則 C: large_L{數字}_area{數字}_pad2.png
                        if self._is_valid_large_component(f):
                             valid_files.append(self._create_meta(full_path, category, item_name, "large_component"))

        return valid_files

    def _create_meta(self, path: str, category: str, item_name: str, img_type: str) -> Dict[str, Any]:
        """封裝 Metadata"""
        return {
            "path": path,
            "metadata": {
                "category": category,          # ATC鈑金
                "item_name": item_name,        # AK0OCVE8...
                "folder_path": os.path.dirname(path),
                "filename": os.path.basename(path),
                "image_type": img_type,        # merged, random, large_component
                "add_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            # 建立一個唯一 ID，避免重複加入
            "id": f"{item_name}_{os.path.basename(path)}" 
        }

# ==========================================
# 模組 3: ChromaDB 管理 (Database Logic)
# ==========================================

class VectorDBManager:
    def __init__(self, persist_path: str, collection_name: str, embedding_fn: BaseEmbeddingFunction):
        self.client = chromadb.PersistentClient(path=persist_path)
        self.embedding_fn = embedding_fn
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )

    def ingest_data(self, data_list: List[Dict[str, Any]], batch_size: int = 10):
        """
        批量寫入數據到 ChromaDB
        """
        total = len(data_list)
        print(f"[Info] Start ingesting {total} images into ChromaDB...")

        for i in range(0, total, batch_size):
            batch = data_list[i : i + batch_size]
            
            ids = [item["id"] for item in batch]
            metadatas = [item["metadata"] for item in batch]
            paths = [item["path"] for item in batch]
            
            # 這裡我們傳入 paths，讓 embedding_fn 內部的 __call__ 去讀取圖片
            # 這種方式稱為 Data-Loader 模式，比預先讀取所有圖片更節省記憶體
            try:
                self.collection.add(
                    ids=ids,
                    metadatas=metadatas,
                    # 注意: 視你的 ChromaDB 版本與 Embedding Function 實作，
                    # 這裡可以傳 images (PIL物件/numpy) 或 uris (路徑)。
                    # 我們的 OpenCLIPEmbedding 實作了接收路徑並讀取的邏輯，所以這裡傳 input=paths
                    # 但在 collection.add 參數中，對應 input 的參數通常是 documents 或 images
                    # 為了語意正確，我們使用 images 參數傳遞路徑 (依賴 embedding function 處理)
                    images=paths 
                )
                print(f"  - Processed batch {i//batch_size + 1}/{(total//batch_size)+1}")
            except Exception as e:
                print(f"[Error] Batch ingestion failed: {e}")

# ==========================================
# 主程式 (Main Execution)
# ==========================================

def main():
    # 設定參數
    ROOT_DIR = "./engineering_images_100dpi_2" # 請修改為實際路徑
    DB_PATH = "./chroma_db_store"
    COLLECTION_NAME = "engineering_parts_vision"
    
    # 1. 初始化 Embedding 模型 (可替換為其他模型或 API)
    # 若要自定義，只需繼承 BaseEmbeddingFunction 即可
    embedding_fn = OpenCLIPEmbedding(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k")
    
    # 2. 掃描檔案
    crawler = FileCrawler(root_dir=ROOT_DIR)
    if not os.path.exists(ROOT_DIR):
        print(f"[Error] Directory not found: {ROOT_DIR}")
        return

    print("[Info] Scanning directory structure...")
    target_files = crawler.scan()
    print(f"[Info] Found {len(target_files)} valid images matching criteria.")
    
    if not target_files:
        print("[Info] No files to ingest.")
        return

    # 3. 存入資料庫
    db_manager = VectorDBManager(DB_PATH, COLLECTION_NAME, embedding_fn)
    db_manager.ingest_data(target_files, batch_size=20)
    
    print("[Success] Ingestion complete.")

if __name__ == "__main__":
    main()