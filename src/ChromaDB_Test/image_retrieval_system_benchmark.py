import os
import re
import random
import datetime
import shutil
import numpy as np
import pandas as pd
import chromadb
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from typing import List, Dict, Any, Tuple
import open_clip
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 設定亂數種子以確保再現性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)

# ==========================================
# 模組 1: Embedding Models (Model Zoo)
# ==========================================

class BaseEmbeddingFunction(chromadb.EmbeddingFunction):
    def get_name(self) -> str:
        """回傳模型名稱，用於建立 Collection"""
        raise NotImplementedError
    
    def name(self) -> str:
        """回傳模型名稱，用於建立 Collection"""
        raise NotImplementedError

class OpenCLIPEmbedding(BaseEmbeddingFunction):
    """OpenCLIP (Vision Transformer, ViT) 模型。
    適合通用性強、語意理解好的場景。
    """
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model_name = f"OpenCLIP_{model_name}"
        print(f"[Init] Loading {self._model_name}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.to(self.device).eval()
    
    # 實作 name() 方法供 ChromaDB 呼叫
    def name(self) -> str:
        return self._model_name

    def __call__(self, input: Any) -> List[List[float]]:
        images = self._prepare_images(input, self.preprocess)
        if not images: return []
        with torch.no_grad():
            features = self.model.encode_image(torch.stack(images).to(self.device))
            features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().tolist()

    def _prepare_images(self, input_data, transform):
        imgs = []
        for item in input_data:
            try:
                img = Image.open(item).convert("RGB") if isinstance(item, str) else item
                imgs.append(transform(img))
            except Exception:
                imgs.append(transform(Image.new('RGB', (224, 224))))
        return imgs

class ResNet50Embedding(BaseEmbeddingFunction):
    """標準 ResNet50 (ImageNet Pretrained) - Baseline"""
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model_name = "ResNet50_ImageNet"
        print(f"[Init] Loading {self._model_name}...")
        # 使用新版 weights 參數
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        self.model = models.resnet50(weights=weights)
        # 移除最後一層 FC，只取特徵
        self.model.fc = nn.Identity()
        self.model.to(self.device).eval()
        self.preprocess = weights.transforms()

    # 實作 name() 方法供 ChromaDB 呼叫
    def name(self) -> str:
        return self._model_name

    def __call__(self, input: Any) -> List[List[float]]:
        imgs = []
        for item in input:
            try:
                img = Image.open(item).convert("RGB") if isinstance(item, str) else item
                imgs.append(self.preprocess(img))
            except Exception:
                # Fallback transform if image fails
                imgs.append(self.preprocess(Image.new('RGB', (224, 224))))
        
        if not imgs: return []
        
        with torch.no_grad():
            tensor = torch.stack(imgs).to(self.device)
            features = self.model(tensor)
            # L2 Normalize
            features = torch.nn.functional.normalize(features, p=2, dim=1)
        return features.cpu().numpy().tolist()

class SimSiamEmbedding(BaseEmbeddingFunction):
    """
    自定義 SimSiam 架構。
    注意：SimSiam 是一個訓練框架。若您有訓練好的 .pth 權重，請傳入 model_path。
    否則此處將初始化一個 SimSiam 結構並加載 ImageNet Backbone 作為演示。
    """
    def __init__(self, model_path: str = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model_name = "Custom_SimSiam_ResNet50"
        print(f"[Init] Loading {self._model_name}...")
        
        # 1. 建立 Backbone (ResNet50)
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # SimSiam 通常使用 backbone output (2048 dim)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1]) 
        
        # 2. 如果有訓練好的權重，在這裡加載
        if model_path and os.path.exists(model_path):
            print(f"[Info] Loading SimSiam weights from {model_path}")
            # 假設權重檔儲存的是 state_dict
            checkpoint = torch.load(model_path, map_location=self.device)
            # 這裡需要根據您儲存權重的方式調整 key (例如移除 'module.' 前綴)
            self.backbone.load_state_dict(checkpoint, strict=False)
        else:
            print("[Info] No custom SimSiam weights provided. Using ImageNet initialization.")

        self.backbone.to(self.device).eval()
        
        # 標準 ImageNet 預處理
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # 實作 name() 方法供 ChromaDB 呼叫
    def name(self) -> str:
        return self._model_name

    def __call__(self, input: Any) -> List[List[float]]:
        imgs = []
        for item in input:
            try:
                img = Image.open(item).convert("RGB") if isinstance(item, str) else item
                imgs.append(self.preprocess(img))
            except Exception:
                imgs.append(self.preprocess(Image.new('RGB', (224, 224))))

        if not imgs: return []
        
        with torch.no_grad():
            tensor = torch.stack(imgs).to(self.device)
            features = self.backbone(tensor)
            features = features.view(features.size(0), -1) # Flatten
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            
        return features.cpu().numpy().tolist()

# ==========================================
# 模組 2: 資料處理與拆分 (Data Splitter)
# ==========================================

class DataManager:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.all_files = []

    def scan_files(self) -> List[Dict]:
        """爬蟲邏輯 (與先前相同，略作精簡)"""
        files_data = []
        if not os.path.exists(self.root_dir): return []
        
        categories = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        for cat in categories:
            cat_path = os.path.join(self.root_dir, cat)
            items = [d for d in os.listdir(cat_path) if os.path.isdir(os.path.join(cat_path, d))]
            for item in items:
                item_path = os.path.join(cat_path, item)
                
                # Helper to add file
                def add_file(f, f_type):
                    full_p = os.path.join(item_path, f)
                    if os.path.isfile(full_p):
                        files_data.append({
                            "path": full_p,
                            "item_name": item,
                            "category": cat,
                            "type": f_type,
                            "id": f"{item}_{f}"
                        })

                # Scan
                try: 
                    fs = os.listdir(item_path)
                    for f in fs:
                        if f == f"{item}_merged.png": add_file(f, "merged")
                        elif re.match(rf"^{re.escape(item)}_random_(?:0[1-9]|1[0-9]|20)\.png$", f): add_file(f, "random")
                    
                    if os.path.exists(os.path.join(item_path, "large_components")):
                        lc_path = os.path.join(item_path, "large_components")
                        for f in os.listdir(lc_path):
                            if re.match(r"^large_L\d+_area\d+_pad2\.png$", f):
                                full_p = os.path.join(lc_path, f)
                                files_data.append({
                                    "path": full_p,
                                    "item_name": item,
                                    "category": cat,
                                    "type": "large_component",
                                    "id": f"{item}_{f}"
                                })
                except OSError: continue
        
        self.all_files = files_data
        return files_data

    def create_split(self, test_size=0.2) -> Tuple[List[Dict], List[Dict]]:
        """
        將資料拆分為：
        1. Gallery (存入 DB 的圖)
        2. Query (用來測試搜尋功能的圖)
        策略：針對每個 item_name，隨機保留 20% 的圖片作為測試用 (Query)，其餘 80% 存入 DB。
        """
        df = pd.DataFrame(self.all_files)
        if df.empty: return [], []
        
        gallery_list = []
        query_list = []

        # Group by item_name to ensure every item is in both sets (if possible)
        for item_name, group in df.groupby("item_name"):
            if len(group) < 2:
                # 如果只有一張圖，只能放 Gallery，不然沒得搜
                gallery_list.extend(group.to_dict('records'))
            else:
                train_set, test_set = train_test_split(group, test_size=test_size, random_state=42)
                gallery_list.extend(train_set.to_dict('records'))
                query_list.extend(test_set.to_dict('records'))
        
        print(f"[Split Info] Total: {len(df)} | Gallery (DB): {len(gallery_list)} | Query (Test): {len(query_list)}")
        return gallery_list, query_list

# ==========================================
# 模組 3: 自動化評估引擎 (Evaluator)
# ==========================================

class Evaluator:
    def __init__(self, db_path: str):
        self.db_path = db_path
        # 每次評估前清理舊 DB 以確保公平
        if os.path.exists(db_path):
            try:
                shutil.rmtree(db_path)
            except OSError as e:
                print(f"[Warning] Could not delete old DB: {e}")
            
        self.client = chromadb.PersistentClient(path=db_path)

    def run_evaluation(self, 
                       embedding_fns: List[BaseEmbeddingFunction], 
                       gallery_data: List[Dict], 
                       query_data: List[Dict]) -> pd.DataFrame:
        
        results = []

        for emb_fn in embedding_fns:
            model_name = emb_fn.name()
            print(f"\n[Evaluating] Model: {model_name}")
            
            # 1. 建立並寫入 Collection (Gallery)
            collection = self.client.get_or_create_collection(
                name=f"eval_{model_name}",
                embedding_function=emb_fn,
                metadata={"hnsw:space": "cosine"}
            )
            
            # 批量寫入
            batch_size = 32
            print(f"  -> Ingesting {len(gallery_data)} images to Gallery...")
            for i in tqdm(range(0, len(gallery_data), batch_size), leave=False):
                batch = gallery_data[i : i + batch_size]
                collection.add(
                    ids=[b["id"] for b in batch],
                    metadatas=[{"item_name": b["item_name"], "path": b["path"]} for b in batch],
                    images=[b["path"] for b in batch]
                )

            # 2. 執行查詢測試 (Query)
            correct_top1 = 0
            correct_top5 = 0
            total_queries = len(query_data)
            
            print(f"  -> Testing {total_queries} queries...")
            
            # 為了效率，可以做 Batch Query，但為了邏輯清晰這裡跑迴圈
            # 生產環境建議改為 batch query_images
            
            # 建立 Query Batch
            q_paths = [q["path"] for q in query_data]
            q_truths = [q["item_name"] for q in query_data]
            
            # Batch Query 執行
            # 分批處理 Query 以避免記憶體溢出
            query_batch_size = 10  
            all_retrieved_metas = []

            for i in range(0, len(q_paths), query_batch_size):
                batch_paths = q_paths[i : i + query_batch_size]
                
                # 手動呼叫 embedding function 取得向量
                # 注意：這裡我們的 emb_fn 接受路徑列表
                batch_embeddings = emb_fn(batch_paths)
                
                batch_results = collection.query(
                    query_embeddings=batch_embeddings, # 使用向量查詢，最穩定
                    n_results=10
                )
                all_retrieved_metas.extend(batch_results['metadatas'])
            
            # 3. 計算指標
            for idx, truth in enumerate(q_truths):
                # 取得回傳的 metadata 列表 (list of dicts)
                retrieved_metas = all_retrieved_metas[idx]
                # 提取預測的 item_names
                pred_names = [m['item_name'] for m in retrieved_metas]
                
                # Top-1 Check
                if pred_names[0] == truth:
                    correct_top1 += 1
                
                # Top-5 Check
                if truth in pred_names[:5]:
                    correct_top5 += 1

            acc_1 = (correct_top1 / total_queries) * 100
            acc_5 = (correct_top5 / total_queries) * 100
            
            print(f"  -> Result: Top-1 Acc: {acc_1:.2f}% | Top-5 Acc: {acc_5:.2f}%")
            
            results.append({
                "Model": model_name,
                "Top-1 Accuracy (%)": round(acc_1, 2),
                "Top-5 Accuracy (%)": round(acc_5, 2),
                "Gallery Size": len(gallery_data),
                "Query Size": total_queries
            })
            
            # 清理 Collection 釋放記憶體 (Optional)
            self.client.delete_collection(name=f"eval_{model_name}")

        return pd.DataFrame(results)

# ==========================================
# 主程式
# ==========================================

def main():
    # 設定路徑
    ROOT_DIR = "./results/batch2/engineering_images_100dpi_2"
    EVAL_DB_PATH = "./chroma_db_evaluation"
    SIMSIAM_WEIGHTS = "./simsiam_resnet50.pth" # 若有訓練好的權重，請放在這
    
    # 1. 準備數據
    print("=== Step 1: Data Preparation ===")
    dm = DataManager(ROOT_DIR)
    files = dm.scan_files()
    if not files:
        print("[Error] No files found.")
        return
        
    # 自動拆分數據集 (80% 入庫, 20% 測試)
    gallery_set, query_set = dm.create_split(test_size=0.2)
    
    # 2. 初始化模型庫
    print("\n=== Step 2: Model Initialization ===")
    models_to_test = [
        # 模型 A: OpenCLIP (SOTA)
        OpenCLIPEmbedding(model_name="ViT-B-32"),
        
        # 模型 B: ResNet50 (Standard Baseline)
        ResNet50Embedding(),
        
        # 模型 C: Custom SimSiam (Advanced)
        SimSiamEmbedding(model_path=SIMSIAM_WEIGHTS)
    ]
    
    # 3. 執行評估
    print("\n=== Step 3: Automated Evaluation Loop ===")
    evaluator = Evaluator(EVAL_DB_PATH)
    result_df = evaluator.run_evaluation(models_to_test, gallery_set, query_set)
    
    # 4. 輸出報告
    print("\n=== Final Evaluation Report ===")
    print(result_df.to_markdown(index=False))
    
    # 額外建議：保存 CSV
    result_df.to_csv("model_evaluation_report.csv", index=False)
    print("\n[Info] Report saved to model_evaluation_report.csv")

if __name__ == "__main__":
    main()