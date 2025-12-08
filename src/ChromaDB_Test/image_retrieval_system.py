import datetime
import os
import re
from typing import Any

import chromadb
import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image


# ==========================================
# 模組 1: Embedding Models (策略模式)
# ==========================================


class BaseEmbeddingFunction(chromadb.EmbeddingFunction):
    def get_name(self) -> str:
        """回傳模型名稱，用於建立 Collection"""
        raise NotImplementedError


class OpenCLIPEmbedding(BaseEmbeddingFunction):
    """
    使用 OpenCLIP (ViT) 模型。
    適合通用性強、語意理解好的場景。
    """

    def __init__(
        self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name_str = f"openclip_{model_name.replace('-', '_')}"
        print(f"[Init] Loading {self.model_name_str} on {self.device}...")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.to(self.device)
        self.model.eval()

    def get_name(self) -> str:
        return self.model_name_str

    def __call__(self, input: Any) -> list[list[float]]:
        images = []
        for item in input:
            if isinstance(item, str):  # File Path
                try:
                    img = Image.open(item).convert("RGB")
                    images.append(self.preprocess(img))
                except Exception:
                    images.append(self.preprocess(Image.new("RGB", (224, 224))))
            elif isinstance(item, Image.Image):  # PIL Object
                images.append(self.preprocess(item))

        if not images:
            return []

        image_tensor = torch.stack(images).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(image_tensor)
            features /= features.norm(dim=-1, keepdim=True)  # Normalize

        return features.cpu().numpy().tolist()


# 若有其他模型 (例如 ResNet, EfficientNet)，可在此擴充 Class
# class ResNetEmbedding(BaseEmbeddingFunction): ...

# ==========================================
# 模組 2: 檔案爬蟲 (保持原邏輯)
# ==========================================


class FileCrawler:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir

    def _is_valid_random_image(self, filename: str, item_name: str) -> bool:
        pattern = rf"^{re.escape(item_name)}_random_(?:0[1-9]|1[0-9]|20)\.png$"
        return bool(re.match(pattern, filename))

    def _is_valid_large_component(self, filename: str) -> bool:
        pattern = r"^large_L\d+_area\d+_pad2\.png$"
        return bool(re.match(pattern, filename))

    def scan(self) -> list[dict[str, Any]]:
        valid_files = []
        if not os.path.exists(self.root_dir):
            return []

        categories = [
            d
            for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ]

        for category in categories:
            cat_path = os.path.join(self.root_dir, category)
            items = [
                d
                for d in os.listdir(cat_path)
                if os.path.isdir(os.path.join(cat_path, d))
            ]

            for item_name in items:
                item_path = os.path.join(cat_path, item_name)
                try:
                    files_in_item = os.listdir(item_path)
                except OSError:
                    continue

                # 收集該 Item 下所有符合條件的圖片
                for f in files_in_item:
                    full_path = os.path.join(item_path, f)
                    if not os.path.isfile(full_path):
                        continue

                    if f == f"{item_name}_merged.png":
                        valid_files.append(
                            self._create_meta(full_path, category, item_name, "merged")
                        )
                    elif self._is_valid_random_image(f, item_name):
                        valid_files.append(
                            self._create_meta(full_path, category, item_name, "random")
                        )

                large_comp_path = os.path.join(item_path, "large_components")
                if os.path.exists(large_comp_path) and os.path.isdir(large_comp_path):
                    for f in os.listdir(large_comp_path):
                        full_path = os.path.join(large_comp_path, f)
                        if self._is_valid_large_component(f):
                            valid_files.append(
                                self._create_meta(
                                    full_path, category, item_name, "large_component"
                                )
                            )
        return valid_files

    def _create_meta(
        self, path: str, category: str, item_name: str, img_type: str
    ) -> dict[str, Any]:
        return {
            "path": path,
            "metadata": {
                "category": category,
                "item_name": item_name,
                "folder_path": os.path.dirname(path).replace(
                    "/large_components", ""
                ),  # 修正路徑以指向父層
                "filename": os.path.basename(path),
                "image_type": img_type,
                "add_date": datetime.datetime.now().strftime("%Y-%m-%d"),
            },
            "id": f"{item_name}_{os.path.basename(path)}",
        }


# ==========================================
# 模組 3: 資料庫管理 (支援多模型 Collection)
# ==========================================


class MultiModelDBManager:
    def __init__(self, persist_path: str):
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collections = {}  # cache

    def get_collection(self, embedding_fn: BaseEmbeddingFunction):
        name = f"eng_parts_{embedding_fn.get_name()}"
        # 使用 Cosine Distance (1 - similarity)
        return self.client.get_or_create_collection(
            name=name,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def ingest(
        self, data: list[dict], embedding_fn: BaseEmbeddingFunction, batch_size=20
    ):
        collection = self.get_collection(embedding_fn)
        # 檢查是否已經有資料，若有則簡單跳過 (實際生產環境可做增量更新)
        if collection.count() > 0:
            print(
                f"[Info] Collection {collection.name} already has {collection.count()} items. Skipping ingestion."
            )
            return

        print(f"[Info] Ingesting {len(data)} items into {collection.name}...")
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            try:
                collection.add(
                    ids=[b["id"] for b in batch],
                    metadatas=[b["metadata"] for b in batch],
                    images=[b["path"] for b in batch],
                )
            except Exception as e:
                print(f"[Error] Batch failed: {e}")


# ==========================================
# 模組 4: 搜尋與分析引擎 (核心邏輯)
# ==========================================


class SearchEngine:
    def __init__(self, db_manager: MultiModelDBManager):
        self.db_manager = db_manager

    def search_image(
        self,
        query_img_path: str,
        embedding_fns: list[BaseEmbeddingFunction],
        top_k_raw: int = 20,
    ) -> None:
        if not os.path.exists(query_img_path):
            print(f"[Error] Query image not found: {query_img_path}")
            return

        print(f"\n{'=' * 60}")
        print(f"🚀 Search Report for: {os.path.basename(query_img_path)}")
        print(f"{'=' * 60}")

        # 針對每個 Embedding Model 執行搜尋並比較
        for emb_fn in embedding_fns:
            collection = self.db_manager.get_collection(emb_fn)

            # 1. 執行查詢
            # query_images 接受路徑 (由 embedding_fn 處理)
            results = collection.query(
                query_images=[query_img_path],
                n_results=top_k_raw,
                include=["metadatas", "distances"],
            )

            # 2. 轉換為 DataFrame 方便聚合
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]

            if not metadatas:
                print(f"Model: {emb_fn.get_name()} - No results found.")
                continue

            df = pd.DataFrame(metadatas)
            # Cosine Distance 轉 Similarity: Similarity = 1 - Distance
            df["similarity"] = 1 - np.array(distances)

            # 3. 聚合邏輯 (Aggregation)
            # 根據 'item_name' 分組，計算: 出現次數, 最高相似度, 平均相似度
            agg_df = (
                df.groupby("item_name")
                .agg(
                    count=("item_name", "count"),
                    max_sim=("similarity", "max"),
                    avg_sim=("similarity", "mean"),
                    folder_path=("folder_path", "first"),  # 取第一個路徑即可
                )
                .sort_values(by="max_sim", ascending=False)
                .head(5)
            )  # 只取前 5 高的「名稱」

            # 4. 輸出結果
            print(f"\n📊 Model: {emb_fn.get_name()}")
            print("-" * 30)

            rank = 1
            for item_name, row in agg_df.iterrows():
                # 重組原始圖片路徑
                original_img_path = os.path.join(
                    row["folder_path"], f"{item_name}_original.png"
                )

                print(f"#{rank} Name: {item_name}")
                print(f"   ├─ Count (in top {top_k_raw}): {row['count']}")
                print(f"   ├─ Max Similarity: {row['max_sim']:.4f}")
                print(f"   ├─ Avg Similarity: {row['avg_sim']:.4f}")
                print(f"   └─ Original Image: {original_img_path}")
                rank += 1
        print("\n")


# ==========================================
# 主程式
# ==========================================


def main():
    # 設定路徑 (請依實際環境修改)
    ROOT_DIR = "./results/batch2/engineering_images_100dpi_2"
    DB_PATH = "./chroma_db_store_multi"
    QUERY_IMAGE = "./test_query.png"  # 請準備一張圖片做測試，或指向資料集內的某張圖

    # 建立測試用的假 query image (如果不存在)
    if not os.path.exists(QUERY_IMAGE) and os.path.exists(ROOT_DIR):
        # 嘗試隨便抓一張圖當作 query
        crawler = FileCrawler(ROOT_DIR)
        files = crawler.scan()
        if files:
            QUERY_IMAGE = files[0]["path"]
            print(f"[Setup] Using existing image as query: {QUERY_IMAGE}")

    # 1. 準備多個 Embedding Models
    # 模型 A: ViT-B-32 (速度快，標準)
    model_a = OpenCLIPEmbedding(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k")
    # 模型 B: ViT-L-14 (更精準，但較慢) -> 這裡為了演示，使用同一類別但不同參數
    # 若無 GPU，建議註解掉第二個模型以節省時間
    model_b = OpenCLIPEmbedding(model_name="ViT-B-16", pretrained="laion2b_s34b_b88k")

    models_to_compare = [model_a, model_b]

    # 2. 掃描檔案
    crawler = FileCrawler(root_dir=ROOT_DIR)
    files_data = crawler.scan()
    print(f"[Info] Found {len(files_data)} images.")

    # 3. 存入資料庫 (針對每個模型分別建立 Collection)
    db = MultiModelDBManager(DB_PATH)
    if files_data:
        for model in models_to_compare:
            db.ingest(files_data, model)

    # 4. 執行以圖搜圖
    engine = SearchEngine(db)
    engine.search_image(QUERY_IMAGE, models_to_compare, top_k_raw=50)


if __name__ == "__main__":
    main()
