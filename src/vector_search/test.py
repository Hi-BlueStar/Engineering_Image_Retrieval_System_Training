import logging
from vector_search import ChromaDBManager, SimSiamFeatureExtractor, RetrievalEngine, WeightedSumStrategy

# 1. 初始化資料庫與模型
# 確保 ./chroma_db 目錄存在或可寫入
db = ChromaDBManager(db_path="./chroma_db", collection_name="engineering_image")

# 若有預訓練權重，指定 model_path；否則使用隨機初始化 (僅供測試)
extractor = SimSiamFeatureExtractor(model_path="outputs/simsiam_exp_01_Run_01_Seed_42_20260130_105404/checkpoints/checkpoint_best.pth", backbone="resnet18", in_channels=1)

import os
import glob
import cv2
import numpy as np

# 2. 建立檢索引擎 (使用加權總和策略)
strategy = WeightedSumStrategy(weights={"main_view": 1.2, "bom_table": 0.8})
engine = RetrievalEngine(db_manager=db, feature_extractor=extractor, aggregation_strategy=strategy)

from vector_search import ImageIndexer
from src.vector_search.visualizer import RetrievalVisualizer

# 3. 建立索引 (Indexing)
print("\n=== 開始建立索引 ===")
indexer = ImageIndexer(db_manager=db, feature_extractor=extractor)

image_root = "data/engineering_images_Clean_100dpi"
# 若要測試部分檔案，可以只傳入 subfolder 或 file list
# 這裡傳入 root 目錄，使用 batch_size=32
# indexer.index 自動處理遞迴、過濾與批次
indexer.index(sources=image_root, batch_size=32)

print("索引建立流程結束。")

# 4. 執行檢索 (Retrieval)
# 挑選其中一張剛索引過的圖片作為 Query
query_image_path = "data/engineering_images_Clean_100dpi/換刀臂/175B30-DSV-L00-10130.png"
print(f"\n=== 執行檢索 Query: {query_image_path} ===")

top_k = 20
if os.path.exists(query_image_path):
    results = engine.retrieve(query_image_path, top_k=top_k)

    # 5. 輸出結果
    print(f"Top {top_k} Results:")
    for rank, res in enumerate(results, 1):
        print(f"{rank}. PDF ID: {res['parent_pdf_id']}, Score: {res['score']:.4f}")
        print(f"   Matched Components: {res['accumulated_matches']}")
        
    # 6. 視覺化
    print(f"正在產生視覺化報告...")
    viz = RetrievalVisualizer(output_dir="outputs/retrieval_vis")
    viz.visualize(query_img_path=query_image_path, results=results, top_k=top_k, filename="retrieval_top20.png")
    print(f"報告已儲存至 outputs/retrieval_vis/retrieval_top20.png")
    
else:
    print(f"Query image {query_image_path} not found.")


# uv run python src/vector_search/test.py