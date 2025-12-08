import os

import chromadb
import numpy as np
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from PIL import Image


# 設定參數 (必須與入庫時的設定一致)
COLLECTION_NAME = "engineering_components_v1"
DB_PATH = "./chroma_db_store"


def init_db():
    """初始化資料庫連線"""
    client = chromadb.PersistentClient(path=DB_PATH)
    embedding_func = OpenCLIPEmbeddingFunction()

    # 取得現有的 Collection
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func,
        data_loader=ImageLoader(),
    )
    return collection


def display_results(query_input, results, search_type="text"):
    """
    格式化並顯示搜尋結果
    """
    print(f"\n{'=' * 20} 搜尋結果 ({search_type}) {'=' * 20}")
    print(f"查詢輸入: {query_input}")
    print("-" * 60)

    # ChromaDB 回傳的結果是 list of list，因為可以一次查詢多筆
    # 這裡我們只取第一筆查詢的結果
    ids = results["ids"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]
    uris = results["uris"][0] if results["uris"] else []

    for rank, (id, dist, meta, uri) in enumerate(zip(ids, distances, metadatas, uris)):
        # 距離越小代表越相似 (Cosine Distance 或 L2)
        similarity_score = 1 - dist  # 簡單轉換為相似度 (僅供參考，視距離度量而定)

        print(f"Rank {rank + 1} | ID: {id[:8]}... | 距離: {dist:.4f}")
        print(f"  📂 檔名: {meta.get('filename')}")
        print(f"  🏷️ 類別: {meta.get('category')} | 料號: {meta.get('part_id')}")
        print(f"  🔗 路徑: {uri}")
        print("-" * 30)


def search_by_text(collection, query_text, n_results=3):
    """
    文字搜圖: 使用自然語言描述尋找圖片
    """
    print(f"\n[系統] 正在執行文字搜尋: '{query_text}' ...")

    results = collection.query(
        query_texts=[query_text],  # Chroma 會自動呼叫 OpenCLIP 將文字轉為向量
        n_results=n_results,
        include=["metadatas", "distances", "uris"],  # 指定回傳欄位
    )

    display_results(query_text, results, search_type="Text")


def search_by_image(collection, query_image_path, n_results=3):
    """
    以圖搜圖: 使用圖片尋找相似圖片
    """
    if not os.path.exists(query_image_path):
        print(f"錯誤: 找不到查詢圖片 {query_image_path}")
        return

    print(f"\n[系統] 正在執行圖片搜尋 (Reference: {query_image_path}) ...")

    # 讀取圖片並轉為 numpy array 供 Chroma 處理
    try:
        image = np.array(Image.open(query_image_path))

        results = collection.query(
            query_images=[image],  # Chroma 會自動呼叫 OpenCLIP 將圖片轉為向量
            n_results=n_results,
            include=["metadatas", "distances", "uris"],
        )

        display_results(query_image_path, results, search_type="Image")

    except Exception as e:
        print(f"圖片處理失敗: {e}")


# --- 主程式 ---
if __name__ == "__main__":
    # 1. 初始化
    try:
        collection = init_db()
        print(f"成功連接資料庫: {COLLECTION_NAME}")
        print(f"資料庫中共有 {collection.count()} 筆資料")
    except Exception as e:
        print(f"資料庫連接失敗: {e}")
        exit()

    # 2. 測試案例：文字搜尋
    # 情境：工程師忘記料號，只記得外觀
    user_query = "大型的ATC鈑金組件"
    search_by_text(collection, user_query)

    # 3. 測試案例：以圖搜圖
    # 情境：假設我們有一張剛拍攝或隨機選取的圖片，想找它的原始設計圖或相似件
    # 注意：這裡請將路徑替換為您硬碟中實際存在的圖片路徑
    sample_image_path = "engineering_images_100dpi_2/ATC鈑金/AK0OCVE8-50060010100/AK0OCVE8-50060010100_random_01.png"

    # 如果檔案存在才執行測試
    if os.path.exists(sample_image_path):
        search_by_image(collection, sample_image_path)
    else:
        print(
            "\n[提示] 請修改 sample_image_path 變數為您實際的圖片路徑以測試以圖搜圖。"
        )
