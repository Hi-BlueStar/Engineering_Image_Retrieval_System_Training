import os

import chromadb
import pandas as pd
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction


# 設定 pandas 顯示選項，確保輸出表格整齊
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.float_format", "{:.4f}".format)


def get_collection(persist_path: str = "./chroma_data"):
    """
    連接到現有的 ChromaDB
    """
    client = chromadb.PersistentClient(path=persist_path)
    embedding_func = OpenCLIPEmbeddingFunction()
    data_loader = ImageLoader()

    # 取得現有的 Collection
    collection = client.get_collection(
        name="engineering_components_gallery",
        embedding_function=embedding_func,
        data_loader=data_loader,
    )
    return collection


def search_and_aggregate(query_image_path: str, collection, top_k: int = 20):
    """
    以圖搜圖，並聚合統計結果
    """
    if not os.path.exists(query_image_path):
        print(f"錯誤: 找不到查詢圖片 {query_image_path}")
        return

    print(f"正在搜尋圖片: {query_image_path} ...")

    # 1. 執行查詢
    # query_uris 會讓 ChromaDB 使用 ImageLoader 自動讀取圖片並透過 OpenCLIP 轉向量
    results = collection.query(
        query_uris=[query_image_path],
        n_results=top_k,
        include=[
            "metadatas",
            "distances",
        ],  # 我們需要 MetaData (取得名稱) 和 Distances (計算相似度)
    )

    # 2. 解析結果
    # ChromaDB 回傳的是 List[List]，因為可以一次查多張，我們只查一張所以取 [0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not metadatas:
        print("未找到任何結果。")
        return

    # 3. 資料處理與相似度轉換
    # ChromaDB 預設回傳「距離 (Distance)」，越小越好。
    # 這裡我們將其轉換為「相似度 (Similarity)」，越高越好 (0~1)。
    # 針對 OpenCLIP (Cosine Distance)，通常 Similarity = 1 - Distance
    parsed_data = []
    for meta, dist in zip(metadatas, distances):
        similarity = 1 - dist
        parsed_data.append(
            {
                "item_name": meta.get("item_name", "Unknown"),
                "category": meta.get("category", "Unknown"),
                "type": meta.get(
                    "type", "unknown"
                ),  # 擷取圖片類型 (merged, random, large_component)
                "similarity": similarity,
                "filename": meta.get("filename", "Unknown"),
            }
        )

    # 轉為 DataFrame 方便計算
    df = pd.DataFrame(parsed_data)

    # 4. 聚合統計 (Aggregation)
    # 根據 item_name 分組，計算各項指標
    stats = (
        df.groupby("item_name")
        .agg(
            category=("category", "first"),  # 取第一個出現的分類
            primary_match_type=(
                "type",
                lambda x: x.value_counts().index[0],
            ),  # 找出出現頻率最高的類型
            count=("item_name", "count"),  # 出現次數
            max_similarity=("similarity", "max"),  # 最高相似度
            avg_similarity=("similarity", "mean"),  # 平均相似度
        )
        .reset_index()
    )

    # 5. 排序策略
    # 這裡採取「混合排序」：先看誰有最高的相似度 (Max Sim)，通常這代表最強的特徵匹配
    # 您也可以改為 sort_values(by='count', ...) 採用「多數決」
    stats_sorted = stats.sort_values(by="max_similarity", ascending=False).reset_index(
        drop=True
    )

    # 6. 輸出結果
    print("\n====== 搜尋結果統計 (前 5 高) ======")
    # 顯示欄位包含新的 primary_match_type
    print(
        stats_sorted[
            ["item_name", "category", "primary_match_type", "count", "max_similarity"]
        ].head(5)
    )

    print("\n====== 詳細分析 (Top 1) ======")
    top_result = stats_sorted.iloc[0]

    # 針對不同匹配類型給出不同解讀
    match_type_desc = {
        "merged": "整體外觀 (Merged)",
        "large_component": "局部特徵 (Large Component)",
        "random_sample": "隨機視角 (Random)",
    }.get(top_result["primary_match_type"], top_result["primary_match_type"])

    print(f"判定結果名稱: {top_result['item_name']}")
    print(f"所屬分類:     {top_result['category']}")
    print(f"主要匹配特徵: {match_type_desc}")  # 顯示匹配來源
    print(f"信心指標:     {top_result['max_similarity']:.4f} (最高相似度)")
    print(
        f"佐證數量:     {top_result['count']} / {top_k} (在前 {top_k} 筆結果中出現的次數)"
    )

    return stats_sorted


if __name__ == "__main__":
    # 範例：請將此路徑替換為您想測試的圖片路徑
    # 它可以是資料庫沒看過的圖片 (例如現場拍攝的照片)
    TEST_IMAGE_PATH = "test_query_image.png"

    # 建立一個假的測試圖來跑流程 (實際使用請換成真實路徑)
    # 若您沒有圖片，這段會報錯，請確保路徑存在
    if not os.path.exists(TEST_IMAGE_PATH):
        # 這裡只是為了演示，隨便抓資料夾裡的一張圖來當 Query
        demo_dir = "engineering_images_100dpi_2/ATC鈑金/AK0OCVE8-50060010100"
        if os.path.exists(demo_dir):
            files = [f for f in os.listdir(demo_dir) if f.endswith("_merged.png")]
            if files:
                TEST_IMAGE_PATH = os.path.join(demo_dir, files[0])

    if os.path.exists(TEST_IMAGE_PATH):
        chroma_collection = get_collection()
        # 搜尋前 20 筆最像的圖片，然後做統計
        search_and_aggregate(TEST_IMAGE_PATH, chroma_collection, top_k=20)
    else:
        print("請設定有效的 TEST_IMAGE_PATH 進行測試")
