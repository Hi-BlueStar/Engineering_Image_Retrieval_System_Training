"""
向量資料庫管理層。
負責與 ChromaDB 進行交互，包含資料的 Upsert 與 Query。
"""
import logging
import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from chromadb.errors import ChromaError

# 設定 Logger
logger = logging.getLogger(__name__)
# 基本設定，若主程式未設定 logging 則會生效
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

class ChromaDBManager:
    """
    ChromaDB 管理器。
    
    負責封裝 ChromaDB Client 的初始化、Collection 管理以及向量數據的寫入與查詢。
    強制啟用持久化存儲。
    """

    REQUIRED_METADATA_KEYS = {"original_filename", "page_num", "component_type", "parent_pdf_id"}

    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "engineering_drawings"):
        """
        初始化 ChromaDBManager。

        Args:
            db_path (str): 資料庫持久化路徑。預設為 "./chroma_db"。
            collection_name (str): Collection 名稱。預設為 "engineering_drawings"。
        """
        self.db_path = db_path
        self.collection_name = collection_name
        
        try:
            logger.info(f"正在初始化 ChromaDB Client，路徑: {self.db_path}")
            # 使用 PersistentClient 啟用持久化
            self.client = chromadb.PersistentClient(path=self.db_path)
            
            # Get or Create Collection
            # 使用 cosine distance 作為預設距離度量
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"成功加載 Collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"ChromaDB 初始化失敗: {e}")
            raise RuntimeError(f"無法初始化 ChromaDB: {e}")

    def upsert_vectors(self, vectors: List[List[float]], metadatas: List[Dict[str, Any]], ids: List[str]):
        """
        新增或更新向量資料 (Upsert)。

        Args:
            vectors (List[List[float]]): 向量列表 (Embedding)。
            metadatas (List[Dict[str, Any]]): Metadata 列表，必須包含必要的鍵值。
            ids (List[str]): 唯一 ID 列表。

        Raises:
            ValueError: 若輸入列表長度不一致或 Metadata 缺少必要鍵值。
            RuntimeError: 若資料庫操作失敗。
        """
        if not (len(vectors) == len(metadatas) == len(ids)):
            raise ValueError("vectors, metadatas, ids 的長度必須一致。")

        # 驗證 Metadata Schema
        for i, meta in enumerate(metadatas):
            missing_keys = self.REQUIRED_METADATA_KEYS - meta.keys()
            if missing_keys:
                error_msg = f"Index {i} 的 Metadata 缺少必要鍵值: {missing_keys}"
                logger.error(error_msg)
                raise ValueError(error_msg)

        try:
            self.collection.upsert(
                embeddings=vectors,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"成功 Upsert {len(ids)} 筆資料到 ChromaDB。")
            
        except ChromaError as e:
            logger.error(f"ChromaDB Upsert 操作失敗: {e}")
            raise RuntimeError(f"資料庫寫入錯誤: {e}")
        except Exception as e:
            logger.error(f"未預期的錯誤: {e}")
            raise

    def query_vectors(self, query_embeddings: List[List[float]], n_results: int = 20, where: Optional[Dict] = None) -> Dict[str, Any]:
        """
        查詢相似向量 (KNN Search)。

        Args:
            query_embeddings (List[List[float]]): 查詢向量列表。
            n_results (int): 返回的 Top-K 數量。
            where (Optional[Dict]): Metadata 過濾條件 (ChromaDB 語法)。

        Returns:
            Dict[str, Any]: 查詢結果字典，包含 'ids', 'distances', 'metadatas', 'documents'。
        """
        try:
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                include=["metadatas", "distances"]
            )
            return results
        except Exception as e:
            logger.error(f"查詢失敗: {e}")
            raise RuntimeError(f"查詢執行錯誤: {e}")
