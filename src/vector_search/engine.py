"""
檢索與聚合核心引擎。
負責協調特徵提取、資料庫搜尋以及結果聚合重排序。
"""
import logging
from typing import List, Dict, Any, Type
import numpy as np
import cv2  # 用於簡易影像讀取與切割
from pathlib import Path
from collections import defaultdict

from src.vector_search.interfaces import ScoreAggregationStrategy
from src.vector_search.database import ChromaDBManager
from src.vector_search.feature_extractor import SimSiamFeatureExtractor
from src.vector_search.utils import extract_rois_from_image

logger = logging.getLogger(__name__)

# =============================================================================
# Aggregation Strategies
# =============================================================================

class WeightedSumStrategy(ScoreAggregationStrategy):
    """
    加權總和策略。
    
    將同一圖紙下的所有元件相似度分數進行加權求和。
    預設所有元件權重為 1.0。
    """
    def __init__(self, weights: Dict[str, float] = None):
        """
        Args:
            weights (Dict[str, float], optional): 元件類型權重表 {component_type: weight}。
        """
        self.weights = weights if weights else {}

    def aggregate(self, component_scores: List[float], metadata_list: List[Dict[str, Any]]) -> float:
        total_score = 0.0
        for score, meta in zip(component_scores, metadata_list):
            c_type = meta.get('component_type', 'default')
            w = self.weights.get(c_type, 1.0)
            total_score += score * w
        return total_score

class MaxPoolingStrategy(ScoreAggregationStrategy):
    """
    最大化策略。
    
    取同一圖紙下所有元件中的最高相似度分數作為代表。
    適用於「只要有一個關鍵元件匹配即可」的情境。
    """
    def aggregate(self, component_scores: List[float], metadata_list: List[Dict[str, Any]]) -> float:
        if not component_scores:
            return 0.0
        return max(component_scores)

# =============================================================================
# Retrieval Engine
# =============================================================================

class RetrievalEngine:
    """
    高精度工程影像檢索引擎。
    
    整合 Feature Extractor 與 ChromaDB，透過策略模式執行多元件聚合檢索。
    """

    def __init__(
        self,
        db_manager: ChromaDBManager,
        feature_extractor: SimSiamFeatureExtractor,
        aggregation_strategy: ScoreAggregationStrategy = None
    ):
        """
        初始化檢索引擎。

        Args:
            db_manager: 向量資料庫管理器實例。
            feature_extractor: 特徵提取器實例。
            aggregation_strategy: 聚合策略實例 (預設為 WeightedSumStrategy)。
        """
        self.db = db_manager
        self.extractor = feature_extractor
        self.strategy = aggregation_strategy if aggregation_strategy else WeightedSumStrategy()

    def set_strategy(self, strategy: ScoreAggregationStrategy):
        """動態切換聚合策略。"""
        logger.info(f"切換聚合策略為: {strategy.__class__.__name__}")
        self.strategy = strategy

    def _split_query_image(self, query_img_path: str) -> List[np.ndarray]:
        """
        將查詢影像切割為元件。
        
        使用與 Indexer 一致的切割邏輯 (src.vector_search.utils.extract_rois_from_image)。
        
        Args:
            query_img_path (str): 查詢影像路徑。

        Returns:
            List[np.ndarray]: 元件影像列表。
        """
        img = cv2.imread(query_img_path)
        if img is None:
            logger.error(f"無法讀取影像: {query_img_path}")
            return []
            
        # 使用共用邏輯切割 ROI
        # extract_rois_from_image 回傳 List[Tuple[img, info]]
        try:
            rois_with_info = extract_rois_from_image(img, top_n=5)
            
            if not rois_with_info:
                # 若無明顯 ROI，退化為使用原圖
                return [img]
                
            # 只回傳影像部分
            return [roi_img for roi_img, info in rois_with_info]
            
        except Exception as e:
            logger.error(f"ROI 切割失敗: {e}，將使用原圖")
            return [img]

    def retrieve(self, query_img_path: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        執行檢索流程。

        1. 輸入查詢圖片 -> 切分為 N 個元件
        2. 對每個元件在 ChromaDB 進行 KNN 搜尋
        3. 根據 parent_pdf_id 進行分組
        4. 呼叫聚合策略計算最終分數
        5. 返回 Top-K 結果

        Args:
            query_img_path: 查詢圖片路徑。
            top_k: 返回結果數量。

        Returns:
            List[Dict]: 排序後的結果列表，每項包含 {parent_pdf_id, score, matched_components_count}。
        """
        logger.info(f"開始檢索: {query_img_path}")
        
        # 1. 切割元件
        components = self._split_query_image(query_img_path)
        if not components:
            return []
            
        # 2. 提取特徵
        embeddings = self.extractor.extract_batch(components)
        if len(embeddings) == 0:
            return []
            
        # 3. 搜尋 (Query ChromaDB)
        # 對每個元件都去搜 Top-K (這裡是對每個 query component 找 Top-K，總共 N * K 候選)
        query_results = self.db.query_vectors(
            query_embeddings=embeddings.tolist(),
            n_results=top_k
        )
        
        # 4. 收集與分組 (Group by parent_pdf_id)
        # query_results 結構: {'ids': [[id1, id2...], [id1, ...]], 'distances': [...], 'metadatas': [...]}
        candidates = defaultdict(lambda: {'scores': [], 'metadatas': []})
        
        num_components = len(embeddings)
        for i in range(num_components):
            batch_ids = query_results['ids'][i]
            batch_dists = query_results['distances'][i]
            batch_metas = query_results['metadatas'][i]
            
            for doc_id, dist, meta in zip(batch_ids, batch_dists, batch_metas):
                if not meta: continue
                
                pdf_id = meta.get('parent_pdf_id')
                if not pdf_id: continue
                
                # 轉換 Distance 為 Similarity Score
                # 假設 Chroma 使用 Cosine Distance (0~2), Sim = 1 - Dist
                similarity = 1.0 - dist
                
                candidates[pdf_id]['scores'].append(similarity)
                candidates[pdf_id]['metadatas'].append(meta)

        # 5. 聚合評分 (Aggregation)
        final_scores = []
        for pdf_id, data in candidates.items():
            final_score = self.strategy.aggregate(data['scores'], data['metadatas'])
            final_scores.append({
                'parent_pdf_id': pdf_id,
                'score': final_score,
                'accumulated_matches': len(data['scores']), # 匹配到的元件總次數
                'details': data['metadatas'] # 除錯用
            })
            
        # 6. 排序 (Re-ranking)
        final_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # 返回 Top-K
        return final_scores[:top_k]
