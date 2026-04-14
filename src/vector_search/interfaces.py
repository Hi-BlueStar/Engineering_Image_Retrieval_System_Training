"""
Vector Search 模組的介面定義檔。
包含策略模式的抽象層與評估器介面。
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class ScoreAggregationStrategy(ABC):
    """
    分數聚合策略的抽象基類 (Abstract Base Class)。
    
    負責定義如何將同一張圖紙(Pdf)下的多個元件(Components)檢索分數聚合為單一分數，
    以便進行最終的圖紙層級排序。
    """
    
    @abstractmethod
    def aggregate(self, component_scores: List[float], metadata_list: List[Dict[str, Any]]) -> float:
        """
        聚合多個元件的分數。

        Args:
            component_scores (List[float]): 該圖紙下所有檢索到的元件相似度分數列表。
            metadata_list (List[Dict[str, Any]]): 對應每個分數的元件 Metadata 列表，
                                                  可用於進階權重計算 (如根據 component_type)。

        Returns:
            float: 聚合後的最終分數 (代表該圖紙與查詢的相似度)。
        """
        pass

class BaseEvaluator(ABC):
    """
    檢索系統評估器的抽象基類。
    
    規範評估方法的介面，確保不同的評估指標 (mAP, Recall@K 等) 遵循統一的調用方式。
    """

    @abstractmethod
    def evaluate(self, query_dataset: Any, ground_truth_map: Dict[str, Any]) -> Dict[str, float]:
        """
        執行評估並返回指標結果。

        Args:
            query_dataset (Any): 查詢資料集，具體型別取決於實作 (可能是 DataLoader 或 List)。
            ground_truth_map (Dict[str, Any]): 地面真值映射表，通常為 {query_id: expected_result}。

        Returns:
            Dict[str, float]: 包含評估指標名稱與數值的字典，例如 {'mAP': 0.85, 'Recall@10': 0.92}。
        """
        pass
