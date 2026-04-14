"""
Vector Search Module
初始化模組，暴露主要類別。
"""

from .interfaces import BaseEvaluator, ScoreAggregationStrategy
from .database import ChromaDBManager
from .feature_extractor import SimSiamFeatureExtractor
from .engine import RetrievalEngine, WeightedSumStrategy, MaxPoolingStrategy
from .indexer import ImageIndexer

__all__ = [
    "BaseEvaluator",
    "ScoreAggregationStrategy",
    "ChromaDBManager",
    "SimSiamFeatureExtractor",
    "RetrievalEngine",
    "WeightedSumStrategy",
    "MaxPoolingStrategy",
    "ImageIndexer",
]
