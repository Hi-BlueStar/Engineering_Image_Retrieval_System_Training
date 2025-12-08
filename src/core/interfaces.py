# src/core/interfaces.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ImageMetadata:
    """統一的圖片 Metadata 結構，方便後續寫入向量資料庫 (如 Milvus/Qdrant)"""

    id: str
    category: str
    source_path: str
    tag: str
    created_at: str  # ISO 8601 format recommended
    extra: dict[str, Any] | None = None


class IMetadataExtractor(ABC):
    """Metadata 提取策略介面"""

    @abstractmethod
    def extract(self, file_path: Path) -> ImageMetadata:
        pass


class IFileFilter(ABC):
    """檔案過濾策略介面"""

    @abstractmethod
    def match(self, file_path: Path) -> bool:
        pass
