# src/utils/metadata.py
import re
from pathlib import Path
from datetime import datetime
from src.core.interfaces import IMetadataExtractor, ImageMetadata

class ATCMetadataExtractor(IMetadataExtractor):
    """針對 ATC 鈑金專案的 Metadata 提取實作"""
    
    def __init__(self):
        # 預編譯 Regex 以提升效能
        self.re_suffix = re.compile(
            r"(_merged|_random_\d+|_large_L\d+_area\d+_pad2)\.png$"
        )

    def _determine_tag(self, filename: str, parent_dir: str) -> str:
        if "large_components" in parent_dir:
            return "large_component"
        if "_merged" in filename:
            return "merged"
        if "_random_" in filename:
            return "random_sample"
        return "unknown"

    def extract(self, file_path: Path) -> ImageMetadata:
        filename = file_path.name
        parent_name = file_path.parent.name
        
        # 1. Category 固定
        category = "ATC鈑金"
        
        # 2. Tag 判斷
        tag = self._determine_tag(filename, parent_name)
        
        # 3. ID 清洗 (移除後綴，保留原始工件名稱)
        # 例如: part123_merged.png -> part123
        clean_id = self.re_suffix.sub("", filename)
        # 若 regex 未匹配到預期後綴（理論上被 filter 擋過不會發生），退回到 stem
        if clean_id == filename: 
            clean_id = file_path.stem

        return ImageMetadata(
            id=clean_id,
            category=category,
            source_path=str(file_path.absolute()),
            tag=tag,
            created_at=datetime.now().isoformat()
        )