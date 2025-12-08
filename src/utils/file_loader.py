# src/utils/file_loader.py
import re
import logging
from pathlib import Path
from typing import List, Generator
from src.core.interfaces import IFileFilter

# 設定 Logger
logger = logging.getLogger(__name__)

class StrictRegexFilter(IFileFilter):
    """
    實作特定專案需求的過濾邏輯：
    1. A: {name}_merged.png
    2. B: {name}_random_01.png ~ 20.png (精確匹配)
    3. C: large_components/{name}_large_L{d}_area{d}_pad2.png
    """
    def __init__(self):
        # 規則 A: Merged
        self.pattern_merged = re.compile(r".+_merged\.png$")
        
        # 規則 B: Random 01-20 (嚴格限制數字範圍)
        # (?:0[1-9]|1[0-9]|20) 確保只匹配 01-20
        self.pattern_random = re.compile(r".+_random_(?:0[1-9]|1[0-9]|20)\.png$")
        
        # 規則 C: Large Components (需檢查檔名與父目錄)
        self.pattern_large = re.compile(r".+_large_L\d+_area\d+_pad2\.png$")

    def match(self, file_path: Path) -> bool:
        filename = file_path.name
        
        # 檢查規則 A
        if self.pattern_merged.match(filename):
            return True
            
        # 檢查規則 B
        if self.pattern_random.match(filename):
            return True
            
        # 檢查規則 C (需同時滿足目錄與檔名)
        if file_path.parent.name == "large_components" and self.pattern_large.match(filename):
            return True
            
        return False

class ImageLoader:
    """負責遍歷檔案系統並應用過濾器"""
    def __init__(self, root_dir: str | Path, file_filter: IFileFilter):
        self.root_dir = Path(root_dir)
        self.filter = file_filter

    def scan(self) -> List[Path]:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory not found: {self.root_dir}")
            
        valid_files: List[Path] = []
        logger.info(f"Starting scan in {self.root_dir}...")

        # 遞迴遍歷所有 png 檔案 (效能優化：先用 glob 縮小範圍)
        # 若有非 png 需求，可改為 rglob('*')
        for file_path in self.root_dir.rglob("*.png"):
            if self.filter.match(file_path):
                valid_files.append(file_path)
        
        logger.info(f"Scan complete. Found {len(valid_files)} valid images.")
        return valid_files