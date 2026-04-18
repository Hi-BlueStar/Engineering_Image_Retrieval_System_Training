"""影像前處理工具模組 (Image Preprocessor)。

封裝 `image_preprocessing_core` (原 `batch_multiprocess2`) 邏輯。
已經將依賴複製至 v2 本地，解除了對上一層目錄的耦合。
"""
import os
from pathlib import Path


class ImagePreprocessor:
    def __init__(self, input_dir: str, output_dir: str, cfg: dict):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.cfg = cfg

    def run(self):
        """執行批次前處理。"""
        try:
            from .image_preprocessing_core import BatchConfig, process_folder
        except ImportError as e:
            print(f"[Error] 無法載入本地影像處理模組: {e}")
            return {"ok": 0, "failed": 0}
            
        bcfg = BatchConfig(
            input_dir=self.input_dir,
            output_root=self.output_dir,
            patterns=(".png", ".jpg", ".jpeg"),
            recursive=True,
            per_image_outdir="{stem}",
            skip_existing=True,
            max_workers=self.cfg.get("preprocess_max_workers", 12),
            top_n=self.cfg.get("preprocess_top_n", 5),
            remove_largest=self.cfg.get("preprocess_remove_largest", True),
            padding=self.cfg.get("preprocess_padding", 2),
            max_attempts=self.cfg.get("preprocess_max_attempts", 400),
            random_count=self.cfg.get("preprocess_random_count", 10),
            write_report_json=True
        )
        
        print("🔬 開始進行影像連通元件分析...")
        report = process_folder(bcfg)
        return report
