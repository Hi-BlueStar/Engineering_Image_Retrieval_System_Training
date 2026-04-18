"""資料準備管線指令 (Data Preparation Pipeline)。

執行：
1. ZIP 解壓縮
2. PDF 轉影像
3. 影像前處理
4. 分割 Train/Val Run
"""
import argparse
import subprocess
from pathlib import Path
from src.config.structured import Config
from src.tools.pdf_converter import PDFConverter
from src.tools.image_preprocessor import ImagePreprocessor
from src.tools.splitter import DatasetSplitter

def main():
    parser = argparse.ArgumentParser(description="Run Data Preparation Pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    args, unknown = parser.parse_known_args()
    
    cfg = Config.load(args.config, unknown)
    
    print("\n" + "="*50)
    print("🚀 啟動資料準備管線 (Phase 1)")
    print("="*50)

    # 1. ZIP
    if not cfg.pipeline.skip_zip_extraction and cfg.data.raw_zip_path:
        zip_p = Path(cfg.data.raw_zip_path)
        pdf_dir = Path(cfg.data.raw_pdf_dir)
        if zip_p.exists() and not pdf_dir.exists():
            print(f"📦 解壓縮 {zip_p} => {pdf_dir}")
            pdf_dir.mkdir(parents=True, exist_ok=True)
            if zip_p.suffix.lower() == ".rar":
                subprocess.run(["7z", "x", str(zip_p), f"-o{pdf_dir}", "-y", "-mmt=on"], check=True)
            else:
                import zipfile
                with zipfile.ZipFile(zip_p, 'r') as z:
                    z.extractall(pdf_dir)

    # 2. PDF to Image
    if not cfg.pipeline.skip_pdf_conversion:
        pdf_conv = PDFConverter(cfg.data.raw_pdf_dir, cfg.data.converted_image_dir, cfg.data.pdf_dpi, cfg.data.pdf_max_workers)
        pdf_conv.run()

    # 3. Preprocess
    if not cfg.pipeline.skip_preprocessing:
        pre_cfg = {
            "preprocess_top_n": cfg.data.preprocess_top_n,
            "preprocess_remove_largest": cfg.data.preprocess_remove_largest,
            "preprocess_padding": cfg.data.preprocess_padding,
            "preprocess_max_attempts": cfg.data.preprocess_max_attempts,
            "preprocess_random_count": cfg.data.preprocess_random_count,
            "preprocess_max_workers": cfg.data.preprocess_max_workers,
        }
        preproc = ImagePreprocessor(cfg.data.converted_image_dir, cfg.data.preprocessed_image_dir, pre_cfg)
        preproc.run()

    # 4. Split
    print("\n📂 正在進行資料分割...")
    splitter = DatasetSplitter(cfg.data.preprocessed_image_dir, cfg.data.dataset_root, cfg.data.split_ratio)
    splitter.scan()
    for i in range(cfg.data.n_runs):
        splitter.split_run(i + 1, cfg.data.base_seed + i)
        
    print("\n✅ 資料準備管線執行完畢。")

if __name__ == "__main__":
    main()
