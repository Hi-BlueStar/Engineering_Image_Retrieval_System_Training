"""
SimSiam 資料處理管線 (Data Processing Pipeline)。

專司：ZIP 解壓、PDF 拆圖、連通元件分析提取，與全域資料集大數據建立。
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core import ConfigManager, get_logger, ConfigError
from src.data.pipeline.pdf_extractor import extract_zip, convert_pdfs
from src.data.pipeline.image_preprocessor import run_image_preprocessing
from src.data.pipeline.splitter import DatasetSplitter

def main():
    try:
        config_path = PROJECT_ROOT / "configs" / "default_config.yaml"
        config = ConfigManager.load_from_yaml(config_path)
        
        logger = get_logger("DataPipeline", log_file=Path(config.data.dataset_root) / "data_pipeline.log")
        logger.info("==== 啟動大數據資料前處理管線 ====")
        
        input_source = Path(config.data.input_source)
        pdf_dir = Path(config.data.raw_pdf_dir)
        
        # 1. 偵測輸入源並解壓
        if not config.pipeline_flags.skip_zip_extraction:
            if input_source.is_file() and input_source.suffix.lower() in [".zip", ".rar"]:
                logger.info(f"偵測到壓縮檔 {input_source}，正在解壓縮...")
                extract_zip(input_source, pdf_dir)
            elif input_source.is_dir():
                 logger.info(f"偵測到資料夾 {input_source}，將直接採用其內容...")
                 pdf_dir = input_source
            else:
                 logger.warning(f"輸入源異常，可能遺失或非支援格式: {input_source}")

        # 2. PDF 轉影像
        img_dir = Path(config.data.converted_image_dir)
        if not config.pipeline_flags.skip_pdf_conversion:
            convert_pdfs(
                pdf_dir=pdf_dir,
                output_dir=img_dir,
                dpi=config.pdf_extraction.get("dpi", 100),
                max_workers=config.pdf_extraction.get("max_workers", 8)
            )
            
        # 3. 影像特徵擷取 (CPU 密集平行化)
        prep_dir = Path(config.data.preprocessed_image_dir)
        if not config.pipeline_flags.skip_preprocessing:
             run_image_preprocessing(
                 config_dict=config.image_preprocessing,
                 input_dir=str(img_dir),
                 output_dir=str(prep_dir)
             )
             
        # 4. 資料集大數據零拷貝切割
        logger.info("建置 DataFrame 索引 CSV...")
        splitter = DatasetSplitter(config)
        for i in range(config.data.n_runs):
            seed = config.data.base_seed + i
            train_csv, val_csv = splitter.split(run_idx=i, seed=seed)
            
        logger.info("==== 資料前處理管線完美結案 ====")
        
    except ConfigError as ce:
        print(f"配置嚴重錯誤: {ce}")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
