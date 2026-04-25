"""資料預處理管線入口 (Data Preparation Pipeline Entry Point)。

============================================================
Pipeline 1: 解壓縮 → PDF 轉圖 → 連通元件前處理 → 資料集分割

此腳本完全獨立於模型訓練，可在不同機器上預先準備資料。
所有步驟均支援透過設定檔跳過已完成的階段。

使用方式::

    python v2/prepare_data.py --config v2/configs/default.yaml
    python v2/prepare_data.py --config v2/configs/default.yaml data.skip_extraction=true
============================================================
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# --- 確保專案根目錄在 Python Path ---
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AppConfig
from src.logger import get_logger, setup_logging
from src.training.timer import TimerCollection
from src.data.pdf_converter import convert_pdfs_to_images

logger = get_logger(__name__)


def main() -> None:
    """資料預處理管線主入口。

    流程：
        1. 解壓縮（可選）
        2. PDF → Image 轉換
        3. 影像前處理（連通元件分析）
        4. 資料集分割（多 Run）
    """
    # --- 解析命令列 ---
    parser = argparse.ArgumentParser(
        description="SimSiam 資料預處理管線",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="v2/configs/default.yaml",
        help="YAML 設定檔路徑 (預設: v2/configs/default.yaml)",
    )
    args, overrides = parser.parse_known_args()

    # --- 載入設定 ---
    cfg = AppConfig.from_yaml(args.config, cli_overrides=overrides)
    cfg.validate()

    # --- 初始化日誌 ---
    setup_logging(
        level=cfg.logging.level,
        log_file=(
            str(Path(cfg.experiment.output_dir) / "prepare_data.log")
            if cfg.logging.log_to_file
            else None
        ),
        use_rich=cfg.logging.use_rich,
    )

    timers = TimerCollection()
    total_timer = timers.create("total_pipeline")
    total_timer.start()

    d = cfg.data

    # ========================================================
    # Step 1: 解壓縮
    # ========================================================
    logger.info("=" * 60)
    logger.info("Step 1: 壓縮檔解壓縮")
    logger.info("=" * 60)

    t = timers.create("step_1_extraction")
    t.start()

    from src.data.extraction import extract_archive

    # 1.1 解壓無標籤訓練資料
    logger.info("解壓無標籤訓練資料...")
    extract_archive(
        archive_path=d.raw_zip_path,
        output_dir=d.raw_pdf_dir,
        skip=d.skip_extraction,
    )

    # 1.2 解壓標註評估資料
    if d.labeled_zip_path:
        logger.info("解壓標註評估資料...")
        extract_archive(
            archive_path=d.labeled_zip_path,
            output_dir=d.raw_labeled_pdf_dir,
            skip=d.skip_labeled_extraction,
        )
    t.stop()

    # ========================================================
    # Step 2: PDF → Image
    # ========================================================
    logger.info("=" * 60)
    logger.info("Step 2: PDF → Image 轉換")
    logger.info("=" * 60)

    t = timers.create("step_2_pdf_conversion")
    t.start()

    # 2.1 轉換無標籤 PDF (平鋪)
    if not d.skip_pdf_conversion:
        logger.info("轉換無標籤 PDF...")
        convert_pdfs_to_images(
            pdf_dir=d.raw_pdf_dir,
            output_dir=d.converted_image_dir,
            dpi=d.pdf_dpi,
            max_workers=d.pdf_max_workers,
            skip=d.skip_pdf_conversion,
            preserve_structure=False,
        )
    else:
        logger.info("跳過無標籤 PDF 轉換 (skip_pdf_conversion=True)")

    # 2.2 轉換標註 PDF (保留目錄結構)
    logger.info("檢查標註 PDF 目錄: %s", d.raw_labeled_pdf_dir)
    if Path(d.raw_labeled_pdf_dir).exists():
        logger.info("標註 PDF 目錄存在")
        if not d.skip_labeled_pdf_conversion:
            logger.info("開始轉換標註 PDF (保留結構)...")
            convert_pdfs_to_images(
                pdf_dir=d.raw_labeled_pdf_dir,
                output_dir=d.converted_labeled_image_dir,
                dpi=d.pdf_dpi,
                max_workers=d.pdf_max_workers,
                skip=d.skip_labeled_pdf_conversion,
                preserve_structure=True,
            )
        else:
            logger.info("跳過標註 PDF 轉換 (skip_labeled_pdf_conversion=True)")
    else:
        logger.warning("標註 PDF 目錄不存在: %s", d.raw_labeled_pdf_dir)
    t.stop()

    # ========================================================
    # Step 3: 影像前處理
    # ========================================================
    logger.info("=" * 60)
    logger.info("Step 3: 影像前處理（連通元件分析）")
    logger.info("=" * 60)

    t = timers.create("step_3_preprocessing")
    t.start()

    from src.data.preprocessing import PreprocessConfig, preprocess_images

    prep_cfg = PreprocessConfig(
        input_dir=d.converted_image_dir,
        output_root=d.preprocessed_image_dir,
        max_workers=d.preprocess_max_workers,
        top_n=d.preprocess_top_n,
        max_bbox_ratio=d.preprocess_max_bbox_ratio,
        min_bbox_area=d.preprocess_min_bbox_area,
        padding=d.preprocess_padding,

        use_connected_components=d.use_connected_components,
        use_topology_analysis=d.use_topology_analysis,
        use_topology_pruning=d.use_topology_pruning,
        topology_pruning_iters=d.topology_pruning_iters,
        topology_pruning_ksize=d.topology_pruning_ksize,
        min_simple_area=d.min_simple_area,
        remove_gifu_logo=d.remove_gifu_logo,
        logo_template_path=d.logo_template_path,
        logo_mask_region=d.logo_mask_region,
    )
    preprocess_images(prep_cfg, skip=d.skip_preprocessing)
    t.stop()

    # ========================================================
    # Step 4: 資料集分割（多 Run）
    # ========================================================
    logger.info("=" * 60)
    logger.info("Step 4: 資料集分割 (%d runs)", d.n_runs)
    logger.info("=" * 60)

    t = timers.create("step_4_dataset_split")
    t.start()

    from src.data.splitter import split_dataset

    for run_idx in range(d.n_runs):
        seed = d.base_seed + run_idx
        run_name = f"Run_{run_idx + 1:02d}_Seed_{seed}"
        n_train, n_test = split_dataset(
            source_root=d.preprocessed_image_dir,
            output_root=d.dataset_dir,
            run_name=run_name,
            split_ratio=d.split_ratio,
            seed=seed,
        )
        logger.info("  [%s] train_stems=%d, test_stems=%d", run_name, n_train, n_test)
    t.stop()

    # ========================================================
    # 總結
    # ========================================================
    total_elapsed = total_timer.stop()

    logger.info("=" * 60)
    logger.info("資料預處理管線完成")
    logger.info("=" * 60)
    logger.info("總耗時: %.1f 分鐘", total_elapsed / 60)

    for item in timers.summary():
        logger.info(
            "  %-30s  淨耗時=%.2fs  牆鐘=%.2fs",
            item["name"],
            item["net_elapsed_sec"],
            item["wall_elapsed_sec"],
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("使用者中斷執行")
    except Exception:
        logger.exception("資料預處理管線異常終止")
        sys.exit(1)
