#!/usr/bin/env python3
"""資料分析入口點 (Data Analysis Entry Point)。

============================================================
提供兩個子命令：

    eda      — 深度探索性資料分析（影像尺寸、像素強度、CC 統計、類別平衡）
    preview  — 前處理管線視覺化預覽（逐步對照圖）

使用範例::

    # 使用預設設定檔執行 EDA
    uv run python v2/analyze_data.py eda --config v2/configs/default.yaml

    # 覆寫資料目錄執行預覽
    uv run python v2/analyze_data.py preview --config v2/configs/default.yaml --n-samples 20 --dpi 400
============================================================
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

# 確保 v2/ 目錄的 src 套件可正常 import
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from rich.console import Console
from rich.panel import Panel

from src.config import AppConfig
from src.logger import setup_logging, get_logger

console = Console()
logger = get_logger(__name__)


# ============================================================
# Sub-command: eda
# ============================================================

def cmd_eda(args: argparse.Namespace, cfg: AppConfig) -> None:
    from src.analysis.eda import EDAAnalyzer

    # 優先序: CLI 參數 > Config 檔案
    data_dir = args.data_dir or cfg.data.converted_image_dir
    output_dir = args.output_dir or str(Path(cfg.experiment.output_dir) / "eda")

    analyzer = EDAAnalyzer(
        data_dir=data_dir,
        output_dir=output_dir,
        sample_n=args.sample_n,
        cc_sample_n=args.cc_sample_n,
        remove_logo=cfg.data.remove_gifu_logo,
        logo_template_path=cfg.data.logo_template_path,
        logo_mask_region=cfg.data.logo_mask_region,
        max_bbox_ratio=cfg.data.preprocess_max_bbox_ratio,
        use_topology_analysis=cfg.data.use_topology_analysis,
        use_topology_pruning=cfg.data.use_topology_pruning,
        topology_pruning_iters=cfg.data.topology_pruning_iters,
        topology_pruning_ksize=cfg.data.topology_pruning_ksize,
        min_simple_area=cfg.data.min_simple_area,
        min_bbox_area=cfg.data.preprocess_min_bbox_area,
        max_workers=cfg.data.preprocess_max_workers,
        seed=cfg.data.base_seed,
    )
    analyzer.run_all()


# ============================================================
# Sub-command: preview
# ============================================================

def cmd_preview(args: argparse.Namespace, cfg: AppConfig) -> None:
    from src.analysis.preview import PreprocessingPreview

    # 優先序: CLI 參數 > Config 檔案
    data_dir = args.data_dir or cfg.data.converted_image_dir
    output_dir = args.output_dir or str(Path(cfg.experiment.output_dir) / "preview")

    # 使用 Config 中的前處理參數
    d = cfg.data
    params = {
        "top_n": d.preprocess_top_n,
        "max_bbox_ratio": d.preprocess_max_bbox_ratio,
        "padding": d.preprocess_padding,

        "use_connected_components": d.use_connected_components,
        "use_topology_analysis": d.use_topology_analysis,
        "use_topology_pruning": d.use_topology_pruning,
        "topology_pruning_iters": d.topology_pruning_iters,
        "topology_pruning_ksize": d.topology_pruning_ksize,
        "min_simple_area": d.min_simple_area,
        "remove_gifu_logo": d.remove_gifu_logo,
        "logo_template_path": d.logo_template_path,
        "logo_mask_region": d.logo_mask_region,
        "min_bbox_area": d.preprocess_min_bbox_area,
        "img_size": cfg.training.img_size,
    }

    preview = PreprocessingPreview(
        input_dir=data_dir,
        n_samples=args.n_samples,
        image_ids=args.image_ids,
        output_dir=output_dir,
        params=params,
        seed=d.base_seed,
        figure_dpi=args.dpi,
    )
    preview.run()


# ============================================================
# Argument parser
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyze_data",
        description="SimSiam v2 — 資料分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # 全域參數
    parser.add_argument(
        "--config",
        type=str,
        default="v2/configs/default.yaml",
        help="YAML 設定檔路徑 (預設: v2/configs/default.yaml)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ---- eda ----
    p_eda = sub.add_parser("eda", help="深度探索性資料分析 (EDA)")
    p_eda.add_argument("--data-dir", help="影像來源目錄 (預設使用 config.data.converted_image_dir)")
    p_eda.add_argument("--output-dir", help="分析結果輸出目錄 (預設使用 outputs/eda)")
    p_eda.add_argument("--sample-n", type=int, default=500, help="像素強度分析取樣數")
    p_eda.add_argument("--cc-sample-n", type=int, default=200, help="CC 分析取樣數")
    p_eda.set_defaults(func=cmd_eda)

    # ---- preview ----
    p_prev = sub.add_parser("preview", help="前處理管線視覺化預覽")
    p_prev.add_argument("--data-dir", help="原始影像目錄 (預設使用 config.data.converted_image_dir)")
    p_prev.add_argument("--n-samples", type=int, default=5, help="隨機抽取影像數")
    p_prev.add_argument("--image-ids", nargs="+", help="指定影像 ID (檔案名稱，不含副檔名) 列表")
    p_prev.add_argument("--output-dir", help="預覽圖輸出目錄 (預設使用 outputs/preview)")
    p_prev.add_argument("--dpi", type=int, default=150, help="輸出圖片 DPI")
    p_prev.set_defaults(func=cmd_preview)

    return parser


# ============================================================
# Entry point
# ============================================================

def main() -> None:
    # --- 1. 解析命令列 (抽取出 --config 與 overrides) ---
    parser = build_parser()
    args, overrides = parser.parse_known_args()

    # 處理誤放在子命令後的 --config (argparse 子解析器限制)
    if "--config" in overrides:
        idx = overrides.index("--config")
        if idx + 1 < len(overrides):
            args.config = overrides[idx + 1]
            logger.debug("從未知參數中擷取到 --config: %s", args.config)

    # 過濾出真正的 dotlist 覆寫 (必須包含 '=' 且不以 '-' 開頭)
    dotlist_overrides = [o for o in overrides if "=" in o and not o.startswith("-")]
    
    # --- 2. 載入設定 ---
    try:
        cfg = AppConfig.from_yaml(args.config, cli_overrides=dotlist_overrides)
        cfg.validate()
    except Exception as e:
        console.print(f"[bold red]載入設定失敗:[/bold red] {e}")
        sys.exit(1)

    # --- 3. 初始化日誌 (依循 prepare_data.py 模式) ---
    setup_logging(
        level=cfg.logging.level,
        log_file=(
            str(Path(cfg.experiment.output_dir) / "analyze_data.log")
            if cfg.logging.log_to_file
            else None
        ),
        use_rich=cfg.logging.use_rich,
        force=True,
    )

    console.print(
        Panel(
            f"[bold]SimSiam v2 — Data Analysis Toolkit[/bold]\n"
            f"Command: [cyan]{args.command}[/cyan] | Config: [cyan]{args.config}[/cyan]",
            border_style="blue",
        )
    )
    logger.info("Starting SimSiam v2 Data Analysis Toolkit")
    
    # 執行子命令 (傳入 args 與 cfg)
    args.func(args, cfg)


if __name__ == "__main__":
    main()
