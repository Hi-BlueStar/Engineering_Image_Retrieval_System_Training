#!/usr/bin/env python3
"""資料分析入口點 (Data Analysis Entry Point)。

============================================================
提供兩個子命令：

    eda      — 深度探索性資料分析（影像尺寸、像素強度、CC 統計、類別平衡）
    preview  — 前處理管線視覺化預覽（逐步對照圖）

使用範例::

    # EDA 分析
    python v2/analyze_data.py eda \\
        --data-dir data/converted_images \\
        --output-dir outputs/eda \\
        --sample-n 500 --cc-sample-n 200

    # 前處理預覽（使用預設參數）
    python v2/analyze_data.py preview \\
        --data-dir data/converted_images \\
        --n-samples 5 \\
        --output-dir outputs/preview

    # 前處理預覽（自訂參數）
    python v2/analyze_data.py preview \\
        --data-dir data/converted_images \\
        --n-samples 8 \\
        --top-n 3 \\
        --no-remove-largest \\
        --no-topology

============================================================
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 確保 v2/ 目錄的 src 套件可正常 import
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from rich.console import Console
from rich.panel import Panel

console = Console()


# ============================================================
# Sub-command: eda
# ============================================================

def cmd_eda(args: argparse.Namespace) -> None:
    from src.analysis.eda import EDAAnalyzer

    analyzer = EDAAnalyzer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        sample_n=args.sample_n,
        cc_sample_n=args.cc_sample_n,
        seed=args.seed,
    )
    analyzer.run_all()


# ============================================================
# Sub-command: preview
# ============================================================

def cmd_preview(args: argparse.Namespace) -> None:
    from src.analysis.preview import PreprocessingPreview

    # 從 CLI 參數組裝 params 字典
    params = {
        "top_n": args.top_n,
        "remove_largest": args.remove_largest,
        "padding": args.padding,
        "max_attempts": args.max_attempts,
        "use_connected_components": args.use_cc,
        "use_topology_analysis": args.use_topology,
        "remove_gifu_logo": args.remove_logo,
        "logo_template_path": args.logo_template,
        "logo_mask_region": None,
    }

    preview = PreprocessingPreview(
        input_dir=args.data_dir,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        params=params,
        seed=args.seed,
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
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- eda ----
    p_eda = sub.add_parser("eda", help="深度探索性資料分析 (EDA)")
    p_eda.add_argument("--data-dir", required=True, help="影像來源目錄（支援類別子目錄）")
    p_eda.add_argument("--output-dir", default="outputs/eda", help="分析結果輸出目錄")
    p_eda.add_argument("--sample-n", type=int, default=500, help="像素強度分析取樣數")
    p_eda.add_argument("--cc-sample-n", type=int, default=200, help="CC 分析取樣數")
    p_eda.add_argument("--seed", type=int, default=42, help="隨機種子")
    p_eda.set_defaults(func=cmd_eda)

    # ---- preview ----
    p_prev = sub.add_parser("preview", help="前處理管線視覺化預覽")
    p_prev.add_argument("--data-dir", required=True, help="原始影像目錄")
    p_prev.add_argument("--n-samples", type=int, default=5, help="隨機抽取影像數")
    p_prev.add_argument("--output-dir", default="outputs/preview", help="預覽圖輸出目錄")
    p_prev.add_argument("--seed", type=int, default=42, help="隨機種子")
    p_prev.add_argument("--dpi", type=int, default=150, help="輸出圖片 DPI")

    # 前處理參數
    p_prev.add_argument("--top-n", type=int, default=5, help="保留 CC 數量上限")
    p_prev.add_argument("--no-remove-largest", dest="remove_largest", action="store_false",
                        help="不移除最大元件（圖框）")
    p_prev.set_defaults(remove_largest=True)
    p_prev.add_argument("--padding", type=int, default=2, help="CC 裁切邊距（像素）")
    p_prev.add_argument("--max-attempts", type=int, default=400, help="隨機排列嘗試次數")
    p_prev.add_argument("--no-cc", dest="use_cc", action="store_false",
                        help="關閉連通元件分析")
    p_prev.set_defaults(use_cc=True)
    p_prev.add_argument("--no-topology", dest="use_topology", action="store_false",
                        help="關閉拓撲感知排序")
    p_prev.set_defaults(use_topology=False)
    p_prev.add_argument("--remove-logo", dest="remove_logo", action="store_true",
                        help="啟用 Logo 移除")
    p_prev.set_defaults(remove_logo=False)
    p_prev.add_argument("--logo-template", default=None, help="Logo 模板圖片路徑")
    p_prev.set_defaults(func=cmd_preview)

    return parser


# ============================================================
# Entry point
# ============================================================

def main() -> None:
    console.print(
        Panel(
            "[bold]SimSiam v2 — Data Analysis Toolkit[/bold]\n"
            "Commands: [cyan]eda[/cyan] | [cyan]preview[/cyan]",
            border_style="blue",
        )
    )
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
