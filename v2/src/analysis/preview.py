"""前處理管線視覺化預覽模組 (Preprocessing Preview Module)。

============================================================
隨機抽取 N 張原始影像，按順序套用前處理步驟，生成對照圖：

    [ 原始圖 ] → [ Logo 移除後 ] → [ 二值化遮罩 ] → [ CC 偵測結果 ] → [ 最終預處理變體 ]

使用 matplotlib 輸出多格對照 PNG，方便在正式批次處理前評估
參數設定效果。支援直接修改 ``params`` 字典進行快速實驗。
============================================================
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager

# --- Matplotlib 中文字體設定 ---
_FONT_PATH = Path(__file__).resolve().parents[2] / "NotoSansTC-VariableFont_wght.ttf"
if not _FONT_PATH.exists():
    _FONT_PATH = Path("v2/NotoSansTC-VariableFont_wght.ttf")

if _FONT_PATH.exists():
    try:
        font_manager.fontManager.addfont(str(_FONT_PATH))
        prop = font_manager.FontProperties(fname=str(_FONT_PATH))
        matplotlib.rcParams["font.family"] = prop.get_name()
        matplotlib.rcParams["axes.unicode_minus"] = False  # 解決負號顯示問題
    except Exception as e:
        from src.logger import get_logger
        get_logger(__name__).warning("無法載入自訂字體 %s: %s", _FONT_PATH, e)

from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table

from src.config import AppConfig
from src.data.logo_removal import remove_logo
from src.data.preprocessing import binarize, discover_components, arrange_crops
from src.data.topology import sort_crops_by_topology
from src.logger import get_logger

console = Console()
logger = get_logger(__name__)

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

_CC_COLORS_BGR = [
    (0, 220, 0),
    (220, 0, 0),
    (0, 0, 220),
    (220, 140, 0),
    (160, 0, 220),
]
_CC_COLORS_RGB = [(r, g, b) for b, g, r in _CC_COLORS_BGR]


class PreprocessingPreview:
    """前處理管線視覺化工具。

    Args:
        input_dir: 原始影像目錄。
        n_samples: 隨機抽取的影像數。
        output_dir: 預覽圖輸出目錄。
        params: 前處理參數字典（覆蓋預設值）。
        seed: 隨機種子。
        figure_dpi: 輸出圖片 DPI。
    """

    def __init__(
        self,
        input_dir: str,
        n_samples: int = 5,
        image_ids: Optional[List[str]] = None,
        output_dir: str = "outputs/preview",
        params: Optional[Dict[str, Any]] = None,
        seed: int = 42,
        figure_dpi: int = 150,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.n_samples = n_samples
        self.image_ids = image_ids
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.figure_dpi = figure_dpi

        # params 由外部傳入（通常在 analyze_data.py 中從 AppConfig 構建）
        self.params = params or {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        """執行預覽：抽樣 → 管線處理 → 儲存對照圖。"""
        images = [
            p for p in self.input_dir.rglob("*")
            if p.suffix.lower() in _IMG_EXTS
        ]

        if not images:
            logger.error("未在 %s 找到任何影像", self.input_dir)
            return

        rng = random.Random(self.seed)
        
        selected_ids: List[Path] = []
        if self.image_ids:
            selected_ids = [
                img for img in images 
                if img.stem in self.image_ids or img.name in self.image_ids
            ]
            
            # 檢查是否有指定的 ID 沒被找到
            found_ids = {img.stem for img in selected_ids} | {img.name for img in selected_ids}
            missing_ids = set(self.image_ids) - found_ids
            if missing_ids:
                logger.warning("以下指定 ID 未找到對應影像: %s", missing_ids)

        # 從剩餘影像中隨機取樣
        remaining_images = [img for img in images if img not in selected_ids]
        random_samples = []
        if self.n_samples > 0 and remaining_images:
            random_samples = rng.sample(remaining_images, min(self.n_samples, len(remaining_images)))

        samples = selected_ids + random_samples

        if not samples:
            logger.warning("無任何影像可供預覽（指定 ID 未找到且未要求隨機取樣）")
            return

        self._print_header(len(images), len(samples), len(selected_ids), len(random_samples))

        logger.info("Generating previews for %d sampled images...", len(samples))
        saved: List[Path] = []
        for i, img_path in enumerate(
            track(samples, description="[cyan]生成預覽圖", console=console)
        ):
            out = self._preview_one(img_path, i)
            if out is not None:
                saved.append(out)

        self._print_footer(saved)
        logger.info("Preprocessing preview finished. %d images saved to %s", len(saved), self.output_dir)

    # ------------------------------------------------------------------ #
    # Per-image pipeline
    # ------------------------------------------------------------------ #

    def _preview_one(self, img_path: Path, idx: int) -> Optional[Path]:
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            logger.warning("無法讀取影像: %s", img_path.name)
            return None

        stages: Dict[str, Any] = {}

        # Stage 0: Original
        stages["original"] = gray.copy()

        # Stage 1: Logo removal
        current = gray.copy()
        if self.params.get("remove_gifu_logo"):
            try:
                current = remove_logo(
                    current,
                    template_path=self.params.get("logo_template_path"),
                    mask_region=self.params.get("logo_mask_region"),
                )
            except Exception as e:
                logger.debug("Logo removal failed for preview: %s", e)
        stages["logo_removed"] = current.copy()

        # Stage 2: Binarization (Otsu)
        binary = binarize(current)
        stages["binary"] = binary.copy()

        # Stage 3: CC detection visualisation
        crops: List[np.ndarray] = []
        labels_info: List[Dict] = []
        
        if self.params.get("use_connected_components"):
            cc_vis, comps = self._extract_cc_vis(binary)
            stages["cc_detection"] = cc_vis
            crops = [c["crop"] for c in comps]
            
            # Prepare labels for legend
            for rank, c in enumerate(comps):
                color_rgb = _CC_COLORS_RGB[rank % len(_CC_COLORS_RGB)]
                labels_info.append({
                    "rank": rank + 1,
                    "area": c["area"],
                    "color": tuple(channel / 255.0 for channel in color_rgb),
                })
        else:
            stages["cc_detection"] = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

        # Stage 4: Final arrangement
        if crops:
            _rng = random.Random(self.seed + idx)
            h, w = gray.shape

            if self.params.get("use_topology_analysis"):
                try:
                    crops = sort_crops_by_topology(crops)
                except Exception as e:
                    logger.debug("Topology sorting failed for preview: %s", e)

            canvas = arrange_crops(crops, h, w, self.params.get("max_attempts", 400), _rng)
            final = 255 - canvas
            stages["final"] = final
        else:
            # Fallback for no CCs or no arrangement
            stages["final"] = current

        out_path = self.output_dir / f"preview_{idx:03d}_{img_path.stem}.png"
        self._generate_figure(img_path, stages, labels_info, out_path)
        return out_path

    # ------------------------------------------------------------------ #
    # CC extraction with visualisation
    # ------------------------------------------------------------------ #

    def _extract_cc_vis(
        self,
        binary: np.ndarray,
    ) -> Tuple[np.ndarray, List[dict]]:
        """Run CC analysis using shared logic and return coloured visualisation + components info."""
        comps = discover_components(
            binary,
            top_n=self.params.get("top_n", 5),
            remove_largest=self.params.get("remove_largest", True),
            padding=self.params.get("padding", 2),
        )

        vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        
        for rank, c in enumerate(comps):
            x1, y1, x2, y2 = c["bbox"]
            color_rgb = _CC_COLORS_RGB[rank % len(_CC_COLORS_RGB)]
            
            cv2.rectangle(vis, (x1, y1), (x2, y2), color_rgb, 2)
            cv2.putText(
                vis,
                f"CC{rank + 1}",
                (x1 + 2, max(y1 + 14, 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color_rgb,
                1,
                cv2.LINE_AA,
            )

        return vis, comps

    # ------------------------------------------------------------------ #
    # Figure generation
    # ------------------------------------------------------------------ #

    def _generate_figure(
        self,
        img_path: Path,
        stages: Dict[str, Any],
        labels_info: List[Dict],
        out_path: Path,
    ) -> None:
        stage_labels = {
            "original": "① 原始影像",
            "logo_removed": "② Logo 移除後",
            "binary": "③ 二值化遮罩",
            "cc_detection": "④ CC 偵測結果",
            "final": "⑤ 最終預處理變體",
        }

        n = len(stages)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5))
        if n == 1:
            axes = [axes]

        for ax, (key, img) in zip(axes, stages.items()):
            if img is None:
                ax.axis("off")
                continue

            if len(img.shape) == 3:
                ax.imshow(img)
            else:
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)

            ax.set_title(stage_labels.get(key, key), fontsize=9, pad=4)
            ax.axis("off")

            # Add CC legend patches on cc_detection panel
            if key == "cc_detection" and labels_info:
                patches = [
                    mpatches.Patch(
                        facecolor=info["color"],
                        label=f"CC{info['rank']} (area={info['area']:,})",
                    )
                    for info in labels_info
                ]
                ax.legend(handles=patches, fontsize=6, loc="lower right", framealpha=0.7)

        fig.suptitle(
            f"{img_path.name}",
            fontsize=10,
            fontweight="bold",
            y=1.01,
        )

        # Parameter box
        param_lines = [
            f"top_n={self.params['top_n']}",
            f"remove_largest={self.params['remove_largest']}",
            f"use_cc={self.params['use_connected_components']}",
            f"use_topo={self.params['use_topology_analysis']}",
        ]
        fig.text(
            0.01, -0.01,
            "  ".join(param_lines),
            fontsize=7,
            color="gray",
            transform=fig.transFigure,
        )

        plt.tight_layout(rect=[0, 0.02, 1, 1])
        plt.savefig(str(out_path), dpi=self.figure_dpi, bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------------ #
    # UI
    # ------------------------------------------------------------------ #

    def _print_header(self, total_images: int, total_samples: int, n_specified: int = 0, n_random: int = 0) -> None:
        param_table = Table(show_header=False, box=None, padding=(0, 1))
        param_table.add_column("param", style="cyan")
        param_table.add_column("value", style="white")
        for k, v in self.params.items():
            if v is not None:
                param_table.add_row(k, str(v))

        sample_info = []
        if n_specified > 0:
            sample_info.append(f"指定 ID: [yellow]{n_specified}[/yellow] 張")
        if n_random > 0:
            sample_info.append(f"隨機取樣: [yellow]{n_random}[/yellow] 張")
        
        info_str = " + ".join(sample_info) if sample_info else "無取樣"

        console.print(
            Panel(
                f"[bold]輸入目錄:[/bold] [cyan]{self.input_dir}[/cyan]\n"
                f"[bold]輸出目錄:[/bold] [cyan]{self.output_dir}[/cyan]\n"
                f"[bold]總影像數:[/bold] [green]{total_images:,}[/green]  "
                f"→  預覽共 [yellow]{total_samples}[/yellow] 張 ({info_str})\n\n"
                "[bold]前處理參數:[/bold]\n" + "\n".join(f"  {k} = {v}" for k, v in self.params.items() if v is not None),
                title="[bold blue] 前處理管線預覽",
                border_style="blue",
            )
        )

    def _print_footer(self, saved: List[Path]) -> None:
        console.print(
            Panel(
                f"[bold green]預覽完成！[/bold green]\n"
                f"已生成 [yellow]{len(saved)}[/yellow] 張對照圖\n"
                f"輸出目錄: [cyan]{self.output_dir}[/cyan]",
                border_style="green",
            )
        )
        for p in saved:
            console.print(f"  [dim]→ {p.name}[/dim]")
