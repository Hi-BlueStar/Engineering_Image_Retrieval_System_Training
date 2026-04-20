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

from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table

console = Console()

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

    # 預設前處理參數（對應 PreprocessConfig 欄位）
    DEFAULT_PARAMS: Dict[str, Any] = {
        "top_n": 5,
        "remove_largest": True,
        "padding": 2,
        "max_attempts": 400,
        "use_connected_components": True,
        "use_topology_analysis": False,  # preview 關閉以加速
        "remove_gifu_logo": False,
        "logo_template_path": None,
        "logo_mask_region": None,
    }

    def __init__(
        self,
        input_dir: str,
        n_samples: int = 5,
        output_dir: str = "outputs/preview",
        params: Optional[Dict[str, Any]] = None,
        seed: int = 42,
        figure_dpi: int = 150,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.n_samples = n_samples
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.figure_dpi = figure_dpi

        self.params = dict(self.DEFAULT_PARAMS)
        if params:
            self.params.update(params)

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
            console.print(f"[red]未在 {self.input_dir} 找到任何影像[/red]")
            return

        rng = random.Random(self.seed)
        samples = rng.sample(images, min(self.n_samples, len(images)))

        self._print_header(len(images), len(samples))

        saved: List[Path] = []
        for i, img_path in enumerate(
            track(samples, description="[cyan]生成預覽圖", console=console)
        ):
            out = self._preview_one(img_path, i)
            if out is not None:
                saved.append(out)

        self._print_footer(saved)

    # ------------------------------------------------------------------ #
    # Per-image pipeline
    # ------------------------------------------------------------------ #

    def _preview_one(self, img_path: Path, idx: int) -> Optional[Path]:
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            console.print(f"  [red]無法讀取: {img_path.name}[/red]")
            return None

        stages: Dict[str, Any] = {}

        # Stage 0: Original
        stages["original"] = gray.copy()

        # Stage 1: Logo removal
        current = gray.copy()
        if self.params["remove_gifu_logo"]:
            try:
                from src.data.logo_removal import remove_logo
                current = remove_logo(
                    current,
                    template_path=self.params.get("logo_template_path"),
                    mask_region=self.params.get("logo_mask_region"),
                )
            except Exception:
                pass
        stages["logo_removed"] = current.copy()

        # Stage 2: Binarization (Otsu)
        _, binary = cv2.threshold(
            current, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        stages["binary"] = binary.copy()

        # Stage 3: CC detection visualisation
        if self.params["use_connected_components"]:
            cc_vis, crops, labels_info = self._extract_cc_vis(binary)
            stages["cc_detection"] = cc_vis
        else:
            stages["cc_detection"] = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
            crops = []
            labels_info = []

        # Stage 4: Final arrangement
        if crops:
            from src.data.preprocessing import _arrange
            _rng = random.Random(self.seed + idx)
            h, w = gray.shape

            if self.params["use_topology_analysis"]:
                try:
                    from src.data.topology import sort_crops_by_topology
                    crops = sort_crops_by_topology(crops)
                except Exception:
                    pass

            canvas = _arrange(crops, h, w, self.params["max_attempts"], _rng)
            final = 255 - canvas
            stages["final"] = final
        elif not self.params["use_connected_components"]:
            stages["final"] = current
        else:
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
    ) -> Tuple[np.ndarray, List[np.ndarray], List[Dict]]:
        """Run CC analysis and return coloured visualisation + crops."""
        h, w = binary.shape
        top_n = self.params["top_n"]
        remove_largest = self.params["remove_largest"]
        padding = self.params["padding"]

        num_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        crops: List[np.ndarray] = []
        labels_info: List[Dict] = []

        if num_labels <= 1:
            return vis, crops, labels_info

        components = [
            (i, int(stats[i, cv2.CC_STAT_AREA]))
            for i in range(1, num_labels)
        ]
        components.sort(key=lambda x: x[1], reverse=True)

        if remove_largest and len(components) > 1:
            components = components[1:]

        components = components[:top_n]

        for rank, (label_idx, area) in enumerate(components):
            x = int(stats[label_idx, cv2.CC_STAT_LEFT])
            y = int(stats[label_idx, cv2.CC_STAT_TOP])
            cw = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            ch = int(stats[label_idx, cv2.CC_STAT_HEIGHT])

            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w, x + cw + padding)
            y2 = min(h, y + ch + padding)

            if x2 - x1 < 2 or y2 - y1 < 2:
                continue

            color_rgb = _CC_COLORS_RGB[rank % len(_CC_COLORS_RGB)]
            color_bgr = _CC_COLORS_BGR[rank % len(_CC_COLORS_BGR)]
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

            crops.append(binary[y1:y2, x1:x2].copy())
            labels_info.append({
                "rank": rank + 1,
                "area": area,
                "color": tuple(c / 255.0 for c in color_rgb),
                "bbox": (x1, y1, x2, y2),
            })

        return vis, crops, labels_info

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

    def _print_header(self, total_images: int, n_samples: int) -> None:
        param_table = Table(show_header=False, box=None, padding=(0, 1))
        param_table.add_column("param", style="cyan")
        param_table.add_column("value", style="white")
        for k, v in self.params.items():
            if v is not None:
                param_table.add_row(k, str(v))

        console.print(
            Panel(
                f"[bold]輸入目錄:[/bold] [cyan]{self.input_dir}[/cyan]\n"
                f"[bold]輸出目錄:[/bold] [cyan]{self.output_dir}[/cyan]\n"
                f"[bold]總影像數:[/bold] [green]{total_images:,}[/green]  "
                f"→  預覽 [yellow]{n_samples}[/yellow] 張\n\n"
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
