"""前處理管線視覺化預覽模組 (Preprocessing Preview Module)。

============================================================
隨機抽取 N 張原始影像，按順序套用前處理步驟，生成對照圖：

    [連通元件模式]
    原始 → 二值化 → CC 偵測 + Logo 過濾 → 各元件（含 Padding）
    → 訓練輸入（Letterbox）→ 增強示意

    [全圖模式 (use_connected_components=False)]
    原始 → 二值化（視覺參考）→ 前處理後影像（含 Logo 移除）
    → 最終完整影像（含 Padding）→ 訓練輸入（Letterbox）→ 增強示意

前處理參數完全對應 prepare_data.py 的 PreprocessConfig。
============================================================
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image as PILImage

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
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception as e:
        from src.logger import get_logger
        get_logger(__name__).warning("無法載入自訂字體 %s: %s", _FONT_PATH, e)

from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table

from src.data.logo_removal import remove_logo
from src.data.preprocessing import apply_crop_postprocess, binarize, discover_components
from src.data.topology import sort_crops_by_topology
from src.dataset.dataset import Letterbox
from src.dataset.transforms import EngineeringDrawingAugmentation
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


def _letterbox_np(img: np.ndarray, size: int, fill: int = 255) -> np.ndarray:
    """PIL Letterbox → numpy，與 dataset.py Letterbox 完全相同邏輯。"""
    return np.array(Letterbox(size, fill)(PILImage.fromarray(img)))


class PreprocessingPreview:
    """前處理管線視覺化工具。

    前處理邏輯完全對應 ``prepare_data.py`` 的 PreprocessConfig 流程：
    - ``use_connected_components=True``：binarize → discover_components（含 logo 過濾）
      → (可選) sort_crops_by_topology → 裁切元件 → Letterbox
    - ``use_connected_components=False``：(可選) remove_logo → (可選) topology_guided_mask
      → Letterbox

    Args:
        input_dir: 原始影像目錄（對應 converted_image_dir）。
        n_samples: 隨機抽取的影像數。
        output_dir: 預覽圖輸出目錄。
        params: 前處理參數字典，鍵名對應 DataConfig 欄位（由 analyze_data.py 傳入）。
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
            found_ids = {img.stem for img in selected_ids} | {img.name for img in selected_ids}
            missing_ids = set(self.image_ids) - found_ids
            if missing_ids:
                logger.warning("以下指定 ID 未找到對應影像: %s", missing_ids)

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
    # Per-image pipeline（與 preprocessing._process_one 邏輯對應）
    # ------------------------------------------------------------------ #

    def _preview_one(self, img_path: Path, idx: int) -> Optional[Path]:
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            logger.warning("無法讀取影像: %s", img_path.name)
            return None

        stages: Dict[str, Any] = {}
        labels_info: List[Dict] = []

        # Stage 0: 原始影像
        stages["original"] = gray.copy()

        # Stage 1: Otsu 二值化（供視覺參考；非 CC 模式的實際管線不做二值化）
        binary = binarize(gray)
        stages["binary"] = binary.copy()

        crops: List[np.ndarray] = []

        if self.params.get("use_connected_components"):
            # ── CC 模式：與 _process_one 中 use_connected_components=True 路徑一致 ──
            # discover_components 內部處理 logo 過濾與拓撲剪枝
            cc_vis, comps_info = self._extract_cc_vis(binary)
            stages["cc_detection"] = cc_vis
            crops = [c["crop"] for c in comps_info]
            for rank, c in enumerate(comps_info):
                color_rgb = _CC_COLORS_RGB[rank % len(_CC_COLORS_RGB)]
                labels_info.append({
                    "rank": rank + 1,
                    "area": c["area"],
                    "is_complex": c.get("is_complex", False),
                    "n_holes": c.get("n_holes", 0),
                    "color": tuple(ch / 255.0 for ch in color_rgb),
                })

            # 拓撲剪枝歷史視覺化（對應 topology_preserving_pruning 的 history）
            if self.params.get("use_topology_pruning"):
                max_history = max((len(c.get("crop_history", [])) for c in comps_info), default=0)
                h, w = gray.shape
                for step_i in range(max_history):
                    iter_canvas = np.zeros((h, w), dtype=np.uint8)
                    for c in comps_info:
                        hist = c.get("crop_history", [])
                        crop_state = hist[step_i] if step_i < len(hist) else c["crop"]
                        x1, y1, x2, y2 = c["bbox"]
                        px1, py1 = max(0, x1), max(0, y1)
                        px2, py2 = min(w, x2), min(h, y2)
                        cw_s = 0 if x1 >= 0 else -x1
                        ch_s = 0 if y1 >= 0 else -y1
                        iter_canvas[py1:py2, px1:px2] = np.maximum(
                            iter_canvas[py1:py2, px1:px2],
                            crop_state[ch_s : ch_s + (py2 - py1), cw_s : cw_s + (px2 - px1)],
                        )
                    stages[f"pruning_iter_{step_i + 1}"] = iter_canvas
        else:
            # ── 全圖模式：與 _process_one 中 use_connected_components=False 路徑一致 ──
            # 對灰階影像做 logo 移除（非對二值化影像）
            current = gray.copy()
            if self.params.get("remove_gifu_logo"):
                current = remove_logo(
                    current,
                    template_path=self.params.get("logo_template_path"),
                    mask_region=self.params.get("logo_mask_region"),
                    fill_value=255,
                )
            # 以 "preprocessed_gray" 鍵儲存（避免與 CC 模式的 "cc_detection" 混淆）
            stages["preprocessed_gray"] = current
            # crops 維持空列表，後續進入 fallback 路徑

        # ── 有 crops（CC 模式且找到元件）──
        if crops:
            # 依拓撲複雜度排序（對應 prepare_data.py 中 use_topology_analysis 的排序步驟）
            if self.params.get("use_topology_analysis"):
                try:
                    crops = sort_crops_by_topology(crops)
                except Exception as e:
                    logger.debug("Topology sorting failed for preview: %s", e)

            pad = self.params.get("padding", 2)
            for i, crop in enumerate(crops):
                stages[f"comp_{i + 1}"] = apply_crop_postprocess(crop, pad)

            if self.params.get("img_size"):
                size = self.params["img_size"]
                for i, crop in enumerate(crops):
                    postprocessed = apply_crop_postprocess(crop, pad)
                    resized = _letterbox_np(postprocessed, size)
                    stages[f"resized_{i + 1}"] = resized
                    stages[f"aug_sample_{i + 1}"] = self._simulate_augmentation(resized, size)

        else:
            # ── Fallback：CC 模式但無元件，或全圖模式 ──
            if self.params.get("use_connected_components"):
                # CC 模式找不到任何元件：仍對灰階影像做 logo 移除後呈現全圖
                current = gray.copy()
                if self.params.get("remove_gifu_logo"):
                    current = remove_logo(
                        current,
                        template_path=self.params.get("logo_template_path"),
                        mask_region=self.params.get("logo_mask_region"),
                        fill_value=255,
                    )
            else:
                # 全圖模式：直接使用已處理的 preprocessed_gray
                current = stages["preprocessed_gray"]

            # 拓撲感知遮罩（全圖模式，與 _process_one 的 topology_guided_mask 對應）
            if self.params.get("use_topology_analysis"):
                from src.data.topology import topology_guided_mask
                current = topology_guided_mask(current)

            pad = self.params.get("padding", 2)
            padded_current = cv2.copyMakeBorder(
                current, pad, pad, pad, pad,
                cv2.BORDER_CONSTANT, value=255,
            )
            stages["full_image"] = padded_current

            if self.params.get("img_size"):
                size = self.params["img_size"]
                resized = _letterbox_np(padded_current, size)
                stages["resized_full"] = resized
                stages["aug_sample_full"] = self._simulate_augmentation(resized, size)

        out_path = self.output_dir / f"preview_{idx:03d}_{img_path.stem}.png"
        self._generate_figure(img_path, stages, labels_info, out_path)
        return out_path

    def _simulate_augmentation(self, img: np.ndarray, img_size: int) -> np.ndarray:
        """以 EngineeringDrawingAugmentation 生成增強視角（與訓練 CPU 增強路徑完全相同）。"""
        aug = EngineeringDrawingAugmentation(
            img_size=img_size, mean=(0.5,), std=(0.5,), use_augmentation=True,
        )
        pil = PILImage.fromarray(img)
        tensor, _ = aug(pil)
        arr = (tensor.squeeze(0).numpy() * 0.5 + 0.5) * 255
        return np.clip(arr, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------ #
    # CC extraction with visualisation
    # ------------------------------------------------------------------ #

    def _extract_cc_vis(
        self,
        binary: np.ndarray,
    ) -> Tuple[np.ndarray, List[dict]]:
        """呼叫 discover_components（與 prepare_data.py 使用的相同函式）並生成彩色視覺化。

        參數完全對應 preprocessing._process_one 傳入 extract_crops 的參數。
        """
        comps = discover_components(
            binary,
            top_n=self.params.get("top_n", 0),
            max_bbox_ratio=self.params.get("max_bbox_ratio", 0.9),
            min_bbox_area=self.params.get("min_bbox_area", 0),
            padding=self.params.get("padding", 2),
            remove_logo_cfg=self.params.get("remove_gifu_logo", False),
            logo_template_path=self.params.get("logo_template_path"),
            logo_mask_region=self.params.get("logo_mask_region"),
            use_topology_pruning=self.params.get("use_topology_pruning", True),
            topology_pruning_iters=self.params.get("topology_pruning_iters", 3),
            topology_pruning_ksize=self.params.get("topology_pruning_ksize", 2),
            min_simple_area=self.params.get("min_simple_area", 40),
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
        stage_labels: Dict[str, str] = {
            "original": "原始影像",
            "binary": "二值化遮罩 (Otsu)",
            "cc_detection": "CC 偵測 (含 Logo 過濾)",
            # 全圖模式下 logo 移除後的灰階影像
            "preprocessed_gray": "前處理後影像 (Logo 移除)",
            "full_image": "最終完整影像 (含 Padding)",
            "resized_full": f"訓練輸入 (Letterbox {self.params.get('img_size', '?')}px)",
            "aug_sample_full": "增強示意 (SimSiam 雙視角之一)",
        }
        for i in range(20):
            stage_labels[f"pruning_iter_{i + 1}"] = f"拓撲剪枝 Iter {i + 1}"
            stage_labels[f"comp_{i + 1}"] = f"元件 {i + 1} (含 Padding)"
            stage_labels[f"resized_{i + 1}"] = f"訓練輸入 {i + 1} (Letterbox)"
            stage_labels[f"aug_sample_{i + 1}"] = f"增強示意 {i + 1}"

        n = len(stages)
        cols = min(n, 4)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.5 * rows))

        if n == 1:
            axes_flat = [axes]
        elif rows == 1 or cols == 1:
            axes_flat = list(axes) if hasattr(axes, "__iter__") else [axes]
        else:
            axes_flat = axes.flatten()

        for ax, (key, img) in zip(axes_flat[:n], stages.items()):
            if img is None:
                ax.axis("off")
                continue

            if len(img.shape) == 3:
                ax.imshow(img)
            else:
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)

            ax.set_title(stage_labels.get(key, key), fontsize=9, pad=4)
            ax.axis("off")

            if key == "cc_detection" and labels_info:
                patches = [
                    mpatches.Patch(
                        facecolor=info["color"],
                        label=f"CC{info['rank']} (Holes:{info['n_holes']}, Area:{info['area']:,})",
                    )
                    for info in labels_info
                ]
                ax.legend(handles=patches, fontsize=6, loc="lower right", framealpha=0.7)

        for ax in axes_flat[n:]:
            ax.axis("off")

        fig.suptitle(
            f"{img_path.name}",
            fontsize=12,
            fontweight="bold",
            y=1.02 if rows == 1 else 1.05,
        )

        # 參數標注（對應 prepare_data.py PreprocessConfig 欄位）
        param_lines = [
            f"top_n={self.params.get('top_n', 0)}",
            f"use_cc={self.params.get('use_connected_components', True)}",
            f"use_topo={self.params.get('use_topology_analysis', True)}",
            f"use_pruning={self.params.get('use_topology_pruning', True)}",
            f"pruning_iters={self.params.get('topology_pruning_iters', 3)}",
            f"pruning_ksize={self.params.get('topology_pruning_ksize', 2)}",
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
                "[bold]前處理參數 (對應 prepare_data.py PreprocessConfig):[/bold]\n"
                + "\n".join(f"  {k} = {v}" for k, v in self.params.items() if v is not None),
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
