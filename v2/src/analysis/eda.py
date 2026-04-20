"""工程圖資料集探索性分析模組 (Deep EDA Module)。

============================================================
對工程圖 PNG 資料集執行全面的統計分析，涵蓋：

    1. 影像尺寸分佈（寬、高、長寬比）
    2. 像素強度分析（灰階直方圖、雙峰檢測）
    3. 連通元件統計（每張圖的 CC 數量，用於校準 top_n）
    4. 類別平衡分析（各類別影像數量）

輸出：
    - JSON 格式的統計摘要
    - CSV 格式的逐圖統計
    - seaborn/matplotlib 圖表（PNG）
============================================================
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

console = Console()

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_STYLE = "seaborn-v0_8-whitegrid"


class EDAAnalyzer:
    """工程圖資料集 EDA 分析器。

    Args:
        data_dir: 影像來源目錄（支援類別子目錄）。
        output_dir: 分析結果輸出目錄。
        sample_n: 像素強度分析的取樣數量。
        cc_sample_n: 連通元件分析的取樣數量。
        seed: 隨機種子。
    """

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        sample_n: int = 500,
        cc_sample_n: int = 200,
        seed: int = 42,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_n = sample_n
        self.cc_sample_n = cc_sample_n
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self._all_images: Optional[List[Path]] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run_all(self) -> Dict:
        """執行所有分析項目，儲存結果並列印摘要。"""
        images = self._discover_images()

        console.print(
            Panel(
                f"[bold]資料目錄:[/bold] [cyan]{self.data_dir}[/cyan]\n"
                f"[bold]輸出目錄:[/bold] [cyan]{self.output_dir}[/cyan]\n"
                f"[bold]影像總數:[/bold] [green]{len(images):,}[/green]",
                title="[bold blue] SimSiam EDA — 資料探索分析",
                border_style="blue",
            )
        )

        results: Dict = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=4,
        ) as progress:
            steps = [
                ("影像尺寸分析", self._run_size_analysis),
                ("像素強度分析", self._run_intensity_analysis),
                ("連通元件統計", self._run_cc_analysis),
                ("類別平衡分析", self._run_class_balance),
            ]
            task = progress.add_task("EDA 進度", total=len(steps))

            for name, fn in steps:
                progress.update(task, description=f"[bold]{name}")
                results[name] = fn(images)
                progress.advance(task)

        self._save_results(results)
        self._print_summary(results)
        return results

    # ------------------------------------------------------------------ #
    # Analysis steps
    # ------------------------------------------------------------------ #

    def _run_size_analysis(self, images: List[Path]) -> Dict:
        widths, heights, ratios = [], [], []

        for p in images:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            h, w = img.shape
            widths.append(w)
            heights.append(h)
            ratios.append(w / h if h > 0 else 1.0)

        if not widths:
            return {}

        def _stats(arr: List[float]) -> Dict:
            a = np.array(arr)
            return {
                "mean": float(np.mean(a)),
                "std": float(np.std(a)),
                "min": float(np.min(a)),
                "p25": float(np.percentile(a, 25)),
                "p50": float(np.percentile(a, 50)),
                "p75": float(np.percentile(a, 75)),
                "max": float(np.max(a)),
            }

        result = {
            "count": len(widths),
            "width": _stats(widths),
            "height": _stats(heights),
            "aspect_ratio": _stats(ratios),
        }

        self._plot_size_distribution(widths, heights, ratios)
        return result

    def _run_intensity_analysis(self, images: List[Path]) -> Dict:
        sample = images if len(images) <= self.sample_n else self._rng.sample(images, self.sample_n)
        global_hist = np.zeros(256, dtype=np.int64)
        means, stds = [], []

        for p in sample:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
            global_hist += hist.astype(np.int64)
            means.append(float(np.mean(img)))
            stds.append(float(np.std(img)))

        total_px = int(global_hist.sum())
        pct_dark = float(global_hist[:32].sum() / max(total_px, 1))
        pct_bright = float(global_hist[224:].sum() / max(total_px, 1))
        pct_mid = 1.0 - pct_dark - pct_bright

        result = {
            "sample_count": len(sample),
            "mean_pixel": float(np.mean(means)) if means else 0.0,
            "std_pixel": float(np.mean(stds)) if stds else 0.0,
            "pct_dark_0_31": round(pct_dark, 4),
            "pct_midtone_32_223": round(pct_mid, 4),
            "pct_bright_224_255": round(pct_bright, 4),
            "histogram": global_hist.tolist(),
        }

        self._plot_pixel_intensity(global_hist, pct_dark, pct_bright)
        return result

    def _run_cc_analysis(self, images: List[Path]) -> Dict:
        sample = images if len(images) <= self.cc_sample_n else self._rng.sample(images, self.cc_sample_n)
        cc_counts = []

        for p in sample:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            n_components = n_labels - 1  # exclude background

            # Exclude the largest (usually the border frame)
            if n_components > 1:
                areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_labels)]
                areas.sort(reverse=True)
                n_components = len(areas) - 1  # after removing largest

            cc_counts.append(max(0, n_components))

        if not cc_counts:
            return {}

        arr = np.array(cc_counts)
        result = {
            "sample_count": len(cc_counts),
            "mean_cc": float(np.mean(arr)),
            "std_cc": float(np.std(arr)),
            "min_cc": int(np.min(arr)),
            "p25_cc": float(np.percentile(arr, 25)),
            "p50_cc": float(np.percentile(arr, 50)),
            "p75_cc": float(np.percentile(arr, 75)),
            "max_cc": int(np.max(arr)),
            "recommended_top_n": int(np.percentile(arr, 75)),
        }

        self._plot_cc_distribution(cc_counts)
        return result

    def _run_class_balance(self, images: List[Path]) -> Dict:
        class_counts: Dict[str, int] = {}
        for p in images:
            try:
                rel = p.relative_to(self.data_dir)
                class_name = rel.parts[0] if len(rel.parts) > 1 else "_flat"
            except ValueError:
                class_name = "_flat"
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        if not class_counts:
            return {}

        total = sum(class_counts.values())
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        result = {
            "total_images": total,
            "num_classes": len(class_counts),
            "classes": {k: {"count": v, "pct": round(v / total, 4)} for k, v in sorted_classes},
            "imbalance_ratio": round(sorted_classes[0][1] / max(sorted_classes[-1][1], 1), 2),
        }

        self._plot_class_balance(sorted_classes, total)
        return result

    # ------------------------------------------------------------------ #
    # Plotting
    # ------------------------------------------------------------------ #

    def _plot_size_distribution(
        self,
        widths: List[int],
        heights: List[int],
        ratios: List[float],
    ) -> None:
        try:
            plt.style.use(_STYLE)
        except OSError:
            pass

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        palette = sns.color_palette("muted")

        for ax, data, label, color in zip(
            axes,
            [widths, heights, ratios],
            ["Width (px)", "Height (px)", "Aspect Ratio (W/H)"],
            palette[:3],
        ):
            sns.histplot(data, bins=40, ax=ax, color=color, kde=True)
            ax.set_xlabel(label)
            ax.set_ylabel("Count")
            ax.set_title(label)
            ax.axvline(float(np.median(data)), color="red", linestyle="--", alpha=0.7, label=f"Median={np.median(data):.0f}")
            ax.legend(fontsize=8)

        fig.suptitle("Image Size Distribution", fontsize=13, fontweight="bold")
        plt.tight_layout()
        out = self.output_dir / "size_distribution.png"
        plt.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_pixel_intensity(
        self,
        hist: np.ndarray,
        pct_dark: float,
        pct_bright: float,
    ) -> None:
        try:
            plt.style.use(_STYLE)
        except OSError:
            pass

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Raw histogram
        ax = axes[0]
        ax.bar(range(256), hist, width=1, color="steelblue", alpha=0.75)
        ax.set_xlabel("Pixel Value")
        ax.set_ylabel("Frequency")
        ax.set_title("Global Grayscale Histogram")
        ax.axvspan(0, 31, alpha=0.15, color="black", label=f"Dark {pct_dark:.1%}")
        ax.axvspan(224, 255, alpha=0.15, color="yellow", label=f"Bright {pct_bright:.1%}")
        ax.legend(fontsize=8)

        # Normalized log-scale
        ax = axes[1]
        hist_norm = hist / max(hist.sum(), 1)
        ax.bar(range(256), hist_norm, width=1, color="darkorange", alpha=0.75)
        ax.set_yscale("log")
        ax.set_xlabel("Pixel Value")
        ax.set_ylabel("Relative Frequency (log)")
        ax.set_title("Normalized Histogram (log scale)")

        fig.suptitle("Pixel Intensity Analysis", fontsize=13, fontweight="bold")
        plt.tight_layout()
        out = self.output_dir / "pixel_intensity.png"
        plt.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_cc_distribution(self, cc_counts: List[int]) -> None:
        try:
            plt.style.use(_STYLE)
        except OSError:
            pass

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        ax = axes[0]
        max_cc = max(cc_counts) if cc_counts else 20
        bins = min(max_cc + 1, 50)
        sns.histplot(cc_counts, bins=bins, ax=ax, color="mediumseagreen", kde=False)
        ax.axvline(float(np.median(cc_counts)), color="red", linestyle="--", label=f"Median={np.median(cc_counts):.1f}")
        ax.axvline(float(np.percentile(cc_counts, 75)), color="orange", linestyle="--", label=f"P75={np.percentile(cc_counts, 75):.1f}")
        ax.set_xlabel("CC Count per Image (excl. frame)")
        ax.set_ylabel("Image Count")
        ax.set_title("Connected Components per Image")
        ax.legend(fontsize=8)

        ax = axes[1]
        unique, cnts = np.unique(cc_counts, return_counts=True)
        ax.bar(unique, cnts, color="teal", alpha=0.8)
        ax.set_xlabel("CC Count")
        ax.set_ylabel("Image Count")
        ax.set_title("CC Count Frequency (discrete)")

        p75 = int(np.percentile(cc_counts, 75))
        fig.suptitle(f"Connected Components Analysis  |  Recommended top_n = {p75}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        out = self.output_dir / "cc_distribution.png"
        plt.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_class_balance(
        self,
        sorted_classes: List[Tuple[str, int]],
        total: int,
    ) -> None:
        try:
            plt.style.use(_STYLE)
        except OSError:
            pass

        fig, axes = plt.subplots(1, 2, figsize=(max(12, len(sorted_classes) * 0.5 + 4), 5))

        names = [c[0] for c in sorted_classes]
        counts = [c[1] for c in sorted_classes]
        pcts = [c / total * 100 for c in counts]

        ax = axes[0]
        bars = ax.barh(names[::-1], counts[::-1], color=sns.color_palette("tab20", len(names)))
        ax.set_xlabel("Image Count")
        ax.set_title("Class Distribution (count)")
        for bar, cnt in zip(bars, counts[::-1]):
            ax.text(bar.get_width() + total * 0.005, bar.get_y() + bar.get_height() / 2,
                    str(cnt), va="center", fontsize=8)

        ax = axes[1]
        if len(names) <= 10:
            wedges, texts, autotexts = ax.pie(counts, labels=names, autopct="%1.1f%%", startangle=90)
            for at in autotexts:
                at.set_fontsize(8)
        else:
            ax.bar(range(len(counts)), pcts, color=sns.color_palette("tab20", len(names)))
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("Percentage (%)")
        ax.set_title("Class Distribution (%)")

        fig.suptitle(f"Class Balance  |  {len(names)} classes, {total:,} images", fontsize=12, fontweight="bold")
        plt.tight_layout()
        out = self.output_dir / "class_balance.png"
        plt.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #

    def _save_results(self, results: Dict) -> None:
        # JSON summary (without raw histogram list for readability)
        summary = {}
        for k, v in results.items():
            if isinstance(v, dict):
                clean = {kk: vv for kk, vv in v.items() if kk != "histogram"}
                summary[k] = clean
            else:
                summary[k] = v

        json_path = self.output_dir / "eda_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # Full JSON (including histogram)
        full_path = self.output_dir / "eda_full.json"
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # CSV for class balance
        if "類別平衡分析" in results and "classes" in results["類別平衡分析"]:
            csv_path = self.output_dir / "class_balance.csv"
            classes = results["類別平衡分析"]["classes"]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["class", "count", "pct"])
                writer.writeheader()
                for cls_name, cls_data in classes.items():
                    writer.writerow({"class": cls_name, **cls_data})

    def _print_summary(self, results: Dict) -> None:
        # Size table
        if "影像尺寸分析" in results:
            size = results["影像尺寸分析"]
            table = Table(title="影像尺寸統計", border_style="blue", show_header=True)
            table.add_column("指標", style="bold cyan")
            table.add_column("Mean", justify="right")
            table.add_column("Std", justify="right")
            table.add_column("Min", justify="right")
            table.add_column("P50", justify="right")
            table.add_column("Max", justify="right")
            for dim in ["width", "height", "aspect_ratio"]:
                if dim in size:
                    s = size[dim]
                    table.add_row(
                        dim,
                        f"{s['mean']:.1f}",
                        f"{s['std']:.1f}",
                        f"{s['min']:.1f}",
                        f"{s['p50']:.1f}",
                        f"{s['max']:.1f}",
                    )
            console.print(table)

        # Intensity panel
        if "像素強度分析" in results:
            inten = results["像素強度分析"]
            console.print(
                Panel(
                    f"  Dark  (0–31):    [red]{inten.get('pct_dark_0_31', 0):.1%}[/red]\n"
                    f"  Midtone (32–223): [yellow]{inten.get('pct_midtone_32_223', 0):.1%}[/yellow]\n"
                    f"  Bright (224–255): [white]{inten.get('pct_bright_224_255', 0):.1%}[/white]\n"
                    f"  Mean pixel: {inten.get('mean_pixel', 0):.1f}  |  Std: {inten.get('std_pixel', 0):.1f}",
                    title="像素強度分析",
                    border_style="yellow",
                )
            )

        # CC panel
        if "連通元件統計" in results:
            cc = results["連通元件統計"]
            console.print(
                Panel(
                    f"  Mean CC per image:  [green]{cc.get('mean_cc', 0):.1f}[/green]\n"
                    f"  Median CC:          [green]{cc.get('p50_cc', 0):.1f}[/green]\n"
                    f"  P75 CC (→ top_n):   [bold cyan]{cc.get('p75_cc', 0):.0f}[/bold cyan]\n"
                    f"  Max CC observed:    {cc.get('max_cc', 0)}\n"
                    f"  [bold]建議 top_n = {cc.get('recommended_top_n', 5)}[/bold]",
                    title="連通元件統計",
                    border_style="green",
                )
            )

        # Class balance
        if "類別平衡分析" in results:
            bal = results["類別平衡分析"]
            table = Table(title=f"類別平衡 ({bal.get('num_classes', 0)} 類)", border_style="magenta")
            table.add_column("類別", style="cyan")
            table.add_column("影像數", justify="right")
            table.add_column("佔比", justify="right")
            classes = bal.get("classes", {})
            for cls_name, cls_data in list(classes.items())[:20]:  # Show top 20
                pct = cls_data["pct"]
                bar = "█" * int(pct * 30)
                table.add_row(
                    cls_name,
                    str(cls_data["count"]),
                    f"{pct:.1%} {bar}",
                )
            if len(classes) > 20:
                table.add_row("...", f"(+{len(classes) - 20} more)", "")
            console.print(table)

        console.print(
            Panel(
                f"[bold green]分析完成！[/bold green]\n"
                f"結果儲存於: [cyan]{self.output_dir}[/cyan]",
                border_style="green",
            )
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _discover_images(self) -> List[Path]:
        if self._all_images is None:
            self._all_images = [
                p for p in self.data_dir.rglob("*")
                if p.suffix.lower() in _IMG_EXTS
            ]
        return self._all_images
