"""學術圖表繪製腳本 (Academic Analysis Charts for SimSiam Training Pipeline)。

============================================================
從訓練與評估管線產生的各類輸出中提取數據，
繪製學術品質的分析圖表，涵蓋：

    訓練動態分析：
        - Loss 收斂曲線（Train / Val）
        - Feature Std 崩塌監控（SimSiam Collapse Indicator）
        - 學習率排程可視化
        - 每 Epoch 訓練耗時分析

    檢索評估分析：
        - IACS / Inter-class / Contrastive Margin 比較
        - Precision@K 曲線（K = 1, 5, 10, 20, 40, 80）

    前處理效果分析（Raw vs Preprocessed）：
        - 所有指標並列比較
        - 改善量 (Delta) 分析

    消融實驗分析（6 個實驗條件）：
        - 所有指標熱圖（Heatmap）
        - Train / Val Loss 曲線疊加
        - Feature Std 曲線疊加

產生的圖表（存至 --output-dir）：
    fig01_training_dynamics.png
    fig02_retrieval_metrics.png
    fig03_raw_vs_preprocessed.png   （需要 eval_results.json）
    fig04_ablation_heatmap.png      （需要 ablation_summary.json）
    fig05_ablation_loss_curves.png  （需要消融實驗 training_log.csv）
    fig06_ablation_feature_std.png  （需要消融實驗 training_log.csv）

使用方式::

    # 自動偵測（從 outputs/ 目錄搜尋最新實驗）
    python v2/plot_results.py

    # 手動指定所有路徑
    python v2/plot_results.py \\
        --training-log outputs/simsiam_exp_XXXXXX/Run_01_Seed_42/training_log.csv \\
        --summary      outputs/simsiam_exp_XXXXXX/overall_summary.json \\
        --ablation-summary outputs/ablation/ablation_summary.json \\
        --ablation-output-dir outputs/ablation/ \\
        --eval-json    outputs/eval_results.json \\
        --output-dir   outputs/plots/ \\
        --dpi 300
============================================================
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

matplotlib.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          10,
    "axes.titlesize":     12,
    "axes.titleweight":   "bold",
    "axes.labelsize":     10,
    "axes.linewidth":     0.8,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "legend.framealpha":  0.85,
    "legend.edgecolor":   "#cccccc",
    "figure.dpi":         100,
    "figure.facecolor":   "white",
    "axes.facecolor":     "#f9f9f9",
    "axes.grid":          True,
    "grid.color":         "#e0e0e0",
    "grid.linewidth":     0.6,
    "lines.linewidth":    1.6,
    "lines.markersize":   5,
    "savefig.bbox":       "tight",
    "savefig.facecolor":  "white",
})

# ─── 色盤常數 ────────────────────────────────────────────────────────────────

_C = {
    "train":        "#1f77b4",
    "val":          "#ff7f0e",
    "iacs":         "#2ca02c",
    "inter":        "#d62728",
    "margin":       "#9467bd",
    "raw":          "#1f77b4",
    "preprocessed": "#ff7f0e",
    "positive":     "#2ca02c",
    "negative":     "#d62728",
    "neutral":      "#7f7f7f",
    "best":         "#e377c2",
}

# matplotlib tab10 colours for ablation conditions
_ABL_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

_ABLATION_LABELS: Dict[str, str] = {
    "01_baseline":        "Baseline",
    "02_no_topology":     "No Topology",
    "03_no_cc":           "No CC",
    "04_no_augmentation": "No Augmentation",
    "05_no_logo":         "No Logo Removal",
    "06_all_preprocessing": "Full Pipeline",
}

_METRIC_LABELS: Dict[str, Tuple[str, str]] = {
    # key → (display_name, direction)  direction: ↑ = higher better, ↓ = lower better
    "IACS":                  ("IACS",                "↑"),
    "inter_class_avg_sim":   ("Inter-Class Sim",     "↓"),
    "contrastive_margin":    ("Margin",               "↑"),
    "top1_precision":        ("Precision@1",          "↑"),
    "top5_precision":        ("Precision@5",          "↑"),
    "top10_precision":       ("Precision@10",         "↑"),
    "top20_precision":       ("Precision@20",         "↑"),
    "top40_precision":       ("Precision@40",         "↑"),
    "top80_precision":       ("Precision@80",         "↑"),
    "best_val_loss":         ("Best Val Loss",        "↓"),
}

# ─── 資料載入工具 ─────────────────────────────────────────────────────────────


def _load_json(path: Optional[Path]) -> Optional[dict]:
    if path is None or not Path(path).is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_csv(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not Path(path).is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _find_latest_exp_dir(base: str) -> Optional[Path]:
    """在 base 目錄下搜尋最新的實驗目錄（含 overall_summary.json）。"""
    base_path = Path(base)
    if not base_path.is_dir():
        return None
    candidates = sorted(
        [d for d in base_path.iterdir()
         if d.is_dir() and (d / "overall_summary.json").exists()],
        key=lambda d: d.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def _find_training_log(exp_dir: Path) -> Optional[Path]:
    """找到 exp_dir 下最新 Run 的 training_log.csv。"""
    logs = sorted(exp_dir.glob("*/training_log.csv"),
                  key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def _find_ablation_logs(ablation_output_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    搜尋 ablation_output_dir 下所有消融實驗的 training_log.csv。
    回傳 {exp_name: DataFrame}。
    """
    result: Dict[str, pd.DataFrame] = {}
    if not ablation_output_dir.is_dir():
        return result
    for exp_dir in sorted(ablation_output_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        log = _find_training_log(exp_dir)
        if log is None:
            continue
        df = _load_csv(log)
        if df is not None and not df.empty:
            # exp_dir.name 形如 "01_baseline_20241030_120000"
            exp_key = exp_dir.name.split("_20")[0]  # 取 timestamp 前的部分
            # 嘗試直接對應 ABLATION_LABELS
            matched = next(
                (k for k in _ABLATION_LABELS if exp_dir.name.startswith(k)),
                exp_key,
            )
            result[matched] = df
    return result


def _extract_retrieval_metrics(data: dict) -> Tuple[Dict[str, float], List[int]]:
    """從 overall_summary 的 runs[0] 或 retrieval_metrics 中提取指標。"""
    metrics: Dict[str, float] = {}
    top_k: List[int] = []
    for key in ("IACS", "inter_class_avg_sim", "contrastive_margin"):
        if key in data:
            metrics[key] = data[key]
    for key in data:
        if key.startswith("top") and key.endswith("_precision"):
            k = int(key[3:].replace("_precision", ""))
            metrics[key] = data[key]
            top_k.append(k)
    top_k.sort()
    return metrics, top_k


def _label(key: str) -> str:
    info = _METRIC_LABELS.get(key)
    if info:
        return f"{info[0]} {info[1]}"
    return key


# ─── Fig 01: 訓練動態 ─────────────────────────────────────────────────────────


def fig_training_dynamics(
    df: pd.DataFrame,
    proj_dim: int = 2048,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    2×2 圖表：Loss 曲線、Feature Std 崩塌監控、LR 排程、Epoch 耗時。
    """
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.30)

    epochs = df["epoch"].values

    # ── [0,0] Loss 曲線 ────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(epochs, df["train_loss"], color=_C["train"], label="Train Loss", alpha=0.85)
    ax0.plot(epochs, df["val_loss"],   color=_C["val"],   label="Val Loss",
             linestyle="--", alpha=0.85)

    # 標記最佳 epoch
    if "is_best" in df.columns:
        best_mask = df["is_best"].astype(bool)
        best_ep = df.loc[best_mask, "epoch"].values
        best_vl = df.loc[best_mask, "val_loss"].values
        if len(best_ep) > 0:
            best_idx = np.argmin(best_vl)
            ax0.scatter(best_ep[best_idx], best_vl[best_idx],
                        marker="*", s=160, color=_C["best"],
                        zorder=5, label=f"Best (ep {best_ep[best_idx]})")

    ax0.set_title("Training & Validation Loss")
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("SimSiam Loss")
    ax0.legend()

    # ── [0,1] Feature Std 崩塌監控 ───────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    if "train_z_std" in df.columns:
        ax1.plot(epochs, df["train_z_std"], color=_C["train"],
                 label="Train z-std", alpha=0.85)
    if "val_z_std" in df.columns:
        ax1.plot(epochs, df["val_z_std"], color=_C["val"],
                 linestyle="--", label="Val z-std", alpha=0.85)

    ref = 1.0 / math.sqrt(proj_dim)
    ax1.axhline(ref, color=_C["neutral"], linestyle=":", linewidth=1.2,
                label=f"Target 1/√{proj_dim} = {ref:.4f}")
    ax1.set_title("Feature Collapse Monitor (z-std)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Std. Dev. of Projector Output")
    ax1.legend()

    # ── [1,0] Learning Rate 排程 ─────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    if "lr" in df.columns:
        ax2.fill_between(epochs, df["lr"], alpha=0.25, color=_C["train"])
        ax2.plot(epochs, df["lr"], color=_C["train"])
        # 若 lr 跨越多個量級則用 log 軸
        lr_vals = df["lr"].replace(0, np.nan).dropna().values
        if len(lr_vals) > 1 and lr_vals.max() / lr_vals.min() > 50:
            ax2.set_yscale("log")
    ax2.set_title("Learning Rate Schedule")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")
    ax2.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))

    # ── [1,1] 每 Epoch 訓練耗時 ──────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    if "epoch_net_sec" in df.columns and "epoch_wall_sec" in df.columns:
        ax3.bar(epochs, df["epoch_wall_sec"], color=_C["val"],
                alpha=0.55, label="Wall Time", width=0.85)
        ax3.bar(epochs, df["epoch_net_sec"], color=_C["train"],
                alpha=0.85, label="Net Time", width=0.85)
        # 滾動平均線（window=5）
        if len(df) >= 5:
            roll = df["epoch_net_sec"].rolling(5, center=True).mean()
            ax3.plot(epochs, roll, color="black", linewidth=1.4,
                     linestyle="-", label="Net (MA-5)")
    elif "epoch_net_sec" in df.columns:
        ax3.bar(epochs, df["epoch_net_sec"], color=_C["train"], alpha=0.8)

    ax3.set_title("Per-Epoch Training Time")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Time (seconds)")
    ax3.legend()

    fig.suptitle("Training Dynamics Overview", fontsize=14, fontweight="bold", y=1.01)
    _save(fig, save_path)
    return fig


# ─── Fig 02: 檢索評估指標 ─────────────────────────────────────────────────────


def fig_retrieval_metrics(
    metrics: Dict[str, float],
    top_k_values: Optional[List[int]] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    1×2 圖表：IACS / Inter / Margin 橫向長條圖 ＋ Precision@K 折線圖。
    """
    fig, (ax_bar, ax_pk) = plt.subplots(1, 2, figsize=(13, 5))

    # ── 左：相似度指標橫向長條圖 ─────────────────────
    sim_keys = ["IACS", "inter_class_avg_sim", "contrastive_margin"]
    sim_vals = [metrics.get(k, 0.0) for k in sim_keys]
    sim_names = [_METRIC_LABELS[k][0] for k in sim_keys]
    sim_colors = [_C["iacs"], _C["inter"], _C["margin"]]

    bars = ax_bar.barh(sim_names, sim_vals, color=sim_colors,
                       edgecolor="white", height=0.55)
    for bar, val in zip(bars, sim_vals):
        sign = "+" if val >= 0 else ""
        ax_bar.text(
            val + (0.005 if val >= 0 else -0.005),
            bar.get_y() + bar.get_height() / 2,
            f"{sign}{val:.4f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=10, fontweight="bold",
        )

    ax_bar.axvline(0, color="black", linewidth=0.8)
    ax_bar.set_xlim(-0.15, 1.0)
    ax_bar.set_xlabel("Cosine Similarity Value")
    ax_bar.set_title("Similarity-Space Metrics")
    ax_bar.invert_yaxis()

    # ── 右：Precision@K 折線圖 ────────────────────────
    if top_k_values is None:
        top_k_values = sorted(
            int(k[3:].replace("_precision", ""))
            for k in metrics if k.startswith("top") and k.endswith("_precision")
        )

    pk_vals = [metrics.get(f"top{k}_precision", 0.0) for k in top_k_values]
    ax_pk.plot(top_k_values, pk_vals, "o-",
               color=_C["margin"], linewidth=2, markersize=7)
    for k, v in zip(top_k_values, pk_vals):
        ax_pk.annotate(f"{v:.3f}", (k, v), textcoords="offset points",
                       xytext=(0, 7), ha="center", fontsize=8)

    ax_pk.set_ylim(0, 1.05)
    ax_pk.set_xlabel("K")
    ax_pk.set_ylabel("Precision@K")
    ax_pk.set_title("Precision@K Retrieval Curve")
    ax_pk.xaxis.set_major_locator(mticker.FixedLocator(top_k_values))

    fig.suptitle("Retrieval Performance Evaluation", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ─── Fig 03: Raw vs Preprocessed ─────────────────────────────────────────────


def fig_raw_vs_preprocessed(
    eval_data: dict,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    2×2 圖表：相似度指標比較、Precision@K 比較、Delta 分析（各兩組）。
    """
    raw_m = eval_data.get("raw", {}).get("metrics", {})
    pre_m = eval_data.get("preprocessed", {}).get("metrics", {})
    if not raw_m or not pre_m:
        warnings.warn("eval_results.json 缺少 raw 或 preprocessed 指標，跳過 Fig 03")
        return None

    fig = plt.figure(figsize=(15, 9))
    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.32)

    top_k_vals = sorted(
        int(k[3:].replace("_precision", ""))
        for k in raw_m if k.startswith("top") and k.endswith("_precision")
    )

    # ── [0,0] 相似度指標並列長條圖 ───────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    sim_keys = ["IACS", "inter_class_avg_sim", "contrastive_margin"]
    sim_names = [_METRIC_LABELS[k][0] for k in sim_keys]
    x = np.arange(len(sim_keys))
    w = 0.35
    raw_v = [raw_m.get(k, 0) for k in sim_keys]
    pre_v = [pre_m.get(k, 0) for k in sim_keys]

    b1 = ax0.bar(x - w / 2, raw_v, w, label="Raw", color=_C["raw"], alpha=0.85)
    b2 = ax0.bar(x + w / 2, pre_v, w, label="Preprocessed", color=_C["preprocessed"], alpha=0.85)
    ax0.set_xticks(x)
    ax0.set_xticklabels(sim_names, fontsize=9)
    ax0.axhline(0, color="black", linewidth=0.6)
    ax0.set_ylabel("Cosine Similarity")
    ax0.set_title("Similarity Metrics: Raw vs Preprocessed")
    ax0.legend()
    _annotate_bars(ax0, b1)
    _annotate_bars(ax0, b2)

    # ── [0,1] Precision@K 比較折線 ────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    raw_pk  = [raw_m.get(f"top{k}_precision", 0) for k in top_k_vals]
    pre_pk  = [pre_m.get(f"top{k}_precision", 0) for k in top_k_vals]
    ax1.plot(top_k_vals, raw_pk, "o-", color=_C["raw"],
             linewidth=2, markersize=7, label="Raw")
    ax1.plot(top_k_vals, pre_pk, "s--", color=_C["preprocessed"],
             linewidth=2, markersize=7, label="Preprocessed")
    ax1.fill_between(top_k_vals, raw_pk, pre_pk,
                     alpha=0.12, color=_C["positive"])
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel("K")
    ax1.set_ylabel("Precision@K")
    ax1.set_title("Precision@K: Raw vs Preprocessed")
    ax1.legend()
    ax1.xaxis.set_major_locator(mticker.FixedLocator(top_k_vals))

    # ── [1,0] 相似度 Delta 長條圖 ────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    sim_deltas = [pre_m.get(k, 0) - raw_m.get(k, 0) for k in sim_keys]
    colors2 = [_C["positive"] if d >= 0 else _C["negative"] for d in sim_deltas]
    bars2 = ax2.bar(sim_names, sim_deltas, color=colors2, edgecolor="white", alpha=0.85)
    ax2.axhline(0, color="black", linewidth=0.8)
    for bar, val in zip(bars2, sim_deltas):
        sign = "+" if val >= 0 else ""
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + (0.002 if val >= 0 else -0.002),
            f"{sign}{val:.4f}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=9, fontweight="bold",
        )
    ax2.set_ylabel("Δ (Preprocessed − Raw)")
    ax2.set_title("Similarity Metrics Improvement (Δ)")
    ax2.tick_params(axis="x", labelsize=9)

    # ── [1,1] Precision@K Delta 長條圖 ───────────────
    ax3 = fig.add_subplot(gs[1, 1])
    pk_deltas = [
        pre_m.get(f"top{k}_precision", 0) - raw_m.get(f"top{k}_precision", 0)
        for k in top_k_vals
    ]
    colors3 = [_C["positive"] if d >= 0 else _C["negative"] for d in pk_deltas]
    bars3 = ax3.bar([f"P@{k}" for k in top_k_vals], pk_deltas,
                    color=colors3, edgecolor="white", alpha=0.85)
    ax3.axhline(0, color="black", linewidth=0.8)
    for bar, val in zip(bars3, pk_deltas):
        sign = "+" if val >= 0 else ""
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            val + (0.001 if val >= 0 else -0.001),
            f"{sign}{val:.3f}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=8.5,
        )
    ax3.set_ylabel("Δ (Preprocessed − Raw)")
    ax3.set_title("Precision@K Improvement (Δ)")
    ax3.tick_params(axis="x", labelsize=9)

    fig.suptitle("Effect of Data Preprocessing on Retrieval Performance",
                 fontsize=13, fontweight="bold")
    _save(fig, save_path)
    return fig


# ─── Fig 04: 消融實驗指標熱圖 ────────────────────────────────────────────────


def fig_ablation_heatmap(
    ablation_summary: List[dict],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    消融實驗所有指標的 Heatmap（條件×指標）。
    每欄依 min-max 正規化上色，格子內顯示原始值。
    """
    metric_keys = [
        "IACS", "inter_class_avg_sim", "contrastive_margin",
        "top1_precision", "top5_precision", "top10_precision",
        "best_val_loss",
    ]
    col_labels = [
        "IACS↑", "Inter↓", "Margin↑",
        "P@1↑", "P@5↑", "P@10↑",
        "Val Loss↓",
    ]
    # higher-is-better 欄位（用於熱圖方向）
    higher_better = {
        "IACS": True, "inter_class_avg_sim": False, "contrastive_margin": True,
        "top1_precision": True, "top5_precision": True, "top10_precision": True,
        "best_val_loss": False,
    }

    rows = []
    row_labels = []
    for entry in ablation_summary:
        if entry.get("status") == "failed":
            continue
        exp_key = next(
            (k for k in _ABLATION_LABELS if entry.get("exp_name", "").startswith(k)),
            entry.get("exp_name", "Unknown"),
        )
        row_labels.append(_ABLATION_LABELS.get(exp_key, exp_key))
        rows.append([entry.get(k, float("nan")) for k in metric_keys])

    if not rows:
        warnings.warn("ablation_summary 無有效資料，跳過 Fig 04")
        return None

    data = np.array(rows, dtype=float)   # [n_exp, n_metrics]
    norm = np.zeros_like(data)
    for j, key in enumerate(metric_keys):
        col = data[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) < 2:
            norm[:, j] = 0.5
            continue
        col_n = (col - valid.min()) / (valid.max() - valid.min() + 1e-9)
        norm[:, j] = col_n if higher_better[key] else (1 - col_n)

    fig, ax = plt.subplots(figsize=(13, max(4, len(rows) * 0.85 + 2)))
    im = ax.imshow(norm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.xaxis.set_tick_params(labeltop=True, labelbottom=False)

    for i in range(len(row_labels)):
        for j in range(len(metric_keys)):
            val = data[i, j]
            txt = f"{val:.4f}" if not np.isnan(val) else "N/A"
            brightness = norm[i, j]
            color = "black" if 0.2 < brightness < 0.8 else "white"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8.5, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Normalized Score (green = better)", pad=0.02)

    ax.set_title("Ablation Study — Retrieval Metrics Heatmap\n"
                 "(colour normalized per column: green = better performance)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ─── Fig 05: 消融 Loss 曲線疊加 ──────────────────────────────────────────────


def fig_ablation_loss_curves(
    ablation_logs: Dict[str, pd.DataFrame],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    1×2：Train Loss 與 Val Loss 曲線疊加（6 個消融條件）。
    """
    if not ablation_logs:
        warnings.warn("無消融 training_log 資料，跳過 Fig 05")
        return None

    fig, (ax_tr, ax_val) = plt.subplots(1, 2, figsize=(15, 5))

    for idx, (exp_key, df) in enumerate(ablation_logs.items()):
        label = _ABLATION_LABELS.get(exp_key, exp_key)
        color = _ABL_COLORS[idx % len(_ABL_COLORS)]
        ls = ["-", "--", "-.", ":", (0, (3,1,1,1)), (0,(5,2))][idx % 6]
        epochs = df["epoch"].values
        if "train_loss" in df:
            ax_tr.plot(epochs, df["train_loss"], color=color,
                       linestyle=ls, label=label, alpha=0.85)
        if "val_loss" in df:
            ax_val.plot(epochs, df["val_loss"], color=color,
                        linestyle=ls, label=label, alpha=0.85)

    for ax, title in [(ax_tr, "Train Loss"), (ax_val, "Validation Loss")]:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("SimSiam Loss")
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.suptitle("Ablation Study — Loss Convergence Comparison",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ─── Fig 06: 消融 Feature Std 曲線疊加 ───────────────────────────────────────


def fig_ablation_feature_std(
    ablation_logs: Dict[str, pd.DataFrame],
    proj_dim: int = 2048,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    1×2：Train / Val z-std 曲線疊加 + 崩塌參考線。
    """
    if not ablation_logs:
        warnings.warn("無消融 training_log 資料，跳過 Fig 06")
        return None

    fig, (ax_tr, ax_val) = plt.subplots(1, 2, figsize=(15, 5))
    ref = 1.0 / math.sqrt(proj_dim)

    for idx, (exp_key, df) in enumerate(ablation_logs.items()):
        label = _ABLATION_LABELS.get(exp_key, exp_key)
        color = _ABL_COLORS[idx % len(_ABL_COLORS)]
        ls = ["-", "--", "-.", ":", (0, (3,1,1,1)), (0,(5,2))][idx % 6]
        epochs = df["epoch"].values
        if "train_z_std" in df:
            ax_tr.plot(epochs, df["train_z_std"], color=color,
                       linestyle=ls, label=label, alpha=0.85)
        if "val_z_std" in df:
            ax_val.plot(epochs, df["val_z_std"], color=color,
                        linestyle=ls, label=label, alpha=0.85)

    for ax, title in [(ax_tr, "Train z-std"), (ax_val, "Validation z-std")]:
        ax.axhline(ref, color=_C["neutral"], linestyle=":", linewidth=1.5,
                   label=f"Target 1/√{proj_dim} = {ref:.4f}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Std. Dev. of Projector Output")
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.suptitle("Ablation Study — Feature Collapse Monitor",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ─── 輔助函式 ─────────────────────────────────────────────────────────────────


def _annotate_bars(ax: plt.Axes, bars) -> None:
    """在長條頂端標示數值。"""
    for bar in bars:
        val = bar.get_height()
        if not math.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.003,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=7.5,
            )


def _save(fig: plt.Figure, save_path: Optional[Path]) -> None:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=matplotlib.rcParams["savefig.dpi"],
                    bbox_inches="tight")


# ─── CLI 主程式 ──────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SimSiam v2 學術圖表繪製腳本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "範例：\n"
            "  python v2/plot_results.py\n"
            "  python v2/plot_results.py --output-dir outputs/plots --dpi 300\n"
            "  python v2/plot_results.py --eval-json outputs/eval_results.json\n"
        ),
    )
    p.add_argument(
        "--training-log",
        metavar="PATH",
        help="training_log.csv 路徑（未指定時自動從 outputs/ 偵測）",
    )
    p.add_argument(
        "--summary",
        metavar="PATH",
        help="overall_summary.json 路徑",
    )
    p.add_argument(
        "--ablation-summary",
        metavar="PATH",
        default="outputs/ablation/ablation_summary.json",
        help="ablation_summary.json 路徑（預設: outputs/ablation/ablation_summary.json）",
    )
    p.add_argument(
        "--ablation-output-dir",
        metavar="DIR",
        default="outputs/ablation",
        help="消融實驗輸出根目錄，用於載入各條件的 training_log.csv",
    )
    p.add_argument(
        "--eval-json",
        metavar="PATH",
        default="outputs/eval_results.json",
        help="evaluate.py 產生的 eval_results.json 路徑",
    )
    p.add_argument(
        "--outputs-base",
        metavar="DIR",
        default="outputs",
        help="訓練輸出根目錄，用於自動偵測最新實驗（預設: outputs）",
    )
    p.add_argument(
        "--output-dir",
        metavar="DIR",
        default="outputs/plots",
        help="圖表輸出目錄（預設: outputs/plots）",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="輸出解析度 DPI（預設: 200；論文建議 300）",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="不顯示互動視窗，只儲存檔案",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    matplotlib.rcParams["savefig.dpi"] = args.dpi
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated: List[str] = []
    skipped:   List[str] = []

    # ─── 自動偵測訓練資料 ────────────────────────────────────────────────────
    training_log_path = Path(args.training_log) if args.training_log else None
    summary_path      = Path(args.summary)       if args.summary       else None

    if training_log_path is None or summary_path is None:
        latest = _find_latest_exp_dir(args.outputs_base)
        if latest:
            if training_log_path is None:
                training_log_path = _find_training_log(latest)
            if summary_path is None:
                candidate = latest / "overall_summary.json"
                if candidate.exists():
                    summary_path = candidate

    df_train = _load_csv(training_log_path)
    summary  = _load_json(summary_path)

    # 從 summary 取得 proj_dim 與評估指標
    proj_dim = 2048
    run_metrics: Optional[Dict[str, float]] = None
    top_k_values: List[int] = []

    if summary:
        try:
            proj_dim = summary["config"]["model"]["proj_dim"]
        except (KeyError, TypeError):
            pass
        runs = summary.get("runs", [])
        if runs:
            first_run = runs[0]
            run_metrics, top_k_values = _extract_retrieval_metrics(first_run)

    # ─── Fig 01: 訓練動態 ────────────────────────────────────────────────────
    if df_train is not None and not df_train.empty:
        fig = fig_training_dynamics(
            df_train, proj_dim=proj_dim,
            save_path=out_dir / "fig01_training_dynamics.png",
        )
        plt.close(fig)
        generated.append("fig01_training_dynamics.png")
    else:
        skipped.append("fig01_training_dynamics.png (無 training_log.csv)")

    # ─── Fig 02: 檢索評估指標 ────────────────────────────────────────────────
    if run_metrics:
        fig = fig_retrieval_metrics(
            run_metrics, top_k_values=top_k_values or None,
            save_path=out_dir / "fig02_retrieval_metrics.png",
        )
        plt.close(fig)
        generated.append("fig02_retrieval_metrics.png")
    else:
        skipped.append("fig02_retrieval_metrics.png (無 overall_summary.json 評估結果)")

    # ─── Fig 03: Raw vs Preprocessed ────────────────────────────────────────
    eval_data = _load_json(Path(args.eval_json))
    if eval_data and "raw" in eval_data and "preprocessed" in eval_data:
        fig = fig_raw_vs_preprocessed(
            eval_data,
            save_path=out_dir / "fig03_raw_vs_preprocessed.png",
        )
        if fig is not None:
            plt.close(fig)
            generated.append("fig03_raw_vs_preprocessed.png")
    else:
        skipped.append("fig03_raw_vs_preprocessed.png (無 eval_results.json 或缺少 preprocessed 段)")

    # ─── Fig 04: 消融實驗熱圖 ────────────────────────────────────────────────
    ablation_summary = _load_json(Path(args.ablation_summary))
    if ablation_summary:
        fig = fig_ablation_heatmap(
            ablation_summary,
            save_path=out_dir / "fig04_ablation_heatmap.png",
        )
        if fig is not None:
            plt.close(fig)
            generated.append("fig04_ablation_heatmap.png")
    else:
        skipped.append("fig04_ablation_heatmap.png (無 ablation_summary.json)")

    # ─── Fig 05/06: 消融 loss + z_std 曲線 ──────────────────────────────────
    ablation_logs = _find_ablation_logs(Path(args.ablation_output_dir))
    if ablation_logs:
        fig = fig_ablation_loss_curves(
            ablation_logs,
            save_path=out_dir / "fig05_ablation_loss_curves.png",
        )
        if fig is not None:
            plt.close(fig)
            generated.append("fig05_ablation_loss_curves.png")

        fig = fig_ablation_feature_std(
            ablation_logs, proj_dim=proj_dim,
            save_path=out_dir / "fig06_ablation_feature_std.png",
        )
        if fig is not None:
            plt.close(fig)
            generated.append("fig06_ablation_feature_std.png")
    else:
        skipped.append("fig05_ablation_loss_curves.png (無消融 training_log.csv)")
        skipped.append("fig06_ablation_feature_std.png (無消融 training_log.csv)")

    # ─── 摘要報告 ────────────────────────────────────────────────────────────
    _print_report(generated, skipped, out_dir)

    if not args.no_show and generated:
        plt.show()


def _print_report(generated: List[str], skipped: List[str], out_dir: Path) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print("  圖表產生報告")
    print(sep)
    print(f"  輸出目錄: {out_dir.resolve()}")
    print()
    if generated:
        print(f"  ✔ 已產生 ({len(generated)} 張):")
        for name in generated:
            size_kb = (out_dir / name).stat().st_size / 1024
            print(f"      {name}  ({size_kb:.0f} KB)")
    if skipped:
        print()
        print(f"  ✗ 跳過 ({len(skipped)} 張，資料不足):")
        for reason in skipped:
            print(f"      {reason}")
    print(sep)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n中斷。")
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
