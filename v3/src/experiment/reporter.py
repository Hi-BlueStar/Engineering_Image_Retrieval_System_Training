"""Plotly 視覺化報表模組 (Reporter Module)。

============================================================
生成互動式 HTML 報表，用於視覺化訓練過程：

1. **Loss 曲線**：Train vs Val loss 趨勢圖。
2. **特徵標準差曲線**：坍塌監控（Dimensional Collapse）。

設計原則：
    - 純函式介面，只接受 DataFrame，不依賴任何訓練物件。
    - 低耦合：可獨立於 Trainer 與 Tracker 使用。
============================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)


def generate_loss_report(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str = "SimSiam Training Loss",
) -> Path:
    """生成 Loss 曲線的互動式 HTML 報表。

    Args:
        df: 包含 ``epoch``、``train_loss``、``val_loss`` 欄位的
            DataFrame。
        output_path: HTML 輸出路徑。
        title: 圖表標題。

    Returns:
        Path: 報表檔案路徑。

    Raises:
        ImportError: 若 ``plotly`` 未安裝。
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.warning("plotly 未安裝，跳過 Loss 報表生成")
        return Path(output_path)

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=df["epoch"],
            y=df["train_loss"],
            mode="lines+markers",
            name="Train Loss",
            line=dict(color="#3B82F6"),
        )
    )
    if "val_loss" in df.columns:
        fig.add_trace(
            go.Scattergl(
                x=df["epoch"],
                y=df["val_loss"],
                mode="lines+markers",
                name="Val Loss",
                line=dict(color="#F97316"),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title="Negative Cosine Similarity Loss",
        template="plotly_white",
        hovermode="x unified",
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn")
    logger.info("Loss 報表已生成: %s", out)
    return out


def generate_std_report(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str = "Feature Std (z) — Collapse Monitor",
) -> Path:
    """生成特徵標準差曲線的互動式 HTML 報表。

    Args:
        df: 包含 ``epoch``、``train_z_std`` 欄位的 DataFrame。
            可選包含 ``val_z_std``。
        output_path: HTML 輸出路徑。
        title: 圖表標題。

    Returns:
        Path: 報表檔案路徑。

    Raises:
        ImportError: 若 ``plotly`` 未安裝。
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.warning("plotly 未安裝，跳過 Std 報表生成")
        return Path(output_path)

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=df["epoch"],
            y=df["train_z_std"],
            mode="lines+markers",
            name="Train Std",
            line=dict(color="#3B82F6"),
        )
    )

    if "val_z_std" in df.columns and df["val_z_std"].notna().any():
        fig.add_trace(
            go.Scattergl(
                x=df["epoch"],
                y=df["val_z_std"],
                mode="lines+markers",
                name="Val Std",
                line=dict(color="#F97316"),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title="Standard Deviation (Target ≈ 1/√d)",
        template="plotly_white",
        hovermode="x unified",
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn")
    logger.info("Std 報表已生成: %s", out)
    return out


def generate_run_reports(
    df: pd.DataFrame,
    run_dir: str | Path,
    run_name: str,
) -> None:
    """為單一 Run 生成所有報表。

    便利函式：依序生成 Loss 與 Std 報表。

    Args:
        df: 該 Run 的訓練日誌 DataFrame。
        run_dir: Run 的輸出目錄。
        run_name: Run 識別名稱。
    """
    if df.empty:
        logger.warning("Run %s 無訓練日誌，跳過報表生成", run_name)
        return

    run_path = Path(run_dir)
    generate_loss_report(
        df,
        run_path / "training_report.html",
        title=f"SimSiam Training Loss — {run_name}",
    )
    if "train_z_std" in df.columns:
        generate_std_report(
            df,
            run_path / "std_report.html",
            title=f"Feature Std (z) — {run_name}",
        )
