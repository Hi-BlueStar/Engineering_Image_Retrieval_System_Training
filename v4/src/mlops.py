"""MLOps 與實驗追蹤模組 (MLOps & Experiment Tracking Module)。

============================================================
負責記錄訓練期間的所有指標與環境元數據 (Metadata)。

核心功能：
    1. **系統元數據收集**：記錄 Python, PyTorch, CUDA, GPU 型號與記憶體、主機 OS 以及 Git commit 資訊。
    2. **CSV 即時寫入**：每個 Epoch 結束後，自動將 Loss, Val Loss, 特徵標準差 (Monitor Std), LR 附加至 CSV，防範異常中斷。
    3. **Plotly HTML 報告生成**：訓練完成後使用 Plotly WebGL (Scattergl) 引擎渲染損失與特徵坍塌監控曲線，產出無依賴的互動式 HTML 圖表。
============================================================
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch

from .logger import get_logger

logger = get_logger(__name__)


def collect_system_metadata() -> Dict[str, Any]:
    """收集運行環境的系統與硬體規格元數據 (Metadata)"""
    meta: Dict[str, Any] = {}

    # 作業系統與 Python 版本
    meta["python_version"] = platform.python_version()
    meta["os"] = f"{platform.system()} {platform.release()}"
    meta["hostname"] = platform.node()

    # PyTorch 與 CUDA 資訊
    meta["pytorch_version"] = torch.__version__
    meta["cuda_available"] = torch.cuda.is_available()
    meta["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else None

    # CPU 與 RAM
    meta["cpu_count"] = os.cpu_count()
    meta["cpu_model"] = platform.processor() or "unknown"

    try:
        import psutil
        mem = psutil.virtual_memory()
        meta["ram_total_gb"] = round(mem.total / (1024**3), 2)
    except ImportError:
        meta["ram_total_gb"] = None

    # GPU 硬體參數
    if torch.cuda.is_available():
        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                "index": i,
                "name": props.name,
                "total_memory_gb": round(props.total_memory / (1024**3), 2),
            })
        meta["gpus"] = gpus
    else:
        meta["gpus"] = []

    # Git 資訊 (若可用)
    meta["git_commit"] = _get_git_commit_hash()

    return meta


def _get_git_commit_hash() -> Optional[str]:
    """嘗試取得當前代碼的 Git Commit Hash (短代碼)"""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return None


class RunTracker:
    """管理單一訓練運行的指標記錄器 (Run Tracker)"""
    def __init__(self, run_name: str, run_dir: Path, config_dict: Dict[str, Any]) -> None:
        self.run_name = run_name
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_csv_path = self.run_dir / "training_log.csv"
        
        # 儲存該 Run 的設定檔為 json
        with open(self.run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)
            
        # 儲存系統環境元數據
        self.metadata = collect_system_metadata()
        self.metadata["run_name"] = run_name
        self.metadata["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.run_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=4, ensure_ascii=False)

        self._epoch_logs: List[Dict[str, Any]] = []

    def log_epoch(self, metrics: Dict[str, Any]) -> None:
        """記錄單個 Epoch 的訓練指標，並即時附加 (Append) 到 CSV 本地檔案中"""
        self._epoch_logs.append(metrics)
        write_header = not self.log_csv_path.exists()
        
        # 轉為 DataFrame 後附加寫入
        pd.DataFrame([metrics]).to_csv(
            self.log_csv_path,
            mode="a",
            header=write_header,
            index=False,
            encoding="utf-8"
        )


def generate_plotly_report(log_csv_path: str | Path, output_report_path: str | Path, exp_name: str) -> None:
    """讀取 CSV 訓練記錄，利用 Plotly WebGL (Scattergl) 引擎生成無依賴的互動式 HTML 圖表報告"""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.warning("未偵測到 plotly 庫，跳過互動式圖表報告生成。")
        return

    csv_p = Path(log_csv_path)
    if not csv_p.is_file():
        logger.warning("找不到訓練記錄檔 %s，無法生成 HTML 報告。", csv_p)
        return

    try:
        df = pd.read_csv(csv_p)
        if df.empty or "epoch" not in df.columns:
            logger.warning("記錄檔內容為空，無法生成 HTML 報告。")
            return

        # 建立 3 行 1 列的子圖
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                "對稱相似度損失值 (Negative Cosine Similarity Loss)",
                "特徵坍塌監控標準差 (Feature Standard Deviation - z)",
                "學習率變化趨勢 (Learning Rate)"
            )
        )

        # 1. 繪製 Loss 曲線
        fig.add_trace(
            go.Scattergl(
                x=df["epoch"], y=df["train_loss"],
                mode="lines+markers", name="Train Loss",
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        if "val_loss" in df.columns:
            fig.add_trace(
                go.Scattergl(
                    x=df["epoch"], y=df["val_loss"],
                    mode="lines+markers", name="Val Loss",
                    line=dict(color="#ff7f0e", width=2),
                    marker=dict(size=4)
                ),
                row=1, col=1
            )

        # 2. 繪製 Feature Std 曲線
        fig.add_trace(
            go.Scattergl(
                x=df["epoch"], y=df["train_std"],
                mode="lines+markers", name="Train Std",
                line=dict(color="#2ca02c", width=2),
                marker=dict(size=4)
            ),
            row=2, col=1
        )
        if "val_std" in df.columns:
            fig.add_trace(
                go.Scattergl(
                    x=df["epoch"], y=df["val_std"],
                    mode="lines+markers", name="Val Std",
                    line=dict(color="#d62728", width=2),
                    marker=dict(size=4)
                ),
                row=2, col=1
            )
        
        # 標定坍塌臨界警戒線 (std=0.01) 與理論均勻線 (std=0.022)
        fig.add_hline(y=0.022, line_dash="dash", line_color="gray", annotation_text="理論均勻線 (1/sqrt(d))", row=2, col=1)
        fig.add_hline(y=0.010, line_dash="dot", line_color="red", annotation_text="坍塌臨界警戒線", row=2, col=1)

        # 3. 繪製 Learning Rate 曲線
        fig.add_trace(
            go.Scattergl(
                x=df["epoch"], y=df["lr"],
                mode="lines+markers", name="Learning Rate",
                line=dict(color="#9467bd", width=2),
                marker=dict(size=4)
            ),
            row=3, col=1
        )

        fig.update_layout(
            title=f"SimSiam 訓練指標趨勢分析報告 - {exp_name}",
            template="plotly_white",
            height=900,
            hovermode="x unified",
            showlegend=True
        )
        fig.update_xaxes(title_text="Epoch", row=3, col=1)

        out_path = Path(output_report_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_path), include_plotlyjs="cdn")
        logger.info("Plotly HTML 趨勢報告已生成: %s", out_path.resolve())
    except Exception as e:
        logger.error("生成 HTML 報告失敗: %s", e)
