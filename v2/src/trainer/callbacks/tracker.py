"""
即時指標記錄器回調 (Metrics Tracker Callback)。

負責將訓練過程丟入 CSV，並在最終產出 HTML 互動報表。
把報表生成的雜亂程式碼與 Trainer 本身解耦。
"""

import pandas as pd
from pathlib import Path
from ..engine import Callback
from ...core.logger import get_logger

logger = get_logger(__name__)

class MetricsTrackerCallback(Callback):
    """負責寫入 CSV 訓練日誌以及 Plotly 報表生成。"""
    
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logs = []

    def on_epoch_end(self, trainer, epoch: int, metrics: dict):
        rec = {"epoch": epoch, **metrics}
        self.logs.append(rec)
        
        # 即時持久化保護：防禦未預期的中斷導致歷史數據遺失
        df = pd.DataFrame(self.logs)
        df.to_csv(self.run_dir / "training_log.csv", index=False)
        
        # 改用 Logger 輸出當前 Epoch 資訊
        logger.info(
            f"Epoch {epoch:04d} | "
            f"Train Loss: {metrics['train_loss']:.4f} | "
            f"Val Loss: {metrics['val_loss']:.4f} | "
            f"LRate: {metrics['lr']:.2e} | "
            f"Net Time: {metrics['duration']:.1f}s"
        )

    def on_train_end(self, trainer):
        """訓練結束後，為此 Run 生成獨立的 Plotly 互動式報表。"""
        if not self.logs:
            return
            
        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.warning("未安裝 plotly，略過圖表生成程序。")
            return
            
        df = pd.DataFrame(self.logs)
        
        # 繪製 Loss 曲線
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=df["epoch"], y=df["train_loss"], name="Train Loss", mode="lines+markers"))
        fig.add_trace(go.Scattergl(x=df["epoch"], y=df["val_loss"], name="Val Loss", mode="lines+markers"))
        fig.update_layout(title="SimSiam Loss Curve", template="plotly_white")
        fig.write_html(self.run_dir / "loss_curve.html", include_plotlyjs="cdn")
        
        # 繪製 Dimensional Collapse 監控
        fig_std = go.Figure()
        fig_std.add_trace(go.Scattergl(x=df["epoch"], y=df["train_std"], name="Train Std", mode="lines+markers"))
        fig_std.add_trace(go.Scattergl(x=df["epoch"], y=df["val_std"], name="Val Std", mode="lines+markers"))
        fig_std.update_layout(title="Dimensional Collapse Tracker (Standard Deviation)", template="plotly_white")
        fig_std.write_html(self.run_dir / "collapse_std_curve.html", include_plotlyjs="cdn")
        logger.info("視覺化報表生成完畢，已保存至資料夾。")
