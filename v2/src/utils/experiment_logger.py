"""日誌紀錄與實驗視覺化模組 (Experiment Logger)。

管理實驗日誌、Checkpoint 儲存與繪製 Loss 曲線報表。
"""
import json
import os
import platform
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf

from src.config.structured import Config
from src.utils.timer import TimerCollection

# 嘗試使用 Rich 提供更友善的終端輸出
try:
    from rich.console import Console
    console = Console()
except ImportError:
    class Console:
        def print(self, msg, *args, **kwargs):
            print(msg)
    console = Console()


def _collect_system_metadata() -> dict:
    meta = {
        "python_version": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "hostname": platform.node(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cpu_count": os.cpu_count(),
        "cpu_model": platform.processor() or "unknown",
    }
    # 獲取 RAM
    try:
        import psutil
        mem = psutil.virtual_memory()
        meta["ram_total_gb"] = round(mem.total / (1024**3), 2)
    except ImportError:
        meta["ram_total_gb"] = None
        
    # 獲取 GPUs
    if torch.cuda.is_available():
        gpu_list = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_list.append({
                "index": i,
                "name": props.name,
                "total_memory_gb": round(props.total_memory / (1024**3), 2),
            })
        meta["gpus"] = gpu_list
    else:
        meta["gpus"] = []
    return meta


class RunLogger:
    """單次訓練 Run 的 Log 管理器。"""
    
    def __init__(self, run_name: str, run_dir: Path, cfg: Config):
        self.run_name = run_name
        self.run_dir = run_dir
        self.cfg = cfg
        self.ckpt_dir = run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._logs = []

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, train_std: float, val_std: float, lr: float, duration: float):
        self._logs.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_z_std": train_std,
            "val_z_std": val_std,
            "lr": lr,
            "duration": round(duration, 4)
        })
        # 即時持久化
        df = pd.DataFrame(self._logs)
        df.to_csv(self.run_dir / "training_log.csv", index=False)

    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, val_loss: float, is_best: bool = False):
        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss,
            "config": OmegaConf.to_container(self.cfg)
        }
        torch.save(state, self.ckpt_dir / "checkpoint_last.pth")
        
        if epoch % self.cfg.output.save_freq == 0:
            torch.save(state, self.ckpt_dir / f"checkpoint_epoch_{epoch:04d}.pth")
            
        if is_best:
            torch.save(state, self.ckpt_dir / "checkpoint_best.pth")
            console.print(f"      [dim]💾 Best model saved (Val Loss: {val_loss:.4f})[/dim]")

    def generate_report(self):
        if not self._logs: return
        try:
            import plotly.graph_objects as go
        except ImportError:
            return
            
        df = pd.DataFrame(self._logs)
        # Loss Report
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scattergl(x=df["epoch"], y=df["train_loss"], mode="lines+markers", name="Train Loss", line=dict(color="#3B82F6")))
        fig_loss.add_trace(go.Scattergl(x=df["epoch"], y=df["val_loss"], mode="lines+markers", name="Val Loss", line=dict(color="#F97316")))
        fig_loss.update_layout(title=f"SimSiam Training Loss — {self.run_name}", xaxis_title="Epoch", yaxis_title="Loss", template="plotly_white", hovermode="x unified")
        fig_loss.write_html(self.run_dir / "training_report.html", include_plotlyjs="cdn")
        
        # Std Report
        fig_std = go.Figure()
        fig_std.add_trace(go.Scattergl(x=df["epoch"], y=df["train_z_std"], mode="lines+markers", name="Train Std", line=dict(color="#3B82F6")))
        if df["val_z_std"].notna().any():
            fig_std.add_trace(go.Scattergl(x=df["epoch"], y=df["val_z_std"], mode="lines+markers", name="Val Std", line=dict(color="#F97316")))
        fig_std.update_layout(title=f"Feature Std (z) — {self.run_name}", xaxis_title="Epoch", yaxis_title="Standard Deviation", template="plotly_white", hovermode="x unified")
        fig_std.write_html(self.run_dir / "std_report.html", include_plotlyjs="cdn")


class ExperimentLogger:
    """全域實驗記錄器。"""
    
    def __init__(self, cfg: Config, timers: TimerCollection = None):
        self.cfg = cfg
        self.timers = timers or TimerCollection()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = Path(cfg.output.output_dir) / f"{cfg.output.exp_name}_{self.timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save YAML config
        with open(self.exp_dir / "config.yaml", "w", encoding="utf-8") as f:
            f.write(OmegaConf.to_yaml(self.cfg))
            
        # Save Meta
        self.metadata = _collect_system_metadata()
        with open(self.exp_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=4)
            
    def create_run_logger(self, run_name: str) -> RunLogger:
        run_dir = self.exp_dir / run_name
        run_dir.mkdir(exist_ok=True)
        return RunLogger(run_name, run_dir, self.cfg)

    def save_overall_summary(self, run_results: list[dict]):
        summary = {
            "experiment": self.cfg.output.exp_name,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "timing": self.timers.summary(),
            "runs": run_results
        }
        with open(self.exp_dir / "overall_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
