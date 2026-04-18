"""實驗紀錄管理器模組 (Experiment Logger Module)。

============================================================
負責管理實驗的完整生命週期紀錄，包含：

1. **Metadata 收集**：系統資訊（Python / PyTorch / CUDA 版本）、
   硬體規格（CPU / GPU / RAM）、Git commit hash。

2. **訓練日誌**：每個 epoch 的 loss、std、lr、duration 即時寫入 CSV。

3. **Checkpoint 管理**：最佳模型、最新模型、定期快照的儲存策略。

4. **視覺化報表**：使用 Plotly 生成互動式 HTML 報表（Loss 與 Std 曲線）。

5. **計時報告**：整合 TimerCollection 的計時明細，輸出 JSON。

設計考量：
- **即時持久化**：每個 epoch 都立即寫入 CSV，防止訓練中斷導致資料遺失。
- **結構化輸出**：所有產出按固定目錄結構組織，方便跨實驗比較。
============================================================
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from rich.console import Console

from src.training.config import TrainingConfig
from src.training.timer import TimerCollection


# ============================================================
# 系統 Metadata 收集工具
# ============================================================


def _collect_system_metadata() -> dict:
    """收集系統與硬體的完整 Metadata。

    收集的資訊包含：
    - Python 版本、作業系統
    - PyTorch 與 CUDA 版本
    - CPU 規格、RAM 容量
    - GPU 型號與顯存（若有 CUDA）
    - Git commit hash（若在 Git 倉庫中）

    Returns:
        dict: 結構化的系統 metadata 字典。
    """
    meta: dict = {}

    # --- Python 與 OS ---
    meta["python_version"] = platform.python_version()
    meta["os"] = f"{platform.system()} {platform.release()}"
    meta["hostname"] = platform.node()

    # --- PyTorch ---
    meta["pytorch_version"] = torch.__version__
    meta["cuda_available"] = torch.cuda.is_available()
    meta["cuda_version"] = (
        torch.version.cuda if torch.cuda.is_available() else None
    )

    # --- CPU ---
    meta["cpu_count"] = os.cpu_count()
    meta["cpu_model"] = platform.processor() or "unknown"

    # --- RAM ---
    try:
        import psutil

        mem = psutil.virtual_memory()
        meta["ram_total_gb"] = round(mem.total / (1024**3), 2)
    except ImportError:
        meta["ram_total_gb"] = None

    # --- GPU ---
    if torch.cuda.is_available():
        gpu_list = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_list.append(
                {
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(
                        props.total_memory / (1024**3), 2
                    ),
                }
            )
        meta["gpus"] = gpu_list
    else:
        meta["gpus"] = []

    # --- Git ---
    meta["git_commit"] = _get_git_commit_hash()

    return meta


def _get_git_commit_hash() -> str | None:
    """嘗試取得當前 Git commit hash。

    Returns:
        str | None: 7 位短 hash，若非 Git 倉庫或取得失敗則回傳 None。
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


# ============================================================
# 實驗紀錄管理器 (Experiment Logger)
# ============================================================


class ExperimentLogger:
    """管理單一實驗（含多 Run）的完整紀錄。

    此類別負責：
    1. 建立實驗目錄結構
    2. 儲存設定與系統 Metadata
    3. 管理每個 Run 的獨立日誌
    4. 生成視覺化報表

    輸出目錄結構：
        outputs/<exp_name>_<timestamp>/
        ├── config.json              # 完整訓練設定
        ├── metadata.json            # 系統/硬體/實驗 Metadata
        ├── timing_report.json       # 計時明細
        ├── overall_summary.json     # 所有 Run 的結果彙整
        └── Run_01_Seed_42/
            ├── training_log.csv     # 逐 epoch 紀錄
            ├── training_report.html # Plotly Loss 曲線
            ├── std_report.html      # Plotly Std 曲線
            └── checkpoints/
                ├── checkpoint_best.pth
                ├── checkpoint_last.pth
                └── checkpoint_epoch_XXXX.pth

    Args:
        config: 訓練設定物件。
        timers: 計時器集合（可選），用於記錄計時明細。
    """

    def __init__(
        self,
        config: TrainingConfig,
        timers: TimerCollection | None = None,
    ) -> None:
        """初始化實驗紀錄管理器。

        建立實驗根目錄，儲存設定與系統 Metadata。

        Args:
            config: 訓練設定物件。
            timers: 計時器集合，用於最終報告中嵌入計時明細。
        """
        self.cfg = config
        self.timers = timers or TimerCollection()
        self.console = Console()

        # --- 建立實驗根目錄 ---
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = (
            Path(config.output_dir) / f"{config.exp_name}_{self.timestamp}"
        )
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # --- 儲存設定 ---
        config_path = self.experiment_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=4, ensure_ascii=False)

        # --- 儲存系統 Metadata ---
        self.metadata = _collect_system_metadata()
        self.metadata["experiment_name"] = config.exp_name
        self.metadata["timestamp"] = self.timestamp
        self.metadata["experiment_dir"] = str(self.experiment_dir)

        metadata_path = self.experiment_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=4, ensure_ascii=False)

        # --- Run 層級的日誌管理 ---
        self._run_loggers: dict[str, _RunLogger] = {}

    def create_run_logger(self, run_name: str) -> _RunLogger:
        """為指定 Run 建立獨立的日誌管理器。

        每個 Run 有自己的日誌目錄、CSV 紀錄與 checkpoint 目錄。

        Args:
            run_name: Run 的識別名稱（例如 "Run_01_Seed_42"）。

        Returns:
            _RunLogger: 該 Run 的日誌管理器實例。
        """
        run_dir = self.experiment_dir / run_name
        logger = _RunLogger(
            run_name=run_name,
            run_dir=run_dir,
            config=self.cfg,
            console=self.console,
        )
        self._run_loggers[run_name] = logger
        return logger

    def save_timing_report(self) -> Path:
        """儲存計時明細到 JSON 檔案。

        Returns:
            Path: 計時報告的檔案路徑。
        """
        report_path = self.experiment_dir / "timing_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(
                self.timers.summary(),
                f,
                indent=4,
                ensure_ascii=False,
            )
        return report_path

    def save_overall_summary(self, run_results: list[dict]) -> Path:
        """儲存所有 Run 的結果彙整。

        Args:
            run_results: 每個 Run 的結果字典列表。

        Returns:
            Path: 總結報告的檔案路徑。
        """
        summary = {
            "experiment_name": self.cfg.exp_name,
            "timestamp": self.timestamp,
            "n_runs": len(run_results),
            "config": self.cfg.to_dict(),
            "system_metadata": self.metadata,
            "timing": self.timers.summary(),
            "runs": run_results,
        }

        summary_path = self.experiment_dir / "overall_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        return summary_path


# ============================================================
# Run 層級日誌管理器 (Run Logger)
# ============================================================


class _RunLogger:
    """管理單一 Run 的訓練紀錄與 Checkpoint。

    此類別不應由使用者直接實例化，而是透過
    ExperimentLogger.create_run_logger() 建立。

    Args:
        run_name: Run 識別名稱。
        run_dir: 該 Run 的輸出目錄。
        config: 訓練設定。
        console: Rich Console 實例。
    """

    def __init__(
        self,
        run_name: str,
        run_dir: Path,
        config: TrainingConfig,
        console: Console,
    ) -> None:
        """初始化 Run 日誌管理器。

        建立 checkpoint 目錄，初始化日誌列表。

        Args:
            run_name: Run 的識別名稱。
            run_dir: 此 Run 的輸出根目錄。
            config: 訓練設定物件。
            console: Rich Console，用於輸出訊息。
        """
        self.run_name = run_name
        self.run_dir = run_dir
        self.cfg = config
        self.console = console
        self.ckpt_dir = run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self._logs: list[dict] = []

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_z_std: float,
        val_z_std: float,
        lr: float,
        epoch_net_duration: float,
        epoch_wall_duration: float,
    ) -> None:
        """記錄單個 Epoch 的訓練數據。

        每次呼叫都會立即將所有歷史紀錄寫入 CSV，
        確保訓練中斷時不會遺失已完成 epoch 的資料。

        Args:
            epoch: 當前 epoch 編號（1-indexed）。
            train_loss: 訓練集平均 loss。
            val_loss: 驗證集平均 loss。
            train_z_std: 訓練集特徵標準差（坍塌監控）。
            val_z_std: 驗證集特徵標準差。
            lr: 當前學習率。
            epoch_net_duration: 該 epoch 的淨計算耗時（秒）。
            epoch_wall_duration: 該 epoch 的牆鐘耗時（秒）。
        """
        self._logs.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_z_std": train_z_std,
                "val_z_std": val_z_std,
                "lr": lr,
                "epoch_net_sec": round(epoch_net_duration, 4),
                "epoch_wall_sec": round(epoch_wall_duration, 4),
            }
        )
        # --- 即時持久化：防止訓練意外中斷導致資料遺失 ---
        df = pd.DataFrame(self._logs)
        df.to_csv(self.run_dir / "training_log.csv", index=False)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_loss: float,
        is_best: bool = False,
    ) -> None:
        """儲存模型 Checkpoint。

        儲存策略：
        1. **checkpoint_last.pth**：每個 epoch 都覆蓋（確保可恢復最新狀態）。
        2. **checkpoint_epoch_XXXX.pth**：每 save_freq 個 epoch 保留一份。
        3. **checkpoint_best.pth**：當 is_best=True 時額外儲存。

        Args:
            model: 模型實例。
            optimizer: 優化器實例。
            epoch: 當前 epoch。
            val_loss: 當前驗證集 loss。
            is_best: 是否為目前最佳模型。
        """
        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss,
            "config": self.cfg.to_dict(),
        }

        # 1. 最新 checkpoint（每 epoch 覆蓋）
        torch.save(state, self.ckpt_dir / "checkpoint_last.pth")

        # 2. 定期快照
        if epoch % self.cfg.save_freq == 0:
            torch.save(
                state,
                self.ckpt_dir / f"checkpoint_epoch_{epoch:04d}.pth",
            )

        # 3. 最佳模型
        if is_best:
            torch.save(state, self.ckpt_dir / "checkpoint_best.pth")
            self.console.print(
                f"      [dim]💾 Best model saved (Val Loss: {val_loss:.4f})[/dim]"
            )

    def generate_report(self) -> None:
        """使用 Plotly 生成訓練視覺化報表。

        產生兩份 HTML 互動圖表：
        1. training_report.html — Loss 曲線（Train vs Val）
        2. std_report.html — 特徵標準差曲線（坍塌監控）

        Raises:
            ImportError: 若 plotly 未安裝。
        """
        if not self._logs:
            return

        try:
            import plotly.graph_objects as go
        except ImportError:
            self.console.print(
                "[yellow]⚠ plotly 未安裝，跳過報表生成。[/yellow]"
            )
            return

        df = pd.DataFrame(self._logs)

        # --- Loss 曲線 ---
        fig_loss = go.Figure()
        fig_loss.add_trace(
            go.Scattergl(
                x=df["epoch"],
                y=df["train_loss"],
                mode="lines+markers",
                name="Train Loss",
                line=dict(color="#3B82F6"),  # 藍色
            )
        )
        fig_loss.add_trace(
            go.Scattergl(
                x=df["epoch"],
                y=df["val_loss"],
                mode="lines+markers",
                name="Val Loss",
                line=dict(color="#F97316"),  # 橘色
            )
        )
        fig_loss.update_layout(
            title=f"SimSiam Training Loss — {self.run_name}",
            xaxis_title="Epoch",
            yaxis_title="Negative Cosine Similarity Loss",
            template="plotly_white",
            hovermode="x unified",
        )
        fig_loss.write_html(
            self.run_dir / "training_report.html",
            include_plotlyjs="cdn",
        )

        # --- 特徵標準差曲線 ---
        fig_std = go.Figure()
        fig_std.add_trace(
            go.Scattergl(
                x=df["epoch"],
                y=df["train_z_std"],
                mode="lines+markers",
                name="Train Std",
                line=dict(color="#3B82F6"),
            )
        )
        if df["val_z_std"].notna().any():
            fig_std.add_trace(
                go.Scattergl(
                    x=df["epoch"],
                    y=df["val_z_std"],
                    mode="lines+markers",
                    name="Val Std",
                    line=dict(color="#F97316"),
                )
            )
        fig_std.update_layout(
            title=f"Feature Std (z) — {self.run_name}",
            xaxis_title="Epoch",
            yaxis_title="Standard Deviation (Target ≈ 1/√d)",
            template="plotly_white",
            hovermode="x unified",
        )
        fig_std.write_html(
            self.run_dir / "std_report.html",
            include_plotlyjs="cdn",
        )
