"""實驗追蹤模組 (Experiment Tracker Module)。

============================================================
管理單一實驗（含多 Run）的完整紀錄生命週期：

1. **系統 Metadata**：Python/PyTorch/CUDA/GPU/RAM/Git。
2. **訓練日誌**：每 epoch 即時寫入 CSV（防中斷遺失）。
3. **計時報告**：JSON 格式的分步計時明細。
4. **Run 結果彙整**：所有 Run 的總結 JSON。

輸出目錄結構::

    outputs/<exp_name>_<timestamp>/
    ├── config.json
    ├── metadata.json
    ├── timing_report.json
    ├── overall_summary.json
    └── Run_01_Seed_42/
        ├── training_log.csv
        └── checkpoints/
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

from src.config import AppConfig
from src.training.timer import TimerCollection
from src.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 系統 Metadata 收集
# ============================================================


def _collect_system_metadata() -> Dict[str, Any]:
    """收集系統與硬體的完整 Metadata。

    Returns:
        Dict[str, Any]: 結構化系統資訊。
    """
    meta: Dict[str, Any] = {}

    # Python & OS
    meta["python_version"] = platform.python_version()
    meta["os"] = f"{platform.system()} {platform.release()}"
    meta["hostname"] = platform.node()

    # PyTorch
    meta["pytorch_version"] = torch.__version__
    meta["cuda_available"] = torch.cuda.is_available()
    meta["cuda_version"] = (
        torch.version.cuda if torch.cuda.is_available() else None
    )

    # CPU & RAM
    meta["cpu_count"] = os.cpu_count()
    meta["cpu_model"] = platform.processor() or "unknown"

    try:
        import psutil

        mem = psutil.virtual_memory()
        meta["ram_total_gb"] = round(mem.total / (1024**3), 2)
    except ImportError:
        meta["ram_total_gb"] = None

    # GPU
    if torch.cuda.is_available():
        gpu_list = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_list.append(
                {
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                }
            )
        meta["gpus"] = gpu_list
    else:
        meta["gpus"] = []

    # Git
    meta["git_commit"] = _get_git_commit()

    return meta


def _get_git_commit() -> Optional[str]:
    """嘗試取得 Git commit hash。

    Returns:
        Optional[str]: 7 位短 hash 或 ``None``。
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
# ExperimentTracker
# ============================================================


class ExperimentTracker:
    """管理單一實驗的完整紀錄。

    Args:
        config: 應用程式設定。
        timers: 計時器集合。
    """

    def __init__(
        self,
        config: AppConfig,
        timers: Optional[TimerCollection] = None,
    ) -> None:
        self.cfg = config
        self.timers = timers or TimerCollection()

        # --- 建立實驗目錄 ---
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = (
            Path(config.experiment.output_dir)
            / f"{config.experiment.exp_name}_{self.timestamp}"
        )
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # --- 儲存設定 ---
        config_path = self.experiment_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=4, ensure_ascii=False)

        # --- 儲存 Metadata ---
        self.metadata = _collect_system_metadata()
        self.metadata["experiment_name"] = config.experiment.exp_name
        self.metadata["timestamp"] = self.timestamp

        metadata_path = self.experiment_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=4, ensure_ascii=False)

        self._run_trackers: Dict[str, RunTracker] = {}
        logger.info("實驗追蹤器初始化: %s", self.experiment_dir)

    def create_run(self, run_name: str) -> "RunTracker":
        """為指定 Run 建立獨立追蹤器。

        Args:
            run_name: Run 識別名稱。

        Returns:
            RunTracker: Run 追蹤器實例。
        """
        run_dir = self.experiment_dir / run_name
        tracker = RunTracker(run_name=run_name, run_dir=run_dir)
        self._run_trackers[run_name] = tracker
        return tracker

    def save_timing_report(self) -> Path:
        """儲存計時明細至 JSON。

        Returns:
            Path: 報告檔案路徑。
        """
        path = self.experiment_dir / "timing_report.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.timers.summary(), f, indent=4, ensure_ascii=False)
        logger.info("計時報告已儲存: %s", path)
        return path

    def save_summary(self, run_results: List[Dict]) -> Path:
        """儲存所有 Run 結果彙整。

        Args:
            run_results: 每個 Run 的結果字典列表。

        Returns:
            Path: 總結報告路徑。
        """
        summary = {
            "experiment_name": self.cfg.experiment.exp_name,
            "timestamp": self.timestamp,
            "n_runs": len(run_results),
            "config": self.cfg.to_dict(),
            "system_metadata": self.metadata,
            "timing": self.timers.summary(),
            "runs": run_results,
        }

        path = self.experiment_dir / "overall_summary.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        logger.info("總結報告已儲存: %s", path)
        return path


# ============================================================
# RunTracker
# ============================================================


class RunTracker:
    """管理單一 Run 的訓練紀錄。

    不應直接實例化，請透過
    ``ExperimentTracker.create_run()`` 建立。

    Args:
        run_name: Run 識別名稱。
        run_dir: Run 輸出目錄。
    """

    def __init__(self, run_name: str, run_dir: Path) -> None:
        self.run_name = run_name
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._logs: List[Dict[str, Any]] = []

    def log_epoch(self, metrics: Dict[str, Any]) -> None:
        """記錄單個 epoch 的指標並即時持久化至 CSV（append 模式）。

        Args:
            metrics: 包含 ``epoch``、``train_loss``、``val_loss``
                等鍵的指標字典。
        """
        self._logs.append(metrics)
        write_header = len(self._logs) == 1
        pd.DataFrame([metrics]).to_csv(
            self.run_dir / "training_log.csv",
            mode="a",
            header=write_header,
            index=False,
        )

    def get_logs_dataframe(self) -> pd.DataFrame:
        """取得所有 epoch 日誌的 DataFrame。

        Returns:
            pd.DataFrame: 日誌 DataFrame。
        """
        return pd.DataFrame(self._logs)
