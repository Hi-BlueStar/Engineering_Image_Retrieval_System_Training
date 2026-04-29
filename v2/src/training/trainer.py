"""Trainer 核心訓練迴圈模組 (Trainer Module)。

============================================================
封裝完整的 epoch 訓練迴圈，包含：

- 訓練 (``_train_one_epoch``)
- 驗證 (``_evaluate``)
- 混合精度 (AMP) 自動管理
- Checkpoint 儲存（委派給 ``CheckpointManager``）
- 實驗指標記錄（委派給 ``ExperimentTracker``）

設計原則：
    - **依賴注入**：Trainer 不建立模型、優化器或資料載入器，
      所有依賴均由外部注入。
    - **模型無關**：只要符合 ``nn.Module`` 介面（接受 ``x1, x2``
      輸入、回傳 ``p1, p2, z1, z2``），即可無縫替換模型。
    - **損失函數無關**：損失函數從外部傳入。

效能設計：
    - ``torch.amp.autocast`` (FP16)：train 和 evaluate 均啟用。
    - ``optimizer.zero_grad(set_to_none=True)``：減少記憶體寫入。
    - ``non_blocking=True``：張量搬移不阻塞 CPU。
============================================================
"""

from __future__ import annotations

import contextlib
import time
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from src.dataset.gpu_transforms import GPUAugmentation
from src.model.loss import calculate_collapse_std, simsiam_loss
from src.training.checkpoint import CheckpointManager
from src.training.timer import PrecisionTimer
from src.logger import get_logger

logger = get_logger(__name__)
_console = Console()

# Progress bar 每 N batch 才同步一次 loss，避免 per-batch GPU→CPU sync 拖慢 GPU
_PROGRESS_SYNC_EVERY = 20


def _make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}", justify="left"),
        BarColumn(bar_width=38),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[info]}", justify="right"),
        console=_console,
        refresh_per_second=4,
    )


class Trainer:
    """核心訓練迴圈引擎。

    Args:
        model: SimSiam 模型（或任何符合雙視角介面的模型）。
        optimizer: 優化器。
        scheduler: 學習率排程器。
        scaler: ``torch.amp.GradScaler``。
        checkpoint_mgr: Checkpoint 管理器。
        device: 訓練裝置（``"cuda"`` 或 ``"cpu"``）。
        loss_fn: 損失函數，預設為 ``simsiam_loss``。
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scaler: torch.amp.GradScaler,
        checkpoint_mgr: CheckpointManager,
        device: str = "cuda",
        loss_fn: Optional[Callable] = None,
        grad_clip: float = 0.0,
        gpu_aug: Optional[GPUAugmentation] = None,
        max_batches: Optional[int] = None,
        channels_last: bool = False,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.checkpoint_mgr = checkpoint_mgr
        self.device = device
        self.loss_fn = loss_fn or simsiam_loss
        self.grad_clip = grad_clip
        self.gpu_aug = gpu_aug  # None → DataLoader 已提供 (v1, v2)
        self.max_batches = max_batches
        self.channels_last = channels_last
        self._mem_fmt = torch.channels_last if channels_last else torch.contiguous_format

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        *,
        start_epoch: int = 1,
        best_val_loss: float = float("inf"),
        epoch_callback: Optional[Callable[[dict], None]] = None,
    ) -> dict[str, Any]:
        """執行完整的多 epoch 訓練迴圈。

        Args:
            train_loader: 訓練 DataLoader。
            val_loader: 驗證 DataLoader。
            epochs: 總 epoch 數（不含已完成的 epoch）。
            start_epoch: 起始 epoch（Resume 時 > 1）。
            best_val_loss: 已知最佳 val_loss（Resume 時從 checkpoint 傳入）。
            epoch_callback: 每 epoch 結束後的可選回呼函式，
                接受一個包含該 epoch 指標的字典。

        Returns:
            dict: 訓練結果摘要，包含 ``best_val_loss`` 等。
        """
        use_amp = self.scaler.is_enabled() and str(self.device).startswith("cuda")
        has_val = len(val_loader) > 0

        if start_epoch > epochs:
            logger.info("訓練已完成 (start_epoch=%d > epochs=%d)，跳過", start_epoch, epochs)
            return {"best_val_loss": best_val_loss}

        logger.info(
            "開始訓練: epochs=%d, start=%d, AMP=%s, device=%s, has_val=%s",
            epochs,
            start_epoch,
            use_amp,
            self.device,
            has_val,
        )

        remaining = epochs - start_epoch + 1

        with _make_progress() as progress:
            epoch_task = progress.add_task(
                "[cyan]訓練進度",
                total=remaining,
                info="",
            )

            for epoch in range(start_epoch, epochs + 1):
                epoch_timer = PrecisionTimer(f"epoch_{epoch}")
                epoch_timer.start()
                wall_start = time.perf_counter()

                # --- Train ---
                train_task = progress.add_task(
                    f"[green]  ├─ Train [{epoch}/{epochs}]",
                    total=len(train_loader),
                    info="",
                )
                train_loss, train_std = self._train_one_epoch(
                    train_loader, use_amp, progress, train_task
                )
                progress.remove_task(train_task)

                # --- Evaluate ---
                if has_val:
                    eval_task = progress.add_task(
                        f"[yellow]  └─ Eval  [{epoch}/{epochs}]",
                        total=len(val_loader),
                        info="",
                    )
                    val_loss, val_std = self._evaluate(
                        val_loader, use_amp, progress, eval_task
                    )
                    progress.remove_task(eval_task)
                else:
                    val_loss, val_std = train_loss, train_std

                # --- Checkpoint (暫停計時) ---
                epoch_timer.pause()

                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss

                self.checkpoint_mgr.save(
                    self.model, self.optimizer, epoch, val_loss, is_best,
                    scheduler=self.scheduler, scaler=self.scaler,
                )

                epoch_timer.resume()

                # --- Scheduler ---
                self.scheduler.step()

                # --- 停止 epoch 計時 ---
                epoch_net = epoch_timer.stop()
                epoch_wall = time.perf_counter() - wall_start

                # --- 指標收集 ---
                current_lr = self.optimizer.param_groups[0]["lr"]
                metrics = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_z_std": train_std,
                    "val_z_std": val_std,
                    "lr": current_lr,
                    "epoch_net_sec": epoch_net,
                    "epoch_wall_sec": epoch_wall,
                    "is_best": is_best,
                }

                if epoch_callback:
                    epoch_callback(metrics)

                best_marker = " [bold yellow]★[/]" if is_best else ""
                info_str = (
                    f"[green]t={train_loss:.4f}[/] "
                    f"[yellow]v={val_loss:.4f}[/] "
                    f"lr={current_lr:.1e} "
                    f"best={best_val_loss:.4f}{best_marker}"
                )
                progress.update(epoch_task, advance=1, info=info_str)

                # --- 週期性日誌 ---
                if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
                    logger.info(
                        "Epoch %d/%d — train_loss=%.4f, val_loss=%.4f, "
                        "std=%.4f, lr=%.2e, net=%.1fs",
                        epoch,
                        epochs,
                        train_loss,
                        val_loss,
                        train_std,
                        current_lr,
                        epoch_net,
                    )

        _console.print(
            Panel(
                f"[bold green]訓練完成[/bold green]\n"
                f"Best Val Loss: [cyan]{best_val_loss:.6f}[/cyan]",
                title="[bold]Training Summary",
                border_style="green",
            )
        )
        logger.info("訓練完成: best_val_loss=%.4f", best_val_loss)
        return {"best_val_loss": best_val_loss}

    def _train_one_epoch(
        self,
        loader: DataLoader,
        use_amp: bool,
        progress: Optional[Progress] = None,
        task_id: Optional[TaskID] = None,
    ) -> Tuple[float, float]:
        """執行單個 epoch 的訓練。

        Args:
            loader: 訓練 DataLoader。
            use_amp: 是否啟用混合精度。
            progress: Rich Progress 實例（由 fit() 傳入）。
            task_id: 對應的 Progress task ID。

        Returns:
            Tuple[float, float]: ``(avg_loss, avg_std)``。
        """
        self.model.train()
        total_loss = torch.tensor(0.0, device=self.device)
        total_std = torch.tensor(0.0, device=self.device)
        num_batches = 0

        amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp
            else contextlib.nullcontext()
        )

        for i, batch in enumerate(loader):
            if self.max_batches is not None and i >= self.max_batches:
                break
            if self.gpu_aug is not None:
                # GPU 增強模式：batch = raw tensor [B, C, H, W]
                raw = batch.to(self.device, non_blocking=True)
                v1, v2 = self.gpu_aug.create_views(raw)
            else:
                # CPU 增強模式：batch = (view1, view2)
                v1, v2 = batch
                v1 = v1.to(self.device, non_blocking=True)
                v2 = v2.to(self.device, non_blocking=True)

            if self.channels_last:
                v1 = v1.to(memory_format=self._mem_fmt)
                v2 = v2.to(memory_format=self._mem_fmt)

            self.optimizer.zero_grad(set_to_none=True)

            with amp_ctx:
                p1, p2, z1, z2 = self.model(v1, v2)
                loss = self.loss_fn(p1, p2, z1, z2)

            if use_amp:
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                self.optimizer.step()

            total_loss += loss.detach()

            with torch.no_grad():
                batch_std = (
                    calculate_collapse_std(z1) + calculate_collapse_std(z2)
                ) / 2.0
                total_std += batch_std

            num_batches += 1

            if progress is not None and task_id is not None:
                is_last = (i + 1) == len(loader)
                if num_batches % _PROGRESS_SYNC_EVERY == 0 or is_last:
                    avg_loss = (total_loss / num_batches).item()
                    progress.update(
                        task_id,
                        advance=1,
                        info=f"[dim]loss={avg_loss:.4f}[/]",
                    )
                else:
                    progress.update(task_id, advance=1)

        return (
            (total_loss / max(num_batches, 1)).item(),
            (total_std / max(num_batches, 1)).item(),
        )

    @torch.no_grad()
    def _evaluate(
        self,
        loader: DataLoader,
        use_amp: bool,
        progress: Optional[Progress] = None,
        task_id: Optional[TaskID] = None,
    ) -> Tuple[float, float]:
        """驗證模型損失與標準差。

        注意：此處的 loss 為 SSL 收斂指標，
        而非下游分類準確率。

        Args:
            loader: 驗證 DataLoader。
            use_amp: 是否啟用混合精度。
            progress: Rich Progress 實例（由 fit() 傳入）。
            task_id: 對應的 Progress task ID。

        Returns:
            Tuple[float, float]: ``(avg_loss, avg_std)``。
        """
        self.model.eval()
        total_loss = torch.tensor(0.0, device=self.device)
        total_std = torch.tensor(0.0, device=self.device)
        num_batches = 0

        amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp
            else contextlib.nullcontext()
        )

        for i, batch in enumerate(loader):
            if self.max_batches is not None and i >= self.max_batches:
                break
            if self.gpu_aug is not None:
                raw = batch.to(self.device, non_blocking=True)
                v1, v2 = self.gpu_aug.create_views(raw)
            else:
                v1, v2 = batch
                v1 = v1.to(self.device, non_blocking=True)
                v2 = v2.to(self.device, non_blocking=True)

            if self.channels_last:
                v1 = v1.to(memory_format=self._mem_fmt)
                v2 = v2.to(memory_format=self._mem_fmt)

            with amp_ctx:
                p1, p2, z1, z2 = self.model(v1, v2)
                loss = self.loss_fn(p1, p2, z1, z2)

            total_loss += loss.detach()
            batch_std = (
                calculate_collapse_std(z1) + calculate_collapse_std(z2)
            ) / 2.0
            total_std += batch_std
            num_batches += 1

            if progress is not None and task_id is not None:
                is_last = (i + 1) == len(loader)
                if num_batches % _PROGRESS_SYNC_EVERY == 0 or is_last:
                    avg_loss = (total_loss / num_batches).item()
                    progress.update(
                        task_id,
                        advance=1,
                        info=f"[dim]loss={avg_loss:.4f}[/]",
                    )
                else:
                    progress.update(task_id, advance=1)

        return (
            (total_loss / max(num_batches, 1)).item(),
            (total_std / max(num_batches, 1)).item(),
        )
