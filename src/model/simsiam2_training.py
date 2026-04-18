"""
SimSiam 訓練控制器 (Trainer Script)
------------------------------------------------------------------------------
功能：
1. 載入圖片並進行 訓練/驗證 (Train/Val) 分割。
2. 執行 SimSiam 自監督訓練迴圈。
3. 支援自動儲存 Checkpoint (最佳模型與最新模型)。
4. 使用 Plotly 繪製訓練過程的 Loss 曲線並輸出 HTML 報告。

使用方式：
    python simsiam_training.py
"""

import functools
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import psutil
from typing import List, Dict


# 引入核心模組
import simsiam2 as ss
import augmentations as aug
import torch
from rich import box  # noqa: F401
from rich.console import Console
from rich.markup import escape  # noqa: F401
from rich.panel import Panel  # noqa: F401
from rich.progress import (
    BarColumn,  # noqa: F401
    Progress,  # noqa: F401
    SpinnerColumn,  # noqa: F401
    TaskProgressColumn,  # noqa: F401
    TextColumn,  # noqa: F401
    TimeElapsedColumn,  # noqa: F401
)
from rich.table import Table  # noqa: F401
from rich.traceback import install  # noqa: F401


# 安裝 Rich 的錯誤追蹤 (讓 Traceback 變漂亮)
install()

# 初始化全域 Console
console = Console()

# -----------------------------------------------------------------------------
# 1. 訓練參數設定 (Configuration)
# -----------------------------------------------------------------------------


@dataclass
class Config:
    # --- 資料設定 ---
    dataset_root: str = "dataset"  # 資料集根目錄
    # 指定 Run 資料夾內的相對路徑結構
    train_subpath: str = "Component_Dataset/train"
    val_subpath: str = "Component_Dataset/val"

    img_size: int = 512  # 圖片輸入尺寸
    img_exts: tuple = (".jpg", ".png", ".bmp", ".tif", ".webp")  # 支援格式

    # --- 模型設定 ---
    backbone: str = "resnet18"  # 骨幹網路 (resnet18/resnet50)
    pretrained: bool = True  # 是否載入 ImageNet 預訓練權重
    in_channels: int = 1  # 輸入通道數 (1=灰階, 3=RGB)

    # --- 訓練設定 ---
    epochs: int = 200  # 總訓練輪數
    batch_size: int = 64  # 批次大小
    lr: float = 2e-5  # 學習率 (SSL 原本 3e-6 太小，改為 2e-5 以利特徵空間展開)
    weight_decay: float = 1e-5  # 權重衰減
    num_workers: int = 8  # DataLoader Workers (Windows 建議設 0)
    seed: int = 42  # 隨機種子

    # --- 輸出設定 ---
    save_freq: int = 10  # Checkpoint 存檔頻率 (Epochs)
    output_dir: str = "outputs"  # 輸出根目錄
    exp_name: str = "simsiam_exp_01"  # 實驗名稱

    def to_dict(self):
        return asdict(self)


# -----------------------------------------------------------------------------
# 1.5. 裝飾器：計時與資源監控 (Decorators)
# -----------------------------------------------------------------------------


def monitor_resources(func):
    """
    裝飾器：計算函式執行時間，並監控 CPU、RAM 與 GPU (若可用) 的佔用狀況。
    執行完畢後會將效能數據印在 Console 上。
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 重置 GPU 峰值記憶體統計 (確保是這段程式碼的增量)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start_time = time.time()
        start_cpu = psutil.cpu_percent(interval=None)

        try:
            result = func(*args, **kwargs)
        except Exception as e:
            # 錯誤處理：紅色粗體顯示
            console.print(
                f"[bold red]❌ 執行錯誤 ({func.__name__}): {escape(str(e))}[/bold red]"
            )
            raise e

        end_time = time.time()
        duration = end_time - start_time

        # 收集系統資源數據
        cpu_usage = psutil.cpu_percent(interval=None)
        mem_info = psutil.virtual_memory()
        mem_usage = mem_info.percent

        gpu_info = ""
        if torch.cuda.is_available():
            # 取得峰值記憶體 (MB)
            gpu_mem = torch.cuda.max_memory_allocated() / (1024**2)
            gpu_info = f" | [magenta]GPU Mem:[/magenta] {gpu_mem:.0f}MB"

        # 使用 dim 樣式顯示，避免搶奪主要訊息焦點
        console.print(
            f"   ↳ [dim]Completed '{func.__name__}' in {duration:.2f}s "
            f"| CPU: {cpu_usage}% | RAM: {mem_usage}%{gpu_info}[/dim]"
        )

        return result

    return wrapper


# -----------------------------------------------------------------------------
# 2. 輔助功能：圖表與儲存 (Utils)
# -----------------------------------------------------------------------------


class ExperimentManager:
    """管理實驗目錄、日誌記錄與 Checkpoint 儲存。"""

    def __init__(self, config: Config):
        self.cfg = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path(config.output_dir) / f"{config.exp_name}_{self.timestamp}"
        self.ckpt_dir = self.save_dir / "checkpoints"

        # 建立目錄
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # 儲存設定檔
        with open(self.save_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=4, ensure_ascii=False)

        # 初始化日誌 DataFrame
        self.logs: list[dict] = []
        print(f"[Info] 實驗目錄已建立: {self.save_dir}")

    def log_epoch(
        self, epoch: int, train_loss: float, val_loss: float, train_z_std: float, val_z_std: float, lr: float, duration: float
    ):
        """記錄單個 Epoch 的數據。"""
        self.logs.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_z_std": train_z_std,
                "val_z_std": val_z_std,
                "lr": lr,
                "duration": duration,
            }
        )
        # 即時寫入 CSV 防止中斷丟失
        df = pd.DataFrame(self.logs)
        df.to_csv(self.save_dir / "training_log.csv", index=False)

    def save_checkpoint(self, model, optimizer, epoch, loss, is_best=False):
        """儲存模型權重。"""
        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": loss,
            "config": self.cfg.to_dict(),
        }

        # 1. 儲存最新的 Checkpoint (覆蓋舊的)
        last_path = self.ckpt_dir / "checkpoint_last.pth"
        torch.save(state, last_path)

        # 2. 儲存當前 Epoch 的 Checkpoint (保留歷史紀錄)
        # 使用 :04d 補零，方便排序，例如 checkpoint_epoch_0001.pth
        if epoch % self.cfg.save_freq == 0:
            epoch_path = self.ckpt_dir / f"checkpoint_epoch_{epoch:04d}.pth"
            torch.save(state, epoch_path)

        # 3. 如果是最佳模型，額外儲存一份
        if is_best:
            best_path = self.ckpt_dir / "checkpoint_best.pth"
            torch.save(state, best_path)
            print(f"      [Chkpt] Best Model Saved (Val Loss: {loss:.4f})")
            # 使用 dim 顯示較不重要的訊息，避免洗版
            console.print(f"      [dim]Saved epoch {epoch} & Best model (Val Loss: {loss:.4f})[/dim]")

    def generate_report(self):
        """使用 Plotly 生成 HTML 訓練報告。"""
        if not self.logs:
            return

        df = pd.DataFrame(self.logs)

        fig = go.Figure()

        # 繪製 Training Loss
        fig.add_trace(
            go.Scattergl(
                x=df["epoch"],
                y=df["train_loss"],
                mode="lines+markers",
                name="Train Loss",
                line=dict(color="blue"),
            )
        )

        # 繪製 Validation Loss
        fig.add_trace(
            go.Scattergl(
                x=df["epoch"],
                y=df["val_loss"],
                mode="lines+markers",
                name="Val Loss",
                line=dict(color="orange"),
            )
        )

        fig.update_layout(
            title=f"SimSiam Training Loss - {self.cfg.exp_name}",
            xaxis_title="Epoch",
            yaxis_title="Negative Cosine Similarity Loss",
            template="plotly_white",
            hovermode="x unified",
        )

        report_path = self.save_dir / "training_report.html"
        fig.write_html(report_path, include_plotlyjs="cdn")
        print(f"[Info] 訓練報告已生成: {report_path}")

        # --- 生成特徵標準差監控報告 (Feature Standard Deviation) ---
        if "train_z_std" in df.columns:
            fig_std = go.Figure()
            fig_std.add_trace(go.Scattergl(x=df["epoch"], y=df["train_z_std"], mode="lines+markers", name="Train Std", line=dict(color="blue")))
            if "val_z_std" in df.columns and "val_z_std" in df.columns[df["val_z_std"].notnull()]:
                fig_std.add_trace(go.Scattergl(x=df["epoch"], y=df["val_z_std"], mode="lines+markers", name="Val Std", line=dict(color="orange")))
            
            fig_std.update_layout(
                title=f"Feature Standard Deviation (z) - {self.cfg.exp_name}",
                xaxis_title="Epoch",
                yaxis_title="Standard Deviation (Target ~ 1/sqrt(d))",
                template="plotly_white",
                hovermode="x unified",
            )
            report_std_path = self.save_dir / "training_report_std.html"
            fig_std.write_html(report_std_path, include_plotlyjs="cdn")
            print(f"[Info] 特徵標準差報告已生成: {report_std_path}")


# -----------------------------------------------------------------------------
# 3. 訓練流程封裝 (Training Session)
# -----------------------------------------------------------------------------

@monitor_resources
def run_single_session(cfg: Config, run_name: str, train_path: Path, val_path: Path):
    """執行單個 Run 的完整訓練流程"""
    
    # 針對此 Run 建立獨立的 Manager，修改 exp_name 以區分目錄
    current_cfg = Config(**cfg.to_dict()) # 複製一份 Config
    current_cfg.exp_name = f"{cfg.exp_name}_{run_name}" # e.g. simsiam_exp_01_Run_01_Seed_42
    
    manager = ExperimentManager(current_cfg)

    console.print(Panel(
        f"[bold blue]啟動訓練任務: {run_name}[/bold blue]\n"
        f"Train Path: [dim]{train_path}[/dim]\n"
        f"Output Dir: [dim]{manager.save_dir}[/dim]",
        title="Session Start", border_style="blue", box=box.ROUNDED
    ))

    # --- 1. 準備資料 ---
    def prepare_dataloaders():
        with console.status(f"[bold green]正在載入資料: {run_name}...[/bold green]", spinner="dots"):
            def scan_images(folder: Path) -> List[Path]:
                if not folder.exists(): return []
                return [p for p in folder.rglob('*') if p.suffix.lower() in current_cfg.img_exts]

            train_files = scan_images(train_path)
            val_files = scan_images(val_path)
            
            if not train_files:
                raise FileNotFoundError(f"路徑 {escape(str(train_path))} 無圖片")

            # 建立 Transform 與 Loader
            norm_mean, norm_std = [0.5], [0.5]
            train_transform = aug.EngineeringDrawingAugmentation(img_size=current_cfg.img_size, mean=norm_mean, std=norm_std)
            val_transform = aug.EngineeringDrawingAugmentation(img_size=current_cfg.img_size, mean=norm_mean, std=norm_std)
            
            train_ds = ss.UnlabeledImages(train_files, transform=train_transform, grayscale=(current_cfg.in_channels==1))
            val_ds = ss.UnlabeledImages(val_files, transform=val_transform, grayscale=(current_cfg.in_channels==1))
            
            t_dl = torch.utils.data.DataLoader(
                train_ds, batch_size=current_cfg.batch_size, shuffle=True, 
                drop_last=True, num_workers=current_cfg.num_workers, pin_memory=True
            )
            v_dl = torch.utils.data.DataLoader(
                val_ds, batch_size=current_cfg.batch_size, shuffle=False, 
                drop_last=False, num_workers=current_cfg.num_workers, pin_memory=True
            )
            return t_dl, v_dl, len(train_files), len(val_files)

    train_dl, val_dl, n_train, n_val = prepare_dataloaders()
    console.print(f"   ✅ 資料準備完成: Train={n_train}, Val={n_val}")

    # --- 2. 建立模型 ---
    # 每個 Run 都重新初始化模型 (確保獨立性)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model = ss.SimSiam(
        backbone=current_cfg.backbone, 
        pretrained=current_cfg.pretrained, 
        in_channels=current_cfg.in_channels
    ).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"[bold yellow]可訓練參數數量: {trainable_params}[/bold yellow]")
    if trainable_params == 0:
        raise ValueError("模型中沒有任何參數需要更新，請檢查 backbone 是否被凍結。")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=current_cfg.lr, weight_decay=current_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=current_cfg.epochs)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    # --- 3. 開始訓練 ---
    best_val_loss = float('inf')
    
    # 定義 Progress Bar (針對此 Session)
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn("{task.fields[info]}"),
    )

    with progress:
        task_id = progress.add_task(f"[cyan]Run: {run_name}", total=current_cfg.epochs, info="Init...")
        
        for epoch in range(1, current_cfg.epochs + 1):
            epoch_start = time.time()
            
            train_loss, train_std = ss.train_one_epoch(model, train_dl, optimizer, scaler, device)
            val_loss, val_std = ss.evaluate(model, val_dl, device) if len(val_dl) > 0 else (0.0, 0.0)
            
            epoch_dur = time.time() - epoch_start
            current_lr = optimizer.param_groups[0]['lr']
            
            info_text = f"[b]L:[/b]{train_loss:.4f}|[b]V:[/b]{val_loss:.4f}|[b]Std:[/b]{train_std:.4f}"
            progress.update(task_id, advance=1, info=info_text)
            
            manager.log_epoch(epoch, train_loss, val_loss, train_std, val_std, current_lr, epoch_dur)
            
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            manager.save_checkpoint(model, optimizer, epoch, val_loss, is_best)
            scheduler.step()

    # 產生報告
    manager.generate_report()
    
    # 回傳該次運行的摘要，供 Main 彙整
    return {
        "Run": run_name,
        "Best Val Loss": best_val_loss,
        "Log Dir": str(manager.save_dir)
    }

# -----------------------------------------------------------------------------
# 4. 主控程式 (Main Entry)
# -----------------------------------------------------------------------------

def main():
    cfg = Config()
    
    # Windows 防呆
    if os.name == "nt" and cfg.num_workers > 0:
        console.print("[yellow]⚠️  Windows 系統建議將 num_workers 設為 0[/yellow]")

    torch.manual_seed(cfg.seed)
    
    # 1. 掃描 dataset root 下的所有 Run 資料夾
    root_path = Path(cfg.dataset_root)
    if not root_path.exists():
        console.print(f"[bold red]❌ 資料集根目錄不存在: {root_path}[/bold red]")
        return

    # 尋找所有名稱包含 "Run_" 的資料夾並排序
    run_folders = sorted([
        p for p in root_path.iterdir() 
        if p.is_dir() and "Run_" in p.name
    ])

    if not run_folders:
        console.print(f"[bold red]❌ 在 {root_path} 下找不到任何 'Run_*' 資料夾[/bold red]")
        return

    console.print(Panel(
        f"[bold green]已發現 {len(run_folders)} 個訓練任務[/bold green]\n" + 
        "\n".join([f"- {f.name}" for f in run_folders]),
        title="批次排程列表", border_style="green"
    ))

    # 2. 依序執行訓練
    results = []
    total_start = time.time()

    for folder in run_folders:
        train_p = folder / cfg.train_subpath
        val_p = folder / cfg.val_subpath
        
        # 執行單次訓練
        try:
            res = run_single_session(cfg, folder.name, train_p, val_p)
            results.append(res)
        except Exception as e:
            console.print(f"[bold red]❌ 任務 {folder.name} 失敗: {e}[/bold red]")
            results.append({"Run": folder.name, "Best Val Loss": "Failed", "Log Dir": str(e)})

    # 3. 整體總結
    total_time = time.time() - total_start
    
    table = Table(title="批次訓練總結報告", box=box.HEAVY)
    table.add_column("Run Name", style="cyan")
    table.add_column("Best Val Loss", style="green")
    table.add_column("Log Directory", style="dim")

    for r in results:
        loss_display = f"{r['Best Val Loss']:.4f}" if isinstance(r['Best Val Loss'], float) else str(r['Best Val Loss'])
        table.add_row(r['Run'], loss_display, r['Log Dir'])

    console.print("\n")
    console.print(table)
    console.print(f"[bold]總執行時間:[/bold] {total_time/60:.1f} min")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]⚠️  使用者中斷訓練。[/bold yellow]")
    except Exception as e:
        console.print_exception(show_locals=False)