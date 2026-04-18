"""SimSiam 一站式訓練管線 (End-to-End Training Pipeline)。

============================================================
本腳本整合了從原始資料（PDF/ZIP）到 SimSiam 自監督模型訓練完成的
完整管線，包含以下階段：

Phase 1 — 資料準備（只執行一次）：
    Step 1.0: ZIP 解壓縮（若 PDF 資料夾不存在）
    Step 1.1: PDF → 影像（呼叫 pdf_to_image2.run）
    Step 1.2: 影像前處理（呼叫批次連通元件分析）

Phase 2 — 多 Run 訓練迴圈：
    for each Run (seed_i):
        Step 2.1: 資料集分割（train/val）
        Step 2.2: 建立 DataLoader
        Step 2.3: 建立模型與優化器（每 Run 重新初始化）
        Step 2.4: 訓練迴圈（epoch loop，計時暫停 I/O）
        Step 2.5: 儲存結果與報表

Phase 3 — 收尾：
    Step 3.1: 彙整所有 Run 結果
    Step 3.2: 輸出計時明細與實驗紀錄

核心特色：
    - 嚴謹的分函式 / 整體計時（支援暫停非核心 I/O）
    - 完整的實驗紀錄（Metadata、逐 epoch CSV、Plotly HTML 報表）
    - 模組化架構，各階段可獨立跳過
    - 繁體中文 Google Style Docstrings

使用方式：
    uv run python main_training.py

References:
    Chen & He. "Exploring Simple Siamese Representation Learning". CVPR 2021.
============================================================
"""

from __future__ import annotations

import os
import random
import shutil
import sys
import subprocess
import time
import zipfile
from pathlib import Path

import torch
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.traceback import install as install_rich_traceback

# --- 安裝 Rich 美化 Traceback ---
install_rich_traceback(show_locals=False)

# --- 確保 src 目錄在 Python Path 中 ---
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- 專案內部模組 ---
from src.training.config import TrainingConfig
from src.training.experiment_logger import ExperimentLogger
from src.training.timer import PrecisionTimer, TimerCollection

# --- 全域 Console ---
console = Console()


# ============================================================
# Phase 1: 資料準備 (Data Preparation) — 只執行一次
# ============================================================


def _step_1_0_extract_zip(cfg: TrainingConfig, timers: TimerCollection) -> None:
    """Step 1.0: ZIP 解壓縮。

    若 raw_pdf_dir 已存在且包含檔案，則自動跳過。
    否則嘗試從 raw_zip_path 解壓縮至 raw_pdf_dir。

    設計考量：
        解壓縮為 I/O 密集操作，計時器會忠實記錄其耗時。
        若 ZIP 不存在且 PDF 資料夾也不存在，則拋出錯誤。

    Args:
        cfg: 訓練設定。
        timers: 計時器集合。

    Raises:
        FileNotFoundError: 當 ZIP 與 PDF 資料夾皆不存在時。
    """
    timer = timers.create("step_1_0_zip_extraction")

    pdf_dir = Path(cfg.raw_pdf_dir)

    # --- 自動跳過判斷 ---
    if pdf_dir.exists() and any(pdf_dir.rglob("*.pdf")):
        console.print(
            f"  [dim]⏭ 跳過 ZIP 解壓縮：PDF 資料夾已存在 ({pdf_dir})[/dim]"
        )
        return

    if cfg.skip_zip_extraction:
        console.print("  [dim]⏭ 跳過 ZIP 解壓縮（skip_zip_extraction=True）[/dim]")
        return

    if cfg.raw_zip_path is None or not Path(cfg.raw_zip_path).is_file():
        raise FileNotFoundError(
            f"PDF 資料夾 ({pdf_dir}) 不存在，"
            f"且找不到 ZIP 檔案 ({cfg.raw_zip_path})。"
            "請提供有效的 raw_zip_path 或已存在的 raw_pdf_dir。"
        )

    timer.start()
    console.print(f"  📦 正在解壓縮: {cfg.raw_zip_path} → {cfg.raw_pdf_dir}")

    zip_path = Path(cfg.raw_zip_path)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    if zip_path.suffix.lower() == ".rar":
        console.print(f"  📦 [bold yellow]偵測到 RAR 格式[/bold yellow]: 呼叫 7z 並開啟多執行緒參數 (-mmt=on) 加速解壓縮...")
        # 解壓縮瓶頸優化：
        # 1. -mmt=on: 要求 7z 開啟所有 CPU 核心 (視 RAR 是否為固實壓縮而定)
        # 2. stdout 導向 DEVNULL: RAR 壓縮檔常有幾萬個檔案，關閉 stdout 收集可大幅節省記憶體並加速 I/O
        try:
            result = subprocess.run(
                ["7z", "x", str(zip_path), f"-o{pdf_dir}", "-y", "-mmt=on"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"7z 解壓縮失敗 (Error Code {e.returncode}):\n{e.stderr}")
        except FileNotFoundError:
            raise RuntimeError("系統找不到 '7z' 指令，請確保已安裝 p7zip 或相關工具。")
    else:
        # 加速 ZIP 解壓縮：使用 Python ThreadPoolExecutor 進行分塊平行解壓縮
        # (zlib 解壓縮底層為 C 且會釋放 GIL，因此使用 ThreadPool 即可獲得極佳的平行加速比)
        console.print("  📦 [bold cyan]偵測到 ZIP 格式[/bold cyan]: 使用多核心平行加速解壓縮...")
        import concurrent.futures
        
        with zipfile.ZipFile(zip_path, "r") as z_main:
            members = z_main.namelist()
            
        if not members:
            console.print("  [yellow]⚠ ZIP 檔案為空[/yellow]")
            return
            
        # 決定執行緒數量與分塊大小
        max_workers = min(32, (os.cpu_count() or 4) * 2)  # I/O 密集，可多開一些執行緒
        chunk_size = max(1, len(members) // (max_workers * 2))
        chunks = [members[i:i + chunk_size] for i in range(0, len(members), chunk_size)]
        
        def _extract_chunk(chunk_members: list[str]) -> None:
            # 每個 Thread 獨立開啟 ZipFile，避免多執行緒針對相同檔案指標產生競爭 (Race Condition)
            # 在迴圈中處理 chunk 能大幅攤銷重複讀取 ZIP central directory 的開銷
            with zipfile.ZipFile(zip_path, "r") as z_in:
                for member in chunk_members:
                    z_in.extract(member, pdf_dir)

        # 啟動平行處理
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 將 map 轉換為 list 確保所有任務都執行完畢並捕捉可能的 Exception
            list(executor.map(_extract_chunk, chunks))

    elapsed = timer.stop()
    console.print(
        f"  [green]✔[/green] ZIP 解壓縮完成 ({elapsed:.2f}s)"
    )


def _pdf_worker(pdf_path_str: str, root_dir_str: str, out_dir_str: str, dpi: int) -> dict:
    """PDF 轉影像工作常式（供 multiprocessing 使用）。"""
    import fitz
    from pathlib import Path
    
    pdf_path = Path(pdf_path_str)
    root_dir = Path(root_dir_str)
    out_dir = Path(out_dir_str)
    
    scale = dpi / 72.0
    
    try:
        class_label = pdf_path.parent.name
        rel_dir = pdf_path.parent.relative_to(root_dir)
        pdf_stem = pdf_path.stem
        # 替換空格以確保檔名相容性
        fname = f"{pdf_stem.replace(' ', '_')}.png"
        rel_path = rel_dir / fname
        dest_abs = out_dir / rel_path
        
        dest_abs.parent.mkdir(parents=True, exist_ok=True)
        
        with fitz.open(pdf_path) as doc:
            page = doc.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            pix.save(dest_abs.as_posix())
            
        return {
            "status": "success",
            "source_pdf": str(pdf_path),
            "class_label": class_label,
            "image_path": rel_path.as_posix()
        }
    except Exception as e:
        return {
            "status": "error",
            "file": str(pdf_path),
            "error": f"{type(e).__name__}: {e}"
        }


def _step_1_1_pdf_to_image(cfg: TrainingConfig, timers: TimerCollection) -> None:
    """Step 1.1: PDF → 影像轉換。

    使用多行程 (ProcessPoolExecutor) 解析 PDF 為圖片，加速 CPU 密集操作。
    若 converted_image_dir 已存在且包含影像，則自動跳過。

    Args:
        cfg: 訓練設定。
        timers: 計時器集合。

    Raises:
        FileNotFoundError: 當 raw_pdf_dir 不存在時。
    """
    timer = timers.create("step_1_1_pdf_to_image")

    img_dir = Path(cfg.converted_image_dir)

    # --- 自動跳過判斷 ---
    if img_dir.exists() and any(img_dir.rglob("*.png")):
        console.print(
            f"  [dim]⏭ 跳過 PDF 轉換：影像目錄已存在 ({img_dir})[/dim]"
        )
        return

    if cfg.skip_pdf_conversion:
        console.print("  [dim]⏭ 跳過 PDF 轉換（skip_pdf_conversion=True）[/dim]")
        return

    timer.start()
    console.print(
        f"  🖼 正在轉換 PDF → Image "
        f"(DPI={cfg.pdf_dpi}, workers={cfg.pdf_max_workers})"
    )

    import concurrent.futures
    import pandas as pd

    pdf_dir = Path(cfg.raw_pdf_dir)
    pdf_files = list(pdf_dir.rglob("*.pdf"))

    if not pdf_files:
        console.print("  [yellow]⚠ 未發現任何 PDF 檔案[/yellow]")
        return
        
    img_dir.mkdir(parents=True, exist_ok=True)
    
    max_workers = cfg.pdf_max_workers or max(1, (os.cpu_count() or 4) - 1)
    rows_out = []
    failed_count = 0
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_pdf_worker, str(p), str(pdf_dir), str(img_dir), cfg.pdf_dpi)
            for p in pdf_files
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[cyan]轉換 PDF..."),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task_id = progress.add_task("PDF 轉換", total=len(futures))
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res["status"] == "success":
                    rows_out.append((res["source_pdf"], res["class_label"], res["image_path"]))
                else:
                    failed_count += 1
                progress.update(task_id, advance=1)

    df_manifest = pd.DataFrame(rows_out, columns=["source_pdf", "class_label", "image_path"])
    manifest_path = img_dir / "manifest.csv"
    df_manifest.to_csv(manifest_path, index=False)

    elapsed = timer.stop()
    n_images = len(df_manifest)
    if failed_count > 0:
        console.print(f"  [yellow]⚠ 有 {failed_count} 個 PDF 轉換失敗[/yellow]")
        
    console.print(
        f"  [green]✔[/green] PDF 轉換完成：成功 {n_images} 張影像 ({elapsed:.2f}s)"
    )

def _step_1_2_preprocess_images(
    cfg: TrainingConfig, timers: TimerCollection
) -> None:
    """Step 1.2: 影像前處理（批次連通元件分析）。

    呼叫 src.image_preprocessing_batch_multiprocess2.process_folder()
    對轉換後的影像進行連通元件分析、大小組件分離與合併。

    設計考量：
        前處理是 CPU 密集的多程序操作，通常耗時較長。
        此步驟產出的 large_components/ 子資料夾將作為後續訓練的資料來源。

    Args:
        cfg: 訓練設定。
        timers: 計時器集合。
    """
    timer = timers.create("step_1_2_image_preprocessing")

    preprocessed_dir = Path(cfg.preprocessed_image_dir)

    # --- 自動跳過判斷 ---
    if preprocessed_dir.exists() and any(preprocessed_dir.rglob("*.png")):
        console.print(
            f"  [dim]⏭ 跳過影像前處理：已處理目錄已存在 ({preprocessed_dir})[/dim]"
        )
        return

    if cfg.skip_preprocessing:
        console.print("  [dim]⏭ 跳過影像前處理（skip_preprocessing=True）[/dim]")
        return

    timer.start()
    console.print(
        f"  🔬 正在進行影像前處理 (workers={cfg.preprocess_max_workers})"
    )

    from src.image_preprocessing_batch_multiprocess2 import (
        BatchConfig,
        process_folder,
    )

    batch_cfg = BatchConfig(
        input_dir=cfg.converted_image_dir,
        output_root=cfg.preprocessed_image_dir,
        patterns=(".png", ".jpg", ".jpeg"),
        recursive=True,
        per_image_outdir="{stem}",
        skip_existing=True,
        max_workers=cfg.preprocess_max_workers,
        top_n=cfg.preprocess_top_n,
        remove_largest=cfg.preprocess_remove_largest,
        seed=None,
        padding=cfg.preprocess_padding,
        max_attempts=cfg.preprocess_max_attempts,
        random_count=cfg.preprocess_random_count,
        write_report_json=True,
    )
    report = process_folder(batch_cfg)

    elapsed = timer.stop()
    ok_count = report.get("ok", 0)
    fail_count = report.get("failed", 0)
    console.print(
        f"  [green]✔[/green] 影像前處理完成：成功={ok_count}, "
        f"失敗={fail_count} ({elapsed:.2f}s)"
    )


# ============================================================
# Phase 2: 多 Run 訓練迴圈 (Multi-Run Training Loop)
# ============================================================


def _step_2_1_split_dataset(
    cfg: TrainingConfig,
    run_idx: int,
    seed: int,
    timers: TimerCollection,
) -> tuple[Path, Path]:
    """Step 2.1: 資料集分割（train/val）。

    對前處理完成的影像資料進行分層隨機分割，
    每個 Run 使用不同的 seed 以確保統計獨立性。

    使用 src.split_dataset.RichDatasetSplitter 的核心邏輯，
    但此處直接呼叫其分割與複製功能。

    Args:
        cfg: 訓練設定。
        run_idx: Run 索引（0-indexed）。
        seed: 此 Run 的隨機種子。
        timers: 計時器集合。

    Returns:
        tuple[Path, Path]: (train_path, val_path) 二元組。
    """
    timer_name = f"step_2_1_split_run_{run_idx + 1:02d}"
    timer = timers.create(timer_name)

    run_name = f"Run_{run_idx + 1:02d}_Seed_{seed}"
    run_output_dir = Path(cfg.dataset_dir) / run_name

    train_path = run_output_dir / cfg.train_subpath
    val_path = run_output_dir / cfg.val_subpath

    # --- 自動跳過判斷：若 train 資料夾已存在且有檔案 ---
    if train_path.exists() and any(train_path.rglob("*.png")):
        console.print(
            f"    [dim]⏭ 跳過資料集分割：{run_name} 已存在[/dim]"
        )
        return train_path, val_path

    timer.start()
    console.print(f"    📂 正在分割資料集 (seed={seed})...")

    from src.split_dataset import RichDatasetSplitter

    splitter = RichDatasetSplitter(
        source_root=cfg.preprocessed_image_dir,
        output_root=cfg.dataset_dir,
        split_ratio=cfg.split_ratio,
    )

    # --- 掃描資料集結構（取得類別與工件映射） ---
    splitter.scan_dataset_structure()

    # --- 設定隨機種子並執行單次分割 ---
    random.seed(seed)
    current_output_dir = Path(cfg.dataset_dir) / run_name

    # 使用 splitter 的進度條包裝
    from rich.progress import Progress as RichProgress

    with RichProgress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        # 計算總檔案量以設定進度條
        total_files = sum(
            splitter._count_png_recursive(inst)
            for instances in splitter.structure_map.values()
            for inst in instances
        )
        file_task = progress.add_task(
            f"分割 {run_name}...", total=total_files
        )

        for class_name, instances in splitter.structure_map.items():
            instances_shuffled = instances[:]
            random.shuffle(instances_shuffled)

            split_idx = int(len(instances_shuffled) * cfg.split_ratio)
            train_instances = instances_shuffled[:split_idx]
            val_instances = instances_shuffled[split_idx:]

            splitter.copy_files_batch(
                train_instances, "train", current_output_dir, progress, file_task
            )
            splitter.copy_files_batch(
                val_instances, "val", current_output_dir, progress, file_task
            )

    elapsed = timer.stop()

    # 驗證分割結果
    n_train = sum(1 for _ in train_path.rglob("*.png")) if train_path.exists() else 0
    n_val = sum(1 for _ in val_path.rglob("*.png")) if val_path.exists() else 0

    console.print(
        f"    [green]✔[/green] 分割完成：Train={n_train}, Val={n_val} ({elapsed:.2f}s)"
    )
    return train_path, val_path


def _step_2_2_prepare_dataloaders(
    cfg: TrainingConfig,
    train_path: Path,
    val_path: Path,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, int, int]:
    """Step 2.2: 建立訓練與驗證 DataLoader。

    掃描指定路徑下的所有支援格式影像，建立 Dataset 與 DataLoader。
    使用 augmentations.py 中定義的工程圖專用增強策略。

    設計考量：
        - 訓練集使用 EngineeringDrawingAugmentation（隨機增強）
        - 驗證集也使用相同增強（因 SimSiam 需要雙視角，
          驗證時仍需產生兩個增強版本來計算 loss）
        - drop_last=True 避免最後一個不完整 batch 導致 BatchNorm 問題

    Args:
        cfg: 訓練設定。
        train_path: 訓練集影像目錄。
        val_path: 驗證集影像目錄。

    Returns:
        tuple: (train_loader, val_loader, n_train, n_val)。

    Raises:
        FileNotFoundError: 當訓練集路徑下無影像時。
    """
    from src.model.augmentations import EngineeringDrawingAugmentation
    from src.model.simsiam2 import UnlabeledImages

    def scan_images(folder: Path) -> list[Path]:
        """掃描資料夾下所有支援格式的影像檔案。"""
        if not folder.exists():
            return []
        return sorted(
            p for p in folder.rglob("*")
            if p.suffix.lower() in cfg.img_exts
        )

    train_files = scan_images(train_path)
    val_files = scan_images(val_path)

    if not train_files:
        raise FileNotFoundError(
            f"訓練集路徑 {train_path} 下無影像檔案。"
        )

    # --- 建立增強策略 ---
    # ⚠ 邊界條件：灰階圖片的 mean/std 為單通道值
    norm_mean = [0.5]
    norm_std = [0.5]

    train_transform = EngineeringDrawingAugmentation(
        img_size=cfg.img_size, mean=norm_mean, std=norm_std
    )
    val_transform = EngineeringDrawingAugmentation(
        img_size=cfg.img_size, mean=norm_mean, std=norm_std
    )

    # --- 建立 Dataset ---
    is_grayscale = cfg.in_channels == 1
    train_ds = UnlabeledImages(
        train_files, transform=train_transform, grayscale=is_grayscale
    )
    val_ds = UnlabeledImages(
        val_files, transform=val_transform, grayscale=is_grayscale
    )

    # --- 建立 DataLoader ---
    # 防呆：Windows 系統不支援多程序 DataLoader
    effective_workers = 0 if os.name == "nt" else cfg.num_workers

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,  # 避免不完整 batch 導致 BatchNorm 問題
        num_workers=effective_workers,
        pin_memory=True,
        persistent_workers=effective_workers > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=effective_workers,
        pin_memory=True,
        persistent_workers=effective_workers > 0,
    )

    return train_loader, val_loader, len(train_files), len(val_files)


def _step_2_3_build_model(
    cfg: TrainingConfig,
    device: str,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, object, object]:
    """Step 2.3: 建立 SimSiam 模型、優化器與排程器。

    每個 Run 都重新初始化模型與優化器，確保訓練的完全獨立性。

    設計考量：
        - 使用 AdamW 作為優化器（比原論文的 SGD 更穩定）
        - CosineAnnealingLR 學習率排程（smooth decay）
        - GradScaler 用於混合精度訓練（CUDA 環境下自動啟用）

    Args:
        cfg: 訓練設定。
        device: 訓練裝置 ('cuda' 或 'cpu')。

    Returns:
        tuple: (model, optimizer, scheduler, scaler)。

    Raises:
        ValueError: 當模型無可訓練參數時。
    """
    from src.model.simsiam2 import SimSiam

    model = SimSiam(
        backbone=cfg.backbone,
        proj_dim=cfg.proj_dim,
        pred_hidden=cfg.pred_hidden,
        pretrained=cfg.pretrained,
        in_channels=cfg.in_channels,
    ).to(device)

    # --- 驗證可訓練參數 ---
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    if trainable_params == 0:
        raise ValueError(
            "模型中沒有任何可訓練參數，請檢查 backbone 是否被意外凍結。"
        )
    console.print(
        f"    📊 可訓練參數: {trainable_params:,}"
    )

    # --- 優化器 ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # --- 學習率排程 ---
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs
    )

    # --- 混合精度 ---
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    return model, optimizer, scheduler, scaler


def _step_2_4_training_loop(
    cfg: TrainingConfig,
    run_name: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    scaler: object,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: str,
    run_logger: object,
    timers: TimerCollection,
    run_idx: int,
) -> float:
    """Step 2.4: 執行訓練迴圈。

    此函式包含完整的 epoch 迴圈，每個 epoch 包含：
    1. 訓練一個 epoch (train_one_epoch)
    2. 驗證 (evaluate)
    3. 暫停計時 → 儲存 checkpoint → 恢復計時
    4. 記錄日誌

    計時策略：
        - epoch_timer: 記錄每個 epoch 的淨計算時間（排除 checkpoint I/O）
        - 在 save_checkpoint 前後呼叫 pause/resume 確保計時準確

    Args:
        cfg: 訓練設定。
        run_name: Run 的識別名稱。
        model: SimSiam 模型。
        optimizer: 優化器。
        scheduler: 學習率排程器。
        scaler: GradScaler。
        train_loader: 訓練 DataLoader。
        val_loader: 驗證 DataLoader。
        device: 訓練裝置。
        run_logger: Run 日誌管理器。
        timers: 計時器集合。
        run_idx: Run 索引。

    Returns:
        float: 該 Run 的最佳驗證 loss。
    """
    from src.model.simsiam2 import evaluate, train_one_epoch

    best_val_loss = float("inf")
    run_timer_name = f"step_2_4_train_run_{run_idx + 1:02d}"
    run_timer = timers.create(run_timer_name)
    run_timer.start()

    # --- Rich 進度條 ---
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn("{task.fields[info]}"),
        console=console,
    ) as progress:
        task_id = progress.add_task(
            f"[cyan]Run: {run_name}",
            total=cfg.epochs,
            info="Init...",
        )

        for epoch in range(1, cfg.epochs + 1):
            # --- Epoch 計時器 ---
            epoch_timer = PrecisionTimer(f"epoch_{epoch}")
            epoch_timer.start()
            epoch_wall_start = time.perf_counter()

            # --- 訓練 ---
            train_loss, train_std = train_one_epoch(
                model, train_loader, optimizer, scaler, device
            )

            # --- 驗證 ---
            val_loss, val_std = (
                evaluate(model, val_loader, device)
                if len(val_loader) > 0
                else (0.0, 0.0)
            )

            # --- ⚠ 暫停計時：以下為非核心 I/O 操作 ---
            epoch_timer.pause()

            # 判斷是否為最佳模型
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            # 儲存 Checkpoint（計時已暫停）
            run_logger.save_checkpoint(
                model, optimizer, epoch, val_loss, is_best
            )

            # --- 恢復計時 ---
            epoch_timer.resume()

            # 更新學習率
            scheduler.step()

            # 停止 epoch 計時
            epoch_net = epoch_timer.stop()
            epoch_wall = time.perf_counter() - epoch_wall_start

            # --- 記錄日誌 ---
            current_lr = optimizer.param_groups[0]["lr"]
            run_logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_z_std=train_std,
                val_z_std=val_std,
                lr=current_lr,
                epoch_net_duration=epoch_net,
                epoch_wall_duration=epoch_wall,
            )

            # --- 更新進度條 ---
            info_text = (
                f"[b]L:[/b]{train_loss:.4f}|"
                f"[b]V:[/b]{val_loss:.4f}|"
                f"[b]Std:[/b]{train_std:.4f}|"
                f"[b]Net:[/b]{epoch_net:.1f}s"
            )
            progress.update(task_id, advance=1, info=info_text)

    run_timer.stop()
    return best_val_loss


def _run_single_training(
    cfg: TrainingConfig,
    run_idx: int,
    seed: int,
    experiment_logger: ExperimentLogger,
    timers: TimerCollection,
) -> dict:
    """執行單個 Run 的完整訓練流程。

    包含 Step 2.1 ~ 2.5 的完整序列：
    資料集分割 → DataLoader → 模型建立 → 訓練 → 報表。

    Args:
        cfg: 訓練設定。
        run_idx: Run 索引（0-indexed）。
        seed: 此 Run 的隨機種子。
        experiment_logger: 實驗紀錄管理器。
        timers: 計時器集合。

    Returns:
        dict: 該 Run 的結果摘要。
    """
    run_name = f"Run_{run_idx + 1:02d}_Seed_{seed}"
    overall_run_timer = timers.create(f"run_{run_idx + 1:02d}_total")
    overall_run_timer.start()

    console.print(
        Panel(
            f"[bold blue]🚀 啟動訓練任務: {run_name}[/bold blue]",
            title=f"Run {run_idx + 1}/{cfg.n_runs}",
            border_style="blue",
            box=box.ROUNDED,
        )
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"    🖥 裝置: [bold]{device}[/bold]")

    try:
        # --- Step 2.1: 資料集分割 ---
        train_path, val_path = _step_2_1_split_dataset(
            cfg, run_idx, seed, timers
        )

        # --- Step 2.2: 建立 DataLoader ---
        dl_timer = timers.create(f"step_2_2_dataloader_run_{run_idx + 1:02d}")
        dl_timer.start()

        train_loader, val_loader, n_train, n_val = _step_2_2_prepare_dataloaders(
            cfg, train_path, val_path
        )
        dl_timer.stop()
        console.print(
            f"    [green]✔[/green] DataLoader 就緒："
            f"Train={n_train}, Val={n_val}"
        )

        # --- Step 2.3: 建立模型 ---
        # 每 Run 重新初始化，確保獨立性
        torch.manual_seed(seed)
        model, optimizer, scheduler, scaler = _step_2_3_build_model(
            cfg, device
        )

        # --- Step 2.4: 訓練迴圈 ---
        run_logger = experiment_logger.create_run_logger(run_name)
        best_val_loss = _step_2_4_training_loop(
            cfg=cfg,
            run_name=run_name,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            run_logger=run_logger,
            timers=timers,
            run_idx=run_idx,
        )

        # --- Step 2.5: 產生報表 ---
        report_timer = timers.create(f"step_2_5_report_run_{run_idx + 1:02d}")
        report_timer.start()
        run_logger.generate_report()
        report_timer.stop()

        overall_run_timer.stop()

        result = {
            "run_name": run_name,
            "seed": seed,
            "best_val_loss": best_val_loss,
            "n_train": n_train,
            "n_val": n_val,
            "status": "success",
            "log_dir": str(run_logger.run_dir),
        }

        console.print(
            f"    [green]✔[/green] {run_name} 完成："
            f"Best Val Loss = {best_val_loss:.4f}"
        )
        return result

    except Exception as e:
        overall_run_timer.stop()
        console.print(
            f"    [bold red]❌ {run_name} 失敗: {escape(str(e))}[/bold red]"
        )
        return {
            "run_name": run_name,
            "seed": seed,
            "best_val_loss": None,
            "status": "failed",
            "error": str(e),
        }


# ============================================================
# Phase 3: 收尾 (Finalization)
# ============================================================


def _print_final_summary(
    run_results: list[dict],
    timers: TimerCollection,
    experiment_logger: ExperimentLogger,
    total_elapsed: float,
) -> None:
    """Phase 3: 輸出最終總結報告。

    包含所有 Run 的結果彙整表格與計時明細表格。

    Args:
        run_results: 每個 Run 的結果字典列表。
        timers: 計時器集合。
        experiment_logger: 實驗紀錄管理器。
        total_elapsed: 整體牆鐘耗時（秒）。
    """
    console.print()
    console.rule("[bold green]🏆 訓練完成 — 總結報告")

    # --- Run 結果表格 ---
    result_table = Table(
        title="批次訓練結果", box=box.HEAVY, show_lines=True
    )
    result_table.add_column("Run", style="cyan", justify="center")
    result_table.add_column("Seed", style="magenta", justify="center")
    result_table.add_column("Best Val Loss", style="green", justify="right")
    result_table.add_column("Status", justify="center")
    result_table.add_column("Log Dir", style="dim", overflow="fold")

    for r in run_results:
        loss_str = (
            f"{r['best_val_loss']:.4f}"
            if isinstance(r.get("best_val_loss"), float)
            else str(r.get("best_val_loss", "N/A"))
        )
        status_str = (
            "[green]✔[/green]"
            if r.get("status") == "success"
            else f"[red]✗ {r.get('error', '')[: 50]}[/red]"
        )
        result_table.add_row(
            r.get("run_name", ""),
            str(r.get("seed", "")),
            loss_str,
            status_str,
            r.get("log_dir", ""),
        )

    console.print(result_table)

    # --- 計時明細表格 ---
    timing_table = Table(
        title="計時明細", box=box.ROUNDED, show_lines=True
    )
    timing_table.add_column("步驟", style="cyan")
    timing_table.add_column("淨耗時", style="green", justify="right")
    timing_table.add_column("牆鐘耗時", style="yellow", justify="right")
    timing_table.add_column("暫停次數", style="dim", justify="right")

    for row in timers.summary_table_rows():
        timing_table.add_row(*row)

    console.print(timing_table)

    # --- 儲存報告 ---
    timing_path = experiment_logger.save_timing_report()
    summary_path = experiment_logger.save_overall_summary(run_results)

    console.print(
        f"\n[bold]📁 實驗目錄:[/bold] {experiment_logger.experiment_dir}"
    )
    console.print(f"[bold]⏱ 總執行時間:[/bold] {total_elapsed / 60:.1f} min")
    console.print(f"[dim]計時報告: {timing_path}[/dim]")
    console.print(f"[dim]總結報告: {summary_path}[/dim]")


# ============================================================
# 主程式入口 (Main Entry Point)
# ============================================================


def main() -> None:
    """SimSiam 訓練管線的主入口函式。

    完整流程：
    1. Phase 1: 資料準備（ZIP→PDF→Image→前處理，只執行一次）
    2. Phase 2: 多 Run 訓練（每 Run 獨立分割→訓練→儲存）
    3. Phase 3: 收尾（彙整結果→計時報告→實驗紀錄）

    所有設定集中在 TrainingConfig 中修改。
    """
    # ========================================================
    # 設定區（使用者可在此修改參數）
    # ========================================================
    cfg = TrainingConfig(
        # --- 資料來源 ---
        raw_zip_path="data/吉輔提供資料.zip",
        raw_pdf_dir="data/吉輔提供資料",
        converted_image_dir="data/engineering_images_100dpi",
        preprocessed_image_dir="data/preprocessed_images_100dpi",
        dataset_dir="dataset",
        # --- PDF 轉換 ---
        pdf_dpi=100,
        pdf_max_workers=16,
        # --- 影像前處理 ---
        preprocess_top_n=5,
        preprocess_remove_largest=True,
        preprocess_max_workers=12,
        preprocess_random_count=10,
        # --- 資料集分割 ---
        split_ratio=0.8,
        n_runs=5,
        base_seed=42,
        # --- 模型 ---
        backbone="resnet18",
        pretrained=True,
        in_channels=1,
        # --- 訓練 ---
        img_size=512,
        epochs=200,
        batch_size=64,
        lr=2e-5,
        weight_decay=1e-5,
        num_workers=16,
        # --- 輸出 ---
        output_dir="outputs",
        exp_name="simsiam_exp",
        save_freq=10,
        # --- 階段控制 ---
        skip_zip_extraction=True,
        skip_pdf_conversion=False,
        skip_preprocessing=False,
    )

    # --- 驗證設定 ---
    cfg.validate()

    # --- 初始化計時與紀錄 ---
    timers = TimerCollection()
    total_timer = timers.create("total_pipeline")
    total_timer.start()

    experiment_logger = ExperimentLogger(config=cfg, timers=timers)

    # ========================================================
    # Phase 1: 資料準備（只執行一次）
    # ========================================================
    console.rule("[bold cyan]Phase 1: 資料準備")

    phase1_timer = timers.create("phase_1_data_preparation")
    phase1_timer.start()

    _step_1_0_extract_zip(cfg, timers)
    _step_1_1_pdf_to_image(cfg, timers)
    _step_1_2_preprocess_images(cfg, timers)

    phase1_timer.stop()
    console.print(
        f"[green]✔[/green] Phase 1 完成 "
        f"({phase1_timer.elapsed:.2f}s)\n"
    )

    # ========================================================
    # Phase 2: 多 Run 訓練迴圈
    # ========================================================
    console.rule("[bold cyan]Phase 2: 多 Run 訓練")
    console.print(
        f"  排程: {cfg.n_runs} 個 Run, "
        f"base_seed={cfg.base_seed}\n"
    )

    phase2_timer = timers.create("phase_2_multi_run_training")
    phase2_timer.start()

    run_results: list[dict] = []
    for run_idx in range(cfg.n_runs):
        seed = cfg.base_seed + run_idx
        result = _run_single_training(
            cfg=cfg,
            run_idx=run_idx,
            seed=seed,
            experiment_logger=experiment_logger,
            timers=timers,
        )
        run_results.append(result)

    phase2_timer.stop()

    # ========================================================
    # Phase 3: 收尾
    # ========================================================
    total_elapsed = total_timer.stop()

    _print_final_summary(
        run_results=run_results,
        timers=timers,
        experiment_logger=experiment_logger,
        total_elapsed=total_elapsed,
    )


# ============================================================
# 腳本入口
# ============================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print(
            "\n[bold yellow]⚠ 使用者中斷訓練。已完成的紀錄已保存。[/bold yellow]"
        )
    except Exception:
        console.print_exception(show_locals=False)
