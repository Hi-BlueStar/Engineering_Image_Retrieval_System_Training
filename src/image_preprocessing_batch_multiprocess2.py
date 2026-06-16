# batch_runner for image_preprocessing2
from __future__ import annotations

import os
# 強制所有底層數學與平行計算庫在單個子行程中以單執行緒運行，避免執行緒爆炸與 CPU 爭搶死鎖
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import contextlib

# import ctypes
import functools  # 用於裝飾器
import gc  # 引入垃圾回收模組
import json
import multiprocessing as mp
import os

# import queue
import sqlite3  # noqa: F401
import sys
import threading
import time
import traceback
import tracemalloc
from dataclasses import dataclass

# from datetime import datetime
from pathlib import Path


try:
    import psutil  # 強烈建議安裝
except Exception:
    psutil = None

# GPU 監控（選用）
try:
    import pynvml

    _NVML_OK = True
except Exception:
    _NVML_OK = False

from rich import box

# 第三方：視覺處理模組在 image_preprocessing2 內部使用
# 這裡只需要 rich/psutil/pynvml（選用）
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.rule import Rule
from rich.table import Table


# 確保可直接在專案根目錄執行此檔時能找到同資料夾的模組
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

# 匯入新版主流程函式（隨機排列大元件版本）
from image_preprocessing3 import run_pipeline  # noqa: E402


# image_preprocessing2 主要參數的預設值
PIPE_DEFAULTS = {
    "TOP_N": 5,
    "REMOVE_LARGEST": True,
    "SEED": None,
    "PADDING": 2,
    "MAX_ATTEMPTS": 400,
    "RANDOM_COUNT": 10,
}


# ================================
# 設定與工具
# ================================
def _fmt_bytes(n: int | None) -> str:
    if n is None:
        return "N/A"
    units = ["B", "KB", "MB", "GB", "TB"]
    s = float(n)
    for u in units:
        if s < 1024.0:
            return f"{s:.2f} {u}"
        s /= 1024.0
    return f"{s:.2f} PB"


# ================================
# 硬體感知裝飾器 (Resource Gatekeeper)
# ================================
def resource_guard(
    ram_threshold: float = 95.0,
    check_interval: float = 1.0,
):
    """
    硬體資源守門員：
    在執行函式前檢查系統記憶體 (RAM)。
    若超過設定閾值，則暫停執行並等待資源釋放，避免 OOM 崩潰。
    此處理為 CPU 密集型，已移除無意義的 GPU 顯存檢查，避免受其他背景訓練程序干擾。
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 1. 檢查系統記憶體 (RAM)
            if psutil:
                warned = False
                while True:
                    mem = psutil.virtual_memory()
                    if mem.percent < ram_threshold:
                        break
                    # 資源吃緊，進入等待，輸出實體日誌避免無聲卡死
                    if not warned:
                        sys.stderr.write(
                            f"\n[WARN] 行程 {os.getpid()} 偵測到系統 RAM 負載過高 ({mem.percent}%)，子程序暫停等待中...\n"
                        )
                        sys.stderr.flush()
                        warned = True
                    time.sleep(check_interval)

            # 資源充足，放行
            return func(*args, **kwargs)

        return wrapper

    return decorator


@dataclass
class BatchConfig:
    # 掃描與輸出
    input_dir: str | Path
    output_root: str | Path = "results_batch2"
    patterns: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    recursive: bool = True
    per_image_outdir: str = "{stem}"  # 可用 {stem} / {name} / {parent}
    preserve_structure: bool = True  # 是否保留與輸入資料夾相同的子路徑
    skip_existing: bool = True  # 若輸出下已存在 {stem}_merged.* 就略過
    max_workers: int = 1  # >1 啟用多執行緒（I/O 為主）

    # image_preprocessing2 參數
    top_n: int = PIPE_DEFAULTS["TOP_N"]
    remove_largest: bool = PIPE_DEFAULTS["REMOVE_LARGEST"]
    seed: int | None = PIPE_DEFAULTS["SEED"]
    padding: int = PIPE_DEFAULTS["PADDING"]
    max_attempts: int = PIPE_DEFAULTS["MAX_ATTEMPTS"]
    random_count: int = PIPE_DEFAULTS["RANDOM_COUNT"]

    # 報表
    write_report_json: bool = True
    report_json_path: str | Path | None = None  # 預設 output_root / "batch_report.json"

    # 監控
    monitor_interval_sec: float = 0.5
    enable_gpu_monitor: bool = True


@dataclass
class FileStat:
    path: Path
    ok: bool
    duration_sec: float
    rss_delta: int | None = None
    py_peak_bytes: int | None = None
    error: str | None = None
    output_dir: Path | None = None
    saved_files: list[str] | None = None


# ================================
# 資源即時監控（背景執行緒）
# ================================
class ResourceMonitor:
    def __init__(
        self,
        progress: Progress,
        task_id: int,
        interval: float = 0.5,
        enable_gpu: bool = True,
    ):
        self.progress = progress
        self.task_id = task_id
        self.interval = max(0.2, float(interval))
        self.enable_gpu = enable_gpu and _NVML_OK
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

        if self.enable_gpu:
            try:
                pynvml.nvmlInit()
                self._gpu_handles = [
                    pynvml.nvmlDeviceGetHandleByIndex(i)
                    for i in range(pynvml.nvmlDeviceGetCount())
                ]
            except Exception:
                self.enable_gpu = False

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        if self.enable_gpu:
            with contextlib.suppress(Exception):
                pynvml.nvmlShutdown()

    def _run(self):
        proc = psutil.Process(os.getpid()) if psutil else None
        while not self._stop.is_set():
            # CPU / RAM / IO
            if psutil:
                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory().percent
                rss = proc.memory_info().rss if proc else None
                io = psutil.disk_io_counters()
                io_str = (
                    f"R:{_fmt_bytes(io.read_bytes)} W:{_fmt_bytes(io.write_bytes)}"
                    if io
                    else "IO:N/A"
                )
            else:
                cpu = ram = None
                rss = None
                io_str = "psutil 未安裝"

            # GPU
            gpu_str = "GPU:N/A"
            if self.enable_gpu:
                try:
                    parts = []
                    for idx, h in enumerate(self._gpu_handles):
                        util = pynvml.nvmlDeviceGetUtilizationRates(h)
                        meminfo = pynvml.nvmlDeviceGetMemoryInfo(h)
                        parts.append(
                            f"GPU{idx} {util.gpu:>3}% | {_fmt_bytes(meminfo.used)}/\
                                {_fmt_bytes(meminfo.total)}"
                        )
                    gpu_str = " | ".join(parts)
                except Exception:
                    gpu_str = "GPU:不可用"

            text = (
                f"[bold]系統資源[/] | "
                f"CPU: {cpu:.0f}% | RAM: {ram:.0f}% | RSS: {_fmt_bytes(rss)} |\
                    {io_str} | {gpu_str}"
                if psutil
                else "[bold]系統資源[/] | psutil 未安裝"
            )
            # 更新進度列的描述
            self.progress.update(self.task_id, description=text)
            time.sleep(self.interval)


# ================================
# 檔案掃描
# ================================
def _gather_images(cfg: BatchConfig) -> list[Path]:
    root = Path(cfg.input_dir)
    if not root.exists():
        raise FileNotFoundError(f"找不到資料夾：{root}")

    def is_image(p: Path) -> bool:
        return p.suffix.lower() in cfg.patterns

    if cfg.recursive:
        files = [p for p in root.rglob("*") if p.is_file() and is_image(p)]
    else:
        files = [p for p in root.iterdir() if p.is_file() and is_image(p)]
    files.sort()
    return files


def _compute_out_dir(cfg: BatchConfig, img_path: Path) -> Path:
    stem = img_path.stem
    name = img_path.name
    parent = img_path.parent.name
    sub = cfg.per_image_outdir.format(stem=stem, name=name, parent=parent)
    root = Path(cfg.output_root)

    # 依輸入資料夾結構建立對應子資料夾
    rel_parent = None
    if cfg.preserve_structure:
        try:
            rel_parent = img_path.parent.resolve().relative_to(
                Path(cfg.input_dir).resolve()
            )
        except Exception:
            rel_parent = img_path.parent.name

    out_dir = root
    if rel_parent:
        out_dir = out_dir / rel_parent
    if sub:
        out_dir = out_dir / sub
    return out_dir


def _already_done(out_dir: Path, stem: str, suffix: str) -> bool:
    # image_preprocessing2 會在輸出目錄下產生 {stem}_merged.<ext>
    merged_name = f"{stem}_merged{suffix if suffix else '.png'}"
    return (out_dir / merged_name).exists()


# ================================
# [新增/重構] 單一圖片處理函式 (Worker)
# 必須定義在 Top-level 以便 Multiprocessing 序列化
# ================================
def _process_image_task(args: tuple[Path, BatchConfig]) -> FileStat:
    """
    子程序執行的單一任務單元。
    """
    import cv2
    cv2.setNumThreads(0) # 停用 OpenCV 內部多執行緒以避免與 mp.Pool 衝突
    
    img_path, cfg = args

    # 在子程序中重新計算輸出路徑
    out_dir = _compute_out_dir(cfg, img_path)
    suffix = img_path.suffix or ".png"

    # 檢查是否略過
    if cfg.skip_existing and _already_done(out_dir, img_path.stem, suffix):
        return FileStat(
            path=img_path,
            ok=True,
            duration_sec=0.0,
            rss_delta=None,
            py_peak_bytes=None,
            error=None,
            output_dir=out_dir,
            saved_files=None,
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    # 準備 Resource Guard
    # 注意：子程序無法直接使用主程序的 console，這裡傳入 None 或簡單 print
    # 若需嚴格監控，可在此處實例化新的 Console 或僅依賴 Log
    @resource_guard(ram_threshold=95.0)
    def _protected_run(target_path: str, target_out_dir: str):
        return run_pipeline(
            input_path=target_path,
            output_dir=target_out_dir,
            top_n=cfg.top_n,
            remove_largest=cfg.remove_largest,
            seed=cfg.seed,
            padding=cfg.padding,
            max_attempts=cfg.max_attempts,
            random_count=cfg.random_count,
        )

    # 量測與執行
    proc = psutil.Process(os.getpid()) if psutil else None
    rss_before = proc.memory_info().rss if proc else None

    started_here = not tracemalloc.is_tracing()
    if started_here:
        tracemalloc.start()

    t0 = time.perf_counter()
    ok = True
    err_msg = None
    saved_files: list[str] = []

    try:
        # 執行管線
        outputs = _protected_run(str(img_path), str(out_dir))

        # 收集結果
        if isinstance(outputs, dict):
            for key in ("original", "merged"):
                p = outputs.get(key)
                if p:
                    saved_files.append(str(p))
            for p in (
                outputs.get("random", [])
                if isinstance(outputs.get("random"), list)
                else []
            ):
                saved_files.append(str(p))
            if outputs.get("large_dir"):
                saved_files.append(str(outputs["large_dir"]))

        # 簡單驗證
        merged_path = outputs.get("merged") if isinstance(outputs, dict) else None
        if merged_path and not Path(merged_path).exists():
            ok = False
            err_msg = f"處理完成但找不到輸出檔案：{merged_path}"

    except Exception as e:
        ok = False
        err_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"
    finally:
        duration = time.perf_counter() - t0
        current, peak = tracemalloc.get_traced_memory()
        if started_here:
            tracemalloc.stop()
        rss_after = proc.memory_info().rss if proc else None
        rss_delta = (
            (rss_after - rss_before)
            if (psutil and rss_before is not None and rss_after is not None)
            else None
        )

        # 已由 Pool(maxtasksperchild=50) 自動回收進程以釋放記憶體，故免除手動回收開銷
        # gc.collect()

    return FileStat(
        path=img_path,
        ok=ok,
        duration_sec=duration,
        rss_delta=rss_delta,
        py_peak_bytes=peak,
        error=err_msg,
        output_dir=out_dir,
        saved_files=saved_files or None,
    )


# ================================
# 核心：批次處理 (Multiprocessing 版)
# ================================
def process_folder(cfg: BatchConfig) -> dict:
    """
    對資料夾內所有圖片執行 run_pipeline (多程序平行運算)。
    """
    console = Console()
    out_root = Path(cfg.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    report_path = (
        Path(cfg.report_json_path)
        if cfg.report_json_path
        else out_root / "batch_report.json"
    )

    files = _gather_images(cfg)
    if not files:
        console.print(
            f"[yellow]資料夾內未找到圖片：{cfg.input_dir}（副檔名：{cfg.patterns}）[/]"
        )
        return {"count": 0, "ok": 0, "failed": 0, "items": []}

    # 決定並行數：若 cfg.max_workers 為 1 則不開 Pool，方便除錯；否則開 Pool
    # 注意：Image Processing 是 CPU-bound，通常設為 CPU 核心數或是核心數 - 1
    num_procs = min(len(files), max(1, cfg.max_workers))

    console.print(Rule("[bold cyan]批次處理開始 (Multiprocessing)[/]"))
    console.print(f"[b]來源[/]: {Path(cfg.input_dir).resolve()}")
    console.print(f"[b]輸出根目錄[/]: {out_root.resolve()}")
    console.print(
        f"[b]圖片數[/]: {len(files)}  | [b]程序數 (Processes)[/]: {num_procs}"
    )

    # Rich 進度條
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        expand=True,
    )

    results: list[FileStat] = []

    # 內部報告產生器
    def _build_report() -> dict:
        ok_cnt = sum(1 for r in results if r.ok)
        fail_cnt = len(results) - ok_cnt
        return {
            "count": len(results),
            "ok": ok_cnt,
            "failed": fail_cnt,
            "output_root": str(Path(cfg.output_root).resolve()),
            "items": [
                {
                    "path": str(r.path),
                    "ok": r.ok,
                    "duration_sec": r.duration_sec,
                    "rss_delta": r.rss_delta,
                    "py_peak_bytes": r.py_peak_bytes,
                    "error": r.error,
                    "output_dir": str(r.output_dir) if r.output_dir else None,
                    "saved_files": r.saved_files,
                }
                for r in results
            ],
            "params": {
                "top_n": cfg.top_n,
                "remove_largest": cfg.remove_largest,
                "seed": cfg.seed,
                "padding": cfg.padding,
                "max_attempts": cfg.max_attempts,
                "random_count": cfg.random_count,
                "patterns": cfg.patterns,
                "recursive": cfg.recursive,
                "skip_existing": cfg.skip_existing,
                "max_workers": cfg.max_workers,
                "preserve_structure": cfg.preserve_structure,
            },
        }

    def _maybe_write_report(force: bool = False) -> None:
        if not cfg.write_report_json or not results:
            return
        if not force and len(results) % 500 != 0:
            return
        report_path.parent.mkdir(parents=True, exist_ok=True)
        data = _build_report()
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        # 不要在迴圈中頻繁 print，以免洗版

    with progress:
        monitor = None
        task_args = [(p, cfg) for p in files]
        work_task = progress.add_task("[bold]處理圖片[/]", total=len(files))

        try:
            # 2. 建立 Process Pool (使用 spawn 以避免 fork 導致的執行緒死鎖問題，且不設 maxtasksperchild)
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=num_procs) as pool:
                # 3. 啟動資源監控 (此時 Pool 已啟動完成，可安全啟動監控執行緒)
                monitor_task = progress.add_task("系統資源監控中…", total=None)
                monitor = ResourceMonitor(
                    progress,
                    monitor_task,
                    interval=cfg.monitor_interval_sec,
                    enable_gpu=cfg.enable_gpu_monitor,
                )
                monitor.start()

                for res in pool.imap_unordered(
                    _process_image_task, task_args, chunksize=1
                ):
                    results.append(res)

                    # 更新 UI
                    if res.duration_sec == 0.0 and res.ok:
                        desc = f"[dim]略過: {res.path.name}[/]"
                    elif res.ok:
                        desc = f"完成: {res.path.name}"
                    else:
                        desc = f"[red]失敗[/]: {res.path.name}"

                    progress.update(work_task, advance=1, description=desc)
                    _maybe_write_report()

        except KeyboardInterrupt:
            console.print("[red]使用者中斷執行 (KeyboardInterrupt)[/]")
            if 'pool' in locals():
                pool.terminate()
            raise
        except Exception as e:
            console.print(f"[bold red]發生未預期錯誤: {e}[/]")
            traceback.print_exc()
        finally:
            if monitor is not None:
                monitor.stop()

    # 彙整與顯示最終結果
    console.print(Rule("[bold green]處理完成，彙整結果[/]"))
    report = _build_report()
    ok_cnt = report["ok"]
    fail_cnt = report["failed"]

    table = Table(title="批次摘要", box=box.SIMPLE_HEAVY)
    table.add_column("檔名", overflow="fold")
    table.add_column("狀態", justify="center")
    table.add_column("耗時", justify="right")
    table.add_column("ΔRSS (Subproc)", justify="right")
    table.add_column("PyPeak", justify="right")

    # 僅顯示最後 20 筆或全部，避免表格過長
    display_limit = 20
    display_results = (
        results if len(results) <= display_limit else results[-display_limit:]
    )

    for r in display_results:
        table.add_row(
            r.path.name,
            "[green]OK[/]" if r.ok else "[red]FAIL[/]",
            f"{r.duration_sec:.3f}s",
            _fmt_bytes(r.rss_delta),
            _fmt_bytes(r.py_peak_bytes),
        )

    if len(results) > display_limit:
        table.add_row("...", "...", "...", "...", "...")

    console.print(table)
    console.print(
        f"[b]總計[/]: {len(results)} | [green]成功[/]: {ok_cnt} |\
            [red]失敗[/]: {fail_cnt}"
    )

    # 強制寫入最終報表
    _maybe_write_report(force=True)
    console.print(f"[dim]完整報表已儲存至[/]: {report_path}")

    return report


def save_results_to_sqlite(
    report_data: dict, db_path: str | Path = "processing_history.db"
):
    """
    將批次處理的結果報告儲存至 SQLite 資料庫。

    Args:
        report_data (dict): 由 process_folder() 回傳的字典報告。
        db_path (str | Path): SQLite 資料庫檔案路徑。
    """
    db_path = Path(db_path)

    # 定義資料表 Schema
    # 包含執行時間、檔案資訊、處理狀態、效能數據以及使用的參數配置
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS processing_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_name TEXT,
        original_path TEXT,
        output_dir TEXT,
        status TEXT,           -- 'OK' or 'FAIL'
        duration_sec REAL,
        rss_delta INTEGER,     -- 記憶體變化量
        peak_memory INTEGER,   -- Python 峰值記憶體
        error_message TEXT,
        saved_files_json TEXT, -- 產出的檔案列表 (JSON 格式)
        config_json TEXT,      -- 當次執行的參數配置 (JSON 格式)
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    insert_sql = """
    INSERT INTO processing_logs (
        file_name, original_path, output_dir, status, duration_sec,
        rss_delta, peak_memory, error_message, saved_files_json, config_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    # 取得當次批次的設定參數 (序列化為 JSON 字串以便儲存)
    config_json = json.dumps(report_data.get("params", {}), ensure_ascii=False)

    items = report_data.get("items", [])
    if not items:
        return

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 1. 建立資料表
        cursor.execute(create_table_sql)

        # 2. 準備批次寫入的資料
        data_to_insert = []
        for item in items:
            # 處理路徑物件轉字串
            orig_path = str(item.get("path", ""))
            file_name = Path(orig_path).name
            out_dir = item.get("output_dir")
            out_dir_str = str(out_dir) if out_dir else None

            # 狀態轉換
            status = "OK" if item.get("ok") else "FAIL"

            # 序列化 saved_files 列表
            saved_files = item.get("saved_files")
            saved_files_str = (
                json.dumps(saved_files, ensure_ascii=False) if saved_files else None
            )

            data_to_insert.append(
                (
                    file_name,
                    orig_path,
                    out_dir_str,
                    status,
                    item.get("duration_sec", 0.0),
                    item.get("rss_delta"),
                    item.get("py_peak_bytes"),
                    item.get("error"),
                    saved_files_str,
                    config_json,
                )
            )

        # 3. 執行寫入
        cursor.executemany(insert_sql, data_to_insert)
        conn.commit()

        # 使用 Rich Console 顯示成功訊息 (若有的話)
        console = Console()
        console.print(
            f"[bold green]✓ 已將 {len(data_to_insert)} 筆處理紀錄儲存至資料庫：[/] \
                {db_path.resolve()}"
        )

    except sqlite3.Error as e:
        print(f"SQLite 資料庫錯誤: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()


# ================================
# 範例用法（非 CLI，直接以程式呼叫）
# ================================
def demo(
    input_dir: str = "./data/engineering_images_100dpi",
    output_root: str = "./results/batch2",
):
    cfg = BatchConfig(
        input_dir=input_dir,
        output_root=output_root,
        patterns=(".png", ".jpg", ".jpeg"),
        recursive=True,
        # 以檔名建立各自子資料夾，避免 large_components 混在一起
        per_image_outdir="{stem}",
        skip_existing=True,
        max_workers=20,  # 若 I/O 多可嘗試 >1
        top_n=5,
        remove_largest=True,
        seed=None,
        padding=2,
        max_attempts=400,
        random_count=20,
        write_report_json=True,
        report_json_path=None,  # None 則寫到 output_root/batch_report.json
        monitor_interval_sec=0.5,
        enable_gpu_monitor=True,
    )
    # 1. 執行批次處理並取得報告
    report = process_folder(cfg)

    # 2. [新增] 將結果存入 SQLite
    # 資料庫檔案會建立在 output_root 下，方便管理
    db_file = Path(output_root) / "batch_history.db"
    save_results_to_sqlite(report, db_path=db_file)


if __name__ == "__main__":
    demo(
        input_dir="./data/engineering_images_100dpi",
        output_root="./results/batch/engineering_images_100dpi",
    )
