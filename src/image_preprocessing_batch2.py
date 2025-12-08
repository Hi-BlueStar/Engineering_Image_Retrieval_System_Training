# batch_runner for image_preprocessing2
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
import contextlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import os
import sys
import time
import json
import threading
import queue
import traceback
import gc  # 新增：用於強制回收垃圾以獲得準確記憶體數據

import tracemalloc

# 確保可直接在專案根目錄執行此檔時能找到同資料夾的模組
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

# 第三方：視覺處理模組在 image_preprocessing2 內部使用
# 這裡只需要 rich/psutil/pynvml（選用）
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn,
    TaskProgressColumn,
)
from rich.table import Table
from rich.rule import Rule
from rich import box

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

# 匯入新版主流程函式（隨機排列大元件版本）
from image_preprocessing2 import run_pipeline

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
def _fmt_bytes(n: Optional[int]) -> str:
    if n is None:
        return "N/A"
    units = ["B", "KB", "MB", "GB", "TB"]
    s = float(n)
    for u in units:
        if s < 1024.0:
            return f"{s:.2f} {u}"
        s /= 1024.0
    return f"{s:.2f} PB"


@dataclass
class BatchConfig:
    # 掃描與輸出
    input_dir: Union[str, Path]
    output_root: Union[str, Path] = "results_batch2"
    patterns: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    recursive: bool = True
    per_image_outdir: str = "{stem}"  # 可用 {stem} / {name} / {parent}
    preserve_structure: bool = True   # 是否保留與輸入資料夾相同的子路徑
    skip_existing: bool = True        # 若輸出下已存在 {stem}_merged.* 就略過
    max_workers: int = 1              # >1 啟用多執行緒（I/O 為主）

    # image_preprocessing2 參數
    top_n: int = PIPE_DEFAULTS["TOP_N"]
    remove_largest: bool = PIPE_DEFAULTS["REMOVE_LARGEST"]
    seed: Optional[int] = PIPE_DEFAULTS["SEED"]
    padding: int = PIPE_DEFAULTS["PADDING"]
    max_attempts: int = PIPE_DEFAULTS["MAX_ATTEMPTS"]
    random_count: int = PIPE_DEFAULTS["RANDOM_COUNT"]

    # 報表
    write_report_json: bool = True
    report_json_path: Optional[Union[str, Path]] = None  # 預設 output_root / "batch_report.json"

    # 監控
    monitor_interval_sec: float = 0.5
    enable_gpu_monitor: bool = True




@dataclass
class SegmentResult:
    """單一階段的效能數據"""
    name: str
    duration: float
    rss_start: int
    rss_end: int
    rss_delta: int
    peak_trace_diff: int  # 該階段內的 Python 物件記憶體增量峰值


# @dataclass
# class FileStat:
#     path: Path
#     ok: bool
#     duration_sec: float
#     rss_delta: Optional[int] = None
#     py_peak_bytes: Optional[int] = None
#     error: Optional[str] = None
#     output_dir: Optional[Path] = None
#     saved_files: Optional[List[str]] = None

@dataclass
class FileStat:
    """檔案處理結果（增強版）"""
    path: Path
    ok: bool
    total_duration: float
    # 移除舊的單一欄位，改用 segments 紀錄細節
    segments: List[SegmentResult]
    error: Optional[str] = None
    output_dir: Optional[Path] = None
    saved_files: Optional[List[str]] = None

    @property
    def rss_delta_total(self) -> int:
        """計算整個流程的 RSS 記憶體變化 (End of last - Start of first)"""
        if not self.segments:
            return 0
        return self.segments[-1].rss_end - self.segments[0].rss_start

    @property
    def peak_memory(self) -> int:
        """取得所有階段中最高的 Python 記憶體峰值"""
        if not self.segments:
            return 0
        return max(s.peak_trace_diff for s in self.segments)


# 新增：分段效能分析器
class SegmentedProfiler:
    def __init__(self):
        self.results: List[SegmentResult] = []
        self._proc = psutil.Process(os.getpid()) if psutil else None

    @contextlib.contextmanager
    def step(self, name: str):
        """Context Manager: 測量區塊內的記憶體與時間"""
        # 1. 紀錄開始狀態
        t0 = time.perf_counter()
        rss0 = self._proc.memory_info().rss if self._proc else 0
        
        # 啟動 tracemalloc 追蹤峰值
        # 注意：在多執行緒下 tracemalloc 是全域的，數據僅供參考，但對單執行緒非常準確
        tracemalloc.start()
        tracemalloc.reset_peak()
        
        try:
            yield
        finally:
            # 2. 紀錄結束狀態
            t1 = time.perf_counter()
            rss1 = self._proc.memory_info().rss if self._proc else 0
            
            # 取得該區段內的峰值增量
            _, peak_size = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            self.results.append(SegmentResult(
                name=name,
                duration=t1 - t0,
                rss_start=rss0,
                rss_end=rss1,
                rss_delta=rss1 - rss0,
                peak_trace_diff=peak_size
            ))


# ================================
# 資源即時監控（背景執行緒）
# ================================
class ResourceMonitor:
    def __init__(self, progress: Progress, task_id: int, interval: float = 0.5, enable_gpu: bool = True):
        self.progress = progress
        self.task_id = task_id
        self.interval = max(0.2, float(interval))
        self.enable_gpu = enable_gpu and _NVML_OK
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

        if self.enable_gpu:
            try:
                pynvml.nvmlInit()
                self._gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i)
                                     for i in range(pynvml.nvmlDeviceGetCount())]
            except Exception:
                self.enable_gpu = False

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        if self.enable_gpu:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def _run(self):
        proc = psutil.Process(os.getpid()) if psutil else None
        while not self._stop.is_set():
            # CPU / RAM / IO
            if psutil:
                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory().percent
                rss = proc.memory_info().rss if proc else None
                io = psutil.disk_io_counters()
                io_str = f"R:{_fmt_bytes(io.read_bytes)} W:{_fmt_bytes(io.write_bytes)}" if io else "IO:N/A"
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
                            f"GPU{idx} {util.gpu:>3}% | {_fmt_bytes(meminfo.used)}/{_fmt_bytes(meminfo.total)}"
                        )
                    gpu_str = " | ".join(parts)
                except Exception:
                    gpu_str = "GPU:不可用"

            text = (
                f"[bold]系統資源[/] | "
                f"CPU: {cpu:.0f}% | RAM: {ram:.0f}% | RSS: {_fmt_bytes(rss)} | {io_str} | {gpu_str}"
                if psutil else
                f"[bold]系統資源[/] | psutil 未安裝"
            )
            # 更新進度列的描述
            self.progress.update(self.task_id, description=text)
            time.sleep(self.interval)


# ================================
# 檔案掃描
# ================================
def _gather_images(cfg: BatchConfig) -> List[Path]:
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
            rel_parent = img_path.parent.resolve().relative_to(Path(cfg.input_dir).resolve())
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
# 核心：批次處理
# ================================
def process_folder(cfg: BatchConfig) -> Dict:
    """
    對資料夾內所有圖片執行 run_pipeline。
    回傳：包含彙整統計與每檔紀錄的 dict（也可選擇輸出 JSON 報表）。
    """
    console = Console()
    out_root = Path(cfg.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    report_path = Path(cfg.report_json_path) if cfg.report_json_path else out_root / "batch_report.json"

    files = _gather_images(cfg)
    if not files:
        console.print(f"[yellow]資料夾內未找到圖片：{cfg.input_dir}（副檔名：{cfg.patterns}）[/]")
        return {"count": 0, "ok": 0, "failed": 0, "items": []}

    console.print(Rule("[bold cyan]批次處理開始[/]"))
    console.print(f"[b]來源[/]: {Path(cfg.input_dir).resolve()}")
    console.print(f"[b]輸出根目錄[/]: {out_root.resolve()}")
    console.print(f"[b]圖片數[/]: {len(files)}  | [b]執行緒[/]: {cfg.max_workers}")

    # Rich 進度條設定
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

    results: List[FileStat] = []

    def _build_report() -> Dict:
        ok_cnt = sum(1 for r in results if r.ok)
        return {
            "summary": {
                "count": len(results),
                "ok": ok_cnt,
                "failed": len(results) - ok_cnt,
                "output_root": str(Path(cfg.output_root).resolve()),
            },
            "items": [
                {
                    "path": str(r.path),
                    "ok": r.ok,
                    "total_duration": r.total_duration,
                    "rss_delta_total": r.rss_delta_total,
                    "peak_py_bytes": r.peak_memory,
                    "segments": [
                        {
                            "step": s.name,
                            "sec": round(s.duration, 4),
                            "rss_delta": _fmt_bytes(s.rss_delta),
                            "peak_trace": _fmt_bytes(s.peak_trace_diff)
                        } for s in r.segments
                    ],
                    "error": r.error,
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
            }
        }

    def _maybe_write_report(force: bool = False) -> None:
        if not cfg.write_report_json or not results:
            return
        if not force and len(results) % 5 != 0:
            return
        report_path.parent.mkdir(parents=True, exist_ok=True)
        data = _build_report()
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        console.print(f"[dim]已輸出進度報表（{len(results)} 筆）→[/] {report_path}", soft_wrap=True)

    with progress:
        # 系統資源監控 Task（描述會持續被更新）
        monitor_task = progress.add_task("初始化系統資源監控中…", total=None)
        monitor = ResourceMonitor(progress, monitor_task, interval=cfg.monitor_interval_sec,
                                  enable_gpu=cfg.enable_gpu_monitor)
        monitor.start()

        # 掃描 Task
        scan_task = progress.add_task("[bold]掃描檔案[/]", total=len(files))

        # 處理 Task
        work_task = progress.add_task("[bold]處理圖片[/]", total=len(files))

        progress.update(scan_task, advance=len(files))  # 掃描立即完成

        # 執行：可選多執行緒（I/O 為主，OpenCV/NumPy 常釋放 GIL）
        q: "queue.Queue[Path]" = queue.Queue()
        for p in files:
            q.put(p)

        lock = threading.Lock()

        def worker():
            while True:
                try:
                    img_path = q.get_nowait()
                except queue.Empty:
                    break

                out_dir = _compute_out_dir(cfg, img_path)
                suffix = img_path.suffix or ".png"
                if cfg.skip_existing and _already_done(out_dir, img_path.stem, suffix):
                    with lock:
                        results.append(FileStat(
                            path=img_path, ok=True, total_duration=0.0, segments=[],
                            error=None, output_dir=out_dir
                        ))
                        _maybe_write_report()
                    progress.update(work_task, advance=1, description=f"略過（已存在）: {img_path.name}")
                    continue

                out_dir.mkdir(parents=True, exist_ok=True)

                # 量測
                # proc = psutil.Process(os.getpid()) if psutil else None
                # rss_before = proc.memory_info().rss if proc else None

                # started_here = not tracemalloc.is_tracing()
                # if started_here:
                #     tracemalloc.start()
                # t0 = time.perf_counter()
                
                # === 關鍵修改：使用 Profiler 進行分段分析 ===
                # 1. 強制 GC：確保上一張圖的記憶體被釋放，避免數據污染
                gc.collect()
                
                profiler = SegmentedProfiler()
                ok = True
                err_msg = None
                saved_files: List[str] = []
                try:
                    # 分段 1: 準備 (I/O check, setup)
                    with profiler.step("Init"):
                         # 這裡可以放預處理邏輯，目前主要是路徑準備
                         input_str = str(img_path)
                         output_str = str(out_dir)

                    # 分段 2: 核心執行 (Pipeline Execution)
                    with profiler.step("Pipeline"):
                        # 呼叫你的主流程並收集輸出檔案
                        outputs = run_pipeline(
                            input_path=str(img_path),
                            output_dir=str(out_dir),
                            top_n=cfg.top_n,
                            remove_largest=cfg.remove_largest,
                            seed=cfg.seed,
                            padding=cfg.padding,
                            max_attempts=cfg.max_attempts,
                            random_count=cfg.random_count,
                        )
                    
                    # 分段 3: 結果解析 (Result Parsing)
                    with profiler.step("Teardown"):
                        # 收集主要輸出路徑
                        if isinstance(outputs, dict):
                            for key in ("original", "merged"):
                                p = outputs.get(key)
                                if p:
                                    saved_files.append(str(p))
                            for p in outputs.get("random", []) if isinstance(outputs.get("random"), list) else []:
                                saved_files.append(str(p))
                            if outputs.get("large_dir"):
                                saved_files.append(str(outputs["large_dir"]))
                        # 驗證至少有 merged 檔案存在
                        merged_path = outputs.get("merged") if isinstance(outputs, dict) else None
                        if merged_path and not Path(merged_path).exists():
                            ok = False
                            err_msg = f"處理完成但找不到輸出檔案：{merged_path}"
                except Exception as e:
                    ok = False
                    err_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"

                # 計算總耗時
                total_time = sum(s.duration for s in profiler.results)

                with lock:
                    results.append(FileStat(
                        path=img_path,
                        ok=ok,
                        total_duration=total_time,
                        segments=profiler.results,
                        error=err_msg,
                        output_dir=out_dir,
                        saved_files=saved_files or None
                    ))
                    _maybe_write_report()

                if ok:
                    # 顯示記憶體變化 (Pipeline 階段)
                    pipe_stats = next((s for s in profiler.results if s.name == "Pipeline"), None)
                    mem_info = f"RSS:{_fmt_bytes(pipe_stats.rss_delta)}" if pipe_stats else ""
                    progress.update(work_task, advance=1, description=f"完成: {img_path.name} [{mem_info}]")
                else:
                    progress.update(work_task, advance=1, description=f"[red]失敗[/]: {img_path.name}")

        threads: List[threading.Thread] = []
        n_workers = max(1, int(cfg.max_workers))
        for _ in range(n_workers):
            t = threading.Thread(target=worker, daemon=True)
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        # 停止資源監控
        monitor.stop()

    # 彙整
    report = _build_report()
    ok_cnt = report["ok"]
    fail_cnt = report["failed"]
    
    # === 修改 Rich Table 以顯示更多細節 ===
    table = Table(title="批次摘要 (分段分析)", box=box.SIMPLE_HEAVY)
    table.add_column("檔名", overflow="fold")
    table.add_column("狀態", justify="center")
    table.add_column("總耗時", justify="right")
    table.add_column("核心(Pipe)耗時", justify="right")
    table.add_column("核心(Pipe)ΔRSS", justify="right") # 只看核心處理階段的實體記憶體增長
    table.add_column("Python峰值", justify="right")
    for r in results:
        # 找出 Pipeline 階段的數據
        pipe_seg = next((s for s in r.segments if s.name == "Pipeline"), None)
        pipe_dur = f"{pipe_seg.duration:.2f}s" if pipe_seg else "-"
        pipe_rss = _fmt_bytes(pipe_seg.rss_delta) if pipe_seg else "-"

        table.add_row(
            r.path.name,
            "[green]OK[/]" if r.ok else "[red]FAIL[/]",
            f"{r.total_duration:.2f}s",
            pipe_dur,
            pipe_rss,
            _fmt_bytes(r.peak_memory),
        )

    console.print(Rule())
    console.print(table)
    console.print(f"[b]總計[/]: {len(results)} | [green]成功[/]: {ok_cnt} | [red]失敗[/]: {fail_cnt}")

    # 報表輸出（可選，結尾強制寫入一次）
    _maybe_write_report(force=True)

    console.print(Rule("[bold green]批次處理結束[/]"))
    return report


# ================================
# 範例用法（非 CLI，直接以程式呼叫）
# ================================
def demo(input_dir: str = "./data/engineering_images_100dpi", output_root: str = "./results/batch2"):
    cfg = BatchConfig(
        input_dir=input_dir,
        output_root=output_root,
        patterns=(".png", ".jpg", ".jpeg"),
        recursive=True,
        per_image_outdir="{stem}",  # 以檔名建立各自子資料夾，避免 large_components 混在一起
        skip_existing=True,
        max_workers=1,        # 若 I/O 多可嘗試 >1
        top_n=5,
        remove_largest=True,
        seed=None,
        padding=2,
        max_attempts=400,
        random_count=20,
        write_report_json=True,
        report_json_path=None,      # None 則寫到 output_root/batch_report.json
        monitor_interval_sec=0.5,
        enable_gpu_monitor=True,
    )
    process_folder(cfg)


if __name__ == "__main__":
    demo(input_dir="./data/engineering_images_Clean_100dpi", output_root="./results/batch2/engineering_images_Clean_100dpi")
