# batch_runner for image_preprocessing2
from __future__ import annotations

import functools  # 用於裝飾器
import json
import os
import queue
import sys
import threading
import time
import traceback
import tracemalloc
from dataclasses import dataclass
from pathlib import Path


# 確保可直接在專案根目錄執行此檔時能找到同資料夾的模組
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

# 第三方：視覺處理模組在 image_preprocessing2 內部使用
# 這裡只需要 rich/psutil/pynvml（選用）
from rich import box
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
from image_preprocessing3 import run_pipeline


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
    ram_threshold: float = 90.0,
    gpu_mem_threshold: float = 90.0,
    check_interval: float = 1.0,
    console: Console | None = None,
):
    """
    硬體資源守門員：
    在執行函式前檢查系統記憶體 (RAM) 與 GPU 顯存。
    若超過設定閾值，則暫停執行並等待資源釋放，避免 OOM 崩潰。
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 1. 檢查系統記憶體 (RAM)
            if psutil:
                while True:
                    mem = psutil.virtual_memory()
                    if mem.percent < ram_threshold:
                        break
                    # 資源吃緊，進入等待
                    if console:
                        # 顯示一個瞬態訊息 (不會保留在 log)
                        console.print(
                            f"[yellow dim]系統 RAM 負載過高 ({mem.percent}%)，執行緒暫停等待中...[/]",
                            end="\r",
                        )
                    time.sleep(check_interval)

            # 2. 檢查 GPU 顯存 (若有的話)
            if _NVML_OK:
                try:
                    # 簡單策略：檢查第一張 GPU 或所有 GPU 平均
                    # 這裡示範檢查 index 0，若多卡可再擴充
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    while True:
                        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        used_percent = (info.used / info.total) * 100
                        if used_percent < gpu_mem_threshold:
                            break
                        if console:
                            console.print(
                                f"[yellow dim]GPU 顯存負載過高 ({used_percent:.1f}%)，執行緒暫停等待中...[/]",
                                end="\r",
                            )
                        time.sleep(check_interval)
                except Exception:
                    pass  # 獲取失敗則略過檢查

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
                            f"GPU{idx} {util.gpu:>3}% | {_fmt_bytes(meminfo.used)}/{_fmt_bytes(meminfo.total)}"
                        )
                    gpu_str = " | ".join(parts)
                except Exception:
                    gpu_str = "GPU:不可用"

            text = (
                f"[bold]系統資源[/] | "
                f"CPU: {cpu:.0f}% | RAM: {ram:.0f}% | RSS: {_fmt_bytes(rss)} | {io_str} | {gpu_str}"
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
# 核心：批次處理
# ================================
def process_folder(cfg: BatchConfig) -> dict:
    """
    對資料夾內所有圖片執行 run_pipeline。
    回傳：包含彙整統計與每檔紀錄的 dict（也可選擇輸出 JSON 報表）。
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

    console.print(Rule("[bold cyan]批次處理開始[/]"))
    console.print(f"[b]來源[/]: {Path(cfg.input_dir).resolve()}")
    console.print(f"[b]輸出根目錄[/]: {out_root.resolve()}")
    console.print(f"[b]圖片數[/]: {len(files)}  | [b]執行緒[/]: {cfg.max_workers}")

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

    results: list[FileStat] = []

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
        if not force and len(results) % 5 != 0:
            return
        report_path.parent.mkdir(parents=True, exist_ok=True)
        data = _build_report()
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        console.print(
            f"[dim]已輸出進度報表（{len(results)} 筆）→[/] {report_path}",
            soft_wrap=True,
        )

    with progress:
        # 系統資源監控 Task（描述會持續被更新）
        monitor_task = progress.add_task("初始化系統資源監控中…", total=None)
        monitor = ResourceMonitor(
            progress,
            monitor_task,
            interval=cfg.monitor_interval_sec,
            enable_gpu=cfg.enable_gpu_monitor,
        )
        monitor.start()

        # 掃描 Task
        scan_task = progress.add_task("[bold]掃描檔案[/]", total=len(files))

        # 處理 Task
        work_task = progress.add_task("[bold]處理圖片[/]", total=len(files))

        progress.update(scan_task, advance=len(files))  # 掃描立即完成

        # 執行：可選多執行緒（I/O 為主，OpenCV/NumPy 常釋放 GIL）
        q: queue.Queue[Path] = queue.Queue()
        for p in files:
            q.put(p)

        lock = threading.Lock()

        # [新增/重構] 定義受資源監控保護的執行函式
        # 設定閾值：RAM > 90% 或 GPU > 90% 時暫停新任務
        @resource_guard(ram_threshold=90.0, gpu_mem_threshold=90.0, console=console)
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
                        results.append(
                            FileStat(
                                path=img_path,
                                ok=True,
                                duration_sec=0.0,
                                rss_delta=None,
                                py_peak_bytes=None,
                                error=None,
                                output_dir=out_dir,
                            )
                        )
                        _maybe_write_report()
                    progress.update(
                        work_task,
                        advance=1,
                        description=f"略過（已存在）: {img_path.name}",
                    )
                    continue

                out_dir.mkdir(parents=True, exist_ok=True)

                # 量測
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
                    # 呼叫你的主流程並收集輸出檔案
                    outputs = _protected_run(str(img_path), str(out_dir))
                    # 收集主要輸出路徑
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
                    # 驗證至少有 merged 檔案存在
                    merged_path = (
                        outputs.get("merged") if isinstance(outputs, dict) else None
                    )
                    if merged_path and not Path(merged_path).exists():
                        ok = False
                        err_msg = f"處理完成但找不到輸出檔案：{merged_path}"
                except Exception as e:
                    print(e)
                    ok = False
                    err_msg = (
                        f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"
                    )
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

                with lock:
                    results.append(
                        FileStat(
                            path=img_path,
                            ok=ok,
                            duration_sec=duration,
                            rss_delta=rss_delta,
                            py_peak_bytes=peak,
                            error=err_msg,
                            output_dir=out_dir,
                            saved_files=saved_files or None,
                        )
                    )
                    _maybe_write_report()

                # 更新進度列
                if ok:
                    progress.update(
                        work_task, advance=1, description=f"完成: {img_path.name}"
                    )
                else:
                    progress.update(
                        work_task,
                        advance=1,
                        description=f"[red]失敗[/]: {img_path.name}",
                    )

                # ==========================================
                # [新增] 記憶體回收機制
                # ==========================================
                # 1. 顯式刪除可能含有大型數據的變數引用 (如果 outputs 含有影像陣列)
                if "outputs" in locals():
                    del outputs

        threads: list[threading.Thread] = []
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

    table = Table(title="批次摘要", box=box.SIMPLE_HEAVY)
    table.add_column("檔名", overflow="fold")
    table.add_column("狀態", justify="center")
    table.add_column("耗時", justify="right")
    table.add_column("ΔRSS", justify="right")
    table.add_column("Python峰值", justify="right")
    for r in results:
        table.add_row(
            r.path.name,
            "[green]OK[/]" if r.ok else "[red]FAIL[/]",
            f"{r.duration_sec:.3f}s",
            _fmt_bytes(r.rss_delta),
            _fmt_bytes(r.py_peak_bytes),
        )

    console.print(Rule())
    console.print(table)
    console.print(
        f"[b]總計[/]: {len(results)} | [green]成功[/]: {ok_cnt} | [red]失敗[/]: {fail_cnt}"
    )

    # 報表輸出（可選，結尾強制寫入一次）
    _maybe_write_report(force=True)

    console.print(Rule("[bold green]批次處理結束[/]"))
    return report


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
        per_image_outdir="{stem}",  # 以檔名建立各自子資料夾，避免 large_components 混在一起
        skip_existing=True,
        max_workers=8,  # 若 I/O 多可嘗試 >1
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
    process_folder(cfg)


if __name__ == "__main__":
    demo(
        input_dir="./data/engineering_images_Clean_100dpi",
        output_root="./results/batch2/engineering_images_Clean_100dpi",
    )
