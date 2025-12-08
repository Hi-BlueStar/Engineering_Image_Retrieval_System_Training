# batch_runner.py
from __future__ import annotations

import json
import os
import queue
import threading
import time
import traceback
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

from rich import box

# 第三方：視覺處理模組在 image_preprocessing.py 內部使用
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

# 匯入你的主流程函式
from image_preprocessing import DEFAULTS as PIPE_DEFAULTS
from image_preprocessing import run_pipeline  # type: ignore


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


@dataclass
class BatchConfig:
    # 掃描與輸出
    input_dir: str | Path
    output_root: str | Path = "results_batch"
    patterns: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    recursive: bool = True
    per_image_outdir: str = "{stem}"  # 可用 {stem} / {name} / {parent} (此設定已被 _compute_out_dir 修改覆蓋)
    skip_existing: bool = True  # 若輸出下已存在 merged_all_large.png 就略過
    max_workers: int = 1  # >1 啟用多執行緒（I/O 為主）

    # merge_subcomponents_with_topology 參數
    top_n: int = PIPE_DEFAULTS["TOP_N"]
    remove_largest: bool = PIPE_DEFAULTS["REMOVE_LARGEST"]
    iterations: int = PIPE_DEFAULTS["ITERATIONS"]
    save_steps: bool = PIPE_DEFAULTS["SAVE_STEPS"]

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
    # 依照使用者要求：直接輸出到 output_root，不再建立子資料夾
    return Path(cfg.output_root)

    # --- 以下為原始程式碼（已註解） ---
    # stem = img_path.stem
    # name = img_path.name
    # parent = img_path.parent.name
    # # 可用佔位符
    # sub = cfg.per_image_outdir.format(stem=stem, name=name, parent=parent)
    # return Path(cfg.output_root) / sub


def _already_done(out_dir: Path) -> bool:
    # 以「總合輸出」是否存在來判斷
    return (out_dir / "merged" / "merged_all_large.png").exists()


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

        def worker():
            while True:
                try:
                    img_path = q.get_nowait()
                except queue.Empty:
                    break

                out_dir = _compute_out_dir(cfg, img_path)
                if cfg.skip_existing and _already_done(out_dir):
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
                try:
                    # 呼叫你的主流程
                    run_pipeline(
                        input_path=str(img_path),
                        output_dir=str(out_dir),
                        top_n=cfg.top_n,
                        remove_largest=cfg.remove_largest,
                        iterations=cfg.iterations,
                        save_steps=cfg.save_steps,
                        simplify_output=True,
                    )
                except Exception as e:
                    ok = False
                    err_msg = (
                        f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"
                    )
                finally:
                    duration = time.perf_counter() - t0
                    current, peak = tracemalloc.get_traced_memory()
                    if started_here:
                        tracemalloc.stop()
                    rss_after = (
                        proc.memory_info().rss if proc else None if psutil else None
                    )
                    rss_delta = (
                        (rss_after - rss_before)
                        if (psutil and rss_before is not None and rss_after is not None)
                        else None
                    )

                results.append(
                    FileStat(
                        path=img_path,
                        ok=ok,
                        duration_sec=duration,
                        rss_delta=rss_delta,
                        py_peak_bytes=peak,
                        error=err_msg,
                        output_dir=out_dir,
                    )
                )

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
    ok_cnt = sum(1 for r in results if r.ok)
    fail_cnt = sum(1 for r in results if not r.ok)

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

    # 報表輸出（可選）
    report = {
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
            }
            for r in results
        ],
        "params": {
            "top_n": cfg.top_n,
            "remove_largest": cfg.remove_largest,
            "iterations": cfg.iterations,
            "save_steps": cfg.save_steps,
            "patterns": cfg.patterns,
            "recursive": cfg.recursive,
            "skip_existing": cfg.skip_existing,
            "max_workers": cfg.max_workers,
        },
    }

    if cfg.write_report_json:
        json_path = (
            Path(cfg.report_json_path)
            if cfg.report_json_path
            else Path(cfg.output_root) / "batch_report.json"
        )
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        console.print(f"[dim]已輸出報表：[/]{json_path.resolve()}")

    console.print(Rule("[bold green]批次處理結束[/]"))
    return report


# ================================
# 範例用法（非 CLI，直接以程式呼叫）
# ================================
def demo(
    input_dir: str = "./data/engineering_images_100dpi_flat/train",
    output_root: str = "./results/batch",
):
    cfg = BatchConfig(
        input_dir=input_dir,
        output_root=output_root,
        patterns=(".png", ".jpg", ".jpeg"),
        recursive=True,
        per_image_outdir="{stem}",  # 註：此設定已無作用，輸出會直接到 output_root
        skip_existing=True,
        max_workers=8,  # 若 I/O 多可嘗試 >1
        top_n=8,
        remove_largest=True,
        iterations=1,  # ⇠ 可視雜訊調整（例如 3、5、10…）
        save_steps=False,
        write_report_json=True,
        report_json_path=None,  # None 則寫到 output_root/batch_report.json
        monitor_interval_sec=0.5,
        enable_gpu_monitor=True,
    )
    process_folder(cfg)


if __name__ == "__main__":
    demo(
        input_dir="./data/engineering_images_100dpi_flat/train",
        output_root="./results/batch/engineering_images_100dpi_flat/train",
    )
    demo(
        input_dir="./data/engineering_images_100dpi_flat/val",
        output_root="./results/batch/engineering_images_100dpi_flat/val",
    )
    demo(
        input_dir="./data/engineering_images_200dpi_flat/train",
        output_root="./results/batch/engineering_images_200dpi_flat/train",
    )
    demo(
        input_dir="./data/engineering_images_200dpi_flat/val",
        output_root="./results/batch/engineering_images_200dpi_flat/val",
    )
    demo(
        input_dir="./data/engineering_images_400dpi_flat/train",
        output_root="./results/batch/engineering_images_400dpi_flat/train",
    )
    demo(
        input_dir="./data/engineering_images_400dpi_flat/val",
        output_root="./results/batch/engineering_images_400dpi_flat/val",
    )
    demo(
        input_dir="./data/engineering_images_600dpi_flat/train",
        output_root="./results/batch/engineering_images_600dpi_flat/train",
    )
    demo(
        input_dir="./data/engineering_images_600dpi_flat/val",
        output_root="./results/batch/engineering_images_600dpi_flat/val",
    )
