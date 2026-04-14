"""PDF 轉影像工具（支援多執行緒、保留資料夾結構、Rich 進度顯示）。

此模組專為機器學習（ML）與電腦視覺（CV）的資料集前處理而設計：

- 遞迴掃描根目錄 `root_dir`，以父資料夾名稱作為類別標籤（class label）。
- 將每份 PDF 的每一頁視為獨立影像，轉為高解析度 PNG。
- 輸出時依 PDF 在 `root_dir` 的相對路徑建立資料夾，保留原始層級。
- 以多執行緒並行轉換，並以 Rich 呈現動態進度與結果訊息。
- 產生 `manifest.csv`，並於 `run()` 返回對應的 pandas.DataFrame。

使用的主要函式庫：
- PyMuPDF (`fitz`)：PDF 渲染為影像（無需外部 poppler）。
- pandas：生成與輸出資料清單（manifest）。
- rich：現代化的進度與輸出介面。

注意：
- DPI 透過渲染倍率控制（以 72 DPI 為基準等比例縮放）。

Example:
    參見 `if __name__ == '__main__':` 內之 CLI 與範例呼叫。
"""

from __future__ import annotations

import argparse
import os
import threading
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


try:
    import fitz  # PyMuPDF
except Exception as exc:  # pragma: no cover - 匯入期錯誤於執行時提示
    raise ImportError(
        "無法匯入 PyMuPDF（fitz）。請先安裝：pip install pymupdf"
    ) from exc

from rich import print as rprint
from rich.console import Console
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


@dataclass(frozen=True)
class PageTask:
    """單一頁面轉檔任務描述。

    此資料結構描述一頁 PDF（作為獨立影像）之轉檔任務與輸出位置。

    Attributes:
        source_pdf: 原始 PDF 檔案路徑（絕對或相對路徑皆可）。
        class_label: 類別標籤（取自 PDF 之父資料夾名稱）。
        page_index: PDF 頁面索引（0 起算）。
        dest_rel_path: 影像輸出相對於實際 `output_dir` 的相對路徑。
    """

    source_pdf: Path
    class_label: str
    page_index: int
    dest_rel_path: Path


def _validate_inputs(root_dir: Path, output_dir: Path, dpi: int) -> None:
    """驗證輸入參數與路徑。

    Args:
        root_dir: 根目錄，底下包含多個類別資料夾，每個資料夾內含 PDF。
        output_dir: 預期輸出根目錄。
        dpi: 轉檔解析度（DPI）。

    Raises:
        FileNotFoundError: 當 `root_dir` 不存在時。
        ValueError: 當 `dpi` 不合規時。
    """

    if not root_dir.exists() or not root_dir.is_dir():
        raise FileNotFoundError(f"root_dir 不存在或不是資料夾: {root_dir}")
    if dpi <= 0:
        raise ValueError("dpi 必須為正整數。")


def _iter_pdfs(root_dir: Path) -> Iterable[tuple[Path, str]]:
    """遞迴掃描並回傳 PDF 與其類別標籤。

    以 PDF 檔案的父資料夾名稱作為類別標籤（class label）。

    Args:
        root_dir: 包含多個類別資料夾之根目錄。

    Yields:
        Tuple[Path, str]: (pdf_path, class_label) 二元組。
    """

    for pdf in root_dir.rglob("*.pdf"):
        if pdf.is_file():
            yield (pdf, pdf.parent.name)


def _get_pdf_page_count(pdf_path: Path) -> int:
    """取得 PDF 總頁數。

    Args:
        pdf_path: PDF 檔案路徑。

    Returns:
        int: 頁數（>= 1）。

    Raises:
        RuntimeError: 當 PDF 無法開啟或讀取頁數時。
    """

    try:
        with fitz.open(pdf_path) as doc:
            return int(doc.page_count)
    except Exception as exc:  # 轉為更可讀的錯誤型態
        raise RuntimeError(f"讀取 PDF 頁數失敗: {pdf_path}") from exc


def _sanitize_component_for_filename(text: str) -> str:
    """將文字轉為可安全用於檔名的片段（僅用於 flat 模式）。

    僅保留英數字、底線、減號與點，其餘以底線取代，以減少跨平台檔名風險。

    Args:
        text: 任意字串。

    Returns:
        str: 已清理之檔名安全片段。
    """
    # safe = re.sub(r"[^A-Za-z0-9._-]", "_", text)
    # safe = re.sub(r"_+", "_", safe).strip("._-")
    safe = text.replace(" ", "_")

    return safe or ""


def _plan_tasks(
    pdfs_with_labels: list[tuple[Path, str]],
    root_dir: Path,
) -> tuple[dict[Path, list[PageTask]], int]:
    """規劃所有頁面轉檔任務（保持原始資料夾結構）。

    步驟：
    1. 讀取每個 PDF 的頁數，展開成逐頁任務清單。
    2. 依照 PDF 在 `root_dir` 的相對路徑規劃輸出位置，不做資料集拆分。
    3. 以 `source_pdf` 分組，回傳供多執行緒逐檔處理。

    Args:
        pdfs_with_labels: (pdf_path, class_label) 之列表。
        root_dir: 根目錄，用於計算來源相對路徑，確保輸出保留原資料夾結構。

    Returns:
        Tuple[Dict[Path, List[PageTask]], int]:
            - 任務字典：key 為 PDF 檔路徑，value 為該 PDF 的頁面任務列表。
            - 全部頁面任務總數（總影像數）。

    Raises:
        RuntimeError: 若讀取 PDF 頁數發生錯誤。
    """

    tasks_by_pdf: dict[Path, list[PageTask]] = {}
    total_pages = 0

    for pdf_path, class_label in pdfs_with_labels:
        # page_count = _get_pdf_page_count(pdf_path)
        rel_dir = pdf_path.parent.relative_to(root_dir)

        # for page_index in range(page_count):
        #     pdf_stem = pdf_path.stem
        #     page_num = page_index + 1
        #     fname = f"{_sanitize_component_for_filename(pdf_stem)}_page_{page_num}.png"  # noqa: E501
        #     rel_path = rel_dir / fname

        #     task = PageTask(
        #         source_pdf=pdf_path,
        #         class_label=class_label,
        #         page_index=page_index,
        #         dest_rel_path=rel_path,
        #     )
        #     tasks_by_pdf.setdefault(pdf_path, []).append(task)
        #     total_pages += 1
        pdf_stem = pdf_path.stem
        fname = f"{_sanitize_component_for_filename(pdf_stem)}.png"
        rel_path = rel_dir / fname

        task = PageTask(
            source_pdf=pdf_path,
            class_label=class_label,
            page_index=0,
            dest_rel_path=rel_path,
        )
        tasks_by_pdf.setdefault(pdf_path, []).append(task)
        total_pages += 1

    # 為了在單一 PDF 內也有合理的處理順序，依 page_index 排序
    for _pdf_path, plist in tasks_by_pdf.items():
        plist.sort(key=lambda t: t.page_index)

    return tasks_by_pdf, total_pages


def _convert_one_pdf(
    pdf_path: Path,
    tasks: list[PageTask],
    output_dir: Path,
    dpi: int,
    progress: Progress,
    progress_task_id: TaskID,
    rows_out: list[tuple[str, str, str]],
    rows_lock: threading.Lock,
    console: Console,
) -> None:
    """處理單一 PDF（依內含之 PageTask 逐頁轉檔）。

    Args:
        pdf_path: 要處理的 PDF 檔案。
        tasks: 該 PDF 對應的頁面任務列表（已決定 split 與目的路徑）。
        output_dir: 輸出根目錄。
        dpi: 轉檔 DPI（透過縮放矩陣控制）。
        progress: Rich 的 Progress 物件，供更新進度條。
        progress_task_id: 進度條任務 ID。
        rows_out: 供成功頁面寫入 manifest 之共享列表（執行緒安全地 append）。
        rows_lock: 保護 `rows_out` 的鎖。
        console: Rich Console，用於印出狀態訊息。

    Raises:
        Exception: 若內部轉檔時發生未攔截的例外，會由上層處理。
    """

    scale = dpi / 72.0
    success_pages = 0
    failed_pages = 0
    first_error_msg = None

    try:
        doc = fitz.open(pdf_path)
        try:
            for t in tasks:
                # 動態更新目前檔案/頁面至進度條描述
                progress.update(
                    progress_task_id,
                    description=(
                        f"[cyan]處理[/cyan] [yellow]{pdf_path.name}"
                        f"[/yellow] (page {t.page_index + 1})"
                    ),
                )

                try:
                    page = doc.load_page(t.page_index)
                    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)

                    dest_abs = output_dir / t.dest_rel_path
                    dest_abs.parent.mkdir(parents=True, exist_ok=True)
                    pix.save(dest_abs.as_posix())

                    # 寫入 manifest 行：使用相對路徑與標籤、split
                    image_rel_posix = t.dest_rel_path.as_posix()

                    # source_pdf 欄位儲存相對於 root_dir 的相對路徑較佳，但此處
                    # 不保留 root_dir，故以檔名或相對字串表示；由於 spec 未強制，
                    # 以原始路徑字串表達。
                    row = (
                        str(t.source_pdf),  # source_pdf
                        t.class_label,  # class_label
                        image_rel_posix,  # image_path (相對於 output_dir)
                    )
                    with rows_lock:
                        rows_out.append(row)
                    success_pages += 1
                except Exception as page_exc:  # 單頁失敗不影響其他頁
                    failed_pages += 1
                    if first_error_msg is None:
                        first_error_msg = f"{type(page_exc).__name__}: {page_exc}"
                finally:
                    progress.advance(progress_task_id, 1)
        finally:
            doc.close()
    except Exception as open_exc:
        # 若整份 PDF 開啟即失敗，將全部頁面視為失敗
        failed_pages = len(tasks)
        first_error_msg = f"開啟 PDF 失敗: {type(open_exc).__name__}: {open_exc}"
        # 仍把進度推進
        progress.advance(progress_task_id, failed_pages)

    # 完成後印出結果摘要
    if failed_pages == 0:
        rprint(
            f"[green]✓[/green] 處理完成: [cyan]{pdf_path.name}[/cyan] "
            f"([bold]{success_pages}[/bold] 張影像)"
        )
    elif success_pages == 0:
        rprint(
            f"[red]✗[/red] 處理失敗: [yellow]{pdf_path.name}[/yellow] - "
            f"{first_error_msg or '未知錯誤'}"
        )
    else:
        rprint(
            f"[yellow]![/yellow] 部分成功: [cyan]{pdf_path.name}[/cyan] - "
            f"成功 {success_pages} / 失敗 {failed_pages}; "
            f"第一個錯誤: {first_error_msg or '未知'}"
        )


def run(
    root_dir: os.PathLike | str,
    output_dir: os.PathLike | str,
    dpi: int = 300,
    max_workers: int | None = None,
) -> pd.DataFrame:
    """執行 PDF 轉影像並輸出/返回 manifest（不進行資料集拆分）。

    本函式會遞迴掃描 `root_dir` 下的所有 PDF，將每頁轉成 PNG，並在 `output_dir`
    中重建與輸入相同的資料夾結構。最終輸出 `manifest.csv`，並回傳 pandas.DataFrame。

    Args:
        root_dir: 根目錄路徑；其下為類別資料夾，每個資料夾包含若干 PDF。
        output_dir: 輸出根目錄；會依照 PDF 相對於 `root_dir` 的路徑建立相同的資料夾層級。
        dpi: 影像輸出解析度（透過縮放控制渲染倍率）。
        max_workers: 最大執行緒數；預設使用 `os.cpu_count()` 合理值。

    Returns:
        pandas.DataFrame: 含欄位 `source_pdf`, `class_label`, `image_path`。

    Raises:
        FileNotFoundError: 當 `root_dir` 無法存取時。
        ValueError: 當 `dpi` 無效時。
    """  # noqa: E501

    console = Console()

    root_dir = Path(root_dir).resolve()
    output_dir = Path(output_dir).resolve()

    _validate_inputs(root_dir, output_dir, dpi)

    # 掃描 PDF 與類別
    pdfs_with_labels = list(_iter_pdfs(root_dir))
    if not pdfs_with_labels:
        console.print("[yellow]警告[/yellow]：未在 root_dir 發現任何 PDF 檔案。")
        # 空集合仍回傳空 DataFrame，並在輸出目錄建立空 manifest。
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "manifest.csv"
        df_empty = pd.DataFrame(columns=["source_pdf", "class_label", "image_path"])
        df_empty.to_csv(manifest_path, index=False)
        return df_empty

    # 規劃所有頁面任務（保留原資料夾結構）
    tasks_by_pdf, total_pages = _plan_tasks(
        pdfs_with_labels=pdfs_with_labels,
        root_dir=root_dir,
    )

    max_workers = max_workers or max(os.cpu_count() or 1, 1)

    # 準備輸出根目錄
    output_dir.mkdir(parents=True, exist_ok=True)

    console.rule("[bold cyan]PDF ➜ 影像 轉換開始")
    console.print(
        f"根目錄: [cyan]{root_dir}[/cyan]\n"
        f"輸出目錄: [green]{output_dir}[/green]\n"
        f"DPI: [yellow]{dpi}[/yellow]\n"
        f"工作執行緒: [yellow]{max_workers}[/yellow]"
    )

    rows_out: list[tuple[str, str, str]] = []
    rows_lock = threading.Lock()

    # Rich 進度條設定
    progress = Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[bold blue]{task.description}", justify="left"),
        BarColumn(bar_width=None, style="magenta", complete_style="green"),
        MofNCompleteColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=True,
    )

    with progress:
        task_id = progress.add_task("初始化任務…", total=total_pages)

        # 以「每個 PDF 檔」為單位分派工作，內部處理所有頁面
        futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for pdf_path, page_tasks in tasks_by_pdf.items():
                fut = ex.submit(
                    _convert_one_pdf,
                    pdf_path,
                    page_tasks,
                    output_dir,
                    dpi,
                    progress,
                    task_id,
                    rows_out,
                    rows_lock,
                    console,
                )
                futures.append(fut)

            # 逐一等待完成，以捕捉可能的未處理例外（避免靜默失敗）
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as exc:  # 任何未預期錯誤
                    rprint(
                        f"[red]✗[/red] 轉檔工作執行時發生未預期例外："
                        f"[yellow]{type(exc).__name__}: {exc}[/yellow]"
                    )

    # 完成後產生 manifest
    df = pd.DataFrame(rows_out, columns=["source_pdf", "class_label", "image_path"])
    manifest_path = output_dir / "manifest.csv"
    df.to_csv(manifest_path, index=False)

    console.print(
        f"[green]完成[/green]：共輸出 [bold]{len(df)}[/bold] 張影像，"
        f"manifest 已寫入 [cyan]{manifest_path}[/cyan]"
    )
    console.rule("[bold green]處理結束")

    return df


def _build_arg_parser() -> argparse.ArgumentParser:
    """建立命令列介面（CLI）參數解析器。

    Returns:
        argparse.ArgumentParser: 解析器物件。
    """

    parser = argparse.ArgumentParser(
        description=(
            "PDF 轉影像工具（多執行緒 + Rich 進度）\n"
            "- 根目錄需包含類別資料夾，PDF 父資料夾即為該檔的 class label。\n"
            "- 多頁 PDF 每頁視為獨立影像，輸出時保留原資料夾結構。\n"
        )
    )
    parser.add_argument(
        "--root_dir", type=str, required=True, help="資料根目錄（含類別資料夾與 PDF）"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="輸出資料集根目錄"
    )
    parser.add_argument("--dpi", type=int, default=300, help="輸出影像 DPI（預設 300）")
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="最大執行緒數（預設為 CPU 合理值）",
    )
    return parser


if __name__ == "__main__":
    # # CLI 版本：從命令列參數執行
    # parser = _build_arg_parser()
    # args = parser.parse_args()

    # try:
    #     run(
    #         root_dir=args.root_dir,
    #         output_dir=args.output_dir,
    #         dpi=args.dpi,
    #         max_workers=args.max_workers,
    #     )
    # except Exception as e:
    #     rprint(f"[red]✗[/red] 程式執行失敗：[yellow]{type(e).__name__}: {e}[/yellow]")
    #     sys.exit(1)

    # --------------------------------------------------------------
    # 參數寫死版本（供 IDE 快速測試，請依需求修改後取消註解）：
    # --------------------------------------------------------------
    from rich import print as rprint

    try:
        df_manifest = run(
            root_dir=r"./data/吉輔提供資料",  # 根目錄（含類別資料夾與 PDF）
            output_dir=r"./data/engineering_images_100dpi",  # 輸出根目錄
            dpi=100,  # 影像解析度（DPI）
            max_workers=16,  # 預設為 CPU 合理值
        )
        rprint(df_manifest.head())
    except Exception as e:
        rprint(f"[red]✗[/red] 測試執行失敗：[yellow]{type(e).__name__}: {e}[/yellow]")

"""
uv run python src/pdf_to_image2.py --root_dir ./data/吉輔提供資料 --output_dir ./data/engineering_images_100dpi --dpi 100 --max_workers 14

"""  # noqa: E501
