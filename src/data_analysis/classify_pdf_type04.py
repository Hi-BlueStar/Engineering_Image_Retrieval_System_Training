import time
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.theme import Theme


# 設定自定義的主題風格
custom_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
        "vector": "blue",
        "raster": "magenta",
        "vector_draw": "bold yellow",
        "low_res": "bold red reverse",  # 用於低解析度警告
    }
)

console = Console(theme=custom_theme)


def get_page_size_label(width: float, height: float) -> str:
    """
    根據寬高 (points) 判斷紙張規格。允許誤差 +/- 5 points。
    """
    # 確保 width 是短邊，方便比對
    short, long = sorted([width, height])

    sizes = {
        (595, 842): "A4",
        (842, 1191): "A3",
        (1191, 1684): "A2",
        (1684, 2384): "A1",
        (2384, 3370): "A0",
        (420, 595): "A5",
        (612, 792): "Letter",
        (612, 1008): "Legal",
    }

    for (w, h), name in sizes.items():
        if abs(short - w) < 10 and abs(long - h) < 10:
            return name

    return f"{int(short)}x{int(long)}"


def format_size(size_bytes: int) -> str:
    """將 bytes 轉為易讀格式 (KB, MB)"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def classify_pdf_type(file_path: Path, page_sample_limit: int = 5) -> dict:
    """
    分析 PDF 檔案類型、頁面屬性、指紋與解析度。
    """
    try:
        # 0. 檔案指紋 (Fingerprinting)
        stat = file_path.stat()
        file_fingerprint = {
            "size_str": format_size(stat.st_size),
            "created": datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d"),
            "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d"),
        }

        # 使用 context manager 確保檔案會被關閉
        with fitz.open(file_path) as doc:
            total_pages = len(doc)

            if total_pages == 0:
                return {
                    "status": "success",
                    "type": "Empty",
                    "ratio": 0.0,
                    "reason": "PDF 無頁面",
                    "fingerprint": file_fingerprint,
                    "page_attr": {"count": 0, "size": "N/A"},
                    "dpi_stats": {"avg": 0, "status": "N/A"},
                }

            # 1. 頁面屬性採樣 (取第一頁做代表)
            p0 = doc[0]
            page_size_label = get_page_size_label(p0.rect.width, p0.rect.height)
            page_attr = {
                "count": total_pages,
                "size": page_size_label,
                "width": p0.rect.width,
                "height": p0.rect.height,
            }

            # 設定採樣範圍
            pages_to_check = (
                min(total_pages, page_sample_limit)
                if page_sample_limit
                else total_pages
            )
            vector_pages_count = 0
            raster_pages_count = 0
            vector_draw_pages_count = 0

            # DPI 統計容器
            dpi_values = []

            for i in range(pages_to_check):
                page = doc.load_page(i)
                text = page.get_text().strip()
                images = page.get_images()
                drawings = page.get_drawings()

                # DPI 計算邏輯 (針對頁面上的圖片)
                for img in images:
                    xref = img[0]
                    img_width = img[2]  # 像素寬度
                    # 獲取圖片在頁面上的顯示位置 (可能有多次引用)
                    img_rects = page.get_image_rects(xref)
                    for rect in img_rects:
                        if rect.width > 0:
                            # DPI = 像素 / (英吋) = 像素 / (points / 72)
                            dpi = img_width / (rect.width / 72)
                            dpi_values.append(dpi)

                # 判定邏輯
                if len(text) > 3:
                    vector_pages_count += 1
                elif len(images) > 0:
                    raster_pages_count += 1
                elif len(drawings) > 0:
                    vector_draw_pages_count += 1

            # 計算平均 DPI
            avg_dpi = int(sum(dpi_values) / len(dpi_values)) if dpi_values else 0
            dpi_status = "OK"
            if avg_dpi > 0 and avg_dpi < 200:
                dpi_status = "Low Res"  # 低解析度警告

            # 綜合判定類型
            text_ratio = vector_pages_count / pages_to_check

            if vector_pages_count > 0:
                file_type = "Vector"
                reason = f"含文字層 ({vector_pages_count}/{pages_to_check} 頁)"
            elif raster_pages_count > 0:
                file_type = "Raster"
                reason = "掃描/圖片檔"
            elif vector_draw_pages_count > 0:
                file_type = "VectorDrawing"
                reason = "向量繪圖"
            else:
                file_type = "Unknown"
                reason = "空白/未知"

            return {
                "status": "success",
                "type": file_type,
                "ratio": round(text_ratio, 2),
                "reason": reason,
                "fingerprint": file_fingerprint,
                "page_attr": page_attr,
                "dpi_stats": {"avg": avg_dpi, "status": dpi_status},
            }

    except Exception as e:
        return {"status": "error", "msg": str(e)}


def scan_directory(directory: str):
    """
    歷遍資料夾並執行分析，包含 Rich UI 互動。
    """
    target_dir = Path(directory)

    if not target_dir.exists() or not target_dir.is_dir():
        console.print(f"[error]錯誤：路徑 '{directory}' 不存在或不是資料夾！[/error]")
        return

    all_pdfs = []

    console.print(
        Panel(
            f"正在掃描資料夾：[bold cyan]{escape(str(target_dir))}[/bold cyan]",
            title="初始化",
            border_style="blue",
        )
    )

    with console.status(
        "[bold green]正在搜尋 PDF 檔案...[/bold green]", spinner="dots"
    ):
        all_pdfs = list(target_dir.rglob("*.pdf"))
        time.sleep(0.5)

    if not all_pdfs:
        console.print("[warning]未發現任何 PDF 檔案。[/warning]")
        return

    console.print(f"[info]共發現 {len(all_pdfs)} 個 PDF 檔案，準備開始分析...[/info]\n")

    results = []
    vector_count = 0
    raster_count = 0
    vector_draw_count = 0
    low_res_count = 0  # 低解析度計數
    error_count = 0

    progress_layout = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ]

    with Progress(*progress_layout, console=console) as progress:
        task = progress.add_task("[cyan]分析中...", total=len(all_pdfs))

        for pdf_path in all_pdfs:
            display_name = (
                pdf_path.name if len(pdf_path.name) < 30 else pdf_path.name[:27] + "..."
            )
            progress.update(task, description=f"[cyan]分析中: {escape(display_name)}")

            analysis = classify_pdf_type(pdf_path)

            result_entry = {
                "file": pdf_path.name,
                "path": str(pdf_path.relative_to(target_dir)),
                "analysis": analysis,
            }
            results.append(result_entry)

            if analysis["status"] == "success":
                t = analysis["type"]
                if t == "Vector":
                    vector_count += 1
                elif t == "Raster":
                    raster_count += 1
                elif t == "VectorDrawing":
                    vector_draw_count += 1

                # 統計低解析度 (僅針對有點陣圖的檔案)
                if analysis["dpi_stats"]["status"] == "Low Res" and (
                    t == "Raster" or analysis["dpi_stats"]["avg"] > 0
                ):
                    low_res_count += 1
            else:
                error_count += 1

            progress.advance(task)

    # 結果展示表格
    table = Table(title="PDF 深度審計報告", box=box.ROUNDED, show_lines=True)

    table.add_column("檔名", style="cyan", no_wrap=False)
    table.add_column("類型", justify="center")
    table.add_column("大小/頁數", justify="right")
    table.add_column("規格", justify="center")
    table.add_column("DPI (Avg)", justify="right")
    table.add_column("日期", justify="right", style="dim")

    for res in results:
        path_str = escape(res["path"])
        analysis = res["analysis"]

        if analysis["status"] == "success":
            # 準備數據
            type_str = analysis["type"]
            fp = analysis["fingerprint"]
            pa = analysis["page_attr"]
            dpi_info = analysis["dpi_stats"]

            size_page_str = f"{fp['size_str']}\n{pa['count']} 頁"
            date_str = f"C: {fp['created']}\nM: {fp['modified']}"

            # DPI 顯示
            dpi_val = dpi_info["avg"]
            dpi_display = f"{dpi_val} DPI" if dpi_val > 0 else "-"
            if dpi_info["status"] == "Low Res":
                dpi_display = f"[bold red]{dpi_display}[/bold red]"

            # 類型樣式
            if type_str == "Vector":
                type_style = "[bold blue]Vector[/bold blue]"
            elif type_str == "Raster":
                type_style = "[bold magenta]Raster[/bold magenta]"
            elif type_str == "VectorDrawing":
                type_style = "[bold yellow]Draw[/bold yellow]"
            else:
                type_style = "[dim]Unknown[/dim]"

            table.add_row(
                path_str, type_style, size_page_str, pa["size"], dpi_display, date_str
            )
        else:
            err_msg = escape(analysis["msg"])
            table.add_row(
                path_str,
                "[bold red]ERROR[/bold red]",
                "-",
                "-",
                "-",
                f"[red]{err_msg}[/red]",
            )

    console.print("\n")
    console.print(table)

    # 總結報告
    summary_text = (
        f"位置: [u]{escape(str(target_dir))}[/u] | 總數: [bold]{len(all_pdfs)}[/bold]\n"
        f"----------------------------------------\n"
        f"向量文字 (Vector) : [bold blue]{vector_count}[/bold blue]\n"
        f"點陣掃描 (Raster) : [bold magenta]{raster_count}[/bold magenta]\n"
        f"向量繪圖 (Draw)   : [bold yellow]{vector_draw_count}[/bold yellow]\n"
        f"----------------------------------------\n"
        f"低解析度警示 (<200): [bold red]{low_res_count}[/bold red]\n"
        f"分析失敗 (Error)  : [red]{error_count}[/red]"
    )

    console.print(
        Panel(
            summary_text,
            title="[bold green]審計完成[/bold green]",
            subtitle="PDF Analysis Tool v2.0",
            border_style="green",
            box=box.HEAVY_EDGE,
            padding=(1, 2),
        )
    )


if __name__ == "__main__":
    target_directory = "./data/吉輔提供資料Clean"
    scan_directory(target_directory)
