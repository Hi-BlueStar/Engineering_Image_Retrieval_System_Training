import os
import time
import fitz  # PyMuPDF
from pathlib import Path
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn
)
from rich.table import Table
from rich.panel import Panel
from rich.markup import escape
from rich import box
from rich.theme import Theme

# 設定自定義的主題風格
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "vector": "blue",
    "raster": "magenta"
})

console = Console(theme=custom_theme)

def classify_pdf_type(file_path: Path, page_sample_limit: int = 5) -> dict:
    """
    分析 PDF 檔案類型的核心邏輯。
    """
    try:
        # 使用 context manager 確保檔案會被關閉
        with fitz.open(file_path) as doc:
            total_pages = len(doc)
            
            if total_pages == 0:
                return {
                    "status": "success",
                    "type": "Empty",
                    "ratio": 0.0,
                    "reason": "PDF 無頁面"
                }

            pages_to_check = min(total_pages, page_sample_limit) if page_sample_limit else total_pages
            text_content_score = 0
            image_content_score = 0
            
            for i in range(pages_to_check):
                page = doc.load_page(i)
                text = page.get_text().strip()
                images = page.get_images()
                
                # 判定邏輯 (同前次對話)
                if len(text) > 50:
                    text_content_score += 1
                elif len(text) < 10 and len(images) > 0:
                    image_content_score += 1
            
            text_ratio = text_content_score / pages_to_check
            
            # 根據分數決定類型
            if text_ratio > 0.2:
                file_type = "Vector"
                reason = "偵測到顯著文字層"
            elif image_content_score > 0:
                file_type = "Raster"
                reason = "文字極少，以圖片為主"
            else:
                file_type = "Unknown"
                reason = "無文字且無顯著圖片"

            return {
                "status": "success",
                "type": file_type,
                "ratio": round(text_ratio, 2),
                "reason": reason
            }

    except Exception as e:
        # 捕捉所有異常並回傳錯誤訊息
        return {
            "status": "error",
            "msg": str(e)
        }

def scan_directory(directory: str):
    """
    歷遍資料夾並執行分析，包含 Rich UI 互動。
    """
    target_dir = Path(directory)
    
    if not target_dir.exists() or not target_dir.is_dir():
        console.print(f"[error]錯誤：路徑 '{directory}' 不存在或不是資料夾！[/error]")
        return

    all_pdfs = []
    
    # 1. 初始化階段：掃描檔案列表 (使用 Spinner)
    console.print(Panel(f"正在掃描資料夾：[bold cyan]{escape(str(target_dir))}[/bold cyan]", title="初始化", border_style="blue"))
    
    with console.status("[bold green]正在搜尋 PDF 檔案...[/bold green]", spinner="dots"):
        # rglob 會遞迴搜尋所有子資料夾
        all_pdfs = list(target_dir.rglob("*.pdf"))
        # 模擬一點延遲讓使用者看得到動畫 (若檔案很少)
        time.sleep(0.5)

    if not all_pdfs:
        console.print("[warning]未發現任何 PDF 檔案。[/warning]")
        return

    console.print(f"[info]共發現 {len(all_pdfs)} 個 PDF 檔案，準備開始分析...[/info]\n")

    results = []
    vector_count = 0
    raster_count = 0
    error_count = 0

    # 2. 執行過程：處理檔案 (使用 Progress Bar)
    # 自定義進度條樣式
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
            # 顯示當前正在處理的檔名 (截斷過長的檔名)
            display_name = pdf_path.name if len(pdf_path.name) < 30 else pdf_path.name[:27] + "..."
            progress.update(task, description=f"[cyan]分析中: {escape(display_name)}")
            
            # 執行分析
            analysis = classify_pdf_type(pdf_path)
            
            # 儲存結果
            result_entry = {
                "file": pdf_path.name,
                "path": str(pdf_path.relative_to(target_dir)), # 顯示相對路徑
                "analysis": analysis
            }
            results.append(result_entry)

            # 統計數據
            if analysis["status"] == "success":
                if analysis["type"] == "Vector":
                    vector_count += 1
                elif analysis["type"] == "Raster":
                    raster_count += 1
            else:
                error_count += 1
            
            # 更新進度
            progress.advance(task)

    # 3. 結果展示：繪製表格
    table = Table(title="PDF 類型分析報告", box=box.ROUNDED, show_lines=True)

    table.add_column("檔名 (相對路徑)", style="cyan", no_wrap=False)
    table.add_column("類型", justify="center")
    table.add_column("文字覆蓋率", justify="right")
    table.add_column("備註 / 錯誤訊息", style="white")

    for res in results:
        path_str = escape(res["path"])
        analysis = res["analysis"]
        
        if analysis["status"] == "success":
            type_str = analysis["type"]
            ratio_str = f"{analysis['ratio'] * 100:.0f}%"
            reason_str = analysis["reason"]

            # 根據類型上色
            if type_str == "Vector":
                type_style = "[bold blue]向量 (Vector)[/bold blue]"
            elif type_str == "Raster":
                type_style = "[bold magenta]點陣 (Raster)[/bold magenta]"
            else:
                type_style = "[dim]未知 (Unknown)[/dim]"
                
            table.add_row(path_str, type_style, ratio_str, reason_str)
        else:
            # 錯誤處理顯示
            err_msg = escape(analysis["msg"])
            table.add_row(
                path_str, 
                "[bold red]ERROR[/bold red]", 
                "-", 
                f"[bold red]{err_msg}[/bold red]"
            )

    console.print("\n")
    console.print(table)

    # 4. 總結報告面板
    summary_text = (
        f"掃描位置: [u]{escape(str(target_dir))}[/u]\n"
        f"總檔案數: [bold]{len(all_pdfs)}[/bold]\n"
        f"----------------------------------\n"
        f"向量 PDF (Vector): [bold blue]{vector_count}[/bold blue]\n"
        f"點陣 PDF (Raster): [bold magenta]{raster_count}[/bold magenta]\n"
        f"分析失敗 (Error) : [bold red]{error_count}[/bold red]"
    )

    console.print(Panel(
        summary_text, 
        title="[bold green]執行完畢[/bold green]", 
        subtitle="PDF Analysis Tool",
        border_style="green",
        box=box.HEAVY_EDGE, # 重線風格
        padding=(1, 2)
    ))

if __name__ == "__main__":
    # 在此輸入您想要掃描的資料夾路徑，"." 代表當前目錄
    target_directory = "." 
    scan_directory(target_directory)