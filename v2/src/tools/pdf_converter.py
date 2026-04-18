"""PDF 轉影像工具模組。

負責將原始的 PDF 目錄，轉換為可供神經網路讀取的 PNG 圖片。
"""
import concurrent.futures
import os
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd


def _pdf_worker(pdf_path_str: str, root_dir_str: str, out_dir_str: str, dpi: int) -> dict:
    """單點 PDF 轉換工作常式 (Process-safe)。"""
    import fitz  # PyMuPDF
    
    pdf_path = Path(pdf_path_str)
    root_dir = Path(root_dir_str)
    out_dir = Path(out_dir_str)
    scale = dpi / 72.0
    
    try:
        class_label = pdf_path.parent.name
        rel_dir = pdf_path.parent.relative_to(root_dir)
        fname = f"{pdf_path.stem.replace(' ', '_')}.png"
        rel_path = rel_dir / fname
        dest_abs = out_dir / rel_path
        
        dest_abs.parent.mkdir(parents=True, exist_ok=True)
        
        with fitz.open(pdf_path) as doc:
            page = doc.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            pix.save(dest_abs.as_posix())
            
        return {"status": "success", "source_pdf": str(pdf_path), "class_label": class_label, "image_path": rel_path.as_posix()}
    except Exception as e:
        return {"status": "error", "file": str(pdf_path), "error": str(e)}


class PDFConverter:
    def __init__(self, raw_pdf_dir: str, converted_image_dir: str, dpi: int = 100, max_workers: int = 16):
        self.raw_pdf_dir = Path(raw_pdf_dir)
        self.converted_image_dir = Path(converted_image_dir)
        self.dpi = dpi
        self.max_workers = max_workers or max(1, (os.cpu_count() or 4) - 1)

    def run(self) -> Tuple[int, int]:
        """執行轉換。
        
        Returns:
            Tuple[int, int]: (成功數量, 失敗數量)
        """
        if not self.raw_pdf_dir.exists():
            print(f"[Warning] PDF 來源目錄不存在: {self.raw_pdf_dir}")
            return 0, 0

        pdf_files = list(self.raw_pdf_dir.rglob("*.pdf"))
        if not pdf_files:
            print("[Warning] 未發現任何 PDF 檔案。")
            return 0, 0

        self.converted_image_dir.mkdir(parents=True, exist_ok=True)
        
        success_rows = []
        failed_count = 0

        print(f"🖼 開始轉換 {len(pdf_files)} 個 PDF (DPI={self.dpi}, Workers={self.max_workers})...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(_pdf_worker, str(p), str(self.raw_pdf_dir), str(self.converted_image_dir), self.dpi) for p in pdf_files]
            
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                res = future.result()
                if res["status"] == "success":
                    success_rows.append((res["source_pdf"], res["class_label"], res["image_path"]))
                else:
                    failed_count += 1
                    
                if i % 100 == 0:
                    print(f"   已完成: {i}/{len(pdf_files)}")

        df = pd.DataFrame(success_rows, columns=["source_pdf", "class_label", "image_path"])
        df.to_csv(self.converted_image_dir / "manifest.csv", index=False)
        print(f"✔ 轉換完成：成功 {len(success_rows)}, 失敗 {failed_count}")
        return len(success_rows), failed_count
