"""PDF 轉影像模組 (PDF to Image Converter)。

使用 PyMuPDF (fitz) 將 PDF 每頁轉換為灰階 PNG。
多頁 PDF 以 _p000, _p001 … 後綴區分。
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import fitz  # PyMuPDF
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from src.logger import get_logger

logger = get_logger(__name__)


def convert_pdfs_to_images(
    pdf_dir: str,
    output_dir: str,
    dpi: int = 100,
    max_workers: int = 16,
    skip: bool = False,
    preserve_structure: bool = False,
) -> None:
    """將目錄內所有 PDF 批次轉換為灰階 PNG。

    Args:
        pdf_dir: 包含 PDF 的來源目錄（遞迴搜尋）。
        output_dir: PNG 輸出目錄。
        dpi: 轉換解析度。
        max_workers: 執行緒並行數。
        skip: ``True`` 強制跳過此步驟。
        preserve_structure: ``True`` 保留原始目錄結構；``False`` 平鋪輸出。
    """
    if skip:
        logger.info("跳過 PDF 轉換 (skip=True)")
        return

    src_dir = Path(pdf_dir)
    dst_dir = Path(output_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    pdf_files: List[Path] = sorted(
        Path(dp) / fn
        for dp, _, fns in os.walk(src_dir)
        for fn in fns
        if fn.lower().endswith(".pdf")
    )

    if not pdf_files:
        logger.warning("PDF 目錄中無 PDF 檔: %s", src_dir)
        return

    logger.info("開始 PDF 轉換: %d 個 PDF，DPI=%d, preserve_structure=%s", 
                len(pdf_files), dpi, preserve_structure)

    args = []
    for p in pdf_files:
        if preserve_structure:
            rel_path = p.parent.relative_to(src_dir)
            target_dst = dst_dir / rel_path
            target_dst.mkdir(parents=True, exist_ok=True)
        else:
            target_dst = dst_dir
        args.append((p, target_dst, dpi))
    success = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        refresh_per_second=4,
    ) as progress:
        task = progress.add_task("PDF 轉換", total=len(args))
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_convert_one, *a): a[0] for a in args}
            for fut in as_completed(futures):
                pdf_path = futures[fut]
                try:
                    n = fut.result()
                    success += 1
                    if n > 0:
                        logger.debug("轉換完成: %s (%d 頁)", pdf_path.name, n)
                except Exception as exc:
                    logger.error("轉換失敗: %s — %s", pdf_path.name, exc)
                progress.advance(task)

    logger.info("PDF 轉換完成: %d/%d 成功，輸出至 %s", success, len(pdf_files), dst_dir)


def _convert_one(pdf_path: Path, dst_dir: Path, dpi: int) -> int:
    """轉換單一 PDF，回傳轉換頁數；若輸出已存在回傳 0（跳過）。"""
    # 斷點恢復：單頁或多頁第一頁已存在則跳過
    if (dst_dir / f"{pdf_path.stem}.png").exists() or \
       (dst_dir / f"{pdf_path.stem}_p000.png").exists():
        return 0

    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    doc = fitz.open(str(pdf_path))
    n_pages = len(doc)

    for idx, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
        suffix = f"_p{idx:03d}" if n_pages > 1 else ""
        out_path = dst_dir / f"{pdf_path.stem}{suffix}.png"
        pix.save(str(out_path))

    doc.close()
    return n_pages
