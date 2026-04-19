"""壓縮檔解壓縮模組 (Archive Extraction Module)。"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Optional

from src.logger import get_logger

logger = get_logger(__name__)


def extract_archive(
    archive_path: Optional[str],
    output_dir: str,
    skip: bool = False,
) -> None:
    """解壓縮 ZIP 壓縮檔至指定目錄。

    Args:
        archive_path: 壓縮檔路徑；``None`` 表示無壓縮檔，直接跳過。
        output_dir: 解壓目標目錄。
        skip: ``True`` 強制跳過此步驟。
    """
    if skip:
        logger.info("跳過解壓縮 (skip=True)")
        return
    if archive_path is None:
        logger.info("raw_zip_path 為 None，跳過解壓縮")
        return

    src = Path(archive_path)
    if not src.is_file():
        raise FileNotFoundError(f"壓縮檔不存在: {src}")

    dst = Path(output_dir)
    dst.mkdir(parents=True, exist_ok=True)

    suffix = src.suffix.lower()
    if suffix == ".zip":
        _extract_zip(src, dst)
    else:
        raise ValueError(
            f"不支援的壓縮格式: '{suffix}'。目前僅支援 .zip"
        )


def _extract_zip(src: Path, dst: Path) -> None:
    with zipfile.ZipFile(src, "r") as zf:
        members = zf.namelist()
        zf.extractall(dst)
    logger.info(
        "ZIP 解壓完成: %s → %s (%d 個項目)",
        src,
        dst,
        len(members),
    )
