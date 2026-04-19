"""資料集分割模組 (Dataset Splitter Module)。

將前處理輸出依 split_ratio 分為 train / val，
並按 run_name 建立獨立的目錄結構，確保多 Run 可重現。

輸出結構::

    <output_root>/<run_name>/Component_Dataset/train/
    <output_root>/<run_name>/Component_Dataset/val/
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import List

from src.logger import get_logger

logger = get_logger(__name__)

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def split_dataset(
    source_root: str,
    output_root: str,
    run_name: str,
    split_ratio: float = 0.8,
    seed: int = 42,
) -> None:
    """將前處理影像分割為 train / val，並複製至指定 Run 目錄。

    Args:
        source_root: 前處理輸出根目錄（遞迴掃描所有影像）。
        output_root: 資料集輸出根目錄。
        run_name: Run 識別名稱（例如 ``"Run_01_Seed_42"``）。
        split_ratio: 訓練集佔比（0, 1）。
        seed: 隨機種子，確保可重現。
    """
    src = Path(source_root)
    dst_train = Path(output_root) / run_name / "Component_Dataset" / "train"
    dst_val = Path(output_root) / run_name / "Component_Dataset" / "val"
    dst_train.mkdir(parents=True, exist_ok=True)
    dst_val.mkdir(parents=True, exist_ok=True)

    images: List[Path] = sorted(
        p for p in src.rglob("*") if p.suffix.lower() in _IMG_EXTS
    )

    if not images:
        logger.warning("分割來源目錄無影像: %s", src)
        return

    rng = random.Random(seed)
    shuffled = images[:]
    rng.shuffle(shuffled)

    n_train = max(1, int(len(shuffled) * split_ratio))
    train_imgs = shuffled[:n_train]
    val_imgs = shuffled[n_train:]

    _copy_with_unique_names(train_imgs, dst_train, src)
    _copy_with_unique_names(val_imgs, dst_val, src)

    logger.info(
        "資料集分割完成 [%s]: train=%d, val=%d (共 %d 張)",
        run_name,
        len(train_imgs),
        len(val_imgs),
        len(images),
    )


def _copy_with_unique_names(
    images: List[Path],
    dst_dir: Path,
    src_root: Path,
) -> None:
    """複製影像至目標目錄，以相對路徑去除分隔符確保名稱唯一。"""
    for img in images:
        try:
            rel = img.relative_to(src_root)
        except ValueError:
            rel = Path(img.name)

        # 將相對路徑拼接為單一檔名（避免子目錄衝突）
        flat_name = "_".join(rel.parts)
        dst_path = dst_dir / flat_name
        shutil.copy2(img, dst_path)
