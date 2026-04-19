"""資料集分割模組 (Dataset Splitter Module)。

============================================================
將前處理輸出依 split_ratio 分為 train / test，支援兩種模式：

1. **分層抽樣 (Stratified)**（預設）：
   - 以「原始影像 stem」為單位進行分割
   - 確保各類別在 train/test 中的比例與原始資料一致
   - 訓練集包含每個 stem 的所有 arr_* 變體
   - 測試集每個 stem 只取 arr_000.png（代表性影像）

2. **平坦分割 (Flat)**：
   - 無類別子目錄時的向下相容模式
   - 直接隨機分割所有影像

輸出結構::

    <output_root>/<run_name>/Component_Dataset/train/<class>/<stem>/arr_*.png
    <output_root>/<run_name>/Component_Dataset/test/<class>/<stem>/arr_000.png
============================================================
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from src.logger import get_logger

logger = get_logger(__name__)

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def split_dataset(
    source_root: str,
    output_root: str,
    run_name: str,
    split_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[int, int]:
    """分層抽樣分割：保持各類別比例，以 stem 為單位分割。

    自動偵測輸入目錄結構：
    - 若有子目錄 → 視為類別目錄，啟用分層抽樣
    - 若為平坦目錄 → 退化為一般隨機分割

    Args:
        source_root: 前處理輸出根目錄。
        output_root: 資料集輸出根目錄。
        run_name: Run 識別名稱。
        split_ratio: 訓練集佔比（0, 1）。
        seed: 隨機種子。

    Returns:
        Tuple[int, int]: (n_train_stems, n_test_stems) 原始影像數量。
    """
    src = Path(source_root)
    dst_train = Path(output_root) / run_name / "Component_Dataset" / "train"
    dst_test = Path(output_root) / run_name / "Component_Dataset" / "test"
    dst_train.mkdir(parents=True, exist_ok=True)
    dst_test.mkdir(parents=True, exist_ok=True)

    # 偵測是否有類別子目錄
    class_dirs = [d for d in src.iterdir() if d.is_dir()]
    has_classes = bool(class_dirs)

    if has_classes:
        return _stratified_split(src, dst_train, dst_test, split_ratio, seed)
    else:
        return _flat_split(src, dst_train, dst_test, split_ratio, seed)


# ============================================================
# 分層抽樣分割
# ============================================================


def _stratified_split(
    src: Path,
    dst_train: Path,
    dst_test: Path,
    split_ratio: float,
    seed: int,
) -> Tuple[int, int]:
    """以類別子目錄為基礎進行分層抽樣。"""
    class_dirs = sorted(d for d in src.iterdir() if d.is_dir())
    rng = random.Random(seed)

    total_train_stems = 0
    total_test_stems = 0

    for class_dir in class_dirs:
        class_name = class_dir.name
        stems = _discover_stems(class_dir)

        if not stems:
            logger.warning("類別 %s 無 stem 目錄，跳過", class_name)
            continue

        shuffled = stems[:]
        rng.shuffle(shuffled)
        n_train = max(1, int(len(shuffled) * split_ratio))
        train_stems = shuffled[:n_train]
        test_stems = shuffled[n_train:]

        # 訓練集：所有 arr_* 變體
        for stem_dir in train_stems:
            variants = sorted(stem_dir.glob("arr_*.png"))
            if not variants:
                variants = [p for p in stem_dir.iterdir()
                            if p.suffix.lower() in _IMG_EXTS]
            dst_class = dst_train / class_name / stem_dir.name
            dst_class.mkdir(parents=True, exist_ok=True)
            for v in variants:
                shutil.copy2(v, dst_class / v.name)

        # 測試集：每個 stem 取 arr_000.png（最具代表性）
        for stem_dir in test_stems:
            arr_000 = stem_dir / "arr_000.png"
            if not arr_000.exists():
                candidates = sorted(stem_dir.glob("arr_*.png"))
                if not candidates:
                    candidates = [p for p in stem_dir.iterdir()
                                  if p.suffix.lower() in _IMG_EXTS]
                if candidates:
                    arr_000 = candidates[0]
                else:
                    continue
            dst_class = dst_test / class_name / stem_dir.name
            dst_class.mkdir(parents=True, exist_ok=True)
            shutil.copy2(arr_000, dst_class / arr_000.name)

        total_train_stems += len(train_stems)
        total_test_stems += len(test_stems)

        logger.info(
            "  類別 %-30s train=%d stems, test=%d stems",
            class_name,
            len(train_stems),
            len(test_stems),
        )

    logger.info(
        "分層分割完成 [%s]: train_stems=%d, test_stems=%d",
        dst_train.parent.name,
        total_train_stems,
        total_test_stems,
    )
    return total_train_stems, total_test_stems


def _discover_stems(class_dir: Path) -> List[Path]:
    """找出類別目錄下所有 stem 子目錄（包含 arr_*.png 的目錄）。"""
    stems = []
    for d in sorted(class_dir.iterdir()):
        if d.is_dir():
            # stem 目錄：直接包含 arr_*.png 或任意影像
            imgs = list(d.glob("arr_*.png")) or [
                p for p in d.iterdir() if p.suffix.lower() in _IMG_EXTS
            ]
            if imgs:
                stems.append(d)
    return stems


# ============================================================
# 平坦分割（向下相容）
# ============================================================


def _flat_split(
    src: Path,
    dst_train: Path,
    dst_test: Path,
    split_ratio: float,
    seed: int,
) -> Tuple[int, int]:
    """無類別子目錄時的隨機分割。"""
    images: List[Path] = sorted(
        p for p in src.rglob("*") if p.suffix.lower() in _IMG_EXTS
    )

    if not images:
        logger.warning("分割來源目錄無影像: %s", src)
        return 0, 0

    rng = random.Random(seed)
    shuffled = images[:]
    rng.shuffle(shuffled)

    n_train = max(1, int(len(shuffled) * split_ratio))
    train_imgs = shuffled[:n_train]
    test_imgs = shuffled[n_train:]

    _copy_flat(train_imgs, dst_train, src)
    _copy_flat(test_imgs, dst_test, src)

    logger.info(
        "平坦分割完成: train=%d, test=%d (共 %d 張)",
        len(train_imgs),
        len(test_imgs),
        len(images),
    )
    return len(train_imgs), len(test_imgs)


def _copy_flat(images: List[Path], dst_dir: Path, src_root: Path) -> None:
    for img in images:
        try:
            rel = img.relative_to(src_root)
        except ValueError:
            rel = Path(img.name)
        flat_name = "_".join(rel.parts)
        shutil.copy2(img, dst_dir / flat_name)
