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

import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    track,
)

from src.logger import get_logger

logger = get_logger(__name__)

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_MAX_IO_WORKERS = 32


def split_dataset(
    source_root: str,
    output_root: str,
    run_name: str,
    split_ratio: float = 0.8,
    seed: int = 42,
    use_hardlinks: bool = False,
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
        use_hardlinks: True 時以 os.link 硬連結代替複製（同一檔案系統限定）。

    Returns:
        Tuple[int, int]: (n_train_stems, n_test_stems) 原始影像數量。
    """
    src = Path(source_root)
    dst_train = Path(output_root) / run_name / "Component_Dataset" / "train"
    dst_test = Path(output_root) / run_name / "Component_Dataset" / "test"
    dst_train.mkdir(parents=True, exist_ok=True)
    dst_test.mkdir(parents=True, exist_ok=True)

    # 偵測目錄結構
    # 1. 檢查是否為平坦結構 (src/stem/arr_*.png)
    root_stems = _discover_stems(src)
    if root_stems:
        logger.info("檢測到平坦結構 (無類別目錄)，共有 %d 個 stem", len(root_stems))
        return _perform_split({"_": root_stems}, dst_train, dst_test, split_ratio, seed, use_hardlinks)

    # 2. 檢查是否為類別結構 (src/class/stem/arr_*.png)
    class_dirs = sorted(d for d in src.iterdir() if d.is_dir())
    class_to_stems = {}
    for cd in class_dirs:
        stems = _discover_stems(cd)
        if stems:
            class_to_stems[cd.name] = stems

    if class_to_stems:
        logger.info("檢測到類別結構，共有 %d 個類別", len(class_to_stems))
        return _perform_split(class_to_stems, dst_train, dst_test, split_ratio, seed, use_hardlinks)

    # 3. 若都沒找到，嘗試平坦分割模式 (向下相容)
    logger.warning("未偵測到完整的 stem 結構，嘗試執行平坦檔案分割")
    return _flat_split(src, dst_train, dst_test, split_ratio, seed, use_hardlinks)


# ============================================================
# 分層抽樣分割
# ============================================================


def _perform_split(
    class_to_stems: Dict[str, List[Path]],
    dst_train: Path,
    dst_test: Path,
    split_ratio: float,
    seed: int,
    use_hardlinks: bool = False,
) -> Tuple[int, int]:
    """核心分割邏輯：對已分類的 stems 進行分層抽樣。"""
    rng = random.Random(seed)
    total_train_stems = 0
    total_test_stems = 0
    copy_tasks: List[Tuple[Path, Path]] = []

    for class_name, stems in track(class_to_stems.items(), description="[cyan]分割資料"):
        if not stems:
            continue

        shuffled = stems[:]
        rng.shuffle(shuffled)
        n_train = max(1, int(len(shuffled) * split_ratio))
        train_stems = shuffled[:n_train]
        test_stems = shuffled[n_train:]

        # 決定目的地子路徑
        # 如果 class_name 為 "_"，則直接存放在 dst 下 (不建立類別層級)
        def _get_dst_dir(root: Path, class_lvl: str, stem_lvl: str) -> Path:
            if class_lvl == "_":
                return root / stem_lvl
            return root / class_lvl / stem_lvl

        # 訓練集：所有 arr_* 變體
        for stem_dir in train_stems:
            target_dir = _get_dst_dir(dst_train, class_name, stem_dir.name)
            if target_dir.exists() and any(target_dir.iterdir()):
                continue
            variants = sorted(stem_dir.glob("arr_*.png"))
            if not variants:
                variants = [p for p in stem_dir.iterdir()
                            if p.suffix.lower() in _IMG_EXTS]
            target_dir.mkdir(parents=True, exist_ok=True)
            for v in variants:
                dst = target_dir / v.name
                if not dst.exists():
                    copy_tasks.append((v, dst))

        # 測試集：每個 stem 取代表性影像 (arr_000.png)
        for stem_dir in test_stems:
            target_dir = _get_dst_dir(dst_test, class_name, stem_dir.name)
            if target_dir.exists() and any(target_dir.iterdir()):
                continue
            arr_000 = stem_dir / "arr_000.png"
            if not arr_000.exists():
                candidates = sorted(stem_dir.glob("arr_*.png")) or [
                    p for p in stem_dir.iterdir() if p.suffix.lower() in _IMG_EXTS
                ]
                if candidates:
                    arr_000 = candidates[0]
                else:
                    continue
            target_dir.mkdir(parents=True, exist_ok=True)
            dst = target_dir / arr_000.name
            if not dst.exists():
                copy_tasks.append((arr_000, dst))

        total_train_stems += len(train_stems)
        total_test_stems += len(test_stems)

        if class_name != "_":
            logger.info(
                "  類別 %-30s train=%d stems, test=%d stems",
                class_name, len(train_stems), len(test_stems),
            )

    if copy_tasks:
        _parallel_copy(copy_tasks, use_hardlinks=use_hardlinks)

    logger.info(
        "分層分割完成 [%s]: train_stems=%d, test_stems=%d",
        dst_train.parent.name, total_train_stems, total_test_stems,
    )
    return total_train_stems, total_test_stems


def _parallel_copy(tasks: List[Tuple[Path, Path]], use_hardlinks: bool = False) -> None:
    """以 ThreadPoolExecutor 並行複製/硬連結檔案。"""
    def _do(src_dst: Tuple[Path, Path]) -> None:
        src, dst = src_dst
        if dst.exists():
            return
        if use_hardlinks:
            try:
                os.link(src, dst)
                return
            except OSError:
                pass
        shutil.copy2(src, dst)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        refresh_per_second=4,
    ) as progress:
        t = progress.add_task("複製/連結檔案", total=len(tasks))
        with ThreadPoolExecutor(max_workers=_MAX_IO_WORKERS) as pool:
            futs = [pool.submit(_do, task) for task in tasks]
            for fut in as_completed(futs):
                try:
                    fut.result()
                except Exception as exc:
                    logger.warning("檔案複製失敗: %s", exc)
                progress.advance(t)


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
    use_hardlinks: bool = False,
) -> Tuple[int, int]:
    """無類別子目錄時的隨機分割。"""
    images: List[Path] = sorted(
        Path(dp) / fn
        for dp, _, fns in os.walk(src)
        for fn in fns
        if Path(fn).suffix.lower() in _IMG_EXTS
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

    _copy_flat(train_imgs, dst_train, src, use_hardlinks)
    _copy_flat(test_imgs, dst_test, src, use_hardlinks)

    logger.info(
        "平坦分割完成: train=%d, test=%d (共 %d 張)",
        len(train_imgs),
        len(test_imgs),
        len(images),
    )
    return len(train_imgs), len(test_imgs)


def _copy_flat(images: List[Path], dst_dir: Path, src_root: Path, use_hardlinks: bool = False) -> None:
    tasks: List[Tuple[Path, Path]] = []
    for img in images:
        try:
            rel = img.relative_to(src_root)
        except ValueError:
            rel = Path(img.name)
        flat_name = "_".join(rel.parts)
        dst = dst_dir / flat_name
        if not dst.exists():
            tasks.append((img, dst))
    if tasks:
        _parallel_copy(tasks, use_hardlinks=use_hardlinks)
        if not dst.exists():
            tasks.append((img, dst))
    if tasks:
        _parallel_copy(tasks, use_hardlinks=use_hardlinks)
