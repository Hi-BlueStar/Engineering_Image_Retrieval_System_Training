"""影像資料集模組 (Image Dataset Module)。

============================================================
提供兩種資料集類別：

1. **SingleViewDataset**（推薦，GPU 增強模式）：
   - 每次 __getitem__ 回傳單張 resize 後的 raw tensor [C, H, W]
   - Trainer 在 GPU 上呼叫 GPUAugmentation.create_views() 生成雙視角
   - 大幅降低 CPU worker 負擔，解決 CPU 瓶頸
   - 支援三種快取模式：full / partitioned / none（auto 自動選擇）

2. **UnlabeledImageDataset**（向下相容，CPU 增強模式）：
   - 每次 __getitem__ 回傳 (view1, view2) CPU 增強後的雙視角
   - 保留原有行為，供 use_gpu_augmentation=False 時使用
============================================================
"""

from __future__ import annotations

import concurrent.futures
import math
import os
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import psutil
import torch
import torchvision.transforms as T
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from torch.utils.data import Dataset

from src.logger import get_logger

logger = get_logger(__name__)

_IMG_EXTS = {".jpg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# 可用 RAM 中實際可安全使用的上限比例（預留系統餘裕）
_SAFE_RAM_HEADROOM = 0.85


def _probe_available_gb() -> float:
    """回傳目前系統可用 RAM（GB）。"""
    return psutil.virtual_memory().available / 1e9


def _select_cache_mode(n: int, c: int, s: int, memory_fraction: float) -> str:
    """根據資料集大小與可用記憶體自動選擇快取策略。

    Args:
        n: 影像數量。
        c: 通道數。
        s: 影像邊長。
        memory_fraction: 允許佔用可用 RAM 的最大比例。

    Returns:
        ``"full"``、``"partitioned"`` 或 ``"none"``。
    """
    available_gb = _probe_available_gb()
    full_gb = n * c * s * s * 2 / 1e9  # float16 = 2 bytes
    budget_gb = available_gb * min(memory_fraction, _SAFE_RAM_HEADROOM)

    if full_gb <= budget_gb:
        mode = "full"
    elif full_gb > available_gb * 0.05:
        mode = "partitioned"
    else:
        mode = "none"

    logger.info(
        "cache auto-select: available=%.1fGB, full_size=%.1fGB, budget=%.1fGB → mode=%s",
        available_gb, full_gb, budget_gb, mode,
    )
    return mode


class Letterbox:
    """等比例縮放並填充 (Letterboxing)，確保影像不變形。

    Args:
        size: 目標正方形邊長。
        fill: 填充顏色（預設 255 為白色）。
    """

    def __init__(self, size: int, fill: int = 255) -> None:
        self.size = size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img.resize((self.size, self.size), Image.Resampling.BILINEAR)

        scale = self.size / max(w, h)
        nw, nh = int(w * scale), int(h * scale)

        img = img.resize((nw, nh), Image.Resampling.BILINEAR)
        new_img = Image.new(img.mode, (self.size, self.size), self.fill)
        new_img.paste(img, ((self.size - nw) // 2, (self.size - nh) // 2))
        return new_img


class SingleViewDataset(Dataset):
    """GPU 增強模式資料集：回傳 resize 後的單張 raw tensor。

    Worker 只做最輕量的工作（I/O + resize + ToTensor），
    增強在 GPU 上由 GPUAugmentation 完成。

    快取模式（``cache_mode``）：
        - ``"auto"``（預設）：根據可用 RAM 自動選擇以下三種之一。
        - ``"full"``：整個資料集存入單一 shared_memory float16 tensor。
          消除每 epoch disk I/O；記憶體 = N*C*H*W*2 bytes。
        - ``"partitioned"``：依可用 RAM 分成 P 個分區，邊訓練邊輪換載入。
          RAM 峰值 ≈ budget_gb；支援背景預載下一分區。
        - ``"none"``：不快取，每次存取皆從磁碟讀取。

    Args:
        root: 影像根目錄（遞迴搜尋）。
        img_size: 輸出影像邊長（正方形，僅做 resize）。
        img_exts: 支援的影像副檔名列表。
        in_channels: 輸入通道數；``1`` 載入灰階，``3`` 載入 RGB。
        cache_mode: 快取策略（見上方說明）。
        memory_fraction: auto/partitioned 模式下，可用 RAM 的最大使用比例。
        cache_in_memory: 向下相容參數；``True`` 強制 ``"full"``，
            ``False`` 強制 ``"none"``；優先於 ``cache_mode``。
    """

    def __init__(
        self,
        root: Path,
        img_size: int,
        img_exts: List[str],
        in_channels: int = 1,
        cache_mode: str = "auto",
        memory_fraction: float = 0.6,
        cache_in_memory: Optional[bool] = None,
    ) -> None:
        self.root = root
        self.img_size = img_size
        self.in_channels = in_channels
        self._mode = "L" if in_channels == 1 else "RGB"
        self._letterbox = Letterbox(img_size, fill=255)
        self.images = self._scan(img_exts)
        self._transform = T.Compose([
            self._letterbox,
            T.ToTensor(),
        ])

        # 向下相容：cache_in_memory 優先覆寫 cache_mode
        if cache_in_memory is not None:
            cache_mode = "full" if cache_in_memory else "none"

        # auto 模式：根據可用記憶體決定策略
        n = len(self.images)
        c = self.in_channels
        s = self.img_size
        if cache_mode == "auto":
            cache_mode = _select_cache_mode(n, c, s, memory_fraction)

        self._resolved_mode = cache_mode
        self._memory_fraction = memory_fraction

        # 分區狀態（partitioned 模式使用）
        self._partition_offsets: List[int] = []
        self._partition_idx: int = 0
        self._next_cache: Optional[torch.Tensor] = None
        self._preload_event = threading.Event()
        self._preload_lock = threading.Lock()

        # 初始化快取
        self._cache: Optional[torch.Tensor] = None
        if cache_mode == "full":
            self._cache = self._build_full_cache()
        elif cache_mode == "partitioned":
            self._init_partitioned_cache()
        # "none"：_cache 保持 None

        logger.info(
            "SingleViewDataset: root=%s, n=%d, mode=%s, cache_mode=%s, partitions=%d",
            root, n, self._mode, self._resolved_mode, self.n_partitions,
        )

    # ------------------------------------------------------------------
    # 掃描
    # ------------------------------------------------------------------

    def _scan(self, img_exts: List[str]) -> List[Path]:
        ext_set = {e.lower() for e in img_exts}
        return sorted(p for p in self.root.rglob("*") if p.suffix.lower() in ext_set)

    # ------------------------------------------------------------------
    # 全快取（原有邏輯，加入記憶體預檢）
    # ------------------------------------------------------------------

    def _build_full_cache(self) -> torch.Tensor:
        """一次性解碼 + letterbox 為單一連續 float16 tensor。

        在分配前先檢查可用 RAM；不足則 raise RuntimeError（呼叫方應
        改用 partitioned 或 none 模式，而非靜默 OOM）。
        """
        n = len(self.images)
        c = self.in_channels
        s = self.img_size
        expected_gb = n * c * s * s * 2 / 1e9

        available_gb = _probe_available_gb()
        budget_gb = available_gb * min(self._memory_fraction, _SAFE_RAM_HEADROOM)
        if expected_gb > budget_gb:
            raise RuntimeError(
                f"全快取需要 {expected_gb:.2f} GB RAM，但可用預算僅 {budget_gb:.2f} GB "
                f"（可用={available_gb:.2f} GB × fraction={self._memory_fraction}）。"
                "請改用 cache_mode='partitioned' 或 cache_mode='none'。"
            )

        logger.info(
            "SingleViewDataset: 開始 in-RAM 解碼快取 (n=%d, shape=[N,%d,%d,%d], "
            "預期 %.2f GB, root=%s)",
            n, c, s, s, expected_gb, self.root,
        )
        cache = self._load_images_parallel(list(range(n)), n)
        cache.share_memory_()
        logger.info(
            "SingleViewDataset: 全快取完成 — %d 張, %.2f GB",
            n, cache.numel() * 2 / 1e9,
        )
        return cache

    # ------------------------------------------------------------------
    # 分區快取
    # ------------------------------------------------------------------

    def _init_partitioned_cache(self) -> None:
        """初始化分區快取：計算分區邊界，載入分區 0。"""
        n = len(self.images)
        c = self.in_channels
        s = self.img_size
        full_gb = n * c * s * s * 2 / 1e9

        available_gb = _probe_available_gb()
        budget_gb = available_gb * min(self._memory_fraction, _SAFE_RAM_HEADROOM)
        bytes_per_img = c * s * s * 2  # float16
        partition_n = max(1, int(budget_gb * 1e9 / bytes_per_img))
        n_parts = math.ceil(n / partition_n)

        # 建立分區起始索引（含 sentinel）
        self._partition_offsets = list(range(0, n, partition_n)) + [n]
        self._partition_idx = 0

        logger.info(
            "SingleViewDataset: 分區快取初始化 — n=%d, full=%.1fGB, budget=%.1fGB, "
            "partition_n=%d, n_partitions=%d",
            n, full_gb, budget_gb, partition_n, n_parts,
        )

        # 同步載入第 0 分區
        t0 = time.perf_counter()
        self._cache = self._load_partition_sync(0)
        logger.info(
            "SingleViewDataset: 分區 0 載入完成 (%.1fs)",
            time.perf_counter() - t0,
        )

    def _load_partition_sync(self, p_idx: int) -> torch.Tensor:
        """同步載入分區 p_idx，回傳 share_memory_ float16 tensor。"""
        start = self._partition_offsets[p_idx]
        end = self._partition_offsets[p_idx + 1]
        indices = list(range(start, end))
        part_n = end - start
        tensor = self._load_images_parallel(indices, part_n)
        tensor.share_memory_()
        return tensor

    def get_partition_indices(self, p_idx: int) -> List[int]:
        """回傳分區 p_idx 的全域影像索引列表。"""
        start = self._partition_offsets[p_idx]
        end = self._partition_offsets[p_idx + 1]
        return list(range(start, end))

    def activate_partition(self, p_idx: int) -> None:
        """切換至分區 p_idx：若背景預載已完成則直接 swap，否則同步載入。"""
        if self._partition_idx == p_idx:
            return

        with self._preload_lock:
            if self._preload_event.is_set() and self._next_cache is not None:
                # 背景預載已完成，原子 swap
                self._cache = self._next_cache
                self._next_cache = None
                self._preload_event.clear()
                logger.info("分區 %d 已由背景預載切換", p_idx)
            else:
                # 背景尚未完成（或未啟動），同步載入
                logger.info("分區 %d 同步載入中…", p_idx)
                t0 = time.perf_counter()
                self._cache = self._load_partition_sync(p_idx)
                logger.info("分區 %d 同步載入完成 (%.1fs)", p_idx, time.perf_counter() - t0)

        self._partition_idx = p_idx

    def preload_partition_async(self, p_idx: int) -> None:
        """在背景執行緒中預載分區 p_idx（非阻塞）。"""
        if p_idx >= self.n_partitions:
            return

        def _worker():
            logger.info("背景預載分區 %d 開始…", p_idx)
            t0 = time.perf_counter()
            tensor = self._load_partition_sync(p_idx)
            with self._preload_lock:
                self._next_cache = tensor
                self._preload_event.set()
            logger.info("背景預載分區 %d 完成 (%.1fs)", p_idx, time.perf_counter() - t0)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    # ------------------------------------------------------------------
    # 共用：並行讀取影像
    # ------------------------------------------------------------------

    def _load_images_parallel(self, indices: List[int], total: int) -> torch.Tensor:
        """並行載入指定索引的影像，回傳 float16 tensor [total, C, H, W]。"""
        c = self.in_channels
        s = self.img_size
        mode = self._mode
        letterbox = self._letterbox
        images = self.images

        cache = torch.empty((total, c, s, s), dtype=torch.float16)
        log_every = max(1, total // 10)

        def _load_one(local_i: int, global_i: int) -> Tuple[int, torch.Tensor]:
            img = Image.open(images[global_i]).convert(mode)
            img = letterbox(img)
            arr = np.array(img, dtype=np.float32)
            arr /= 255.0
            t = torch.from_numpy(arr)
            if t.ndim == 2:
                t = t.unsqueeze(0)
            else:
                t = t.permute(2, 0, 1)
            return local_i, t.to(torch.float16)

        max_workers = min(16, os.cpu_count() or 4)
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_load_one, local_i, global_i): local_i
                for local_i, global_i in enumerate(indices)
            }
            for future in concurrent.futures.as_completed(futures):
                local_i, t = future.result()
                cache[local_i] = t
                completed += 1
                if completed % log_every == 0 or completed == total:
                    logger.info("  載入進度: %d/%d (%.0f%%)", completed, total, 100.0 * completed / total)

        return cache

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_partitions(self) -> int:
        """分區總數；非分區模式回傳 1。"""
        if self._resolved_mode == "partitioned" and self._partition_offsets:
            return len(self._partition_offsets) - 1
        return 1

    # ------------------------------------------------------------------
    # Dataset 介面
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self._cache is not None:
            if self._resolved_mode == "partitioned":
                # 分區模式：將全域 idx 轉換為分區內部偏移
                local_idx = idx - self._partition_offsets[self._partition_idx]
                if 0 <= local_idx < self._cache.shape[0]:
                    return self._cache[local_idx].float()
                # 越界防禦：fallback 到磁碟（不應發生，但避免靜默錯誤）
                logger.warning(
                    "分區邊界越界：global_idx=%d, partition=%d [%d,%d)，改由磁碟載入",
                    idx, self._partition_idx,
                    self._partition_offsets[self._partition_idx],
                    self._partition_offsets[self._partition_idx + 1],
                )
            else:
                # 全快取：直接索引
                return self._cache[idx].float()

        # 無快取 / 分區 fallback：原 PIL 路徑
        img = Image.open(self.images[idx]).convert(self._mode)
        return self._transform(img)


class UnlabeledImageDataset(Dataset):
    """CPU 增強模式資料集（向下相容）：回傳 (view1, view2) 增強雙視角。

    Args:
        root: 影像根目錄（遞迴搜尋）。
        img_exts: 支援的影像副檔名列表（含點，例如 ``[".png"]``）。
        transform: 接受 PIL 影像並回傳 ``(view1, view2)`` 的可呼叫物件。
        in_channels: 輸入通道數；``1`` 載入灰階，其他值載入 RGB。
    """

    def __init__(
        self,
        root: Path,
        img_exts: List[str],
        transform,
        in_channels: int = 1,
    ) -> None:
        self.root = root
        self.transform = transform
        self._mode = "L" if in_channels == 1 else "RGB"
        self.images = self._scan(img_exts)

        logger.info(
            "UnlabeledImageDataset: root=%s, n=%d, mode=%s",
            root, len(self.images), self._mode,
        )

    def _scan(self, img_exts: List[str]) -> List[Path]:
        ext_set = {e.lower() for e in img_exts}
        return sorted(p for p in self.root.rglob("*") if p.suffix.lower() in ext_set)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.images[idx]).convert(self._mode)
        return self.transform(img)
