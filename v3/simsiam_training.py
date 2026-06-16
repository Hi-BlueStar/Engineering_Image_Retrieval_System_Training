#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SimSiam 自監督學習工程圖訓練與評估 (v3) - 腳本化版本
本腳本根據《實驗計畫書.pdf》與 v3/simsiam_training.ipynb 設計，
將資料讀取、解壓、PDF轉換、前處理、資料分割、模型架構、優化器、學習率、訓練與驗證 Loss 收斂指標等
全部修改為與 src/main_training.py 完全一致，僅保留 Kornia GPUAugmentation 在 v3 中進行自監督特徵學習。
"""

import os
import sys
import time
import random
import json
import contextlib
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, asdict

# --- 專案路徑與衝突 import 處理機制 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 1. 確保根目錄在 sys.path 中
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 2. 預先載入 parent src 的重要模組
import src.split_dataset
from src.training.experiment_logger import ExperimentLogger
from src.training.timer import PrecisionTimer, TimerCollection

# 3. 載入 v3/src/logger 並立即對應，防範子模組導入時發生錯誤
import v3.src.logger
sys.modules['src.logger'] = v3.src.logger

import v3.src.data.logo_removal
import v3.src.data.topology

sys.modules['src.data'] = v3.src.data
sys.modules['src.data.logo_removal'] = v3.src.data.logo_removal
sys.modules['src.data.topology'] = v3.src.data.topology

# 4. 載入其餘 v3.src 子模組
import v3.src.data.extraction
import v3.src.data.pdf_converter
import v3.src.data.preprocessing
import v3.src.data.splitter

# 5. 動態對應所有已載入的 v3.src 子模組至 src
for name, module in list(sys.modules.items()):
    if name.startswith('v3.src'):
        mapped_name = name.replace('v3.src', 'src', 1)
        if mapped_name not in sys.modules:
            sys.modules[mapped_name] = module

# 6. 載入 v3 資料處理模組
from v3.src.data.extraction import extract_archive
from v3.src.data.pdf_converter import convert_pdfs_to_images
from v3.src.data.preprocessing import PreprocessConfig, preprocess_images
from v3.src.data.splitter import split_dataset

# --- 載入剩餘第三方套件 ---
import numpy as np
import cv2
import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models

from rich.console import Console
from rich.markup import escape
from rich import box
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.traceback import install

# 安裝 Rich 的錯誤追蹤，初始化 Console
install(show_locals=False)
console = Console()


# -----------------------------------------------------------------------------
# 1. 訓練與資料設定 (Configuration)
# -----------------------------------------------------------------------------

@dataclass
class Config:
    # --- 資料來源設定 ---
    raw_zip_path: str = "data/吉輔提供資料.zip"
    raw_pdf_dir: str = "data/吉輔提供資料"
    converted_image_dir: str = "data/engineering_images_100dpi"
    preprocessed_image_dir: str = "data/preprocessed_images_100dpi"
    dataset_dir: str = "dataset"

    # --- PDF 轉換設定 ---
    pdf_dpi: int = 100
    pdf_max_workers: int = 16

    # --- 影像前處理設定 ---
    preprocess_top_n: int = 5
    preprocess_remove_largest: bool = True
    preprocess_padding: int = 2
    preprocess_max_attempts: int = 400
    preprocess_random_count: int = 10
    preprocess_max_workers: int = 12

    # --- 資料集分割設定 ---
    split_ratio: float = 0.8
    n_runs: int = 1
    base_seed: int = 42

    train_subpath: str = "Component_Dataset/train"
    val_subpath: str = "Component_Dataset/val"

    # --- 模型設定 ---
    backbone: str = "resnet18"
    pretrained: bool = True
    in_channels: int = 1
    proj_dim: int = 2048
    pred_hidden: int = 512

    # --- 訓練與超參數設定 ---
    img_size: int = 512
    epochs: int = 200
    batch_size: int = 64
    lr: float = 2e-5
    weight_decay: float = 1e-5
    num_workers: int = 8
    img_exts: tuple = (".jpg", ".png", ".bmp", ".tif", ".webp")

    # --- 輸出與日誌設定 ---
    output_dir: str = "outputs"
    exp_name: str = "simsiam_exp_v3"
    save_freq: int = 10

    # --- 管道階段控制旗標 ---
    skip_zip_extraction: bool = True
    skip_pdf_conversion: bool = False
    skip_preprocessing: bool = False

    # --- 離線增強設定 ---
    offline_aug: bool = False
    num_aug_versions: int = 20

    def to_dict(self):
        return asdict(self)

    def validate(self):
        if self.pdf_dpi <= 0:
            raise ValueError(f"pdf_dpi 必須為正整數，收到 {self.pdf_dpi}")
        if not 0.0 < self.split_ratio < 1.0:
            raise ValueError(f"split_ratio 必須在 (0, 1) 之間，收到 {self.split_ratio}")
        if self.n_runs < 1:
            raise ValueError(f"n_runs 必須 >= 1，收到 {self.n_runs}")
        if self.epochs < 1:
            raise ValueError(f"epochs 必須 >= 1，收到 {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size 必須 >= 1，收到 {self.batch_size}")
        if self.lr <= 0:
            raise ValueError(f"lr 必須為正數，收到 {self.lr}")
        if self.backbone not in ("resnet18", "resnet50"):
            raise ValueError(f"backbone 僅支援 'resnet18' 或 'resnet50'，收到 '{self.backbone}'")


# -----------------------------------------------------------------------------
# 2. SimSiam 模型架構與損失函數 (對齊 baseline src)
# -----------------------------------------------------------------------------

def _mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    bn_last: bool = True,
    dropout: float = 0.0,
) -> nn.Sequential:
    layers = [
        nn.Linear(in_dim, hidden_dim, bias=False),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers += [nn.Linear(hidden_dim, out_dim, bias=False)]
    if bn_last:
        layers.append(nn.BatchNorm1d(out_dim, affine=True))
    return nn.Sequential(*layers)


class SimSiam(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        proj_dim: int = 2048,
        pred_hidden: int = 512,
        dropout: float = 0.0,
        pretrained: bool = False,
        in_channels: int = 1,
    ):
        super().__init__()

        # 1. 建立 Backbone
        if backbone == "resnet18":
            net = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
        elif backbone == "resnet50":
            net = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            )
        else:
            raise NotImplementedError(f"Unsupported backbone: {backbone}")

        # 修改第一層卷積以適應輸入通道
        if in_channels != 3:
            old_conv = net.conv1
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            if pretrained and in_channels == 1:
                with torch.no_grad():
                    new_conv.weight[:] = old_conv.weight.sum(dim=1, keepdim=True) / 3.0
            net.conv1 = new_conv

        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net

        # 2. 建立 Projector (隱藏層 2048)
        self.projector = _mlp(feat_dim, 2048, proj_dim, bn_last=True, dropout=dropout)

        # 3. 建立 Predictor
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_hidden, bias=False),
            nn.BatchNorm1d(pred_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(pred_hidden, proj_dim),
        )

        # 4. 初始化
        for m in list(self.projector.modules()) + list(self.predictor.modules()):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        f1 = self.backbone(x1)
        f2 = self.backbone(x2)

        z1 = self.projector(f1)
        z2 = self.projector(f2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return p1, p2, z1.detach(), z2.detach()


def D(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return -(p * z).sum(dim=1).mean()


def calculate_collapse_std(z: torch.Tensor) -> float:
    z_norm = F.normalize(z, dim=1)
    return z_norm.std(dim=0).mean().item()


# -----------------------------------------------------------------------------
# 3. GPU 自監督資料增強 (Kornia-based GPU Augmentation)
# -----------------------------------------------------------------------------

class GPUAugmentation(nn.Module):
    """
    v3 保留的核心元件：基於 Kornia 在 GPU 端進行的自監督資料增強管線。
    為了與 baseline 一致，採用標準的 [0.5] mean / std 正規化。
    """
    def __init__(self, img_size: int = 512, use_augmentation: bool = True, in_channels: int = 1):
        super().__init__()
        self.img_size = img_size
        self.use_augmentation = use_augmentation
        self.in_channels = in_channels
        
        self._mean = torch.tensor([0.5] * in_channels)
        self._std = torch.tensor([0.5] * in_channels)
        
        try:
            import kornia
            import kornia.augmentation as K
            self._has_kornia = True
        except ImportError:
            self._has_kornia = False

        if self._has_kornia:
            self._aug = self._build_aug() 
        else:
            self._aug = None
            
    def _build_aug(self) -> nn.Module:
        import kornia.augmentation as K
        from kornia.constants import Resample
        
        mean = self._mean
        std = self._std
        
        if self.use_augmentation:
            return K.AugmentationSequential(
                K.RandomInvert(p=0.5, same_on_batch=False),
                K.RandomHorizontalFlip(p=0.5, same_on_batch=False),
                K.RandomVerticalFlip(p=0.5, same_on_batch=False),
                K.RandomAffine(
                    degrees=15.0,
                    translate=(0.1, 0.1),
                    resample=Resample.BILINEAR.name,
                    padding_mode='zeros',
                    p=0.7,
                    same_on_batch=False
                ),
                K.RandomResizedCrop(
                    size=(self.img_size, self.img_size),
                    resample=Resample.BILINEAR.name,
                    align_corners=None,
                    p=0.5,
                    same_on_batch=False
                ),
                K.Normalize(mean=mean, std=std),
                data_keys=["input"],
            )
        else:
            return K.AugmentationSequential(
                K.Resize((self.img_size, self.img_size), resample=Resample.BILINEAR.name),
                K.Normalize(mean=mean, std=std),
                data_keys=["input"],
            )
            
    def create_views(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._aug is not None:
            b = x.shape[0]
            x2 = torch.cat([x, x], dim=0)
            out = self._aug(x2)
            v1, v2 = out[:b], out[b:]
        else:
            v1 = self._manual_normalize(x)
            v2 = v1
        return v1, v2
        
    def _manual_normalize(self, x: torch.Tensor) -> torch.Tensor:
        x_resized = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        mean = self._mean.to(x.device).view(1, -1, 1, 1)
        std = self._std.to(x.device).view(1, -1, 1, 1)
        return (x_resized - mean) / std
        
    def to(self, *args, **kwargs) -> "GPUAugmentation":
        super().to(*args, **kwargs)
        self._mean = self._mean.to(*args, **kwargs)
        self._std = self._std.to(*args, **kwargs)
        if self._aug is not None:
            self._aug = self._aug.to(*args, **kwargs)
        return self


# -----------------------------------------------------------------------------
# 4. 簡化之 SingleImageDataset 資料集
# -----------------------------------------------------------------------------

class SingleImageDataset(torch.utils.data.Dataset):
    """
    自定義 SingleImageDataset：僅載入單張灰階影像並縮放，或載入預先生成的離線 NPZ 增強影像。
    """
    def __init__(self, paths: List[Path], transform, grayscale: bool = True, offline_aug: bool = False, augmented_root: Path = None):
        self.paths = list(paths)
        self.transform = transform
        self.grayscale = grayscale
        self.mode = "L" if grayscale else "RGB"
        self.offline_aug = offline_aug
        self.augmented_root = augmented_root
        
        # Setup normalization parameters matching GPUAugmentation
        in_channels = 1 if grayscale else 3
        self._mean = torch.tensor([0.5] * in_channels).view(-1, 1, 1)
        self._std = torch.tensor([0.5] * in_channels).view(-1, 1, 1)
        
        # Resolve project root to derive relative paths
        self.project_root = Path(__file__).resolve().parent.parent

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Any:
        p = self.paths[idx]
        
        if self.offline_aug and self.augmented_root is not None:
            try:
                # Resolve unique npz path matching pregeneration to avoid collisions
                try:
                    rel_path = p.relative_to(self.project_root / "data")
                except ValueError:
                    rel_path = Path(p.name)
                
                npz_path = self.augmented_root / rel_path.parent / f"{rel_path.stem}.npz"
                
                if npz_path.exists():
                    # Load compressed NPZ array [V, C, H, W]
                    data = np.load(npz_path)["data"]
                    num_versions = data.shape[0]
                    
                    # Randomly sample 2 different views
                    import random
                    v1_idx, v2_idx = random.sample(range(num_versions), 2)
                    
                    # Convert to float32 Tensor in [0, 1]
                    v1 = torch.from_numpy(data[v1_idx].astype(np.float32) / 255.0)
                    v2 = torch.from_numpy(data[v2_idx].astype(np.float32) / 255.0)
                    
                    # Apply standard mean/std normalization [0.5]
                    v1 = (v1 - self._mean) / self._std
                    v2 = (v2 - self._mean) / self._std
                    
                    return v1, v2
            except Exception as e:
                print(f"Error loading offline NPZ {p}: {e}, falling back to online mode.")

        # Online Mode fallback
        try:
            img = Image.open(p)
            img = img.convert(self.mode)
            return self.transform(img)
        except Exception as e:
            print(f"Error loading {p}: {e}")
            dummy = Image.new(self.mode, (512, 512), 255)
            return self.transform(dummy)


# -----------------------------------------------------------------------------
# 5. Phase 1: 資料準備 (Data Preparation)
# -----------------------------------------------------------------------------

def _step_1_0_extract_zip(cfg: Config, timers: TimerCollection) -> None:
    timer = timers.create("step_1_0_zip_extraction")

    pdf_dir = Path(cfg.raw_pdf_dir)
    if pdf_dir.exists() and any(pdf_dir.rglob("*.pdf")):
        console.print(
            f"  [dim]⏭ 跳過 ZIP 解壓縮：PDF 資料夾已存在 ({pdf_dir})[/dim]"
        )
        return

    if cfg.skip_zip_extraction:
        console.print("  [dim]⏭ 跳過 ZIP 解壓縮（skip_zip_extraction=True）[/dim]")
        return

    timer.start()
    console.print(f"  📦 正在解壓縮: {cfg.raw_zip_path} → {cfg.raw_pdf_dir}")
    extract_archive(archive_path=cfg.raw_zip_path, output_dir=cfg.raw_pdf_dir, skip=cfg.skip_zip_extraction)
    elapsed = timer.stop()
    console.print(
        f"  [green]✔[/green] ZIP 解壓縮完成 ({elapsed:.2f}s)"
    )


def _step_1_1_pdf_to_image(cfg: Config, timers: TimerCollection) -> None:
    timer = timers.create("step_1_1_pdf_to_image")
    img_dir = Path(cfg.converted_image_dir)

    if img_dir.exists() and any(img_dir.rglob("*.png")):
        console.print(
            f"  [dim]⏭ 跳過 PDF 轉換：影像目錄已存在 ({img_dir})[/dim]"
        )
        return

    if cfg.skip_pdf_conversion:
        console.print("  [dim]⏭ 跳過 PDF 轉換（skip_pdf_conversion=True）[/dim]")
        return

    timer.start()
    console.print(
        f"  🖼 正在轉換 PDF → Image (DPI={cfg.pdf_dpi}, workers={cfg.pdf_max_workers})"
    )
    convert_pdfs_to_images(
        pdf_dir=cfg.raw_pdf_dir,
        output_dir=cfg.converted_image_dir,
        dpi=cfg.pdf_dpi,
        max_workers=cfg.pdf_max_workers,
        skip=cfg.skip_pdf_conversion,
        preserve_structure=True
    )
    elapsed = timer.stop()
    console.print(
        f"  [green]✔[/green] PDF 轉換完成 ({elapsed:.2f}s)"
    )


def _step_1_2_preprocess_images(cfg: Config, timers: TimerCollection) -> None:
    timer = timers.create("step_1_2_image_preprocessing")
    preprocessed_dir = Path(cfg.preprocessed_image_dir)

    if preprocessed_dir.exists() and any(preprocessed_dir.rglob("*.png")):
        console.print(
            f"  [dim]⏭ 跳過影像前處理：已處理目錄已存在 ({preprocessed_dir})[/dim]"
        )
        return

    if cfg.skip_preprocessing:
        console.print("  [dim]⏭ 跳過影像前處理（skip_preprocessing=True）[/dim]")
        return

    timer.start()
    console.print(
        f"  🔬 正在進行影像前處理 (workers={cfg.preprocess_max_workers})"
    )

    prep_cfg = PreprocessConfig(
        input_dir=cfg.converted_image_dir,
        output_root=cfg.preprocessed_image_dir,
        max_workers=cfg.preprocess_max_workers,
        top_n=cfg.preprocess_top_n,
        max_bbox_ratio=0.8 if cfg.preprocess_remove_largest else 1.0,
        padding=cfg.preprocess_padding,
        remove_gifu_logo=True,
        use_connected_components=True,
    )
    preprocess_images(prep_cfg, skip=cfg.skip_preprocessing)

    elapsed = timer.stop()
    console.print(
        f"  [green]✔[/green] 影像前處理完成 ({elapsed:.2f}s)"
    )


# -----------------------------------------------------------------------------
# 6. Phase 2: 多 Run 訓練迴圈
# -----------------------------------------------------------------------------

def _step_2_1_split_dataset(
    cfg: Config,
    run_idx: int,
    seed: int,
    timers: TimerCollection,
) -> Tuple[Path, Path]:
    timer_name = f"step_2_1_split_run_{run_idx + 1:02d}"
    timer = timers.create(timer_name)

    run_name = f"Run_{run_idx + 1:02d}_Seed_{seed}"
    run_output_dir = Path(cfg.dataset_dir) / run_name

    train_path = run_output_dir / cfg.train_subpath
    val_path = run_output_dir / cfg.val_subpath

    if train_path.exists() and any(train_path.rglob("*.png")):
        console.print(
            f"    [dim]⏭ 跳過資料集分割：{run_name} 已存在[/dim]"
        )
        return train_path, val_path

    timer.start()
    console.print(f"    📂 正在分割資料集 (seed={seed})...")

    # 呼叫與 v3 一致的 splitter
    split_dataset(
        source_root=cfg.preprocessed_image_dir,
        output_root=cfg.dataset_dir,
        run_name=run_name,
        split_ratio=cfg.split_ratio,
        seed=seed,
    )

    elapsed = timer.stop()
    n_train = sum(1 for _ in train_path.rglob("*.png")) if train_path.exists() else 0
    n_val = sum(1 for _ in val_path.rglob("*.png")) if val_path.exists() else 0

    console.print(
        f"    [green]✔[/green] 分割完成：Train={n_train}, Val={n_val} ({elapsed:.2f}s)"
    )
    return train_path, val_path


def _step_2_2_prepare_dataloaders(
    cfg: Config,
    train_path: Path,
    val_path: Path,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, int, int]:
    
    def scan_images(folder: Path) -> List[Path]:
        if not folder.exists():
            return []
        return sorted(
            p for p in folder.rglob("*")
            if p.suffix.lower() in cfg.img_exts
        )

    train_files = scan_images(train_path)
    val_files = scan_images(val_path)

    if not train_files:
        raise FileNotFoundError(f"訓練集路徑 {train_path} 下無影像檔案。")

    if cfg.offline_aug:
        from v3.src.data.offline_aug import pregenerate_offline_npz_augmentations
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pregenerate_offline_npz_augmentations(
            cfg=cfg,
            train_files=train_files,
            dataset_class=SingleImageDataset,
            gpu_aug_class=GPUAugmentation,
            device=device
        )

    train_transform = T.Compose([
        T.Resize((cfg.img_size, cfg.img_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor()
    ])
    val_transform = T.Compose([
        T.Resize((cfg.img_size, cfg.img_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor()
    ])

    is_grayscale = cfg.in_channels == 1
    
    project_root = Path(cfg.preprocessed_image_dir).resolve().parent.parent
    augmented_root = project_root / "data" / "augmented_npz" / cfg.exp_name
    
    train_ds = SingleImageDataset(
        train_files,
        transform=train_transform,
        grayscale=is_grayscale,
        offline_aug=cfg.offline_aug,
        augmented_root=augmented_root
    )
    val_ds = SingleImageDataset(
        val_files,
        transform=val_transform,
        grayscale=is_grayscale,
        offline_aug=False
    )

    effective_workers = 0 if os.name == "nt" else cfg.num_workers

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=effective_workers,
        pin_memory=True,
        persistent_workers=effective_workers > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=effective_workers,
        pin_memory=True,
        persistent_workers=effective_workers > 0,
    )

    return train_loader, val_loader, len(train_files), len(val_files)


def _step_2_3_build_model(
    cfg: Config,
    device: str,
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, object, object]:
    
    # 建立模型實例
    model = SimSiam(
        backbone=cfg.backbone,
        proj_dim=cfg.proj_dim,
        pred_hidden=cfg.pred_hidden,
        pretrained=cfg.pretrained,
        in_channels=cfg.in_channels,
    ).to(device)

    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    if trainable_params == 0:
        raise ValueError("模型中沒有任何可訓練參數，請檢查 backbone 是否被意外凍結。")
        
    console.print(f"    📊 可訓練參數: {trainable_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs
    )

    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    return model, optimizer, scheduler, scaler


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    gpu_aug: GPUAugmentation,
    device: str,
) -> Tuple[float, float]:
    model.train()
    gpu_aug.train()
    total_loss = 0.0
    total_std = 0.0
    num_batches = 0

    use_amp = scaler is not None and str(device).startswith("cuda")
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp
        else contextlib.nullcontext()
    )

    for x in loader:
        if isinstance(x, (list, tuple)):
            v1, v2 = x
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)
        else:
            x = x.to(device, non_blocking=True)
            v1, v2 = gpu_aug.create_views(x)

        optimizer.zero_grad(set_to_none=True)

        with amp_ctx:
            p1, p2, z1, z2 = model(v1, v2)
            loss = 0.5 * (D(p1, z2) + D(p2, z1))

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            batch_std = (calculate_collapse_std(z1) + calculate_collapse_std(z2)) / 2.0
            total_std += batch_std

        num_batches += 1

    return total_loss / max(num_batches, 1), total_std / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    gpu_aug: GPUAugmentation,
    device: str,
) -> Tuple[float, float]:
    model.eval()
    gpu_aug.train()
    total_loss = 0.0
    total_std = 0.0
    num_batches = 0

    for x in loader:
        if isinstance(x, (list, tuple)):
            v1, v2 = x
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)
        else:
            x = x.to(device, non_blocking=True)
            v1, v2 = gpu_aug.create_views(x)

        p1, p2, z1, z2 = model(v1, v2)
        loss = 0.5 * (D(p1, z2) + D(p2, z1))
        total_loss += loss.item()

        batch_std = (calculate_collapse_std(z1) + calculate_collapse_std(z2)) / 2.0
        total_std += batch_std

        num_batches += 1

    return total_loss / max(num_batches, 1), total_std / max(num_batches, 1)


def _step_2_4_training_loop(
    cfg: Config,
    run_name: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    scaler: object,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: str,
    run_logger: object,
    timers: TimerCollection,
    run_idx: int,
) -> float:
    use_aug = (cfg.exp_name != "Baseline")
    gpu_aug = GPUAugmentation(
        img_size=cfg.img_size,
        use_augmentation=use_aug,
        in_channels=cfg.in_channels
    ).to(device)

    best_val_loss = float("inf")
    run_timer_name = f"step_2_4_train_run_{run_idx + 1:02d}"
    run_timer = timers.create(run_timer_name)
    run_timer.start()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn("{task.fields[info]}"),
        console=console,
    ) as progress:
        task_id = progress.add_task(
            f"[cyan]Run: {run_name}",
            total=cfg.epochs,
            info="Init...",
        )

        for epoch in range(1, cfg.epochs + 1):
            epoch_timer = PrecisionTimer(f"epoch_{epoch}")
            epoch_timer.start()
            epoch_wall_start = time.perf_counter()

            # 訓練 (單一 epoch，使用 GPU 自監督增強)
            train_loss, train_std = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                gpu_aug=gpu_aug,
                device=device
            )

            # 自監督驗證 Loss 指標計算
            val_loss, val_std = (
                evaluate(
                    model=model,
                    loader=val_loader,
                    gpu_aug=gpu_aug,
                    device=device
                )
                if len(val_loader) > 0
                else (0.0, 0.0)
            )

            epoch_timer.pause()

            # 判斷是否為最佳模型並儲存 checkpoint (以 Val Loss 最小化為基準)
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            run_logger.save_checkpoint(model, optimizer, epoch, val_loss, is_best)

            epoch_timer.resume()
            scheduler.step()

            epoch_net = epoch_timer.stop()
            epoch_wall = time.perf_counter() - epoch_wall_start

            current_lr = optimizer.param_groups[0]["lr"]
            run_logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_z_std=train_std,
                val_z_std=val_std,
                lr=current_lr,
                epoch_net_duration=epoch_net,
                epoch_wall_duration=epoch_wall,
            )

            info_text = (
                f"[b]L:[/b]{train_loss:.4f}|"
                f"[b]V:[/b]{val_loss:.4f}|"
                f"[b]Std:[/b]{train_std:.4f}|"
                f"[b]Net:[/b]{epoch_net:.1f}s"
            )
            progress.update(task_id, advance=1, info=info_text)

    run_timer.stop()
    return best_val_loss


def _run_single_training(
    cfg: Config,
    run_idx: int,
    seed: int,
    experiment_logger: ExperimentLogger,
    timers: TimerCollection,
) -> Dict:
    run_name = f"Run_{run_idx + 1:02d}_Seed_{seed}"
    train_path, val_path = _step_2_1_split_dataset(cfg, run_idx, seed, timers)

    train_loader, val_loader, n_train, n_val = _step_2_2_prepare_dataloaders(cfg, train_path, val_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, optimizer, scheduler, scaler = _step_2_3_build_model(cfg, device)

    run_logger = experiment_logger.create_run_logger(run_name)

    try:
        best_val_loss = _step_2_4_training_loop(
            cfg=cfg,
            run_name=run_name,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            run_logger=run_logger,
            timers=timers,
            run_idx=run_idx,
        )
        status = "success"
        error_msg = ""
    except Exception as e:
        console.print_exception(show_locals=False)
        best_val_loss = float("inf")
        status = "error"
        error_msg = str(e)

    # 與 baseline 一致在 single training 結束後印出 HTML report
    if status == "success":
        try:
            run_logger.generate_report()
        except Exception as report_err:
            console.print(f"      [yellow]⚠ 報表產生失敗: {report_err}[/yellow]")

    return {
        "run_name": run_name,
        "seed": seed,
        "best_val_loss": best_val_loss,
        "status": status,
        "error": error_msg,
        "log_dir": str(run_logger.run_dir) if 'run_logger' in locals() else "",
    }


# -----------------------------------------------------------------------------
# 7. Phase 3: 收尾與彙整報告
# -----------------------------------------------------------------------------

def _print_final_summary(
    run_results: List[Dict],
    timers: TimerCollection,
    experiment_logger: ExperimentLogger,
    total_elapsed: float,
) -> None:
    console.print()
    console.rule("[bold green]🏆 v3 自監督訓練完成 — 總結報告")

    result_table = Table(
        title="批次訓練結果", box=box.HEAVY, show_lines=True
    )
    result_table.add_column("Run", style="cyan", justify="center")
    result_table.add_column("Seed", style="magenta", justify="center")
    result_table.add_column("Best Val Loss", style="green", justify="right")
    result_table.add_column("Status", justify="center")
    result_table.add_column("Log Dir", style="dim", overflow="fold")

    for r in run_results:
        loss_str = (
            f"{r['best_val_loss']:.4f}"
            if isinstance(r.get("best_val_loss"), float)
            else str(r.get("best_val_loss", "N/A"))
        )
        status_str = (
            "[green]✔[/green]"
            if r.get("status") == "success"
            else f"[red]✗ {r.get('error', '')[:50]}[/red]"
        )
        result_table.add_row(
            r.get("run_name", ""),
            str(r.get("seed", "")),
            loss_str,
            status_str,
            r.get("log_dir", ""),
        )

    console.print(result_table)

    timing_table = Table(
        title="計時明細", box=box.ROUNDED, show_lines=True
    )
    timing_table.add_column("步驟", style="cyan")
    timing_table.add_column("淨耗時", style="green", justify="right")
    timing_table.add_column("牆鐘耗時", style="yellow", justify="right")
    timing_table.add_column("暫停次數", style="dim", justify="right")

    for row in timers.summary_table_rows():
        timing_table.add_row(*row)

    console.print(timing_table)

    timing_path = experiment_logger.save_timing_report()
    summary_path = experiment_logger.save_overall_summary(run_results)

    console.print(f"\n[bold]📁 實驗目錄:[/bold] {experiment_logger.experiment_dir}")
    console.print(f"[bold]⏱ 總執行時間:[/bold] {total_elapsed / 60:.1f} min")
    console.print(f"[dim]計時報告: {timing_path}[/dim]")
    console.print(f"[dim]總結報告: {summary_path}[/dim]")


# -----------------------------------------------------------------------------
# 8. 主入口 (Main Entrance)
# -----------------------------------------------------------------------------

def main() -> None:
    # 支援透過 argparse 解析參數以供使用者下達指令
    import argparse
    parser = argparse.ArgumentParser(description="SimSiam Engineering Image Retrieval Training Pipeline CLI")
    
    # 運行模式
    parser.add_argument("--prepare_data", action="store_true", help="僅執行 ZIP 解壓、PDF 轉檔與資料清洗前處理（不訓練）")
    
    # 訓練超參數設定 (單次訓練時生效)
    parser.add_argument("--dataset_type", type=str, default="T_small", choices=["T_small", "T_large"], help="訓練資料集規模")
    parser.add_argument("--experiment_type", type=str, default="Exp_A", choices=["Baseline", "Exp_A", "Exp_B"], help="對照組別設定")
    parser.add_argument("--epochs", type=int, default=200, help="總訓練輪數 Epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--base_lr", type=float, default=2e-5, help="基礎學習率")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="優化器 L2 權重衰減因子")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader 載入器線程數")
    parser.add_argument("--save_dir", type=str, default=None, help="模型權重與 Logs 輸出目錄")
    parser.add_argument("--n_runs", type=int, default=1, help="多 Run 訓練的重複次數")
    parser.add_argument("--offline_aug", action="store_true", help="是否在訓練前執行離線資料增強")
    parser.add_argument("--num_aug_versions", type=int, default=20, help="離線資料增強版本數量")
    
    args, unknown = parser.parse_known_args()
    
    cfg = Config()
    
    # 覆寫配置
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.base_lr
    cfg.weight_decay = args.weight_decay
    cfg.num_workers = args.num_workers
    cfg.n_runs = args.n_runs
    cfg.offline_aug = args.offline_aug
    cfg.num_aug_versions = args.num_aug_versions
    if args.save_dir:
        cfg.output_dir = args.save_dir
    
    # 根據 dataset_type 和 experiment_type 修改資料夾與命名
    cfg.exp_name = f"simsiam_{args.experiment_type}_{args.dataset_type}"
    if args.dataset_type == "T_small":
        cfg.raw_zip_path = "data/吉輔提供資料.zip"
        cfg.raw_pdf_dir = "data/raw_pdfs_small"
        cfg.converted_image_dir = "data/converted_images_small"
        cfg.preprocessed_image_dir = "data/preprocessed_images_small"
    else:
        cfg.raw_zip_path = "data/PDF.zip"
        cfg.raw_pdf_dir = "data/raw_pdfs"
        cfg.converted_image_dir = "data/converted_images"
        cfg.preprocessed_image_dir = "data/preprocessed_images"
    
    cfg.validate()

    def get_nested_dir(base_dir: str) -> str:
        path = Path(base_dir)
        if not path.exists():
            return base_dir
        subdirs = [p for p in path.iterdir() if p.is_dir()]
        files = [p for p in path.iterdir() if p.is_file()]
        if len(subdirs) == 1 and not files:
            return str(subdirs[0])
        return base_dir

    timers = TimerCollection()
    total_timer = timers.create("total_pipeline")
    total_timer.start()

    # 建立 ExperimentLogger
    experiment_logger = ExperimentLogger(config=cfg, timers=timers)

    # ========================================================
    # Phase 1: 資料準備 (解壓與前處理)
    # ========================================================
    console.rule("[bold cyan]Phase 1: 資料準備")
    phase1_timer = timers.create("phase_1_data_preparation")
    phase1_timer.start()

    # 偵測是否需強制解壓
    cfg.skip_zip_extraction = False if args.prepare_data else True
    cfg.skip_pdf_conversion = False if args.prepare_data else True
    cfg.skip_preprocessing = False if args.prepare_data else True

    _step_1_0_extract_zip(cfg, timers)

    # Resolve nested directories dynamically
    cfg.raw_pdf_dir = get_nested_dir(cfg.raw_pdf_dir)
    cfg.converted_image_dir = get_nested_dir(cfg.converted_image_dir)
    cfg.preprocessed_image_dir = get_nested_dir(cfg.preprocessed_image_dir)

    _step_1_1_pdf_to_image(cfg, timers)
    _step_1_2_preprocess_images(cfg, timers)

    phase1_timer.stop()
    console.print(f"[green]✔[/green] Phase 1 完成 ({phase1_timer.elapsed:.2f}s)\n")

    if args.prepare_data:
        if args.offline_aug:
            console.print("[bold cyan]🚀 開始進行離線資料增強 pre-generation...")
            train_path, val_path = _step_2_1_split_dataset(cfg, 0, cfg.base_seed, timers)
            _step_2_2_prepare_dataloaders(cfg, train_path, val_path)
            console.print("[green]✔ 離線資料增強 pre-generation 完成！[/green]")
        console.print("[green]✔ 資料準備與前處理管線完成，由於指定了 --prepare_data，將在此結束（不進行訓練）。[/green]")
        return

    # ========================================================
    # Phase 2: 多 Run 訓練
    # ========================================================
    console.rule("[bold cyan]Phase 2: 多 Run 訓練")
    console.print(f"  排程: {cfg.n_runs} 個 Run, base_seed={cfg.base_seed}\n")

    phase2_timer = timers.create("phase_2_multi_run_training")
    phase2_timer.start()

    run_results = []
    for run_idx in range(cfg.n_runs):
        seed = cfg.base_seed + run_idx
        result = _run_single_training(
            cfg=cfg,
            run_idx=run_idx,
            seed=seed,
            experiment_logger=experiment_logger,
            timers=timers
        )
        run_results.append(result)

    phase2_timer.stop()

    # ========================================================
    # Phase 3: 收尾
    # ========================================================
    total_elapsed = total_timer.stop()

    _print_final_summary(
        run_results=run_results,
        timers=timers,
        experiment_logger=experiment_logger,
        total_elapsed=total_elapsed
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]⚠ 使用者中斷訓練。已完成的紀錄已保存。[/bold yellow]")
    except Exception:
        console.print_exception(show_locals=False)
