"""設定管理與驗證模組 (Configuration Module)。

============================================================
使用 OmegaConf 載入與組合 YAML 設定檔，並轉換為強型別的 dataclass
結構。支援命令列 (CLI) 選項覆寫（dotlist 語法）。

設計優點：
    1. **單一真相來源**：所有資料、模型、訓練超參數集中管理。
    2. ** Fail-Fast 驗證**：在載入後立即檢查所有參數的合法性，防止異常中斷。
    3. **自適應快取設定**：包含是否直接載入 `.npz` 快取的選項。
============================================================
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from omegaconf import DictConfig, OmegaConf

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class DataConfig:
    """資料管線與前處理設定"""
    dataset_root: str = "data"
    raw_zip_path: Optional[str] = "data/吉輔提供資料.zip"  # 預設原始零件 PDF 壓縮包
    raw_pdf_dir: str = "data/raw_pdfs"
    converted_image_dir: str = "data/converted_images"
    preprocessed_image_dir: str = "data/preprocessed_images"
    
    # PDF 轉圖解析度與並行執行緒數
    pdf_dpi: int = 100
    pdf_max_workers: int = 16
    
    # 影像前處理參數
    preprocess_top_n: int = 5
    preprocess_max_bbox_ratio: float = 0.9
    preprocess_padding: int = 2
    preprocess_max_workers: int = 12
    
    # Logo 擦除
    remove_gifu_logo: bool = True
    logo_template_path: Optional[str] = "data/Gifu_logo.png"
    logo_mask_region: Optional[List[float]] = field(default_factory=lambda: [0.0, 0.9, 0.2, 1.0])
    
    # 資料集劃分與快取
    dataset_dir: str = "dataset"
    split_ratio: float = 0.8
    base_seed: int = 42
    
    # 核心 v4 加速：快取路徑與是否直接讀取
    cache_path: str = "dataset/dataset_cache.json"
    load_cached: bool = False  # CLI 若設為 True，直接讀取 cache_path
    
    # 影像標準化參數 (預設為吉輔 CAD 白底線條圖的統計值)
    norm_mean: float = 0.0394
    norm_std: float = 0.1752


@dataclass
class ModelConfig:
    """SimSiam 模型架構與權重設定"""
    backbone: str = "resnet18"  # resnet18 或 resnet50
    pretrained: bool = True
    in_channels: int = 1        # 工程圖預設為灰階 (1)
    proj_dim: int = 2048
    pred_hidden: int = 512


@dataclass
class TrainingConfig:
    """訓練超參數與引擎控制"""
    img_size: int = 512        # 經由 CPU Letterbox 輸出的恆定尺寸
    epochs: int = 200
    batch_size: int = 64
    lr: float = 2e-5
    weight_decay: float = 1e-5
    num_workers: int = 8
    
    # AMP 精度選擇，v4 預設為 bfloat16 (若為 False，可回退到 float16/float32)
    use_bf16: bool = True
    
    # 梯度剪裁
    grad_clip: float = 1.0
    
    # 是否啟用 torch.compile 加速資料增強與 forward
    compile_model: bool = True

    # 是否啟用資料增強
    use_augmentation: bool = True

    # 特徵維度標準差安全斷路器閾值，低於此值即判定發生塌陷並停止訓練
    collapse_threshold: float = 0.005


@dataclass
class ExperimentConfig:
    """實驗追蹤與報告設定"""
    output_dir: str = "outputs_v4"
    exp_name: str = "simsiam_v4"
    save_freq: int = 10         # 每幾 epoch 儲存 checkpoint
    eval_freq: int = 10         # 每幾 epoch 進行一次 LOO 評估
    log_file: str = "training.log"


@dataclass
class AppConfig:
    """頂層設定類別，整合所有子設定"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str,
        cli_overrides: Optional[Sequence[str]] = None,
    ) -> AppConfig:
        """從 YAML 檔載入設定，並支援 CLI 參數進行覆寫"""
        yaml_path_obj = Path(yaml_path)
        if not yaml_path_obj.is_file():
            # 若 YAML 檔不存在，則建立預設設定並自動存檔，以便後續讀取
            logger.warning("未找到設定檔 %s，將採用系統預設值進行初始化", yaml_path)
            default_cfg = OmegaConf.structured(cls)
            yaml_path_obj.parent.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(default_cfg, yaml_path_obj)
            file_cfg = default_cfg
        else:
            file_cfg = OmegaConf.load(yaml_path)

        # 合併 CLI 覆寫參數 (格式如 data.load_cached_npz=True)
        if cli_overrides:
            override_cfg = OmegaConf.from_dotlist(list(cli_overrides))
            file_cfg = OmegaConf.merge(file_cfg, override_cfg)

        schema = OmegaConf.structured(cls)
        merged: DictConfig = OmegaConf.merge(schema, file_cfg)
        obj: AppConfig = OmegaConf.to_object(merged)  # type: ignore

        logger.info("設定載入成功 (包含 %d 個 CLI 參數覆寫)", len(cli_overrides or []))
        return obj

    def to_dict(self) -> dict:
        """將設定物件轉換成純 Python 字典，便於 JSON 序列化與日誌記錄"""
        return asdict(self)

    def validate(self) -> None:
        """驗證參數合理性"""
        d = self.data
        m = self.model
        t = self.training

        # 驗證資料設定
        if d.pdf_dpi <= 0:
            raise ValueError(f"data.pdf_dpi 必須為正整數，收到: {d.pdf_dpi}")
        if not (0.0 < d.preprocess_max_bbox_ratio <= 1.0):
            raise ValueError(f"data.preprocess_max_bbox_ratio 範圍必須在 (0, 1] 之間，收到: {d.preprocess_max_bbox_ratio}")
        if not (0.0 < d.split_ratio < 1.0):
            raise ValueError(f"data.split_ratio 範圍必須在 (0, 1) 之間，收到: {d.split_ratio}")

        # 驗證模型設定
        if m.backbone not in ("resnet18", "resnet50"):
            raise ValueError(f"model.backbone 僅支援 'resnet18' 或 'resnet50'，收到: {m.backbone}")
        if m.in_channels < 1:
            raise ValueError(f"model.in_channels 必須大於等於 1，收到: {m.in_channels}")

        # 驗證訓練超參數
        if t.epochs < 1:
            raise ValueError(f"training.epochs 必須大於等於 1，收到: {t.epochs}")
        if t.batch_size < 1:
            raise ValueError(f"training.batch_size 必須大於等於 1，收到: {t.batch_size}")
        if t.lr <= 0:
            raise ValueError(f"training.lr 必須為正數，收到: {t.lr}")
        if t.img_size != 512:
            logger.warning("提示: 建議 training.img_size 設為 512 以發揮 Letterbox 優化效果，目前設定為: %d", t.img_size)

        logger.info("超參數設定驗證通過")
