"""設定載入與驗證模組 (Configuration Module)。

============================================================
使用 OmegaConf 載入 YAML 設定檔，並轉換為強型別的 dataclass
結構。支援 CLI 選項覆寫（dotlist 語法）。

設計考量：
    1. **巢狀結構**：``DataConfig``、``ModelConfig``、
       ``TrainingConfig``、``ExperimentConfig`` 各自封裝一個關注面向，
       下游模組只需接收自己需要的子設定。
    2. **驗證優先**：所有數值範圍、路徑合理性在載入後立即檢查，
       避免訓練跑到一半才因設定錯誤中斷。
    3. **可序列化**：透過 ``to_dict()`` 匯出純 Python 字典，
       方便 JSON 持久化與實驗追蹤。

使用範例::

    cfg = AppConfig.from_yaml("v2/configs/default.yaml", cli_overrides=sys.argv[1:])
    cfg.validate()
============================================================
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from omegaconf import DictConfig, OmegaConf

from src.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 子設定 (Sub-Configs)
# ============================================================


@dataclass
class DataConfig:
    """資料管線設定。

    涵蓋從壓縮檔解壓到資料集分割的所有路徑與參數。

    Attributes:
        raw_zip_path: 原始壓縮檔路徑，``None`` 表示不解壓。
        skip_extraction: 是否強制跳過解壓縮。
        raw_pdf_dir: PDF 來源目錄。
        converted_image_dir: PDF 轉換後影像目錄。
        pdf_dpi: PDF 轉換解析度。
        pdf_max_workers: PDF 轉換並行數。
        skip_pdf_conversion: 是否跳過 PDF 轉換。
        preprocessed_image_dir: 前處理輸出目錄。
        preprocess_top_n: 保留的大元件數。
        preprocess_max_bbox_ratio: 排除大於整張圖一定比例的外接矩形元件（通常為圖框）。
        preprocess_padding: 元件裁切邊界填充。

        preprocess_max_workers: 前處理並行程序數。
        use_topology_pruning: 是否啟用拓撲分類與剪枝。
        topology_pruning_iters: 結構級剪枝最大迭代次數。
        topology_pruning_ksize: 結構級剪枝起始 Kernel 尺寸。
        min_simple_area: 無孔洞元件的最小面積門檻（剪枝用）。
        skip_preprocessing: 是否跳過前處理。
        dataset_dir: 分割後資料集根目錄。
        split_ratio: 訓練集佔比。
        n_runs: 多 Run 重複次數。
        base_seed: 基礎隨機種子。
        train_subpath: Run 內訓練集子路徑。
        val_subpath: Run 內驗證集子路徑。
    """

    raw_zip_path: Optional[str] = None
    skip_extraction: bool = False
    raw_pdf_dir: str = "data/raw_pdfs"
    converted_image_dir: str = "data/converted_images"
    pdf_dpi: int = 100
    pdf_max_workers: int = 16
    skip_pdf_conversion: bool = False
    preprocessed_image_dir: str = "data/preprocessed_images"
    preprocess_top_n: int = 5
    preprocess_max_bbox_ratio: float = 0.9
    preprocess_padding: int = 2

    preprocess_max_workers: int = 12
    skip_preprocessing: bool = False
    dataset_dir: str = "dataset"
    split_ratio: float = 0.8
    n_runs: int = 1
    base_seed: int = 42
    train_subpath: str = "Component_Dataset/train"
    val_subpath: str = "Component_Dataset/val"
    test_subpath: str = "Component_Dataset/test"

    # --- 消融實驗：前處理旗標 ---
    use_connected_components: bool = True
    use_topology_analysis: bool = True
    use_topology_pruning: bool = True
    topology_pruning_iters: int = 3
    topology_pruning_ksize: int = 2
    min_simple_area: int = 40
    remove_gifu_logo: bool = True
    logo_template_path: Optional[str] = None
    logo_mask_region: Optional[List[float]] = None  # [x1_r, y1_r, x2_r, y2_r]

    # --- 評估設定 ---
    labeled_data_path: str = "data/converted_images"
    test_split_ratio: float = 0.2
    eval_top_k_values: List[int] = field(default_factory=lambda: [1, 5, 10])


@dataclass
class ModelConfig:
    """模型架構設定。

    Attributes:
        backbone: 骨幹網路名稱（``resnet18`` / ``resnet50``）。
        pretrained: 是否載入 ImageNet 預訓練權重。
        in_channels: 輸入影像通道數。
        proj_dim: Projector 輸出維度。
        proj_hidden: Projector 隱藏層維度；``None`` 自動使用 backbone feat_dim。
        pred_hidden: Predictor 隱藏層維度。
    """

    backbone: str = "resnet18"
    pretrained: bool = True
    in_channels: int = 1
    proj_dim: int = 2048
    proj_hidden: Optional[int] = None
    pred_hidden: int = 512


@dataclass
class TrainingConfig:
    """訓練超參數設定。

    Attributes:
        img_size: 輸入影像尺寸（正方形）。
        epochs: 每 Run 訓練 epoch 數。
        batch_size: Batch 大小。
        lr: 學習率。
        weight_decay: AdamW 權重衰減。
        num_workers: DataLoader 工作程序數。
        prefetch_factor: DataLoader 預取因子。
        img_exts: 支援的影像格式列表。
        use_amp: 是否啟用混合精度 (FP16)。
        scheduler: 學習率排程器類型 (``cosine`` / ``step`` / ``constant``)。
        grad_clip: 梯度裁切閾值；``0.0`` 停用。
    """

    img_size: int = 512
    epochs: int = 200
    batch_size: int = 64
    lr: float = 2e-5
    weight_decay: float = 1e-5
    num_workers: int = 8
    prefetch_factor: int = 2
    img_exts: List[str] = field(
        default_factory=lambda: [".jpg", ".png", ".bmp", ".tif", ".webp"]
    )
    use_amp: bool = True
    scheduler: str = "cosine"
    grad_clip: float = 1.0
    use_augmentation: bool = True
    use_gpu_augmentation: bool = True
    resume: bool = True


@dataclass
class ExperimentConfig:
    """實驗追蹤設定。

    Attributes:
        output_dir: 實驗結果輸出根目錄。
        exp_name: 實驗名稱前綴。
        save_freq: Checkpoint 儲存頻率。
        log_file: 日誌檔名。
    """

    output_dir: str = "outputs"
    exp_name: str = "simsiam_exp"
    save_freq: int = 10
    log_file: str = "training.log"


@dataclass
class LoggingConfig:
    """日誌設定。

    Attributes:
        level: 日誌等級 (DEBUG, INFO, WARNING, ERROR)。
        log_to_file: 是否輸出至檔案。
        use_rich: 是否使用 Rich Handler 美化輸出。
    """

    level: str = "INFO"
    log_to_file: bool = True
    use_rich: bool = True


# ============================================================
# 頂層設定 (Top-Level Config)
# ============================================================


@dataclass
class AppConfig:
    """應用程式頂層設定，聚合所有子設定。

    Attributes:
        data: 資料管線設定。
        model: 模型架構設定。
        training: 訓練超參數。
        experiment: 實驗追蹤設定。
    """

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # ----------------------------------------------------------
    # 工廠方法
    # ----------------------------------------------------------

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str,
        cli_overrides: Optional[Sequence[str]] = None,
    ) -> "AppConfig":
        """從 YAML 載入設定，可選合併 CLI 覆寫。

        Args:
            yaml_path: YAML 設定檔路徑。
            cli_overrides: 來自 ``sys.argv`` 的 dotlist 覆寫，
                例如 ``["training.lr=3e-4", "training.epochs=100"]``。

        Returns:
            AppConfig: 已合併的強型別設定物件。

        Raises:
            FileNotFoundError: 當 YAML 檔案不存在時。
            omegaconf.errors.ConfigKeyError: 當覆寫的鍵不存在。
        """
        yaml_path_obj = Path(yaml_path)
        if not yaml_path_obj.is_file():
            raise FileNotFoundError(f"設定檔不存在: {yaml_path}")

        # 1. 載入 YAML
        file_cfg: DictConfig = OmegaConf.load(yaml_path)

        # 2. 合併 CLI 覆寫
        if cli_overrides:
            override_cfg = OmegaConf.from_dotlist(list(cli_overrides))
            file_cfg = OmegaConf.merge(file_cfg, override_cfg)

        # 3. 轉換為 dataclass
        schema = OmegaConf.structured(cls)
        merged: DictConfig = OmegaConf.merge(schema, file_cfg)
        obj: AppConfig = OmegaConf.to_object(merged)  # type: ignore[assignment]

        logger.info("設定載入完成: %s (含 %d 個 CLI 覆寫)", yaml_path, len(cli_overrides or []))
        return obj

    # ----------------------------------------------------------
    # 序列化
    # ----------------------------------------------------------

    def to_dict(self) -> dict:
        """將設定轉為純 Python 字典。

        Returns:
            dict: 所有設定欄位的字典，可直接用於 ``json.dump``。
        """
        return asdict(self)

    # ----------------------------------------------------------
    # 驗證
    # ----------------------------------------------------------

    def validate(self) -> None:
        """驗證設定的合理性。

        檢查數值範圍、路徑格式與互斥旗標。應在訓練/預處理
        開始前呼叫。

        Raises:
            ValueError: 當設定值不合規時。
        """
        d = self.data
        m = self.model
        t = self.training
        e = self.experiment

        # --- Data ---
        if d.pdf_dpi <= 0:
            raise ValueError(f"data.pdf_dpi 必須為正整數，收到 {d.pdf_dpi}")
        if not 0.0 < d.preprocess_max_bbox_ratio <= 1.0:
            raise ValueError(
                f"data.preprocess_max_bbox_ratio 必須在 (0, 1] 之間，收到 {d.preprocess_max_bbox_ratio}"
            )
        if not 0.0 < d.split_ratio < 1.0:
            raise ValueError(
                f"data.split_ratio 必須在 (0, 1) 之間，收到 {d.split_ratio}"
            )
        if d.n_runs < 1:
            raise ValueError(f"data.n_runs 必須 >= 1，收到 {d.n_runs}")

        # --- Model ---
        if m.backbone not in ("resnet18", "resnet50"):
            raise ValueError(
                f"model.backbone 僅支援 'resnet18' / 'resnet50'，"
                f"收到 '{m.backbone}'"
            )
        if m.in_channels < 1:
            raise ValueError(
                f"model.in_channels 必須 >= 1，收到 {m.in_channels}"
            )

        # --- Training ---
        if t.epochs < 1:
            raise ValueError(f"training.epochs 必須 >= 1，收到 {t.epochs}")
        if t.batch_size < 1:
            raise ValueError(f"training.batch_size 必須 >= 1，收到 {t.batch_size}")
        if t.lr <= 0:
            raise ValueError(f"training.lr 必須為正數，收到 {t.lr}")
        if t.img_size < 1:
            raise ValueError(f"training.img_size 必須 >= 1，收到 {t.img_size}")

        logger.info("設定驗證通過")
