"""結構化設定定義 (Structured Configuration)。

============================================================
負責定義設定檔的資料結構，利於 IDE 自動補全與型別檢查。
使用 OmegaConf 來融合 YAML 檔案與命令列參數。
============================================================
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from omegaconf import OmegaConf


@dataclass
class DataConfig:
    raw_zip_path: str | None = None
    raw_pdf_dir: str = "data/raw_pdfs"
    converted_image_dir: str = "data/converted_images"
    preprocessed_image_dir: str = "data/preprocessed_images"
    dataset_root: str = "dataset"
    
    pdf_dpi: int = 100
    pdf_max_workers: int = 16
    
    preprocess_top_n: int = 5
    preprocess_remove_largest: bool = True
    preprocess_padding: int = 2
    preprocess_max_attempts: int = 400
    preprocess_random_count: int = 10
    preprocess_max_workers: int = 12
    
    split_ratio: float = 0.8
    n_runs: int = 5
    base_seed: int = 42
    train_subpath: str = "Component_Dataset/train"
    val_subpath: str = "Component_Dataset/val"


@dataclass
class ModelConfig:
    backbone: str = "resnet18"
    pretrained: bool = True
    in_channels: int = 1
    proj_dim: int = 2048
    pred_hidden: int = 512


@dataclass
class TrainingConfig:
    img_size: int = 512
    epochs: int = 200
    batch_size: int = 64
    lr: float = 2e-5
    weight_decay: float = 1e-5
    num_workers: int = 8
    img_exts: List[str] = field(default_factory=lambda: [".jpg", ".png", ".bmp", ".tif", ".webp"])


@dataclass
class OutputConfig:
    output_dir: str = "outputs"
    exp_name: str = "simsiam_v2_exp"
    save_freq: int = 10


@dataclass
class PipelineConfig:
    skip_zip_extraction: bool = False
    skip_pdf_conversion: bool = False
    skip_preprocessing: bool = False


@dataclass
class Config:
    """全域配置入口。"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    @classmethod
    def load(cls, yaml_path: str | Path, cli_args: list[str] | None = None) -> "Config":
        """從 YAML 與 Command-Line 載入設定。
        
        Args:
            yaml_path: YAML 設定檔路徑。
            cli_args: 命令列參數列表（例如 sys.argv[1:]）。
            
        Returns:
            Config: 融合後的結構化設定物件。
        """
        base_schema = OmegaConf.structured(cls)
        yaml_conf = OmegaConf.load(str(yaml_path))
        merged = OmegaConf.merge(base_schema, yaml_conf)
        
        if cli_args:
            cli_conf = OmegaConf.from_cli(cli_args)
            merged = OmegaConf.merge(merged, cli_conf)
            
        return merged # type: ignore
