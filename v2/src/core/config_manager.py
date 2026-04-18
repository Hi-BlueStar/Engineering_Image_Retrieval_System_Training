"""
配置管理器 (Configuration Manager)。

負責載入 YAML 設定檔，並映射成唯讀 (Read-only) 的 Dataclass 結構。
確保系統設計的「單一真相來源 (Single Source of Truth)」。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

try:
    import yaml
except ImportError:
    pass

from .exceptions import ConfigError

@dataclass(frozen=True)
class DataConfig:
    """資料層相關配置"""
    input_source: str
    raw_pdf_dir: str
    converted_image_dir: str
    preprocessed_image_dir: str
    dataset_root: str
    train_subpath: str
    val_subpath: str
    split_ratio: float
    base_seed: int
    n_runs: int
    use_csv_index: bool = True

@dataclass(frozen=True)
class PipelineFlagsConfig:
    """流程跳過標記"""
    skip_zip_extraction: bool
    skip_pdf_conversion: bool
    skip_preprocessing: bool

@dataclass(frozen=True)
class ModelConfig:
    """模型硬體參數"""
    backbone: str
    pretrained: bool
    in_channels: int
    proj_dim: int
    pred_hidden: int

@dataclass(frozen=True)
class TrainingConfig:
    """超參數與輸出配置"""
    img_size: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    num_workers: int
    save_freq: int
    output_dir: str
    exp_name: str
    resume_from: str | None = None

@dataclass(frozen=True)
class ConfigManager:
    """總配置管理者，整合所有設定，並凍結以防訓練中途被修改。"""
    data: DataConfig
    pipeline_flags: PipelineFlagsConfig
    pdf_extraction: Dict[str, Any]
    image_preprocessing: Dict[str, Any]
    model: ModelConfig
    training: TrainingConfig

    @classmethod
    def load_from_yaml(cls, yaml_path: str | Path) -> "ConfigManager":
        """從 YAML 檔案讀取並建構配置管理物件。

        Args:
            yaml_path (str | Path): YAML 設定檔的路徑。

        Returns:
            ConfigManager: 凍結的配置管理實例。

        Raises:
            ConfigError: 當找不到檔案或解析失敗時拋出。
        """
        path = Path(yaml_path)
        if not path.is_file():
            raise ConfigError(f"設定檔 {path} 不存在。")
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_config = yaml.safe_load(f)
            
            return cls(
                data=DataConfig(**raw_config.get("data", {})),
                pipeline_flags=PipelineFlagsConfig(**raw_config.get("pipeline_flags", {})),
                pdf_extraction=raw_config.get("pdf_extraction", {}),
                image_preprocessing=raw_config.get("image_preprocessing", {}),
                model=ModelConfig(**raw_config.get("model", {})),
                training=TrainingConfig(**raw_config.get("training", {}))
            )
        except Exception as e:
            raise ConfigError(f"讀取 YAML 檔案 {yaml_path} 時發生錯誤: {e}")
