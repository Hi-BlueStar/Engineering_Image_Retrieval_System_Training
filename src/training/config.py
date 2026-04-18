"""訓練設定模組 (Training Configuration Module)。

============================================================
集中管理 SimSiam 訓練管線的所有可調參數。

設計考量：
1. **單一真相來源 (Single Source of Truth)**：所有超參數、路徑、
   控制旗標皆定義於 TrainingConfig dataclass，避免散落各處。

2. **可序列化**：透過 to_dict() 可完整匯出為 JSON，
   確保實驗可完全重現。

3. **階段控制**：skip_* 旗標允許跳過已完成的前處理步驟，
   直接從中間階段開始訓練。
============================================================
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


# ============================================================
# 訓練設定 (Training Configuration)
# ============================================================


@dataclass
class TrainingConfig:
    """SimSiam 訓練管線的完整設定。

    涵蓋從原始資料（PDF/ZIP）到模型訓練完成的所有參數。
    以 dataclass 實作以確保型別安全與可序列化。

    設計原則：
        - 所有路徑參數使用 str（而非 Path），以便 JSON 序列化。
        - 布林旗標控制各階段是否執行，支援從中間狀態恢復。
        - 預設值反映最常見的使用情境。

    Attributes:
        請參閱各欄位的行內文件說明。
    """

    # --------------------------------------------------------
    # 資料來源設定 (Data Source)
    # --------------------------------------------------------

    raw_zip_path: str | None = None
    """原始 ZIP 壓縮檔路徑。當 raw_pdf_dir 不存在時，會嘗試從此路徑解壓縮。"""

    raw_pdf_dir: str = "data/raw_pdfs"
    """已分類的 PDF 資料夾路徑。PDF 應按類別放置於子資料夾中。"""

    # --------------------------------------------------------
    # 資料管線中繼路徑 (Pipeline Intermediate Paths)
    # --------------------------------------------------------

    converted_image_dir: str = "data/converted_images"
    """PDF 轉換後的影像輸出目錄。"""

    preprocessed_image_dir: str = "data/preprocessed_images"
    """影像前處理（連通元件分析等）後的輸出目錄。"""

    dataset_dir: str = "dataset"
    """資料集分割的輸出根目錄。每個 Run 會在此目錄下建立子資料夾。"""

    # --------------------------------------------------------
    # PDF 轉換設定 (PDF Conversion)
    # --------------------------------------------------------

    pdf_dpi: int = 100
    """PDF 轉換的解析度 (DPI)。100 DPI 適合工程圖的訓練用途。"""

    pdf_max_workers: int = 16
    """PDF 轉換的最大並行執行緒數。"""

    # --------------------------------------------------------
    # 影像前處理設定 (Image Preprocessing)
    # --------------------------------------------------------

    preprocess_top_n: int = 5
    """前處理時保留的大元件數。"""

    preprocess_remove_largest: bool = True
    """是否移除面積最大的元件（通常是圖框/邊框）。"""

    preprocess_padding: int = 2
    """元件裁切時的邊界填充像素數。"""

    preprocess_max_attempts: int = 400
    """隨機排列時每個元件的最大嘗試放置次數。"""

    preprocess_random_count: int = 10
    """每張影像產生的隨機排列變體數。"""

    preprocess_max_workers: int = 12
    """前處理的最大並行程序數。"""

    # --------------------------------------------------------
    # 資料集分割設定 (Dataset Split)
    # --------------------------------------------------------

    split_ratio: float = 0.8
    """訓練集佔總資料的比例。"""

    n_runs: int = 5
    """多 Run 訓練的重複次數。每個 Run 使用不同的隨機種子進行分割。"""

    base_seed: int = 42
    """基礎隨機種子。第 i 個 Run 使用 base_seed + i。"""

    # --- 資料集內部路徑結構 ---

    train_subpath: str = "Component_Dataset/train"
    """Run 資料夾內，訓練集的相對路徑。"""

    val_subpath: str = "Component_Dataset/val"
    """Run 資料夾內，驗證集的相對路徑。"""

    # --------------------------------------------------------
    # 模型設定 (Model Architecture)
    # --------------------------------------------------------

    backbone: str = "resnet18"
    """骨幹網路選擇。支援 'resnet18' 或 'resnet50'。"""

    pretrained: bool = True
    """是否載入 ImageNet 預訓練權重。對工程圖灰階輸入，
    會將 RGB 權重平均化為單通道。"""

    in_channels: int = 1
    """輸入圖片通道數。1 = 灰階（工程圖預設），3 = RGB。"""

    proj_dim: int = 2048
    """Projector 輸出維度（即 embedding 維度）。"""

    pred_hidden: int = 512
    """Predictor 隱藏層維度（瓶頸層）。"""

    # --------------------------------------------------------
    # 訓練設定 (Training Hyperparameters)
    # --------------------------------------------------------

    img_size: int = 512
    """圖片輸入尺寸（正方形，單位：像素）。"""

    epochs: int = 200
    """每個 Run 的訓練 epoch 數。"""

    batch_size: int = 64
    """每個 batch 的樣本數。"""

    lr: float = 2e-5
    """學習率。SSL 原論文建議值較高，此處針對工程圖微調。"""

    weight_decay: float = 1e-5
    """AdamW 的權重衰減係數。"""

    num_workers: int = 8
    """DataLoader 的工作程序數。Windows 建議設為 0。"""

    img_exts: tuple[str, ...] = (".jpg", ".png", ".bmp", ".tif", ".webp")
    """支援的影像檔案副檔名。"""

    # --------------------------------------------------------
    # 輸出設定 (Output)
    # --------------------------------------------------------

    output_dir: str = "outputs"
    """實驗結果的輸出根目錄。"""

    exp_name: str = "simsiam_exp"
    """實驗名稱前綴，用於建立輸出子目錄。"""

    save_freq: int = 10
    """Checkpoint 儲存頻率（每 N 個 epoch 儲存一次）。"""

    # --------------------------------------------------------
    # 階段控制旗標 (Pipeline Phase Control)
    # --------------------------------------------------------

    skip_zip_extraction: bool = False
    """是否跳過 ZIP 解壓縮。若 raw_pdf_dir 已存在且有內容，自動跳過。"""

    skip_pdf_conversion: bool = False
    """是否跳過 PDF→影像轉換。若 converted_image_dir 已存在且有內容，自動跳過。"""

    skip_preprocessing: bool = False
    """是否跳過影像前處理。若 preprocessed_image_dir 已存在且有內容，自動跳過。"""

    def to_dict(self) -> dict:
        """將設定轉為字典，便於 JSON 序列化。

        Returns:
            dict: 所有設定欄位的字典表示。
                  tuple 類型會自動轉為 list。
        """
        d = asdict(self)
        # dataclass asdict 會自動將 tuple 轉為 list，滿足 JSON 需求
        return d

    def validate(self) -> None:
        """驗證設定的合理性。

        檢查數值範圍、路徑格式等基本約束條件。
        建議在訓練開始前呼叫。

        Raises:
            ValueError: 當設定值不合規時。
        """
        if self.pdf_dpi <= 0:
            raise ValueError(f"pdf_dpi 必須為正整數，收到 {self.pdf_dpi}")
        if not 0.0 < self.split_ratio < 1.0:
            raise ValueError(
                f"split_ratio 必須在 (0, 1) 之間，收到 {self.split_ratio}"
            )
        if self.n_runs < 1:
            raise ValueError(f"n_runs 必須 >= 1，收到 {self.n_runs}")
        if self.epochs < 1:
            raise ValueError(f"epochs 必須 >= 1，收到 {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size 必須 >= 1，收到 {self.batch_size}")
        if self.lr <= 0:
            raise ValueError(f"lr 必須為正數，收到 {self.lr}")
        if self.backbone not in ("resnet18", "resnet50"):
            raise ValueError(
                f"backbone 僅支援 'resnet18' 或 'resnet50'，收到 '{self.backbone}'"
            )
