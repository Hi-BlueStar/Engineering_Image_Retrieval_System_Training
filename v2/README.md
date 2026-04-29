# SimSiam v2 — 工程圖自監督表徵學習訓練管線

> **Engineering Drawing Self-Supervised Representation Learning Pipeline**

基於 [SimSiam (Chen & He, CVPR 2021)](https://arxiv.org/abs/2011.10566) 的模組化訓練系統，專為工程圖 (CAD / Line Art) 檢索場景優化。

---

## 📋 目錄

1. [專案簡介與架構圖說](#1-專案簡介與架構圖說)
2. [環境依賴與安裝指南](#2-環境依賴與安裝指南)
3. [快速開始 (Quick Start)](#3-快速開始-quick-start)
4. [效能優化說明](#4-效能優化說明)
5. [函數與模組目錄 (API Reference)](#5-函數與模組目錄-api-reference)
6. [進階使用](#6-進階使用)

---

## 1. 專案簡介與架構圖說

### 1.1 背景

本專案將原始的單檔式 SimSiam 訓練腳本重構為「高內聚、低耦合」的模組化系統，遵循以下工程準則：

- **職責分離**：設定、資料、模型、訓練、實驗追蹤五大模組完全解耦。
- **極致效能**：AMP 混合精度、多進程資料處理、高效 DataLoader。
- **統一日誌**：全面禁用 `print()`，使用結構化 `logging` 模組。
- **SOLID 原則**：依賴注入、工廠模式、開放封閉擴展設計。

### 1.2 雙管線架構

系統分為兩條**完全獨立**的管線：

```markdown
┌─────────────────────────────────────────────────────────────┐
│  Pipeline 1: 資料預處理 (prepare_data.py)                     │
│                                                             │
│  ZIP/RAR ─→ PDF→Image ─→ CC分析與拓撲剪枝 ─→ Train/Val 分割      │
│  (多線程)    (多進程)      (多進程)        (分層隨機)           │
└─────────────────────────────────────────────────────────────┘
                            ↓ 檔案系統（唯一契約）
┌─────────────────────────────────────────────────────────────┐
│  Pipeline 2: 模型訓練 (train.py)                              │
│                                                             │
│  DataLoader ─→ SimSiam 訓練 ─→ Checkpoint ─→ 報表            │
│  (pin_memory,   (AMP FP16)     (best/last)   (Plotly HTML)  │
│   prefetch)                                                 │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 專案目錄結構

```markdown
v2/
├── configs/
│   └── default.yaml              # 統一 YAML 設定 (OmegaConf)
├── src/
│   ├── config.py                 # 設定載入與驗證
│   ├── logger.py                 # 統一 logging 工廠
│   ├── data/                     # 資料處理模組
│   │   ├── extraction.py         #   ZIP/RAR 解壓縮
│   │   ├── pdf_converter.py      #   PDF→Image 多進程轉換
│   │   ├── preprocessing.py      #   連通元件分析與拓撲剪枝
│   │   ├── topology.py           #   拓撲特徵計算與保拓撲剪枝
│   │   └── splitter.py           #   資料集分割
│   ├── dataset/                  # PyTorch Dataset & DataLoader
│   │   ├── transforms.py         #   資料增強策略
│   │   ├── dataset.py            #   UnlabeledImageDataset
│   │   └── dataloader.py         #   DataLoader 工廠
│   ├── model/                    # 模型架構
│   │   ├── backbone.py           #   Backbone 工廠 (Registry)
│   │   ├── simsiam.py            #   SimSiam nn.Module
│   │   └── loss.py               #   損失函數 + 坍塌監控
│   ├── training/                 # 訓練引擎
│   │   ├── trainer.py            #   Trainer (DI + AMP)
│   │   ├── checkpoint.py         #   Checkpoint 管理器
│   │   └── timer.py              #   精密計時器
│   └── experiment/               # 實驗追蹤
│       ├── tracker.py            #   ExperimentTracker + RunTracker
│       └── reporter.py           #   Plotly 視覺化報表
├── prepare_data.py               # Pipeline 1 入口
├── train.py                      # Pipeline 2 入口
├── requirements.txt              # 依賴清單
└── README.md                     # 本文件
```

---

## 2. 環境依賴與安裝指南

### 2.1 系統需求

| 項目 | 最低需求 | 建議 |
| ------ | --------- | ----- |
| Python | 3.10+ | 3.11+ |
| PyTorch | 2.0+ | 2.2+ (含 CUDA) |
| GPU | 可選 | NVIDIA GPU + CUDA 12.x |
| RAM | 8 GB | 16+ GB |
| 磁碟 | 視資料集大小 | SSD 推薦 |

### 2.2 安裝步驟

```bash
# 1. 在專案根目錄建立虛擬環境（建議使用 uv）
uv venv .venv --python 3.11
source .venv/bin/activate

# 2. 安裝依賴
uv pip install -r v2/requirements.txt

# 或使用 pip：
# pip install -r v2/requirements.txt
```

### 2.3 額外依賴

- **7z (p7zip)**：解壓 RAR 格式所需，`apt install p7zip-full`。
- **PyMuPDF (fitz)**：PDF 轉換引擎，已包含在 requirements.txt。

---

## 3. 快速開始 (Quick Start)

### 3.1 設定檔

編輯 `configs/default.yaml`，修改資料路徑與超參數：

```yaml
data:
  raw_zip_path: "data/your_data.zip"     # 壓縮檔路徑（或 null）
  raw_pdf_dir: "data/your_pdfs"          # PDF 資料夾
  dataset_dir: "dataset"                 # 輸出目錄

training:
  epochs: 200
  batch_size: 64
  lr: 2.0e-5
```

### 3.2 執行資料預處理

```bash
sudo rm -rf data/converted_images data/preprocessed_images data/raw_pdfs dataset
```

temp：

```bash
uv run python v2/prepare_data.py --config v2/configs/default.yaml data.skip_extraction=true data.skip_pdf_conversion=true data.skip_preprocessing=true
uv run python v2/train.py --config v2/configs/default.yaml training.max_batches=200
```

```bash
# 視覺化與檢查前處理效果
uv run python v2/analyze_data.py preview --config v2/configs/default.yaml --n-samples 20 --dpi 400
```

```bash
# 完整流程：解壓 → PDF轉圖 → 前處理 → 分割
uv run python v2/prepare_data.py --config v2/configs/default.yaml

# 跳過已完成的步驟（自動偵測 + CLI 覆寫）
uv run python v2/prepare_data.py --config v2/configs/default.yaml data.skip_extraction=true
```

### 3.3 執行訓練

```bash
# 測試訓練流程：
uv run python v2/train.py --config v2/configs/minimal_test.yaml
```

```bash
# 基本訓練
uv run python v2/train.py --config v2/configs/default.yaml

# 覆寫超參數
uv run python v2/train.py --config v2/configs/default.yaml \
    training.lr=3e-4 \
    training.epochs=100 \
    training.batch_size=32
```

### 3.4 查看結果

訓練完成後，結果會儲存在 `outputs/` 目錄：

```markdown
outputs/simsiam_exp_20260419_143000/
├── config.json                # 完整設定備份
├── metadata.json              # 系統/硬體資訊
├── timing_report.json         # 分步計時明細
├── overall_summary.json       # 所有 Run 結果彙整
└── Run_01_Seed_42/
    ├── training_log.csv       # 逐 epoch 指標
    ├── training_report.html   # Loss 曲線（互動式）
    ├── std_report.html        # 坍塌監控圖
    └── checkpoints/
        ├── checkpoint_best.pth
        ├── checkpoint_last.pth
        └── checkpoint_epoch_0010.pth
```

---

## 4. 效能優化說明

### 4.1 I/O 與資料管線

| 最佳化 | 模組 | 說明 |
| -------- | ------ | ------ |
| 多線程 ZIP 解壓 | `data/extraction.py` | `ThreadPoolExecutor` 分塊平行，zlib 釋放 GIL |
| 7z 多核 RAR | `data/extraction.py` | `-mmt=on` 啟用所有 CPU 核心 |
| 多進程 PDF 渲染 | `data/pdf_converter.py` | `ProcessPoolExecutor` 繞過 GIL |
| 多進程前處理 | `data/preprocessing.py` | `multiprocessing.Pool` + 資源守門員 |
| `pin_memory=True` | `dataset/dataloader.py` | 加速 CPU→GPU 資料搬移 |
| `prefetch_factor=2` | `dataset/dataloader.py` | 預取下一批次資料 |
| `persistent_workers` | `dataset/dataloader.py` | 避免每 epoch 重啟 worker |

### 4.2 GPU 運算

| 最佳化 | 模組 | 說明 |
| -------- | ------ | ------ |
| AMP (FP16) | `training/trainer.py` | `torch.amp.autocast` + `GradScaler`，train 與 eval 均啟用 |
| `set_to_none=True` | `training/trainer.py` | 梯度清零使用 None 取代零張量 |
| `non_blocking=True` | `training/trainer.py` | 張量搬移不阻塞 CPU |
| 向量化損失 | `model/loss.py` | `F.normalize` + 廣播點積，無 Python 迴圈 |
| `drop_last=True` | `dataset/dataloader.py` | 避免不完整 batch 導致 BatchNorm 異常 |

### 4.3 拓撲感知預處理 (Topology-aware Preprocessing)

| 功能 | 模組 | 說明 |
| -------- | ------ | ------ |
| 拓撲分類 | `data/topology.py` | 依據孔洞數 ($\beta_1$) 將元件分類為 Complex/Simple |
| 拓撲剪枝 | `data/preprocessing.py` | 移除無孔洞的細小噪點，保留核心結構 |
| 遞進式保拓撲清理 | `data/topology.py` | 遞增 Kernel 尺寸進行多次形態學清理，每次自動驗證拓撲不變性，實現最大化去噪 |

---

## 5. 函數與模組目錄 (API Reference)

### 5.1 `src/config.py` — 設定管理

```python
AppConfig.from_yaml(yaml_path, cli_overrides=None) -> AppConfig
# 從 YAML 載入設定，合併 CLI 覆寫

cfg.validate() -> None
# 驗證所有參數的合理性

cfg.to_dict() -> dict
# 序列化為 JSON 相容字典
```

### 5.2 `src/logger.py` — 日誌工廠

```python
setup_logging(level="INFO", log_file=None, use_rich=False) -> None
# 配置根 Logger（入口腳本呼叫一次）

get_logger(name) -> logging.Logger
# 取得命名 Logger（各模組呼叫）
```

### 5.3 `src/data/` — 資料處理

```python
# 解壓縮
extract_archive(archive_path, output_dir, skip=False) -> bool

# PDF 轉換
convert_pdfs_to_images(pdf_dir, output_dir, dpi=100, max_workers=None, skip=False) -> int

# 影像前處理 (包含 CC 分析、拓撲分類、Logo 移除與剪枝)
preprocess_images(config: PreprocessConfig, skip=False) -> dict

# 資料集分割
split_dataset(source_root, output_root, run_name, split_ratio=0.8, seed=42) -> tuple[Path, Path]
```

### 5.4 `src/dataset/` — 資料載入

```python
# DataLoader 工廠
create_dataloaders(train_path, val_path, cfg, in_channels=1) -> tuple[DataLoader, DataLoader, int, int]

# 增強策略
EngineeringDrawingAugmentation(img_size=512, mean=[0.5], std=[0.5])
# __call__(img) -> (view1, view2)

make_inference_transform(img_size=512, mean=[0.5], std=[0.5]) -> T.Compose
```

### 5.5 `src/model/` — 模型架構

```python
# Backbone 工廠
create_backbone(name, pretrained=False, in_channels=3) -> tuple[nn.Module, int]

# SimSiam 模型
model = SimSiam(backbone="resnet18", proj_dim=2048, pred_hidden=512, pretrained=True, in_channels=1)
p1, p2, z1, z2 = model(x1, x2)

# 損失函數
loss = simsiam_loss(p1, p2, z1, z2) -> torch.Tensor
std = calculate_collapse_std(z) -> float
```

### 5.6 `src/training/` — 訓練引擎

```python
# Trainer（依賴注入）
trainer = Trainer(model, optimizer, scheduler, scaler, checkpoint_mgr, device)
result = trainer.fit(train_loader, val_loader, epochs=200, epoch_callback=fn)

# Checkpoint 管理
ckpt_mgr = CheckpointManager(ckpt_dir, save_freq=10, config_dict=cfg.to_dict())
ckpt_mgr.save(model, optimizer, epoch, val_loss, is_best)
state = ckpt_mgr.load(path, model, optimizer)
```

### 5.7 `src/experiment/` — 實驗追蹤

```python
# 實驗追蹤器
tracker = ExperimentTracker(config, timers)
run_tracker = tracker.create_run("Run_01_Seed_42")
run_tracker.log_epoch(metrics_dict)
tracker.save_summary(run_results)

# Plotly 報表
generate_run_reports(df, run_dir, run_name)
```

---

## 6. 進階使用

### 6.1 新增自定義 Backbone

在 `src/model/backbone.py` 中註冊新的 backbone：

```python
from src.model.backbone import _register

@_register("efficientnet_b0")
def _efficientnet_b0(pretrained: bool):
    import torchvision.models as models
    net = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    )
    feat_dim = net.classifier[1].in_features
    net.classifier = nn.Identity()
    return net, feat_dim
```

然後在 YAML 中使用：

```yaml
model:
  backbone: "efficientnet_b0"
```

**不需要修改 Trainer 或任何其他模組。**

### 6.2 自定義增強策略

繼承或替換 `EngineeringDrawingAugmentation`：

```python
class MyCustomAugmentation:
    def __init__(self, img_size: int):
        # 自定義增強邏輯
        ...

    def __call__(self, img) -> tuple[torch.Tensor, torch.Tensor]:
        return view1, view2  # 回傳雙視角
```

在 `src/dataset/dataloader.py` 中替換 transform 建構邏輯即可。

### 6.3 自定義損失函數

透過 Trainer 的 `loss_fn` 參數注入：

```python
def my_custom_loss(p1, p2, z1, z2):
    # 自定義損失計算
    return loss

trainer = Trainer(
    model, optimizer, scheduler, scaler, ckpt_mgr,
    loss_fn=my_custom_loss,
)
```

### 6.4 從 Checkpoint 恢復訓練

```python
from src.training.checkpoint import CheckpointManager

ckpt_mgr = CheckpointManager("path/to/checkpoints")
state = ckpt_mgr.load("checkpoint_last.pth", model, optimizer)
start_epoch = state["epoch"]
```

### 6.5 僅執行部分預處理步驟

```bash
# 只做 PDF 轉換（跳過解壓和前處理）
python v2/prepare_data.py --config v2/configs/default.yaml \
    data.skip_extraction=true \
    data.skip_preprocessing=true

# 只做資料集分割（跳過所有前處理）
python v2/prepare_data.py --config v2/configs/default.yaml \
    data.skip_extraction=true \
    data.skip_pdf_conversion=true \
    data.skip_preprocessing=true
```

---

## 📄 License

MIT License

## 📚 References

- Chen, X., & He, K. (2021). *Exploring Simple Siamese Representation Learning*. CVPR 2021.
