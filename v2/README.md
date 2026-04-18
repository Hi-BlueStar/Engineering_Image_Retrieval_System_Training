# 工程影像檢索系統 - 架構重構版 (v2)

此專案為使用 SimSiam 建立深度學習模型的端到端管線，完全符合 MLOps 標準的模組化架構。專為理解並萃取工程圖 (Line Art/CAD) 之表徵設計。

## 💡 架構設計準則

基於物件導向 (SOLID) 的高內聚、低耦合原則：

- **Config**: 透過 `OmegaConf` 與 `YAML` 達成程式與配置的解耦。
- **Data vs Model**: `Dataset` 負責與檔案系統溝通，`Model` 僅處理 PyTorch Tensor。
- **Trainer**: 控制深度學習 Epoch 的正向與反向傳播邏輯，不干涉資料來源。
- **Tools**: 不在主要 Pipeline 內的輔助工具 (如前處理與 PDF 轉圖) 被封裝為獨立元件。

---

## 📂 目錄結構

```text
v2/
├── configs/                  # [配置層] YAML 設定檔
│   └── default.yaml          # 預設參數 (Hyperparameters)
├── src/                      # [核心模組層]
│   ├── config/               # 結構化 Config 設定 (DataClass)
│   ├── data/                 # Dataset 與 Dataloader 工廠模式，包含資料增強
│   ├── engine/               # Trainer 訓練迴圈控制器
│   ├── models/               # SimSiam 網路與 backbone 模型
│   ├── tools/                # 資料準備的輔助工具 (Splitter, PDF Converter)
│   └── utils/                # 儀表紀錄 (Metrics, Experiment Logger, Timer)
├── scripts/                  # [執行入口]
│   ├── prepare_data.py       # Phase 1: 資料前置處理腳本
│   └── train.py              # Phase 2&3: 主訓練腳本
└── README.md
```

---

## 🛠️ 環境依賴與安裝指南

1. **Python 版本**: 建議使用 `>= 3.9`。
2. **依賴套件**:
   - `torch`, `torchvision` (PyTorch 生態系核心)
   - `omegaconf` (參數解析)
   - `pandas`, `plotly` (實驗報表產生)
   - `pymupdf` (`fitz` 用於 PDF 轉換)
   - `rich` (終端機介面美化)
   - `psutil`, `pynvml` (資源監控 - 可選)

### 安裝指令範例 (使用 Uv / pip)

```bash
# 若有 uv 可用
uv pip install torch torchvision omegaconf pandas plotly pymupdf rich psutil

# 或傳統 pip
pip install torch torchvision omegaconf pandas plotly pymupdf rich psutil
```

---

## 🚀 快速開始 (Quick Start)

請確認終端機的當前目錄為 `v2` 或您的專案根目錄，並將 PYTHONPATH 設定好 (使用 `python -m v2...`)。

### 階段 1：資料準備 (PDF 轉換、前處理與 Dataset 分割)

```bash
# 根據 default.yaml 中的設定進行所有資料的前置準備工作
python scripts/prepare_data.py
```

### 階段 2：啟動模型訓練

```bash
# 開始跑多 Run 的 SimSiam 訓練 (依照 YAML 設定)
python scripts/train.py

# ★ 進階：利用 OmegaConf 動態覆寫參數而不用改程式碼
python scripts/train.py training.batch_size=32 model.backbone=resnet50 learning_rate=1e-4
```

---

## 📖 核心 API 快速參考

若想呼叫此專案中的內部邏輯，可以參考以下範例：

```python
from v2.src.models.simsiam import SimSiam
from v2.src.engine.trainer import SimSiamTrainer

# 1. 初始化模型 (支援單通道工程圖優化)
model = SimSiam(backbone_name='resnet18', in_channels=1)

# 2. 建立封裝好的 Trainer
trainer = SimSiamTrainer(model=model, optimizer=optimizer, device="cuda")

# 3. 執行
loss, std = trainer.train_one_epoch(train_dataloader)
```

## 🔮 進階擴充 (如何更換資料集或模型)

- **新增資料增強**: 修改 `src/data/transforms.py` 內邏輯。
