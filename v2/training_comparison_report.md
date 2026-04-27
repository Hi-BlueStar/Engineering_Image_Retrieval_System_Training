# SimSiam 訓練系統深度對比分析報告 (Pipeline 1 vs Pipeline 2)

本報告針對原始訓練腳本 (`simsiam2_training.py`) 與第二代工程化管線 (`v2`) 的實作細節進行全面性對比，涵蓋設定管理、資料管線、模型架構、訓練流程及評估機制。

---

## 1. 架構設計思維 (Architecture Philosophy)

| 特性 | Pipeline 1 (simsiam2_training.py) | Pipeline 2 (v2) |
| :--- | :--- | :--- |
| **設計模式** | **腳本導向 (Script-based)**：核心邏輯集中在少數檔案中，適合快速原型開發。 | **模組化工程 (Modularized)**：將設定、資料、模型、訓練、實驗追蹤解耦，適合長期維護與大規模實驗。 |
| **依賴管理** | 內部定義輔助類別 (如 `ExperimentManager`)。 | 採用 **依賴注入 (Dependency Injection)**：`Trainer` 不負責建立物件，只負責執行流程。 |
| **代碼組織** | 邏輯與執行混合。 | 嚴格區分核心庫 (`src/`) 與執行入口 (`train.py`, `prepare_data.py`)。 |

---

## 2. 設定管理 (Configuration Management)

| 特性 | Pipeline 1 | Pipeline 2 |
| :--- | :--- | :--- |
| **載入方式** | `dataclass` 硬編碼預設值。 | **YAML + OmegaConf**：支援階層式設定檔與 CLI 動態覆寫 (如 `training.lr=1e-4`)。 |
| **類型安全** | 基本 Python 類型檢查。 | **強型別 AppConfig**：在訓練開始前進行深度驗證 (Validation)，如路徑檢查、數值合法性。 |
| **持久化** | 儲存為 `config.json`。 | 儲存為 `config.yaml` 並包含實驗 Metadata。 |

---

## 3. 資料管線 (Data Pipeline)

| 特性 | Pipeline 1 | Pipeline 2 |
| :--- | :--- | :--- |
| **前處理** | 基本影像載入與轉換。 | **完整預處理流程**：包含連通元件分析 (Connected Components)、拓撲剪枝 (Topology Pruning)、自動 Logo 移除。 |
| **資料增強** | **CPU 增強**：由 DataLoader workers 逐張處理，容易產生 CPU 瓶頸。 | **GPU 加速增強 (Kornia)**：將增強移至 GPU 批次執行，大幅提升訓練吞吐量。 |
| **縮放策略** | CPU 端直接 resize 為目標尺寸。 | **兩階段策略**：CPU 預縮放 (1024) 保留細節，GPU 再進行隨機裁切 (RandomResizedCrop)，減少失真。 |
| **資料集結構** | 簡單的 Run 資料夾掃描。 | 支援從原始 ZIP 解壓、自動分割 (Split) 到生成標準化資料集目錄的完整自動化。 |

---

## 4. 模型架構 (Model Architecture)

| 特性 | Pipeline 1 | Pipeline 2 |
| :--- | :--- | :--- |
| **核心架構** | SimSiam (Backbone + Projector + Predictor)。 | 相同，但 Projector/Predictor 的維度與層數透過設定檔參數化。 |
| **優化技術** | 標準 PyTorch 模型。 | **torch.compile (PT 2.0+)**：支援二進位編譯優化，提升 GPU 執行效率。 |
| **輸入通道** | 支援灰階 (1ch) 與 RGB (3ch)。 | 相同，且在修改第一層 Conv2d 權重時有更嚴謹的預訓練權重處理。 |

---

## 5. 訓練流程與優化 (Training & Optimization)

| 特性 | Pipeline 1 | Pipeline 2 |
| :--- | :--- | :--- |
| **訓練引擎** | 腳本內的 `run_single_session` 函數。 | **Trainer 類別**：封裝了 AMP、梯度裁切、計時器、回呼機制。 |
| **混合精度** | 未顯式優化 (或基本實現)。 | **深度整合 torch.amp**：包含自動縮放 (GradScaler) 與 Unscale 處理。 |
| **梯度管理** | 標準反向傳播。 | 支援 **梯度裁切 (Gradient Clipping)**，防止變形、形態學增強後的數值不穩定。 |
| **斷點恢復** | 無自動化 Resume。 | **自動 Resume 機制**：`CheckpointManager` 會自動尋找最新的 `.pth` 檔案並還原模型、優化器、排程器與 Epoch 進度。 |

---

## 6. 實驗追蹤與日誌 (Experiment Tracking & Logging)

| 特性 | Pipeline 1 | Pipeline 2 |
| :--- | :--- | :--- |
| **進度顯示** | `rich.progress` (基本)。 | **進階 Rich UI**：包含嵌套式進度條 (Epoch/Batch)、即時 Loss/LR/Std 顯示、剩餘時間預估。 |
| **日誌記錄** | Plotly HTML 報告 + CSV。 | **ExperimentTracker**：結構化儲存所有 Run 的指標、計時報告、環境 Metadata、CSV 與視覺化圖表。 |
| **效能分析** | 基本 `time.time()`。 | **PrecisionTimer**：精確記錄網路訓練時間與 I/O 牆鐘時間，用於分析瓶頸。 |

---

## 7. 評估機制 (Evaluation)

| 特性 | Pipeline 1 | Pipeline 2 |
| :--- | :--- | :--- |
| **驗證指標** | 僅 Negative Cosine Similarity (SSL Loss)。 | **雙重驗證**：除了 SSL Loss，還包含**實際檢索評估**。 |
| **下遊任務** | 無。 | **Retrieval Evaluator**：提取標註資料特徵，計算 IACS (類別內相似度)、Margin (對比度)、**Precision@K** (1, 5, 10...)。 |

---

## 總結

`simsiam2_training.py` 是一個成功的 **Proof of Concept (PoC)**，驗證了工程圖 SimSiam 的可行性；而 `v2` 則是一個**生產級的訓練框架**，它解決了 Pipeline 1 在大規模數據下的效能瓶頸 (GPU Augmentation)、穩定性問題 (Gradient Clipping, Validation) 以及實驗管理困難 (OmegaConf, Tracker)。

渲染此報告：[training_comparison_report.md](file:///home/master-user/Desktop/Engineering_Image_Retrieval_System_Training/v2/training_comparison_report.md)
