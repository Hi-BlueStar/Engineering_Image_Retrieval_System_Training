# Engineering Image Retrieval System (SimSiam Training Pipeline v2)

本專案為針對工程圖 (CAD/Line Art) 特性所設計的高效能、高擴展性 SimSiam 自監督學習框架。
經歷了數次重大深度重構，本架構現已徹底解決大資料集下的硬碟與記憶體瓶頸，升級為符合生產環境標準的「雙端點 (Dual Entry)」大數據引擎流水線。

---

## 1. 專案簡介與架構圖說 (Architecture Overview)

系統全面解耦，摒除以往混用的單一進入點，改採「資料前處理」與「模型訓練」分離的設計，並導入**零拷貝大數據索引 (Zero-Copy CSV Index)** 革命。

### 架構模組

* **雙主控端點 (`process_data.py` 與 `train.py`)**：取代舊時代混雜的 `main.py`，無論您要清洗資料還是純粹執行顯卡運算，皆各司其職互不干擾。
* **配置中心 (`configs/` 與 `src.core`)**：`default_config.yaml` 支援聰明的單一 `input_source` (無論丟入 ZIP、RAR 還是已解壓目錄皆自動判定)。所有參數掛載為不變的 Dataclass。
* **資料管線 (`src.data`)**：搭載「零拷貝」切分器。當處理數百萬張圖片時，不再使用緩慢的 `shutil.copy` 塞爆硬碟，而是生成絕對路徑的 Pandas `.csv` 以微秒級傳遞給 Dataloader；徹底防禦 PyTorch 多執行緒 RNG 污染 (Worker RNG Bug Fix)。
* **訓練引擎 (`src.trainer`)**：通用性的訓練主迴圈 `TrainerEngine`。不僅搭載 AMP 精度加速，更內部整合了 VRAM OOM 即時預防與斷點續傳 (Resume from Checkpoint) 安全機制。

### 系統流向圖 (Execution Flow)

```text
[ configs/default_config.yaml ] (Single Source of Truth)
         │
         ├─────────────────────────────────────────┐
         ▼                                         ▼
[ process_data.py ]                      [ train.py ]
         │                                         │
         ├──> 處理 Zip/Rar 解壓與 PDF 分離           ├──> (載入 CSV 索引)
         ├──> CV2 平行特徵分析 (Batch Queue防爆)      ├──> DataLoader (防 Worker 種子重疊 + DMA加速)
         ├──> DatasetSplitter (分層抽樣)           ├──> SimSiamModel (PyTorch Neural Net)
         │                                         │
         └──> [輸出]: train_index.csv              └──> TrainerEngine (Fit Loop)
              [輸出]: val_index.csv                       ├──> 內建 VRAM 掃描掃除、斷點續傳讀取
                                                          ├──> AMP 自動混和精度
                                                          └──> 觸發 Callbacks (報告、Checkpoint)
```

---

## 2. 環境依賴與安裝指南 (Environment Dependencies)

請確保您的環境具備 `uv` 或 `pip` 作為套件管理器。

1. **基礎安裝**

   ```bash
   # 建議使用 Python 3.10+ 環境
   pip install torch torchvision
   ```

2. **必備第三方套件 (處理報表與資料)**

   ```bash
   pip install pyyaml rich pandas plotly opencv-python
   ```

3. **PDF 解壓與圖片轉化 (可選，若資料已轉成 png 則免)**

   ```bash
   pip install pymupdf  # (fitz 庫)
   # 若需支援 RAR 解壓縮，請確保系統已安裝 7z (apt install p7zip-full)
   ```

---

## 3. 快速開始 (Quick Start)

所有管線切換皆由 `configs/default_config.yaml` 控制。現在管線完全分開操作，使得多卡伺服器除錯變得極為明確。

### 步驟 A: 大數據資料集建立 (只需執行一次)

```bash
# 系統將讀取 yaml 中的 input_source，自動解壓縮、找連通元件，並切分生成訓練用 CSV
python v2/process_data.py
```

### 步驟 B: 神經網路模型訓練

```bash
# 直接讀取上方建立好的 CSV 開始 GPU 暴力迭代
python v2/train.py
```

### ✨ 其他實用操作

* **斷點續傳 (Crash Recovery)**：伺服器若無預警中斷，只要在 `yaml` 內將 `resume_from` 指向 `outputs/.../checkpoint_last.pth`，再次執行 `train.py` 即可原木原樣接續權重、優化器進度與 Epoch 刻度。
* **快速檢視報表**：每次實驗會在 `outputs/` 目錄產生 `loss_curve.html` (互動式下降折線分析) 以及 `collapse_std_curve.html` (用以分析是否發生自監督特徵降維崩塌)。

---

## 4. 特殊效能優化與防禦性說明 (Performance & Safety Hardening)

1. **大資料零拷貝切割 (Zero-Copy CSV Split)**:
   傳統的切分會在驗證集與訓練集之間創造數以十萬計的圖片複本，癱瘓 I/O。V2 導入 Pandas `.to_csv()` 只將絕對路徑送給 DataLoader，速度提升百倍並解省巨幅容量。
2. **PyTorch 多進程 RNG 強制隔離 (Worker RNG Bug Fix)**:
   由於 SimSiam 高度依賴 Data Augmentation 製造兩面特徵，V2 設計了 `_worker_init_fn` 強制打亂 PyTorch fork 時共用 Seed 的致命 Bug，保證特徵不會撞車。
3. **動態 VRAM 監控與 OOM 掃除 (VRAM Defense)**:
   `system_monitor.py` 現在嵌入於訓練迴圈中，每個 Epoch 檢查並呼叫 `empty_cache()` 清除碎裂記憶區塊。
4. **驗證集絕對決定性雙視角 (Deterministic Validation Output)**:
   解決了過去 Evaluation 時，亂數造成的 Baseline 浮動假象，取而代之的是強制的 `Horizontal Flip` 鏡像雙視角，既維持了兩個 Input 的硬性需求，也達到 Loss 震盪完美的純淨對比。

---

## 5. 進階使用與 API 參考 (Advanced Customization)

得益於超低耦合設計：

### 欲新增或更換自定義模型結構 (例如 MoCo, BYOL)?

1. 您不需修改任何 `TrainerEngine` 的基礎邏輯！
2. 只要在 `src.models` 目錄下繼承 `BaseModel` 類別，實作其抽象方法 `forward(x1, x2)` 回傳所需的向量。
3. 如果 Loss 的公式不同，在 `src.models.loss.py` 中新增公式即可，並在 `engine.py` 抽換一行呼叫即可完美運作。

### 欲掛載新版指標監控 (例如 Wandb 或是 TensorBoard)?

在 `src/trainer/callbacks/` 建立一個新檔案如 `wandb_logger.py`，繼承 `Callback` 介面，並實做 `on_epoch_end`。然後在 `train.py` 中的 `callbacks = [...]` 中注入，完全不需要改動主訓練迴圈。
