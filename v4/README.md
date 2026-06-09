# SimSiam v4 工程圖紙自監督學習訓練系統使用說明

本系統為工程圖紙（CAD、零件線條圖）檢索系統的**自監督學習 (Self-Supervised Learning, SSL)** 訓練核心 `v4` 版本。

針對大圖像與幾何特徵敏感的特點，`v4` 在架構設計上實施了**非同步預取資料流**與 **GPU 端向量化隨機資料增強**，並全面解耦了代碼拓撲，實現極限吞吐能效（GPU 利用率平均 90%+）與高內聚的系統模組化。

---

## 🚀 核心優化特點

1.  **CPU Letterbox 等比例縮放**：在 CPU 階段將連通域分割出的零件圖以 CUBIC 插值等比例縮放，短邊對稱填充白色背景 (255)，規整為恆定的 $512 \times 512$ 灰階 Tensor。此舉可完全避免 `torch.compile` 對動態 Tensor 形狀反覆編譯的系統開銷。
2.  **`.npz` 快取與自動防呆資料管線**：
    *   **一鍵秒載**：系統會將資料集分割後的 Train/Val 圖像矩陣打包快取為單一 `dataset_cache.npz`。直接加載快取僅需 1 秒，完全消除了讀取數萬張隨機磁碟檔案的 I/O 瓶頸。
    *   **防呆備用**：若快取不存在，系統會向上檢查，自動執行「壓縮檔解壓 $\to$ PDF 轉 PNG $\to$ Logo 擦除與連通域裁切」的完整預處理，並重新生成快取。
3.  **非同步 GPU 預取器 (Double Buffering)**：透過鎖頁記憶體 (Pinned Memory) 與獨立 CUDA Stream 實作 `GPUPrefetcher`，在 GPU 計算當前 Batch 的同時，非同步傳輸下一個 Batch，使 PCIe 複製延遲與運算完全重疊。
4.  **AMP BF16 (Bfloat16)**：直接使用 BF16 混合精度訓練。BF16 的寬廣動態範圍確保對比學習點積計算不溢出，且無須 `GradScaler` 即可穩定收斂，大幅降低顯存開銷。
5.  **模型拓撲與損失解耦**：模型拆分為 `SimSiamEncoder` 與 `SimSiamPredictor`；損失函數模組 `SimSiamLossCriterion` 嚴格執行 `z.detach()` (Stop-gradient)，防止表徵空間坍塌。

---

## 📁 模組目錄結構

```markdown
v4/
├── main.py                     # CLI 主控制入口 (整合前處理與訓練排程)
├── README.md                   # 本說明文件
└── src/
    ├── __init__.py             # 模組初始化
    ├── config.py               # 強型別設定管理 (dataclass + 驗證器)
    ├── logger.py               # 控制台與檔案日誌 (Rich 整合)
    ├── data_pipeline.py        # 前處理、Letterbox、.npz 快取、DataLoader 與 GPUPrefetcher
    ├── models.py               # SimSiam 模型拓撲 (Encoder + Predictor) 與 BF16 支援
    ├── criterion.py            # 損失函數、Stop-Gradient 與坍塌監控
    ├── trainer.py              # Trainer 引擎 (AMP BF16 混合精度訓練循環)
    └── mlops.py                # Plotly HTML 報告生成、CSV 指標記錄器
```

---

## 🛠️ CLI 使用指南

所有指令應於專案根目錄下執行，並將 `PYTHONPATH` 設為當前目錄以順利匯入模組。

### 1. 執行單元測試驗證
在進行任何訓練前，可使用以下指令執行單元測試，以檢查 Letterbox 尺寸、預取型別轉換與 Stop-gradient 機制是否正常：
```bash
PYTHONPATH=. uv run python /home/master-user/.gemini/antigravity-ide/scratch/verify_v4.py
```

### 2. 極速快取訓練模式 (推薦)
若您已生成 `dataset_cache.npz` 快取檔，可加上 `--load_cached_npz` 旗標直接讀取快取，省略所有 PDF 轉檔與前處理時間：
```bash
PYTHONPATH=. uv run python v4/main.py --load_cached_npz
```

### 3. 一站式防呆前處理與訓練模式
若您是首次下載資料或要重新生成資料集，**不加** `--load_cached_npz` 即可。系統會自動檢測上游路徑並依序完成 PDF 轉換與零件圖前處理，最後自動產出快取並啟動訓練：
```bash
PYTHONPATH=. uv run python v4/main.py
```

### 4. CLI 參數動態覆寫
啟動器支援透過命令列直接覆寫主要的超參數設定：
*   `--epochs`: 訓練的總 Epoch 數量（例如 `--epochs 100`）。
*   `--batch_size`: 批次大小（例如 `--batch_size 32`）。
*   `--lr`: 學習率（例如 `--lr 3e-5`）。
*   `--backbone`: 骨幹網路種類，可選 `resnet18` 或 `resnet50`（例如 `--backbone resnet50`）。
*   `--use_bf16`: 是否啟用 AMP BF16 訓練，可選 `true` 或 `false`（例如 `--use_bf16 false`）。
*   `--compile_model`: 是否對模型啟用編譯加速，可選 `true` 或 `false`（例如 `--compile_model true`）。

**覆寫範例**：
```bash
PYTHONPATH=. uv run python v4/main.py --load_cached_npz --epochs 150 --batch_size 32 --lr 1e-5 --use_bf16 true --compile_model true
```

---

## 📊 輸出產物說明

訓練啟動後，將在指定輸出路徑（預設為 `outputs_v4/simsiam_v4/`）生成以下實驗檔案：

*   `config.json`：該次執行的完整超參數配置。
*   `metadata.json`：系統硬體、作業系統、CUDA 與 Git commit 版本的元數據。
*   `training_log.csv`：每個 Epoch 的詳細指標記錄（包括 `train_loss`, `val_loss`, `train_std`, `val_std`, `lr`, `duration`），即時寫入，防範訓練中斷。
*   `training_report.html`：**Plotly WebGL 互動式分析圖表**。您可以使用本機瀏覽器開啟此檔案，點選或拖曳縮放互動式檢視對稱 Cosine Loss、特徵標準差（坍塌監控）與 Learning Rate 的收斂趨勢。
*   `checkpoints/`：
    *   `checkpoint_latest.pth`：最新保存的 Epoch 權重。
    *   `checkpoint_best.pth`：驗證損失最小的最佳模型權重。
    *   `checkpoint_epoch_xxxx.pth`：根據 `save_freq` 儲存的週期歷史模型。
