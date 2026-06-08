# SimSiam 自監督學習工程圖檢索系統 (v3) 實戰教學

本教學旨在引導您在 Docker 容器化開發環境中，使用已腳本化之 `v3/simsiam_training.py` 執行完整自監督學習工程圖特徵提取與檢索檢驗的訓練管線。

---

## 目錄
1. [環境配置與 Docker 指令操作](#1-環境配置與-docker-指令操作)
2. [系統架構設計](#2-系統架構設計)
3. [資料預處理管線 (Data Preparation Pipeline)](#3-資料預處理管線-data-preparation-pipeline)
4. [驗證評估與檢索指標 (LOO & Macro-mAP)](#4-驗證評估與檢索指標-loo--macro-map)
5. [訓練優化與 collapse 監控](#5-訓練優化與-collapse-監控)
6. [指令執行範例](#6-指令執行範例)

---

## 1. 環境配置與 Docker 指令操作

本專案使用 Docker 以確保與 CUDA GPU 驅動以及特定相依套件（如 PyTorch、Kornia、OpenCV 等）之版本相容性。為維持在長途訓練中的穩定度，我們結合 `tmux` 虛擬終端工具進行背景程序管理。

請在主機端依序執行以下指令建置並啟動您的開發環境：

### 步驟 1：建置開發鏡像
```bash
docker build -t="engineering_image_retrieval_system_dev:v3.0" .
```
> **說明**：
> - 啟用 Docker BuildKit 建置引擎，基於 `nvidia/cuda:12.9.1` 基礎鏡像建置。
> - 在 Builder 階段利用 `uv` 極速下載並封裝所有的 Python 套件依賴，並建置在 `/opt/venv` 虛擬環境中，以避免汙染宿主機。

### 步驟 2：清理舊容器與背景運行新容器
```bash
docker compose down
docker compose up -d
```
> **說明**：
> - `docker compose down`：清理過往殘留容器、網卡與磁碟配置。
> - `docker compose up -d`：以背景守護進程 (Detached) 模式啟動。將主機端專案根目錄映射掛載至容器內的 `/workspace` 目錄，並配置所有 GPU（`count: all`）以及 100GB 的共享記憶體（`shm_size: '100gb'`，此設定對 PyTorch 多線程 DataLoader 之穩定性至關重要）。

### 步驟 3：建立 tmux 背景終端
```bash
tmux new
```
> **說明**：
> - 建立一個新的 `tmux` 會話（通常預設為編號 `0`），預防因為 SSH 斷線或本機終端關閉導致訓練任務中途中斷。

### 步驟 4：重新連接 tmux 會話
```bash
tmux attach -t 0
```
> **說明**：
> - 連接回剛才建立的會話。若未來不幸斷線，隨時可以使用此指令重回訓練現場。

### 步驟 5：進入 Docker 容器內部
```bash
docker exec -ti engineering_image_retrieval_system_dev bash
```
> **說明**：
> - `exec -ti`：以互動模式進入正在運行中、名為 `engineering_image_retrieval_system_dev` 的容器內部 bash shell。

### 步驟 6：在容器內使用 uv 執行 Python 腳本
```bash
uv run python -m v3.simsiam_training [ARGS...]
```
> **說明**：
> - `uv run`：自動尋找已在環境變數激活的 `/opt/venv` 虛擬環境中執行 Python，兼具隔離性與高效能。

---

## 2. 系統架構設計

本系統核心基於 **SimSiam (Simple Siamese)** 自監督對比學習模型，專門針對工程圖的線條與拓樸特徵進行對稱特徵學習：

```mermaid
graph TD
    X[輸入影像群組] -->|資料擴增 v1| X1[視角 1 (View 1)]
    X -->|資料擴增 v2| X2[視角 2 (View 2)]
    
    X1 --> Backbone1[ResNet18 Backbone]
    X2 --> Backbone2[ResNet18 Backbone]
    
    Backbone1 -->|f1| Proj1[3層 Projector]
    Backbone2 -->|f2| Proj2[3層 Projector]
    
    Proj1 -->|z1| Pred1[2層 Predictor]
    Proj2 -->|z2| Pred2[2層 Predictor]
    
    Pred1 -->|p1| Loss1[負餘弦損失 D]
    Proj2 -->|z2| Loss1
    
    Pred2 -->|p2| Loss2[負餘弦損失 D]
    Proj1 -->|z1| Loss2
    
    style Loss1 fill:#f9f,stroke:#333,stroke-width:2px
    style Loss2 fill:#f9f,stroke:#333,stroke-width:2px
```

### 防止特徵崩塌 (Representation Collapse) 的雙重安全機制
自監督學習極易陷入所有圖片特徵都輸出成常數的「坍塌」陷阱。SimSiam 使用以下學理機制避免：
1. **不對稱結構 (Asymmetric Architecture)**：
   - 骨幹網路 (Backbone) 採用 `ResNet18`，第一層卷積已修改以相容單通道灰階圖。
   - **Projector** (投影層)：由 3 層 MLP 組成。隱藏層為 512 維，輸出 $d=2048$ 維。所有全連接層皆含有批次正規化 (BN)，但**輸出層不使用 ReLU**。
   - **Predictor** (預測層)：由 2 層 MLP 組成瓶頸 (Bottleneck) 設計。輸入/輸出為 2048 維，隱藏層縮減為 128 維，隱藏層含 BN+ReLU，但**輸出層無 BN 且無 ReLU**。
2. **停止梯度回傳 (Stop-Gradient)**：
   - 損失函數定義為對稱負餘弦相似度平均：
     $$ L = \frac{1}{2} \mathcal{D}(p_1, \text{stop\_gradient}(z_2)) + \frac{1}{2} \mathcal{D}(p_2, \text{stop\_gradient}(z_1)) $$
   - 在 PyTorch 中，通過對目標表徵執行 `z2.detach()`，截斷其中一個分支的梯度反向傳播。

---

## 3. 資料預處理管線 (Data Preparation Pipeline)

工程圖與一般的自然影像（如 ImageNet）具有本質上的差異：其特徵主要由背景中的黑白色塊分佈、連通輪廓、線條交點所決定。因此，設計了嚴密的預處理管線：

### A. Gifu Logo 擦除
工程圖角落常帶有製造商 Gifu (吉輔) 的 Logo。此 Logo 普遍出現在所有工程圖中，若不予以移除，SimSiam 會優先學到該局部 Logo，從而造成特徵特化。
- **定位方式**：優先使用預設角落遮罩，其次使用 OpenCV 模板匹配 (Template Matching)，最後降級為角落高密度二值像素像素密度自動偵測。
- **擦除方式**：將偵測出的 bounding box 區域以純白色（$255$）填充。

### B. 連通元件 (Connected Component, CC) 提取與裁切
大張的工程圖紙包含許多框架線、標題欄與多個零件示意圖。為了聚焦在單一幾何零件上：
1. 套用 **Otsu 二值化反轉**（背景為黑 $0$，前景線條為白 $255$）。
2. 使用 8-連通域分析（`cv2.connectedComponentsWithStats`）提取出獨立圖形。
3. **圖框過濾**：排除包夾 Logo 區域的元件，並利用面積比例限制過濾掉大於整張圖比例 `0.8` 的外框架線。
4. **輸出子圖**：依 bounding box 面積大小排序，提取前 `top_n`（預設 5）個元件，並還原成「白底黑線」格式（外圍補 `padding=2` 邊框後輸出成 `comp_*.png`）。

### C. 專家篩選 (Expert Filter) 與資料集劃分
為了評估特徵提取的檢索能力，從評估集抽取了 50 張查詢種子 (Seeds)。
- 在 Notebook 模式中，提供互動式介面供專家確認 Ground Truth (GT) 零件子圖，並自動混入 500 張干擾項背景圖，建置驗證集 $V$。
- 在 CLI 腳本模式中，為了避免卡死，自動退化為程式化篩選 (Programmatic Fallback)——透過無標籤特徵與目錄結構自動映射，自動化完成對齊並切分出 `T_small`、`T_large` 訓練集與獨立的驗證集 `V`。

---

## 4. 驗證評估與檢索指標 (LOO & Macro-mAP)

在自監督訓練過程中，系統會定期在不參與訓練的獨立驗證集 $V$ 上進行 Leave-One-Out 檢索測試：

```
驗證集 V = [ 50 組專家定義之幾何相似圖集 + 800 張背景干擾圖 ]
```

### Leave-One-Out (LOO) 檢索策略
- 輪流將驗證集 $V$ 中的每一張影像當作查詢鍵（Query），剩下的 $|V|-1$ 張影像作為檢索庫（Gallery）。
- 計算 Query 與 Gallery 之間餘弦相似度並進行降序排序。

### 核心評估指標
1. **Macro-mAP (組間平均精度均值)**：
   - 對 50 個形狀類別各自求算組內的 Mean Average Precision (mAP) 後，再進行算術平均。
   - 該指標能有效避免因為某些類別樣本過多或過少而導致的統計偏差，是判斷特徵空間好壞的最終標準。
2. **Margin (特徵差距)**：
   - 組內相似度平均值與組外相似度平均值之差（$IACS - Inter$）。Margin 越大，代表特徵分群效果越好。
3. **Top-1 Accuracy**：
   - 檢索結果中，最相似的第一張圖是否與 Query 同類別。

### I/O 瓶頸優化：記憶體快取 (RAM Caching)
- 由於驗證集 $V$ 大小固定且評估頻繁，驗證集在初始化時會利用 Python 的 `ThreadPoolExecutor` 多線程並行將所有圖片進行 Letterbox 縮放與 ToTensor 轉換，**一次性全載入 RAM 中**。
- 這徹底消除了訓練期間頻繁讀取 SSD/HDD 產生的 I/O 瓶頸，使得 LOO 特徵檢索時間從數十秒降至接近 0 秒。

---

## 5. 訓練優化與 collapse 監控

### A. 線性學習率縮放規則 (Linear Scaling Rule)
根據 SimSiam 理論，當 Batch Size 變動時，學習率應同步縮放。
- 實際學習率計算：
  $$ \text{Scaled LR} = \text{base\_lr} \times \frac{\text{Batch Size}}{256} $$
- 當您在小資料集 $T_{small}$ 訓練時（Batch Size 預設為 32），基礎學習率 `base_lr=0.05` 將自動縮放為 `0.00625`。在大資料集 $T_{large}$（Batch Size 預設 128）時，縮放為 `0.025`。

### B. SGD 與餘弦衰減
優化器使用帶有動量的 SGD（Momentum=0.9，Weight Decay=1e-4），並採用 `CosineAnnealingLR` 將學習率在 100 Epochs 內平滑衰減至 0。
- 為了確保穩定性，**BN 的權重 (gamma, beta) 與 Bias (偏置) 不參與 L2 權重衰減**。

### C. 混合精度加速 (AMP)
在 CUDA 環境下自動啟用半精度混合計算（`torch.amp.autocast`），以獲得 2 倍以上的訓練吞吐量。

### D. 特徵坍塌即時監控
在每個 Batch 的 Forward 階段，程式會計算特徵維度 $z$ 在正規化後的標準差：
- 在 $d=2048$ 的特徵空間中，若特徵隨機均勻分布，其各維度標準差均值應接近於 $\frac{1}{\sqrt{d}} \approx 0.022$。
- 若標準差跌破警告閾值 `--collapse_warning_threshold`（預設為 0.01），代表模型輸出開始趨同（即崩塌中），腳本會發出警告：
  `[⚠️ 警告] Epoch XXX: 特徵標準差 (0.0042) 低於臨界值 0.01`

---

## 6. 指令執行範例

所有操作均以腳本 `v3/simsiam_training.py` 為核心，支援多種分流模式：

### 模式 A：執行全套資料預處理（解壓、轉檔、裁切、建置）
若您是第一次下載專案，或修改了前處理參數，請執行：
```bash
# 在 Docker tmux 視窗內執行
uv run python -m v3.simsiam_training --prepare_data
```
> 若欲手動變更 PDF 解析度、前處理裁切框等，可加掛參數：
> `uv run python -m v3.simsiam_training --prepare_data --top_n 3 --padding 4`

### 模式 B：執行單次子實驗訓練
依據實驗計畫書，您可以自由指定資料集規模與對照組別。
- **Baseline**：無前處理、無擴增（複製產生兩個完全相同的 view）。
- **Exp_A**：連通域前處理、Kornia GPU 雙視角隨機增強。
- **Exp_B**：無前處理、Kornia GPU 雙視角隨機增強。

範例：在小資料集（$T_{small}$）上跑實驗 A（對照組 A）：
```bash
uv run python -m v3.simsiam_training --dataset_type T_small --experiment_type Exp_A --epochs 100
```
> **產出結果**：
> - 訓練完畢後，結果會存放在 `outputs_v3_local/Exp_A_T_small/` 目錄中。
> - `training_log.csv`：包含每一輪的 Loss、特徵 Std 以及 validation 指標。
> - `summary.json`：存放最佳的 Macro-mAP 與耗時。
> - `training_report.html`：**互動式 Plotly 趨勢 HTML 報告**，可下載至本機以瀏覽器開啟，動態縮放查看 Loss 衰減與 Macro-mAP 之演進曲線。

### 模式 C：自動依序跑完 6 組對照子實驗
如果您想完全自動化重現《實驗計畫書》規定的所有實驗，請執行：
```bash
uv run python -m v3.simsiam_training --run_all --epochs 100
```
> **運行流程**：
> 1. 自動順序執行 Baseline_T_small -> Baseline_T_large -> Exp_A_T_small -> Exp_A_T_large -> Exp_B_T_small -> Exp_B_T_large。
> 2. 結束後，會在 `outputs_v3_local/experiments_comparison_report.md` 生成 Markdown 比較表格，以便一目了然分析前處理、資料增強及資料規模對特徵表示的影響。

---
祝您訓練順利！如有問題，請參閱腳本原始碼中詳細的中文註解與 Config 參數說明。
