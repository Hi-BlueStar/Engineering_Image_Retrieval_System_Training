# Note

## 1. Docker Image

SSH 生成
在 WSL 使用以下指令生成 SSH key

```bash
ssh-keygen -t rsa -b 4096 -f ./id_rsa -N ""
```

將 WSL 中的私鑰複製到 Windows (請根據您的實際路徑調整)

```powershell
cp "\\wsl.localhost\Ubuntu\home\yuestar\Engineering_Image_Retrieval_System\id_rsa" "$env:USERPROFILE\.ssh\id_rsa_docker"
```

清除舊的指紋記錄

```powershell
ssh-keygen -R "[localhost]:2222"
```

```bash
docker build -t="engineering_image_retrieval_system_dev:v3.0" .

docker compose down

docker compose up -d

# docker run -d --gpus all -p 2222:22 --name engineering_image_retrieval_system_dev engineering_image_retrieval_system_dev:v3.0
```

```bash
/opt/venv/bin/python
```

---

## 2. 流程使用檔案

```markdown
src/data_analysis/classify_pdf_type.py ->
src/pdf_to_image2.py -> src/image_preprocessing_batch_multiprocess2.py -> src/split_dataset.py -> src/model/simsiam2_training.py ->
```

## 執行

```bash
tmux new
```

1. 訓練

```bash
uv run python src/model/simsiam2_training.py
```

## 3. ruff 檢查

* **檢查 Import 順序：**

  ```bash
  ruff check .
  ```

  如果順序不對，Ruff 會報錯（例如 `I001 Import block is un-sorted or un-formatted`）。

* **自動修復 Import (Magic Command)：**
  這是您每天會用到的指令，它會自動排序 import 並移除未使用的引用：

  ```bash
  ruff check --fix .
  ```
