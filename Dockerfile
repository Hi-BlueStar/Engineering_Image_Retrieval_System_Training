# syntax=docker/dockerfile:1
# 保持啟用新版語法

# ==========================================
# Stage 1: Builder (編譯與依賴安裝層)
# 用途：負責下載、編譯、安裝所有 Python 套件
# ==========================================
# 使用 NVIDIA 官方提供的 CUDA 基礎映像檔 (依需求選擇版本)
# devel 版本包含編譯器 (nvcc)，runtime 版本僅包含執行庫
# 是要用來跑 Python / AI 專案 (如 PyTorch, TensorFlow)， 請絕對不要使用 13.1.0-base。 您應該選擇 12.9.1-cudnn-runtime (或者對應版本的 devel 進行編譯)。
# 原因： AI 框架高度依賴 cuDNN 和 cuBLAS (包含在 runtime 中)。若使用 base 版，您必須自己在 Dockerfile 中手動下載並安裝這些庫，過程極其繁瑣且容易出錯。
FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04 AS builder

# 避免安裝過程中的互動式提示
ENV DEBIAN_FRONTEND=noninteractive

# --- 1. 系統依賴安裝 ---
# 更新系統並安裝 Python 及必要工具
# 注意：雖然 uv 可以管理 Python 版本，但在 CUDA 環境下，
# 保留系統層級的 Python 開發庫 (python3-dev) 對於某些需要編譯的套件仍有幫助
RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3 \
    python3-dev \
    python-is-python3 \
    git \
    wget \
    ca-certificates

# --- 2. 安裝 uv ---
# 最佳實踐：從官方映像檔複製二進位檔，無需手動 curl 安裝
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# --- 3. 準備環境變數與目錄 ---
ENV UV_PROJECT_ENVIRONMENT="/opt/venv"
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_CACHE_DIR=/root/.cache/uv

WORKDIR /build

# --- 4. 安裝 Python 依賴 ---
# 先複製依賴定義檔，利用 Docker Layer Caching
COPY pyproject.toml .python-version uv.lock* ./

# 執行同步
# 注意：這裡不需要安裝 GUI 庫 (libgl1 等)，因為這裡只負責產生 Python 環境
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# ==========================================
# Stage 2: Final (最終執行層)
# 用途：提供乾淨、輕量的執行環境
# ==========================================
FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04 AS final

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/venv/bin:$PATH"

# --- 1. 系統依賴 (執行層) ---
# 只安裝「執行」所需的輕量套件，不含 GCC 或 Headers
# 這些是您原本列出的 GUI 和 Runtime 必要庫
RUN rm -f /etc/apt/apt.conf.d/docker-clean; \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python-is-python3 \
    ca-certificates \
    libgl1 \
    libglx-mesa0 \
    openssh-server \
    sudo \
    libegl1 \
    libdbus-1-3 \
    tmux \
    curl \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# --- 2. SSH 金鑰配置 ---
RUN mkdir /var/run/sshd && \
    mkdir -p /root/.ssh && \
    chmod 700 /root/.ssh

# 將本地的公鑰複製進容器 (請確保 id_rsa.pub 與 Dockerfile 在同一目錄)
COPY id_rsa.pub /root/.ssh/authorized_keys

# 設定嚴格權限 (SSH 規定若權限過開會拒絕連線)
RUN chmod 600 /root/.ssh/authorized_keys

# 修改 SSH 伺服器配置：
# 1. 允許 Root 透過金鑰登入，但禁止密碼
# 2. 停用密碼驗證 (PasswordAuthentication no)
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config && \
    sed -i 's/PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication no/PasswordAuthentication no/' /etc/ssh/sshd_config

# --- 3. 環境變數繼承修復 (關鍵) ---
# 確保透過 SSH 進來時，還能抓到 CUDA 路徑與 uv 的虛擬環境
RUN echo "export PATH=$PATH" >> /etc/profile && \
    echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> /etc/profile && \
    echo "source /etc/profile" >> /root/.bashrc

# --- 4. 安裝 uv 到最終環境 ---
# 我們直接從官方 image 複製，或者也可以從 builder 階段複製
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# --- 5. 複製編譯好的虛擬環境 ---
# 這是縮小體積的關鍵：只從 builder 階段複製 /opt/venv
COPY --from=builder /opt/venv /opt/venv

# --- 6. 專案建置與依賴同步 ---
# 設定工作目錄
WORKDIR /app

# 暴露 Port 22
EXPOSE 22

# --- 7. (選用) 驗證環境 ---
# 可以在這裡加入一行檢查，確保 CUDA 可被 Torch 抓到 (若 requirements 有 torch)
RUN python -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.version.cuda}')" || true

# --- 8. 啟動 SSH ---
# 使用 sh -c 同時啟動 SSH 並保持容器不退出
CMD ["sh", "-c", "/usr/sbin/sshd && sleep infinity"]
