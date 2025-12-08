# 使用 NVIDIA 官方提供的 CUDA 基礎映像檔 (依需求選擇版本)
# devel 版本包含編譯器 (nvcc)，runtime 版本僅包含執行庫
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# 避免安裝過程中的互動式提示
ENV DEBIAN_FRONTEND=noninteractive

# 更新系統並安裝 Python 及必要工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libegl1-mesa \
    libxkbcommon-x11-0 \
    libxcb-cursor0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-xfixes0 \
    libx11-xcb-dev \
    libdbus-1-3

# 建立 python 別名 (將 python3 對應到 python)
RUN ln -s /usr/bin/python3 /usr/bin/python

# 升級 pip
RUN pip3 install --upgrade pip