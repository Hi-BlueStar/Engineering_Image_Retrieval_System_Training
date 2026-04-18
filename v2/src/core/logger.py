"""
全域日誌系統 (Unified Logger)。

完全取代原本各處散落的 `print()`，使用標準 `logging` 建立統一格式，
支援同時輸出到 Console 與 File。
"""

import logging
import sys
from pathlib import Path
from rich.logging import RichHandler

def setup_logger(name: str, log_file: Path | str | None = None, level: int = logging.INFO) -> logging.Logger:
    """初始化並設定標籤化的全局 Logger，並與 Rich 美化整合。"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 如果已經有 handlers，避免重複添加
    if logger.handlers:
        return logger

    # 對於 Console 輸出，採用 Rich UI 以恢復精美體驗
    console_handler = RichHandler(rich_tracebacks=True, show_time=False, show_path=False, markup=True)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    # 對於 File 存檔，強制保留時間戳以確保 MLOps 紀錄追蹤
    if log_file:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(module)s:%(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# 提供一個全域共用呼叫接口
get_logger = setup_logger
