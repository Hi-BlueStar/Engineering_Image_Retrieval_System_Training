"""統一日誌工廠模組 (Unified Logging Factory Module)。

============================================================
提供全專案的統一日誌介面，全面取代散落的 ``print()`` 呼叫。

設計特點：
    - 每個模組透過 ``get_logger(__name__)`` 取得獨立的 Logger。
    - 所有 Logger 共享統一的格式與 Handler 配置。
    - 支援主控台 (Rich/StreamHandler) 與檔案 (RotatingFileHandler) 雙重輸出。
    - 抑制不必要的第三方套件雜訊（如 PIL、fitz）。
============================================================
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# 預設日誌輸出格式
_DEFAULT_FMT = "[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

# 模組層級旗標，用以防範重複配置根 Logger
_ROOT_CONFIGURED = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    use_rich: bool = True,
    force: bool = False,
) -> None:
    """配置根 Logger 的 Handler 與格式。

    此功能應在應用程式進入點（如 `v4/main.py`）最早的位置呼叫一次。
    後續所有呼叫 `get_logger()` 的 Logger 皆會繼承此設定。

    Args:
        level: 日誌等級字串，例如 "INFO"、"DEBUG"。
        log_file: 日誌檔案路徑。若為 None 則不寫入檔案。
        use_rich: 是否使用 `rich.logging.RichHandler` 提供語法高亮。
        force: 是否強制清除現有配置並重新設定。
    """
    global _ROOT_CONFIGURED
    if _ROOT_CONFIGURED and not force:
        return

    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"無效的日誌等級: {level!r}")

    root_logger = logging.getLogger()
    
    if force:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    _ROOT_CONFIGURED = True
    root_logger.setLevel(numeric_level)

    # --- 主控台 (Console) 輸出配置 ---
    if use_rich:
        try:
            from rich.logging import RichHandler

            console_handler = RichHandler(
                level=numeric_level,
                show_time=True,
                show_path=False,
                markup=True,
                rich_tracebacks=True,
            )
        except ImportError:
            console_handler = _make_stream_handler(numeric_level)
    else:
        console_handler = _make_stream_handler(numeric_level)

    root_logger.addHandler(console_handler)

    # --- 檔案 (File) 輸出配置 ---
    if log_file is not None:
        try:
            file_path = Path(log_file)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                filename=str(file_path),
                maxBytes=10 * 1024 * 1024,  # 10 MB 限制
                backupCount=3,
                encoding="utf-8",
            )
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(
                logging.Formatter(fmt=_DEFAULT_FMT, datefmt=_DEFAULT_DATEFMT)
            )
            root_logger.addHandler(file_handler)
        except Exception as e:
            print(f"警告: 無法初始化日誌檔案 {log_file}: {e}", file=sys.stderr)

    # --- 抑制第三方庫的冗長日誌 ---
    for noisy in ("PIL", "matplotlib", "urllib3", "fitz"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """取得命名 Logger 實例。

    若根 Logger 尚未設定，會自動套用基本的 Stream 輸出配置，以防出現
    "No handlers could be found" 的系統警告。

    Args:
        name: Logger 名稱，慣用 __name__。

    Returns:
        logging.Logger: 已配置的 Logger 實例。
    """
    if not _ROOT_CONFIGURED:
        setup_logging(level="INFO", use_rich=False)
    return logging.getLogger(name)


def _make_stream_handler(level: int) -> logging.StreamHandler:
    """建立標準 StreamHandler 的輔助函式。"""
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(fmt=_DEFAULT_FMT, datefmt=_DEFAULT_DATEFMT)
    )
    return handler
