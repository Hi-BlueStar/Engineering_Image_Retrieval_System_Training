"""統一日誌工廠模組 (Unified Logging Factory Module)。

============================================================
提供全專案的統一日誌介面，全面取代散落的 ``print()`` 與
``console.print()`` 呼叫。

核心設計：
    - 每個模組透過 ``get_logger(__name__)`` 取得獨立的 Logger。
    - 所有 Logger 共享統一的格式與 Handler 配置。
    - 支援雙輸出：Console (StreamHandler) + File (RotatingFileHandler)。
    - 可選整合 Rich Handler 以保留美觀的 Console 輸出。

使用範例::

    from src.logger import get_logger
    logger = get_logger(__name__)
    logger.info("模型初始化完成，可訓練參數: %d", param_count)

============================================================
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# 預設格式常數
_DEFAULT_FMT = "[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

# 模組層級旗標：確保根 handler 只配置一次
_ROOT_CONFIGURED = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    use_rich: bool = False,
) -> None:
    """配置根 Logger 的 Handler 與格式。

    此函式應在應用程式入口（``prepare_data.py`` 或 ``train.py``）
    最早的位置呼叫一次。後續所有透過 ``get_logger()`` 取得的 Logger
    都會繼承此處的配置。

    Args:
        level: 日誌等級字串，例如 ``"INFO"``、``"DEBUG"``。
        log_file: 日誌檔案路徑。若為 ``None`` 則不輸出至檔案。
        use_rich: 是否使用 ``rich.logging.RichHandler`` 取代標準
            ``StreamHandler``。需要已安裝 ``rich`` 套件。

    Raises:
        ValueError: 當 ``level`` 不是合法的日誌等級時。
    """
    global _ROOT_CONFIGURED
    if _ROOT_CONFIGURED:
        return
    _ROOT_CONFIGURED = True

    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"無效的日誌等級: {level!r}")

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # --- Console Handler ---
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

    # --- File Handler ---
    if log_file is not None:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            filename=str(file_path),
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(
            logging.Formatter(fmt=_DEFAULT_FMT, datefmt=_DEFAULT_DATEFMT)
        )
        root_logger.addHandler(file_handler)

    # --- 抑制第三方庫的噪音 ---
    for noisy in ("PIL", "matplotlib", "urllib3", "fitz"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """取得命名 Logger 實例。

    如果根 Logger 尚未配置（例如在單元測試或 Notebook 中直接
    匯入模組），會自動套用一個最小配置以避免
    ``No handlers could be found`` 警告。

    Args:
        name: Logger 名稱，慣例使用 ``__name__``。

    Returns:
        logging.Logger: 已配置的 Logger。
    """
    if not _ROOT_CONFIGURED:
        setup_logging(level="INFO", use_rich=False)
    return logging.getLogger(name)


def _make_stream_handler(level: int) -> logging.StreamHandler:
    """建立標準 StreamHandler。

    Args:
        level: 日誌等級整數值。

    Returns:
        logging.StreamHandler: 已配置的 handler。
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(fmt=_DEFAULT_FMT, datefmt=_DEFAULT_DATEFMT)
    )
    return handler
