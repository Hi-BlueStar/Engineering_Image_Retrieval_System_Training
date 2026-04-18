"""精密計時器模組 (Precision Timer Module)。

============================================================
本模組提供支援「暫停 / 恢復」的高精度計時器，專為機器學習訓練管線設計。

核心設計考量：
1. **淨耗時 vs. 牆鐘時間**：訓練 epoch 的核心計算時間（淨耗時）
   需要排除 checkpoint 儲存、日誌寫入等 I/O 操作，才能準確衡量
   模型的訓練效能。本計時器透過 pause/resume 機制實現此區分。

2. **階層式計時**：`TimerCollection` 管理多個命名計時器，
   可在最終報告中呈現整體與各函式的耗時分佈。

3. **精度保證**：使用 `time.perf_counter()` 確保奈秒級精度，
   避免 `time.time()` 受系統時鐘調整影響。

使用範例：
    >>> timer = PrecisionTimer("train_epoch")
    >>> timer.start()
    >>> # ... 核心訓練邏輯 ...
    >>> timer.pause()    # 暫停：開始儲存 checkpoint
    >>> save_checkpoint()
    >>> timer.resume()   # 恢復：checkpoint 儲存完成
    >>> elapsed = timer.stop()

    >>> tc = TimerCollection()
    >>> t = tc.create("data_loading")
    >>> t.start(); load_data(); t.stop()
    >>> tc.summary()  # 取得所有計時器報告
============================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


# ============================================================
# 精密計時器 (Precision Timer)
# ============================================================


@dataclass
class TimerRecord:
    """單一計時器的最終紀錄。

    用於 TimerCollection 彙整報告時回傳的結構化資料。

    Attributes:
        name: 計時器名稱（對應操作名稱，例如 "train_epoch_01"）。
        net_elapsed: 淨耗時（秒），排除所有暫停時段。
        wall_elapsed: 牆鐘總耗時（秒），含暫停時段。
        pause_count: 暫停次數，可用於評估 I/O 操作的頻率。
        total_paused: 累計暫停時間（秒）。
    """

    name: str
    net_elapsed: float
    wall_elapsed: float
    pause_count: int
    total_paused: float


class PrecisionTimer:
    """支援暫停 / 恢復的高精度計時器。

    設計動機：
        在 ML 訓練迴圈中，我們通常只關心「核心計算」的耗時，
        而非包含 checkpoint 儲存、Rich console 輸出等 I/O 的總耗時。
        此計時器透過 pause() / resume() 明確分離兩者。

    狀態機：
        IDLE → start() → RUNNING ⇄ pause()/resume() → stop() → IDLE

    Args:
        name: 計時器名稱，用於報告中辨識。

    Raises:
        RuntimeError: 當操作順序不正確時（例如未 start 就 stop）。
    """

    # --- 狀態常數 ---
    _STATE_IDLE = "idle"
    _STATE_RUNNING = "running"
    _STATE_PAUSED = "paused"

    def __init__(self, name: str = "unnamed") -> None:
        """初始化計時器。

        Args:
            name: 計時器名稱，用於日誌輸出與報告辨識。
        """
        self.name = name
        self._state: str = self._STATE_IDLE

        # --- 內部時間追蹤 ---
        self._start_wall: float = 0.0       # 牆鐘起始時間
        self._accumulated: float = 0.0      # 累計的「運行中」淨耗時
        self._segment_start: float = 0.0    # 當前運行區段的起始時間
        self._pause_count: int = 0          # 暫停次數
        self._total_paused: float = 0.0     # 累計暫停時間
        self._pause_start: float = 0.0      # 暫停區段起始時間

    def start(self) -> None:
        """啟動計時器。

        將狀態從 IDLE 轉為 RUNNING，記錄起始時間。
        若計時器已在運行中，則靜默忽略（冪等操作）。

        Raises:
            RuntimeError: 當計時器處於 PAUSED 狀態時（應先 resume 或 stop）。
        """
        if self._state == self._STATE_RUNNING:
            return  # 冪等：重複 start 不影響
        if self._state == self._STATE_PAUSED:
            raise RuntimeError(
                f"計時器 '{self.name}' 處於暫停狀態，請先 resume() 或 stop()。"
            )

        now = time.perf_counter()
        self._state = self._STATE_RUNNING
        self._start_wall = now
        self._segment_start = now
        self._accumulated = 0.0
        self._pause_count = 0
        self._total_paused = 0.0

    def stop(self) -> float:
        """停止計時器並回傳淨耗時。

        無論計時器處於 RUNNING 或 PAUSED 狀態，均可呼叫 stop()。
        若為 PAUSED，最後一段暫停時間會計入 total_paused。

        Returns:
            float: 淨耗時（秒），排除所有暫停時段。

        Raises:
            RuntimeError: 當計時器尚未啟動（IDLE 狀態）時。
        """
        if self._state == self._STATE_IDLE:
            raise RuntimeError(
                f"計時器 '{self.name}' 尚未啟動，無法 stop()。"
            )

        now = time.perf_counter()

        if self._state == self._STATE_RUNNING:
            # 結算最後一段運行時間
            self._accumulated += now - self._segment_start
        elif self._state == self._STATE_PAUSED:
            # ⚠ 邊界條件：在暫停狀態直接 stop，最後一段暫停計入 total_paused
            self._total_paused += now - self._pause_start

        self._state = self._STATE_IDLE
        return self._accumulated

    def pause(self) -> None:
        """暫停計時器。

        將當前運行區段的時間累加至 accumulated，並記錄暫停起始時間。
        僅在 RUNNING 狀態下有效；其他狀態靜默忽略。
        """
        if self._state != self._STATE_RUNNING:
            return  # 防呆：非運行中不做任何事

        now = time.perf_counter()
        # 結算當前運行區段
        self._accumulated += now - self._segment_start
        self._pause_start = now
        self._pause_count += 1
        self._state = self._STATE_PAUSED

    def resume(self) -> None:
        """恢復計時器（從暫停中恢復）。

        記錄暫停時間並開啟新的運行區段。
        僅在 PAUSED 狀態下有效；其他狀態靜默忽略。
        """
        if self._state != self._STATE_PAUSED:
            return  # 防呆：非暫停中不做任何事

        now = time.perf_counter()
        self._total_paused += now - self._pause_start
        self._segment_start = now
        self._state = self._STATE_RUNNING

    @property
    def elapsed(self) -> float:
        """取得目前累計的淨耗時（不含暫停時段）。

        可在計時器運行中隨時查詢，不影響計時狀態。

        Returns:
            float: 目前淨耗時（秒）。
        """
        if self._state == self._STATE_RUNNING:
            return self._accumulated + (time.perf_counter() - self._segment_start)
        return self._accumulated

    @property
    def wall_elapsed(self) -> float:
        """取得牆鐘總耗時（含暫停時段）。

        Returns:
            float: 從 start() 至今的總牆鐘時間（秒）。
                   若計時器未啟動，回傳 0.0。
        """
        if self._state == self._STATE_IDLE and self._start_wall == 0.0:
            return 0.0
        return time.perf_counter() - self._start_wall

    def to_record(self) -> TimerRecord:
        """將當前計時器狀態轉換為 TimerRecord。

        通常在 stop() 之後呼叫，以取得結構化報告資料。

        Returns:
            TimerRecord: 包含名稱、淨耗時、牆鐘耗時等資訊的紀錄物件。
        """
        wall = (
            time.perf_counter() - self._start_wall
            if self._start_wall > 0
            else 0.0
        )
        return TimerRecord(
            name=self.name,
            net_elapsed=self._accumulated,
            wall_elapsed=wall,
            pause_count=self._pause_count,
            total_paused=self._total_paused,
        )


# ============================================================
# 計時器集合 (Timer Collection)
# ============================================================


class TimerCollection:
    """管理多個命名計時器的集合，用於產生階層式計時報告。

    設計動機：
        一個訓練管線通常包含多個可計時的子步驟
        （如 data_loading, forward_pass, checkpoint_save 等）。
        此類別提供統一的管理介面，並在最終產生結構化的計時報告。

    使用範例：
        >>> tc = TimerCollection()
        >>> t1 = tc.create("phase_1_data_prep")
        >>> t1.start(); prepare_data(); t1.stop()
        >>> t2 = tc.create("phase_2_training")
        >>> t2.start(); train(); t2.stop()
        >>> report = tc.summary()
    """

    def __init__(self) -> None:
        """初始化空的計時器集合。"""
        self._timers: dict[str, PrecisionTimer] = {}
        self._order: list[str] = []  # 保持建立順序，方便報告排序

    def create(self, name: str) -> PrecisionTimer:
        """建立並註冊一個新的命名計時器。

        若同名計時器已存在，會覆蓋先前的計時器。
        這是刻意設計：允許在多 Run 場景中重用相同名稱。

        Args:
            name: 計時器名稱，建議使用有意義的描述
                  （例如 "run_01_training", "data_split"）。

        Returns:
            PrecisionTimer: 新建立的計時器實例。
        """
        timer = PrecisionTimer(name=name)
        self._timers[name] = timer
        if name not in self._order:
            self._order.append(name)
        return timer

    def get(self, name: str) -> PrecisionTimer | None:
        """取得指定名稱的計時器。

        Args:
            name: 計時器名稱。

        Returns:
            PrecisionTimer | None: 計時器實例，若不存在則回傳 None。
        """
        return self._timers.get(name)

    def summary(self) -> list[dict]:
        """產生所有計時器的結構化摘要報告。

        按照計時器建立順序排列，每個計時器回傳一個字典，
        包含名稱、淨耗時、牆鐘耗時、暫停次數等資訊。

        Returns:
            list[dict]: 計時器報告列表，每個元素為：
                {
                    "name": str,
                    "net_elapsed_sec": float,
                    "wall_elapsed_sec": float,
                    "pause_count": int,
                    "total_paused_sec": float,
                }
        """
        results = []
        for name in self._order:
            timer = self._timers.get(name)
            if timer is None:
                continue
            rec = timer.to_record()
            results.append(
                {
                    "name": rec.name,
                    "net_elapsed_sec": round(rec.net_elapsed, 6),
                    "wall_elapsed_sec": round(rec.wall_elapsed, 6),
                    "pause_count": rec.pause_count,
                    "total_paused_sec": round(rec.total_paused, 6),
                }
            )
        return results

    def summary_table_rows(self) -> list[tuple[str, str, str, str]]:
        """產生可直接餵入 Rich Table 的行資料。

        Returns:
            list[tuple[str, str, str, str]]: 每列為
                (名稱, 淨耗時, 牆鐘耗時, 暫停次數)。
        """
        rows = []
        for item in self.summary():
            rows.append((
                item["name"],
                f"{item['net_elapsed_sec']:.3f}s",
                f"{item['wall_elapsed_sec']:.3f}s",
                str(item["pause_count"]),
            ))
        return rows
