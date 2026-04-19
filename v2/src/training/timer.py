"""精密計時器模組 (Precision Timer Module)。

============================================================
提供支援「暫停 / 恢復」的高精度計時器，專為 ML 訓練管線設計。

核心設計：
    - **淨耗時 vs. 牆鐘時間**：透過 pause/resume 排除
      checkpoint 儲存等非核心 I/O 操作。
    - **階層式計時**：``TimerCollection`` 管理多個命名計時器。
    - **精度保證**：使用 ``time.perf_counter()``。

使用範例::

    timer = PrecisionTimer("train_epoch")
    timer.start()
    # ... 核心訓練 ...
    timer.pause()   # checkpoint I/O
    save_checkpoint()
    timer.resume()
    elapsed = timer.stop()
============================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class TimerRecord:
    """單一計時器的結構化紀錄。

    Attributes:
        name: 計時器名稱。
        net_elapsed: 淨耗時（秒），排除暫停。
        wall_elapsed: 牆鐘總耗時（秒）。
        pause_count: 暫停次數。
        total_paused: 累計暫停時間（秒）。
    """

    name: str
    net_elapsed: float
    wall_elapsed: float
    pause_count: int
    total_paused: float


class PrecisionTimer:
    """支援暫停/恢復的高精度計時器。

    狀態機::

        IDLE → start() → RUNNING ⇄ pause()/resume() → stop() → IDLE

    Args:
        name: 計時器名稱。

    Raises:
        RuntimeError: 操作順序不正確時。
    """

    _IDLE = "idle"
    _RUNNING = "running"
    _PAUSED = "paused"

    def __init__(self, name: str = "unnamed") -> None:
        self.name = name
        self._state: str = self._IDLE
        self._start_wall: float = 0.0
        self._accumulated: float = 0.0
        self._segment_start: float = 0.0
        self._pause_count: int = 0
        self._total_paused: float = 0.0
        self._pause_start: float = 0.0

    def start(self) -> None:
        """啟動計時器。

        冪等操作：重複 start 不影響。

        Raises:
            RuntimeError: 計時器處於 PAUSED 狀態時。
        """
        if self._state == self._RUNNING:
            return
        if self._state == self._PAUSED:
            raise RuntimeError(
                f"計時器 '{self.name}' 處於暫停狀態，請先 resume() 或 stop()。"
            )
        now = time.perf_counter()
        self._state = self._RUNNING
        self._start_wall = now
        self._segment_start = now
        self._accumulated = 0.0
        self._pause_count = 0
        self._total_paused = 0.0

    def stop(self) -> float:
        """停止計時器並回傳淨耗時。

        Returns:
            float: 淨耗時（秒）。

        Raises:
            RuntimeError: 計時器未啟動時。
        """
        if self._state == self._IDLE:
            raise RuntimeError(
                f"計時器 '{self.name}' 尚未啟動，無法 stop()。"
            )
        now = time.perf_counter()
        if self._state == self._RUNNING:
            self._accumulated += now - self._segment_start
        elif self._state == self._PAUSED:
            self._total_paused += now - self._pause_start
        self._state = self._IDLE
        return self._accumulated

    def pause(self) -> None:
        """暫停計時器。僅在 RUNNING 狀態有效。"""
        if self._state != self._RUNNING:
            return
        now = time.perf_counter()
        self._accumulated += now - self._segment_start
        self._pause_start = now
        self._pause_count += 1
        self._state = self._PAUSED

    def resume(self) -> None:
        """恢復計時器。僅在 PAUSED 狀態有效。"""
        if self._state != self._PAUSED:
            return
        now = time.perf_counter()
        self._total_paused += now - self._pause_start
        self._segment_start = now
        self._state = self._RUNNING

    @property
    def elapsed(self) -> float:
        """取得目前累計淨耗時（不影響狀態）。

        Returns:
            float: 淨耗時（秒）。
        """
        if self._state == self._RUNNING:
            return self._accumulated + (
                time.perf_counter() - self._segment_start
            )
        return self._accumulated

    @property
    def wall_elapsed(self) -> float:
        """取得牆鐘總耗時。

        Returns:
            float: 總耗時（秒）。
        """
        if self._state == self._IDLE and self._start_wall == 0.0:
            return 0.0
        return time.perf_counter() - self._start_wall

    def to_record(self) -> TimerRecord:
        """轉換為結構化紀錄。

        Returns:
            TimerRecord: 計時紀錄。
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


class TimerCollection:
    """管理多個命名計時器的集合。

    使用範例::

        tc = TimerCollection()
        t = tc.create("data_loading")
        t.start(); load_data(); t.stop()
        report = tc.summary()
    """

    def __init__(self) -> None:
        self._timers: dict[str, PrecisionTimer] = {}
        self._order: list[str] = []

    def create(self, name: str) -> PrecisionTimer:
        """建立並註冊命名計時器。

        Args:
            name: 計時器名稱。

        Returns:
            PrecisionTimer: 新建立的計時器。
        """
        timer = PrecisionTimer(name=name)
        self._timers[name] = timer
        if name not in self._order:
            self._order.append(name)
        return timer

    def get(self, name: str) -> PrecisionTimer | None:
        """取得指定計時器。

        Args:
            name: 計時器名稱。

        Returns:
            PrecisionTimer | None: 計時器（若存在）。
        """
        return self._timers.get(name)

    def summary(self) -> list[dict]:
        """產生所有計時器的結構化報告。

        Returns:
            list[dict]: 按建立順序排列的計時報告。
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
