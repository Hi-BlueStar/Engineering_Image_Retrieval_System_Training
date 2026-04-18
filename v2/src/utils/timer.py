"""精準計時器模組 (Precision Timer)。

支援「暫停 / 恢復」的高精度計時器，用於區分核心訓練時間與 I/O 等非核心時間。
"""
import time
from dataclasses import dataclass


@dataclass
class TimerRecord:
    name: str
    net_elapsed: float
    wall_elapsed: float
    pause_count: int
    total_paused: float


class PrecisionTimer:
    _STATE_IDLE = "idle"
    _STATE_RUNNING = "running"
    _STATE_PAUSED = "paused"

    def __init__(self, name: str = "unnamed"):
        self.name = name
        self._state = self._STATE_IDLE
        self._start_wall = 0.0
        self._accumulated = 0.0
        self._segment_start = 0.0
        self._pause_count = 0
        self._total_paused = 0.0
        self._pause_start = 0.0

    def start(self):
        if self._state == self._STATE_RUNNING:
            return
        if self._state == self._STATE_PAUSED:
            raise RuntimeError(f"計時器 '{self.name}' 處於暫停狀態。")
        now = time.perf_counter()
        self._state = self._STATE_RUNNING
        self._start_wall = now
        self._segment_start = now
        self._accumulated = 0.0
        self._pause_count = 0
        self._total_paused = 0.0

    def stop(self) -> float:
        if self._state == self._STATE_IDLE:
             return 0.0
        now = time.perf_counter()
        if self._state == self._STATE_RUNNING:
            self._accumulated += now - self._segment_start
        elif self._state == self._STATE_PAUSED:
            self._total_paused += now - self._pause_start
        self._state = self._STATE_IDLE
        return self._accumulated

    def pause(self):
        if self._state != self._STATE_RUNNING:
            return
        now = time.perf_counter()
        self._accumulated += now - self._segment_start
        self._pause_start = now
        self._pause_count += 1
        self._state = self._STATE_PAUSED

    def resume(self):
        if self._state != self._STATE_PAUSED:
            return
        now = time.perf_counter()
        self._total_paused += now - self._pause_start
        self._segment_start = now
        self._state = self._STATE_RUNNING

    def to_record(self) -> TimerRecord:
        wall = (time.perf_counter() - self._start_wall) if self._start_wall > 0 else 0.0
        return TimerRecord(
            name=self.name,
            net_elapsed=self._accumulated,
            wall_elapsed=wall,
            pause_count=self._pause_count,
            total_paused=self._total_paused,
        )


class TimerCollection:
    def __init__(self):
        self._timers = {}
        self._order = []

    def create(self, name: str) -> PrecisionTimer:
        timer = PrecisionTimer(name=name)
        self._timers[name] = timer
        if name not in self._order:
            self._order.append(name)
        return timer

    def summary(self) -> list[dict]:
        results = []
        for name in self._order:
            timer = self._timers.get(name)
            if not timer: continue
            rec = timer.to_record()
            results.append({
                "name": rec.name,
                "net_elapsed_sec": round(rec.net_elapsed, 6),
                "wall_elapsed_sec": round(rec.wall_elapsed, 6),
                "pause_count": rec.pause_count,
                "total_paused_sec": round(rec.total_paused, 6),
            })
        return results
