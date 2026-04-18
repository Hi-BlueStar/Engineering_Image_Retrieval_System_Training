"""
精確計時器模組 (Precision Timer)。

提供支援暫停 (Pause) / 恢復 (Resume) 的計時器，
用以扣除 IO 或 Checkpoint 寫入耗時，取得純淨的運算神經網路時間。
"""

import time

class PrecisionTimer:
    """用以測量不包含資料夾 IO、Checkpoints 寫入等操作的純計算時間。"""
    def __init__(self):
        self._start_time = None
        self._total_time = 0.0
        self._is_running = False

    def start(self):
        self._start_time = time.perf_counter()
        self._is_running = True

    def pause(self):
        if self._is_running and self._start_time is not None:
            self._total_time += time.perf_counter() - self._start_time
            self._is_running = False

    def resume(self):
        if not self._is_running:
            self._start_time = time.perf_counter()
            self._is_running = True

    def stop(self) -> float:
        self.pause()
        return self._total_time

    def reset(self):
        self._total_time = 0.0
        self._is_running = False
