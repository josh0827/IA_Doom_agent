import time
from collections import deque


class FPSCounter:
    def __init__(self, window: int = 30):
        self.times = deque(maxlen=window)
        self.last = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        self.times.append(now - self.last)
        self.last = now
        if not self.times:
            return 0.0
        return len(self.times) / sum(self.times)
