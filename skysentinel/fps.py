"""Frame-rate control: cap processing speed, and measure the achieved rate.

The cap lets you trade speed for cost (throttle a fast machine, or keep several
cameras within budget); removing it lets a high-FPS input run at full temporal
resolution, which the tracker and trajectory classifier use for precision.
"""

from __future__ import annotations

import time


class FpsController:
    def __init__(self, max_fps: float = 0.0):
        self.max_fps = float(max_fps)
        self._next: float | None = None

    def set_max(self, max_fps: float) -> None:
        self.max_fps = float(max_fps)
        self._next = None

    def delay(self, now: float) -> float:
        """Seconds to wait before the next frame to honour the cap (0 = none)."""
        if self.max_fps <= 0:
            return 0.0
        period = 1.0 / self.max_fps
        if self._next is None:
            self._next = now + period
            return 0.0
        wait = self._next - now
        self._next = (now if wait < 0 else self._next) + period
        return max(0.0, wait)

    def wait(self) -> None:
        d = self.delay(time.time())
        if d > 0:
            time.sleep(d)


class FpsMeter:
    def __init__(self, smoothing: float = 0.9):
        self.smoothing = smoothing
        self._t: float | None = None
        self.value = 0.0

    def tick(self) -> float:
        now = time.time()
        if self._t is not None:
            dt = now - self._t
            if dt > 0:
                inst = 1.0 / dt
                self.value = inst if self.value == 0 else self.smoothing * self.value + (1 - self.smoothing) * inst
        self._t = now
        return self.value
