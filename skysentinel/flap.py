"""Wingbeat verifier: the second check that separates birds from drones.

A bird flaps, so its silhouette changes shape in a *periodic* way — the bounding
box width and area rise and fall at the wingbeat rate. A drone is rigid: nothing
on it oscillates, so its silhouette stays steady apart from drift toward or away
from the camera (a smooth trend, not an oscillation).

This measures how much of a track's shape signal sits in a single oscillation. A
strong, clean periodicity means flapping -> bird; a flat spectrum means a rigid
body -> drone. Pure NumPy, so it runs anywhere and is unit-tested.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def periodicity(signal: Sequence[float]) -> float:
    """Fraction of a signal's power held by its single strongest oscillation,
    after removing mean and linear trend. ~1.0 = clean periodic (flapping),
    near 0 = noise or steady (rigid)."""
    x = np.asarray(signal, dtype=np.float64)
    n = x.size
    if n < 6:
        return 0.0
    t = np.arange(n)
    a, b = np.polyfit(t, x, 1)        # remove drift (e.g. approaching the camera)
    x = x - (a * t + b)
    if np.allclose(x, 0):
        return 0.0
    xw = x * np.hanning(n)
    power = np.abs(np.fft.rfft(xw)) ** 2
    total = power[1:].sum()           # all non-DC power
    osc = power[2:]                   # bins >= 2 cycles: a real oscillation, not slow drift
    if total <= 0 or osc.size == 0:
        return 0.0
    return float(osc.max() / total)


def _aspect(boxes: np.ndarray) -> np.ndarray:
    w = np.clip(boxes[:, 2] - boxes[:, 0], 1e-3, None)
    h = np.clip(boxes[:, 3] - boxes[:, 1], 1e-3, None)
    return h / w


def _area(boxes: np.ndarray) -> np.ndarray:
    w = np.clip(boxes[:, 2] - boxes[:, 0], 1e-3, None)
    h = np.clip(boxes[:, 3] - boxes[:, 1], 1e-3, None)
    return w * h


class FlapVerifier:
    def __init__(self, min_len: int = 10, strength_thresh: float = 0.45):
        self.min_len = min_len
        self.strength_thresh = strength_thresh

    def strength(self, box_history: Sequence[Sequence[float]]) -> float:
        """Flap strength in [0,1] from a track's bounding-box history."""
        boxes = np.asarray(box_history, dtype=np.float64).reshape(-1, 4)
        if len(boxes) < self.min_len:
            return 0.0
        return max(periodicity(_aspect(boxes)), periodicity(_area(boxes)))

    def is_bird(self, box_history: Sequence[Sequence[float]]) -> bool:
        return self.strength(box_history) >= self.strength_thresh
