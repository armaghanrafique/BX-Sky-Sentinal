"""Motion-based aerial detector: the fallback when no model weights are available.

Background subtraction plus shape/area filtering flags small moving objects
against the sky. It is coarse, but it guarantees the system still detects
something when a neural model cannot be loaded.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


class MotionDetector:
    def __init__(self, min_area: int = 9, max_area_ratio: float = 0.05, history: int = 200):
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.history = history
        self._bg = None

    def _ensure(self):
        if self._bg is None:
            import cv2

            self._bg = cv2.createBackgroundSubtractorMOG2(history=self.history, varThreshold=25, detectShadows=False)
        return self._bg

    def detect(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        import cv2

        bg = self._ensure()
        h, w = frame_bgr.shape[:2]
        mask = bg.apply(frame_bgr)
        mask = cv2.medianBlur(mask, 3)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes, scores = [], []
        max_area = self.max_area_ratio * w * h
        for c in contours:
            a = cv2.contourArea(c)
            if a < self.min_area or a > max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            boxes.append([x, y, x + bw, y + bh])
            scores.append(min(0.6, 0.3 + a / max(max_area, 1.0)))
        if not boxes:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), []
        return (
            np.asarray(boxes, np.float32),
            np.asarray(scores, np.float32),
            ["motion"] * len(boxes),
        )
