"""Fake detectors so the ensemble/pipeline are tested without torch or cv2."""

import numpy as np

from skysentinel.config import ModelConfig


class FakeDetector:
    def __init__(self, boxes, scores, labels, weight=1.0, name="fake"):
        self._b = np.asarray(boxes, np.float32).reshape(-1, 4)
        self._s = np.asarray(scores, np.float32).reshape(-1)
        self._l = list(labels)
        self.cfg = ModelConfig(name=name, weights="(fake)", weight=weight)

    def detect(self, frame):
        return self._b.copy(), self._s.copy(), list(self._l)
