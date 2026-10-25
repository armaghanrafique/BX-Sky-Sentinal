"""Builds the detector set from config, with a motion-based fallback.

If no YOLO weight loads (missing files, no ultralytics/torch), the system falls
back to the motion detector so detection still runs end to end.
"""

from __future__ import annotations

import logging
from typing import List

from .config import Config, ModelConfig
from .detector import YoloDetector
from .ensemble import Ensemble
from .motion import MotionDetector

log = logging.getLogger("skysentinel.models")


class MotionAdapter:
    """Wraps MotionDetector with the detector interface the ensemble expects."""

    def __init__(self, weight: float = 0.5):
        self._m = MotionDetector()
        self.cfg = ModelConfig(name="motion", weights="(motion)", weight=weight, conf=0.0)

    def detect(self, frame):
        return self._m.detect(frame)


def build_detectors(cfg: Config) -> List:
    detectors = []
    for m in cfg.enabled_models():
        d = YoloDetector(m, device=cfg.device, half=cfg.half)
        if d.load():
            log.info("loaded model %s (%s) on %s", m.name, m.weights, d.device)
            detectors.append(d)
        else:
            log.warning("could not load model %s (%s) - skipping", m.name, m.weights)
    if not detectors and cfg.motion_fallback:
        log.warning("no neural models available - using motion-detection fallback")
        detectors.append(MotionAdapter())
    return detectors


def build_ensemble(cfg: Config) -> Ensemble:
    return Ensemble(
        build_detectors(cfg),
        fuse_iou=cfg.fuse_iou,
        nms_iou=cfg.nms_iou,
        min_support=cfg.min_support,
        min_score=cfg.min_score,
        max_det=cfg.max_det,
    )
