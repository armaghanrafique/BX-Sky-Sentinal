"""YOLO detector wrapper (Ultralytics), with device/precision selection and a
drone-class filter. Ultralytics is imported lazily so the rest of the system
and the tests run without it installed."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .config import ModelConfig

# Class names (in common drone-trained weights, or COCO look-alikes) that count
# as an aerial target. Anything matching is kept; the rest is dropped.
DRONE_ALIASES = {"drone", "uav", "quadcopter", "quadrotor", "aircraft", "airplane", "aeroplane", "helicopter", "bird"}


def resolve_device(device: str) -> str:
    if device and device != "auto":
        return device
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class YoloDetector:
    def __init__(self, cfg: ModelConfig, device: str = "auto", half: bool = True):
        self.cfg = cfg
        self.device = resolve_device(device)
        self.half = half and self.device.startswith("cuda")
        self._model = None
        self._names: dict = {}
        self.ok = False

    def load(self) -> bool:
        try:
            from ultralytics import YOLO

            from .logging_setup import quiet_ultralytics

            quiet_ultralytics()   # ultralytics raises its own log level on import
            self._model = YOLO(self.cfg.weights)
            self._names = dict(self._model.names) if hasattr(self._model, "names") else {}
            self.ok = True
        except Exception:
            self.ok = False
        return self.ok

    def _keep_class(self, class_id: int) -> bool:
        if self.cfg.classes is not None:
            return class_id in self.cfg.classes
        name = str(self._names.get(class_id, "")).lower()
        return (not self._names) or (name in DRONE_ALIASES)

    def detect(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        if not self.ok:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), []
        kwargs = dict(imgsz=self.cfg.imgsz, conf=self.cfg.conf, device=self.device, verbose=False)
        if self.half:                 # only pass half on CUDA; avoids a per-frame deprecation log
            kwargs["half"] = True
        res = self._model.predict(frame_bgr, **kwargs)[0]
        boxes, scores, labels = [], [], []
        for b in res.boxes:
            cid = int(b.cls[0])
            if not self._keep_class(cid):
                continue
            boxes.append(b.xyxy[0].tolist())
            scores.append(float(b.conf[0]))
            labels.append(str(self._names.get(cid, self.cfg.name)))
        if not boxes:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), []
        return np.asarray(boxes, np.float32), np.asarray(scores, np.float32), labels
