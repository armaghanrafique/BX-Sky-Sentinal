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

    def _predict(self, img: np.ndarray, ox: int = 0, oy: int = 0):
        kwargs = dict(imgsz=self.cfg.imgsz, conf=self.cfg.conf, device=self.device, verbose=False)
        if self.half:                 # only pass half on CUDA; avoids a per-frame deprecation log
            kwargs["half"] = True
        res = self._model.predict(img, **kwargs)[0]
        out = []
        for b in res.boxes:
            cid = int(b.cls[0])
            if not self._keep_class(cid):
                continue
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            out.append(([x1 + ox, y1 + oy, x2 + ox, y2 + oy], float(b.conf[0]),
                        str(self._names.get(cid, self.cfg.name))))
        return out

    def _tiles(self, w: int, h: int):
        """Top-left corners of overlapping tiles covering a w x h frame."""
        t = self.cfg.tile_size
        step = max(1, int(t * (1 - self.cfg.tile_overlap)))
        xs = list(range(0, max(1, w - t + 1), step)) or [0]
        ys = list(range(0, max(1, h - t + 1), step)) or [0]
        if xs[-1] != w - t and w > t:
            xs.append(w - t)
        if ys[-1] != h - t and h > t:
            ys.append(h - t)
        return [(x, y) for y in ys for x in xs]

    def detect(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        if not self.ok:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), []
        h, w = frame_bgr.shape[:2]
        found = []
        if self.cfg.tile and (w > self.cfg.tile_size or h > self.cfg.tile_size):
            # Run the whole frame (catches large/near targets) plus each tile at
            # full resolution (catches small / far targets). Boxes are merged by
            # the ensemble's fusion + NMS, so tile-overlap duplicates collapse.
            found += self._predict(frame_bgr)
            t = self.cfg.tile_size
            for ox, oy in self._tiles(w, h):
                crop = frame_bgr[oy:oy + t, ox:ox + t]
                if crop.size:
                    found += self._predict(crop, ox, oy)
        else:
            found += self._predict(frame_bgr)
        if not found:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), []
        boxes = np.asarray([f[0] for f in found], np.float32)
        scores = np.asarray([f[1] for f in found], np.float32)
        labels = [f[2] for f in found]
        return boxes, scores, labels
