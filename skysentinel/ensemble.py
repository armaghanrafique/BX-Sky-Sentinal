"""Runs several detectors on a frame and fuses their boxes into one result.

A detector is any object with `detect(frame) -> (boxes, scores, labels)` and a
`.cfg.weight`. This keeps the ensemble testable with simple fakes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .geometry import nms, weighted_boxes_fusion


@dataclass
class Detections:
    boxes: np.ndarray
    scores: np.ndarray
    labels: List[str]
    support: np.ndarray  # how many models agreed on each box

    @classmethod
    def empty(cls) -> "Detections":
        return cls(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), [], np.zeros((0,), np.int32))

    def __len__(self) -> int:
        return len(self.boxes)


class Ensemble:
    def __init__(self, detectors: Sequence, fuse_iou=0.55, nms_iou=0.5,
                 min_support=1, min_score=0.3, max_det=100):
        self.detectors = list(detectors)
        self.fuse_iou = fuse_iou
        self.nms_iou = nms_iou
        self.min_support = min_support
        self.min_score = min_score
        self.max_det = max_det

    def detect(self, frame) -> Detections:
        boxes_list, scores_list, weights, label_pool = [], [], [], []
        for det in self.detectors:
            b, s, lbls = det.detect(frame)
            boxes_list.append(np.asarray(b, np.float32).reshape(-1, 4))
            scores_list.append(np.asarray(s, np.float32).reshape(-1))
            weights.append(getattr(getattr(det, "cfg", None), "weight", 1.0))
            label_pool.extend(zip(np.asarray(b, np.float32).reshape(-1, 4).tolist(), lbls))

        fb, fs, support = weighted_boxes_fusion(
            boxes_list, scores_list, weights=weights, iou_thresh=self.fuse_iou
        )
        if len(fb) == 0:
            return Detections.empty()

        keep = (fs >= self.min_score) & (support >= self.min_support)
        fb, fs, support = fb[keep], fs[keep], support[keep]
        if len(fb) == 0:
            return Detections.empty()

        order = nms(fb, fs, self.nms_iou)[: self.max_det]
        fb, fs, support = fb[order], fs[order], support[order]
        labels = [self._label_for(box, label_pool) for box in fb]
        return Detections(fb, fs, labels, support)

    @staticmethod
    def _label_for(box, label_pool) -> str:
        from .geometry import iou

        best, best_iou = "drone", 0.0
        for cand_box, lbl in label_pool:
            v = iou(box, cand_box)
            if v > best_iou:
                best, best_iou = lbl, v
        return best
