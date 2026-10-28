"""Per-frame flow: ensemble detect (every Nth frame) -> track -> overlay.

Detection is the slow step, so it runs every `frame_stride` frames; the tracker
carries boxes on the frames in between, which keeps the live view smooth and
fast while staying accurate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .classify import TrajectoryClassifier
from .config import Config
from .ensemble import Ensemble, Detections
from .flap import FlapVerifier
from .overlay import draw_hud, draw_tracks
from .tracker import IoUTracker, Track


@dataclass
class FrameResult:
    frame: np.ndarray
    tracks: List[Track]
    detections: Detections
    fps: float
    index: int


class DetectionPipeline:
    def __init__(self, ensemble: Ensemble, cfg: Config, name: str = ""):
        self.ensemble = ensemble
        self.cfg = cfg
        self.name = name
        self.tracker = IoUTracker(cfg.track_max_age, cfg.track_iou, cfg.track_min_hits, cfg.track_max_dist)
        self.classifier = TrajectoryClassifier(min_len=cfg.trajectory_min_len) if cfg.trajectory_classify else None
        self.flap = FlapVerifier(cfg.flap_min_len, cfg.flap_strength) if cfg.flap_verify else None
        self._i = 0
        self._last_det = Detections.empty()
        self._fps = 0.0
        self._t = time.time()

    def _tick_fps(self) -> None:
        now = time.time()
        dt = now - self._t
        if dt > 0:
            inst = 1.0 / dt
            self._fps = inst if self._fps == 0 else 0.9 * self._fps + 0.1 * inst
        self._t = now

    def process(self, frame: np.ndarray, draw: bool = True) -> FrameResult:
        self._i += 1
        if self.cfg.frame_stride <= 1 or self._i % self.cfg.frame_stride == 1 or len(self._last_det) == 0:
            dets = self.ensemble.detect(frame)
            self._last_det = dets
        else:
            dets = self._last_det
        tracks = self.tracker.update(dets.boxes, dets.scores, dets.labels)
        for t in tracks:
            if self.classifier is not None:
                prof = self.classifier.profile(t.history)
                t.kind, t.kind_score = prof.kind, prof.score
            else:
                # No motion classification: every confirmed aerial target is a drone.
                t.kind = "drone"
            # A clear wingbeat oscillation overrules everything and marks the
            # target a bird (rigid UAVs never trip this).
            if self.flap is not None:
                t.flap = self.flap.strength(t.history)
                if t.flap >= self.cfg.flap_strength:
                    t.kind, t.kind_score = "bird", max(t.kind_score, t.flap)
        if self.cfg.drones_only:
            tracks = [t for t in tracks if t.kind != "bird"]
        self._tick_fps()
        if draw:
            draw_tracks(frame, tracks)
            draw_hud(frame, self._fps, len(tracks), self.name)
        return FrameResult(frame, tracks, dets, round(self._fps, 1), self._i)
