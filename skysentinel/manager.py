"""Multi-camera manager: one capture + pipeline per camera, each on its own
thread, sharing one ensemble. The web UI reads the latest annotated frame and
live track counts per camera."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger("skysentinel.camera")

from .config import CameraConfig, Config
from .ensemble import Ensemble
from .fps import FpsController
from .pipeline import DetectionPipeline
from .sources import VideoSource


@dataclass
class CameraState:
    id: str
    name: str
    source: str
    online: bool = False
    fps: float = 0.0
    targets: int = 0
    last_frame: Optional[np.ndarray] = None
    error: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock)


class CameraWorker:
    def __init__(self, cam: CameraConfig, ensemble: Ensemble, cfg: Config):
        self.cfg = cfg
        self.cam = cam
        self.pipeline = DetectionPipeline(ensemble, cfg, name=cam.name or cam.id)
        self.state = CameraState(cam.id, cam.name or cam.id, cam.source)
        self._src: Optional[VideoSource] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.fps_ctl = FpsController(cfg.target_fps)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        self._src = VideoSource(self.cam.source)
        if not self._src.open():
            self.state.error = "could not open source"
            self.state.online = False
            log.warning("camera '%s' offline: could not open source", self.state.name)
            return
        self.state.online = True
        log.info("camera '%s' online", self.state.name)
        prev_targets = 0
        while not self._stop.is_set():
            pkt = self._src.read()
            if not pkt.ok:
                if self._src.is_live:
                    time.sleep(0.05)
                    continue
                break
            if pkt.frame is None:
                time.sleep(0.005)
                continue
            result = self.pipeline.process(pkt.frame)
            n = len(result.tracks)
            if n > 0 and prev_targets == 0:
                log.warning("DRONE DETECTED on '%s' (%d target%s)", self.state.name, n, "" if n == 1 else "s")
            elif n == 0 and prev_targets > 0:
                log.info("'%s' clear", self.state.name)
            prev_targets = n
            with self.state._lock:
                self.state.last_frame = result.frame
                self.state.fps = result.fps
                self.state.targets = n
            self.fps_ctl.wait()
        self.state.online = False

    def snapshot(self) -> Optional[np.ndarray]:
        with self.state._lock:
            return None if self.state.last_frame is None else self.state.last_frame.copy()

    def stop(self) -> None:
        self._stop.set()
        if self._src:
            self._src.release()
        if self._thread:
            self._thread.join(timeout=2.0)


class CameraManager:
    def __init__(self, ensemble: Ensemble, cfg: Config):
        self.ensemble = ensemble
        self.cfg = cfg
        self.workers: Dict[str, CameraWorker] = {}

    def add(self, cam: CameraConfig) -> None:
        if cam.id in self.workers:
            self.remove(cam.id)
        w = CameraWorker(cam, self.ensemble, self.cfg)
        self.workers[cam.id] = w
        w.start()

    def remove(self, cam_id: str) -> None:
        w = self.workers.pop(cam_id, None)
        if w:
            w.stop()

    def states(self) -> List[CameraState]:
        return [w.state for w in self.workers.values()]

    def worker(self, cam_id: str) -> Optional[CameraWorker]:
        return self.workers.get(cam_id)

    def set_ensemble(self, ensemble: Ensemble) -> None:
        """Swap the detector set live; every camera picks it up on its next frame."""
        self.ensemble = ensemble
        for w in self.workers.values():
            w.pipeline.ensemble = ensemble

    def set_mode(self, mode: str) -> None:
        """Switch Fast/Balanced/Precise live. Stride, cap and agreement update at
        once; input size (imgsz) applies when a camera is (re)started."""
        self.cfg.apply_mode(mode)
        self.ensemble.min_support = self.cfg.min_support
        for w in self.workers.values():
            w.fps_ctl.set_max(self.cfg.target_fps)

    def start_all(self) -> None:
        for cam in self.cfg.cameras:
            if cam.enabled:
                self.add(cam)

    def stop_all(self) -> None:
        for cam_id in list(self.workers):
            self.remove(cam_id)
