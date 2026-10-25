"""Video source handling: files, RTSP/RTMP/HTTP streams, and local webcams.

Capture runs on a background thread that always holds the newest frame, so a
slow consumer never backs up a live stream (it drops stale frames instead).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

STREAM_SCHEMES = ("rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://")


def classify_source(source: str) -> str:
    """Return one of: 'webcam', 'stream', 'file'."""
    s = str(source).strip()
    if s.isdigit():
        return "webcam"
    if s.lower().startswith(STREAM_SCHEMES):
        return "stream"
    return "file"


def normalize_source(source: str):
    """Webcam indices become ints for OpenCV; everything else stays a string."""
    s = str(source).strip()
    return int(s) if s.isdigit() else s


@dataclass
class FramePacket:
    ok: bool
    frame: object = None     # numpy BGR array
    index: int = 0
    timestamp: float = 0.0


class VideoSource:
    """Threaded reader. Live streams/webcams keep only the latest frame;
    files are read in order so none are skipped."""

    def __init__(self, source: str, reconnect: bool = True):
        self.source = source
        self.kind = classify_source(source)
        self.is_live = self.kind in ("stream", "webcam")
        self.reconnect = reconnect and self.is_live
        self._cap = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest: Optional[FramePacket] = None
        self._stop = threading.Event()
        self._count = 0
        self._opened = False

    def open(self) -> bool:
        import cv2

        self._cap = cv2.VideoCapture(normalize_source(self.source))
        self._opened = bool(self._cap and self._cap.isOpened())
        if self._opened and self.is_live:
            self._thread = threading.Thread(target=self._pump, daemon=True)
            self._thread.start()
        return self._opened

    @property
    def fps(self) -> float:
        import cv2

        if not self._cap:
            return 0.0
        v = self._cap.get(cv2.CAP_PROP_FPS)
        return float(v) if v and v > 0 else 0.0

    def _pump(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                if self.reconnect and not self._stop.is_set():
                    self._reopen()
                    continue
                with self._lock:
                    self._latest = FramePacket(ok=False)
                break
            self._count += 1
            with self._lock:
                self._latest = FramePacket(True, frame, self._count, time.time())

    def _reopen(self) -> None:
        import cv2

        try:
            self._cap.release()
        except Exception:
            pass
        time.sleep(1.0)
        self._cap = cv2.VideoCapture(normalize_source(self.source))

    def read(self) -> FramePacket:
        if self.is_live:
            with self._lock:
                return self._latest or FramePacket(ok=False)
        ok, frame = self._cap.read()
        if not ok:
            return FramePacket(ok=False)
        self._count += 1
        return FramePacket(True, frame, self._count, time.time())

    def release(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
