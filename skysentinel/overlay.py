"""Draws detection boxes, track IDs, and an FPS/HUD badge onto frames."""

from __future__ import annotations

from typing import List

import numpy as np

from .tracker import Track

RED = (60, 60, 235)
AMBER = (40, 190, 235)
WHITE = (245, 245, 245)
DARK = (20, 24, 30)


def draw_tracks(frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
    import cv2

    drawn = [t for t in tracks if getattr(t, "kind", "pending") != "bird"]
    # A full "#id Drone/UAV 0.00" caption dwarfs a drone that is only a few pixels
    # across, and a sky full of them becomes an unreadable wall of text. So the
    # caption is only drawn in full beside a large, close target; small targets
    # (the usual case) get just their id, and a dense swarm gets boxes alone. The
    # HUD always shows the running total regardless.
    big_label = len(drawn) <= 20

    for t in drawn:
        kind = getattr(t, "kind", "pending")
        x1, y1, x2, y2 = [int(v) for v in t.box]
        size = max(x2 - x1, y2 - y1)
        # Confirmed drone is solid red; a not-yet-confirmed aerial target is amber.
        col = RED if kind == "drone" else AMBER
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2 if size >= 14 else 1)
        if len(drawn) > 60:                   # dense swarm: boxes only, stay readable
            continue
        if big_label and size >= 34:
            label = "Drone/UAV" if kind == "drone" else "Drone/UAV?"
            tag, scale = f"#{t.id} {label} {t.score:.2f}", 0.5
        else:
            tag, scale = f"#{t.id}", 0.4      # compact id for small / numerous targets
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        ly = y1 - 4 if y1 - th - 6 >= 0 else y2 + th + 4      # flip below if no room above
        cv2.rectangle(frame, (x1, ly - th - 2), (x1 + tw + 6, ly + 2), col, -1)
        cv2.putText(frame, tag, (x1 + 3, ly), cv2.FONT_HERSHEY_SIMPLEX, scale, DARK, 1, cv2.LINE_AA)
    return frame


def draw_detections(frame: np.ndarray, boxes, scores, labels) -> np.ndarray:
    """Draw raw detections on a single image (no tracking/motion available)."""
    import cv2

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), RED, 2)
        tag = f"Drone/UAV {float(score):.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), RED, -1)
        cv2.putText(frame, tag, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    return frame


def draw_hud(frame: np.ndarray, fps: float, count: int, name: str = "") -> np.ndarray:
    import cv2

    h, w = frame.shape[:2]
    label = f"{name}  " if name else ""
    text = f"{label}FPS {fps:4.1f}  |  targets {count}"
    cv2.rectangle(frame, (0, 0), (w, 26), DARK, -1)
    cv2.putText(frame, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.56, WHITE, 1, cv2.LINE_AA)
    if count:
        cv2.circle(frame, (w - 16, 13), 6, RED, -1)
    return frame
