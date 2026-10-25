"""Trajectory / kinematics classifier: tells drones from birds by how they move.

Appearance alone is weak for small aerial objects, so this scores each track's
motion over time. Drones fly straight or hold a steady hover at near-constant
speed with a stable shape; birds wander, vary speed, bob vertically while
flapping, and their bounding box pulses as the wings move.

Pure NumPy so it is testable without a GPU or model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass
class MotionProfile:
    kind: str            # "drone" | "bird" | "uncertain" | "pending"
    score: float         # 0 = bird-like, 1 = drone-like
    straightness: float
    speed_cv: float
    heading_jitter: float
    hover_ratio: float
    vertical_osc: float
    aspect_osc: float
    n: int


def _centroids(boxes: Sequence[Sequence[float]]) -> np.ndarray:
    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], axis=1)


def _aspect(boxes: Sequence[Sequence[float]]) -> np.ndarray:
    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    w = np.clip(b[:, 2] - b[:, 0], 1e-3, None)
    h = np.clip(b[:, 3] - b[:, 1], 1e-3, None)
    return h / w


def straightness(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    path = float(seg.sum())
    net = float(np.linalg.norm(points[-1] - points[0]))
    return net / path if path > 1e-6 else 0.0


def speed_profile(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.zeros((0,), np.float32)
    return np.linalg.norm(np.diff(points, axis=0), axis=1).astype(np.float32)


def heading_jitter(points: np.ndarray) -> float:
    """Mean absolute turn angle (radians) between consecutive motion vectors."""
    v = np.diff(points, axis=0)
    mag = np.linalg.norm(v, axis=1)
    v = v[mag > 1e-3]
    if len(v) < 2:
        return 0.0
    ang = np.arctan2(v[:, 1], v[:, 0])
    d = np.abs(np.diff(ang))
    d = np.minimum(d, 2 * np.pi - d)
    return float(np.mean(d))


def vertical_oscillation(points: np.ndarray) -> float:
    """Rate of up/down direction reversals — flapping birds bob, drones don't."""
    if len(points) < 3:
        return 0.0
    vy = np.diff(points[:, 1])
    vy = vy[np.abs(vy) > 1e-2]
    if len(vy) < 2:
        return 0.0
    sign_changes = int(np.sum(np.diff(np.sign(vy)) != 0))
    return sign_changes / (len(vy) - 1)


def _cv(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    m = float(np.mean(x))
    return float(np.std(x) / m) if m > 1e-6 else 0.0


class TrajectoryClassifier:
    def __init__(self, min_len=8, hover_speed=1.5, min_motion=0.5,
                 w_straight=0.30, w_speed=0.25, w_heading=0.20, w_vosc=0.15, w_aspect=0.10,
                 drone_thresh=0.60, bird_thresh=0.40):
        self.min_len = min_len
        self.hover_speed = hover_speed
        self.min_motion = min_motion
        self.w = dict(straight=w_straight, speed=w_speed, heading=w_heading, vosc=w_vosc, aspect=w_aspect)
        self.drone_thresh = drone_thresh
        self.bird_thresh = bird_thresh

    def profile(self, boxes: Sequence[Sequence[float]]) -> MotionProfile:
        n = len(boxes)
        if n < self.min_len:
            return MotionProfile("pending", 0.5, 0, 0, 0, 0, 0, 0, n)

        pts = _centroids(boxes)
        speeds = speed_profile(pts)
        straight = straightness(pts)
        scv = _cv(speeds)
        jitter = heading_jitter(pts)
        vosc = vertical_oscillation(pts)
        aosc = _cv(_aspect(boxes))
        hover = float(np.mean(speeds < self.hover_speed)) if len(speeds) else 0.0

        moving = float(np.mean(speeds)) if len(speeds) else 0.0

        # Each sub-score is 1.0 for drone-like, 0.0 for bird-like.
        s_straight = straight                                   # straight path -> drone
        s_speed = 1.0 - min(scv / 0.8, 1.0)                     # uniform speed -> drone
        s_heading = 1.0 - min(jitter / (np.pi / 2), 1.0)       # smooth heading -> drone
        s_vosc = 1.0 - min(vosc / 0.5, 1.0)                    # no bobbing -> drone
        s_aspect = 1.0 - min(aosc / 0.25, 1.0)                 # stable shape -> drone

        score = (
            self.w["straight"] * s_straight
            + self.w["speed"] * s_speed
            + self.w["heading"] * s_heading
            + self.w["vosc"] * s_vosc
            + self.w["aspect"] * s_aspect
        )

        # A sustained hover in open sky is strong drone evidence.
        if moving < self.min_motion and hover > 0.8:
            score = max(score, 0.75)

        if score >= self.drone_thresh:
            kind = "drone"
        elif score <= self.bird_thresh:
            kind = "bird"
        else:
            kind = "uncertain"

        return MotionProfile(kind, round(float(score), 3), round(straight, 3), round(scv, 3),
                             round(jitter, 3), round(hover, 3), round(vosc, 3), round(aosc, 3), n)
