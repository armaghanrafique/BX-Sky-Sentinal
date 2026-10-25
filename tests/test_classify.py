import numpy as np

from skysentinel.classify import (
    TrajectoryClassifier,
    heading_jitter,
    straightness,
    vertical_oscillation,
)


def _straight_track(n=20, step=6.0, size=14.0):
    # constant velocity, straight line, stable box -> drone-like
    boxes = []
    for i in range(n):
        cx, cy = 50 + i * step, 50 + i * step
        boxes.append([cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2])
    return boxes


def _erratic_track(n=20):
    # wandering path, bobbing, pulsing aspect -> bird-like
    rng = np.random.default_rng(0)
    boxes = []
    cx, cy = 100.0, 100.0
    for i in range(n):
        cx += rng.uniform(-6, 6)
        cy += (-1) ** i * 5 + rng.uniform(-3, 3)      # vertical bob
        w = 16 + 6 * (i % 2)                           # wing flap -> aspect pulse
        h = 10
        boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return boxes


def _hover_track(n=20, size=14.0):
    boxes = []
    for _ in range(n):
        boxes.append([100 - size / 2, 100 - size / 2, 100 + size / 2, 100 + size / 2])
    return boxes


def test_straightness_bounds():
    pts = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], float)
    assert abs(straightness(pts) - 1.0) < 1e-6
    zig = np.array([[0, 0], [1, 0], [0, 0], [1, 0]], float)
    assert straightness(zig) < 0.5


def test_heading_jitter_zero_for_straight():
    pts = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], float)
    assert heading_jitter(pts) < 1e-6


def test_vertical_oscillation_detects_bobbing():
    pts = np.array([[0, 0], [1, 5], [2, 0], [3, 5], [4, 0]], float)
    assert vertical_oscillation(pts) > 0.5


def test_straight_track_classified_drone():
    c = TrajectoryClassifier(min_len=8)
    prof = c.profile(_straight_track())
    assert prof.kind == "drone"
    assert prof.score >= 0.6


def test_erratic_track_classified_bird():
    c = TrajectoryClassifier(min_len=8)
    prof = c.profile(_erratic_track())
    assert prof.kind == "bird"
    assert prof.score <= 0.4


def test_hover_classified_drone():
    c = TrajectoryClassifier(min_len=8)
    prof = c.profile(_hover_track())
    assert prof.kind == "drone"
    assert prof.hover_ratio > 0.8


def test_short_track_is_pending():
    c = TrajectoryClassifier(min_len=8)
    prof = c.profile(_straight_track(n=4))
    assert prof.kind == "pending"
