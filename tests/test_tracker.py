import numpy as np

from skysentinel.tracker import IoUTracker


def _b(x):
    return np.array([[x, 0, x + 20, 20]], np.float32)


def test_track_confirmed_after_min_hits():
    tr = IoUTracker(max_age=5, iou_thresh=0.3, min_hits=3)
    out = tr.update(_b(0), np.array([0.9]), ["drone"])
    assert out == []  # not yet confirmed
    tr.update(_b(1), np.array([0.9]), ["drone"])
    out = tr.update(_b(2), np.array([0.9]), ["drone"])
    assert len(out) == 1
    assert out[0].id == 1 and out[0].confirmed


def test_track_keeps_id_across_frames():
    tr = IoUTracker(min_hits=1)
    a = tr.update(_b(0), np.array([0.9]), ["drone"])[0].id
    b = tr.update(_b(3), np.array([0.9]), ["drone"])[0].id
    assert a == b


def test_separate_objects_get_separate_ids():
    tr = IoUTracker(min_hits=1)
    tr.update(_b(0), np.array([0.9]), ["drone"])
    two = tr.update(
        np.array([[0, 0, 20, 20], [300, 0, 320, 20]], np.float32),
        np.array([0.9, 0.8]), ["drone", "drone"],
    )
    assert len({t.id for t in two}) == 2


def test_track_expires_after_max_age():
    tr = IoUTracker(max_age=2, min_hits=1)
    tr.update(_b(0), np.array([0.9]), ["drone"])
    for _ in range(4):
        tr.update(np.zeros((0, 4), np.float32), np.zeros((0,)), [])
    assert tr.tracks == []
