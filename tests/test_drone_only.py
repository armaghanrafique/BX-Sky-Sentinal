import numpy as np

from skysentinel.config import Config, ModelConfig
from skysentinel.ensemble import Ensemble
from skysentinel.pipeline import DetectionPipeline


class MovingDetector:
    """Replays a scripted path of boxes, one per frame, so the pipeline's
    trajectory classifier can judge it as drone or bird."""

    def __init__(self, boxes):
        self._boxes = boxes
        self.cfg = ModelConfig(name="moving", weights="(fake)", weight=1.0)
        self._i = -1

    def detect(self, frame):
        self._i = min(self._i + 1, len(self._boxes) - 1)
        b = np.asarray([self._boxes[self._i]], np.float32)
        return b, np.asarray([0.9], np.float32), ["uav"]


def _run(boxes, drones_only):
    cfg = Config.load(overrides={"frame_stride": 1, "min_support": 1, "min_score": 0.0})
    cfg.track_min_hits = 1
    cfg.trajectory_classify = True        # these tests exercise motion-based bird/drone telling
    cfg.trajectory_min_len = 8
    cfg.drones_only = drones_only
    pipe = DetectionPipeline(Ensemble([MovingDetector(boxes)], min_support=1, min_score=0.0), cfg)
    last = []
    frame = np.zeros((300, 400, 3), np.uint8)
    for _ in boxes:
        last = pipe.process(frame, draw=False).tracks
    return last


def _bird_path(n=22):
    boxes = []
    cx, cy = 100.0, 100.0
    rng = np.random.default_rng(1)
    for i in range(n):
        cx += rng.uniform(-6, 6)
        cy += (-1) ** i * 6 + rng.uniform(-3, 3)      # bobbing
        w = 16 + 6 * (i % 2)                           # wing-flap aspect pulse
        boxes.append([cx - w / 2, cy - 5, cx + w / 2, cy + 5])
    return boxes


def _drone_path(n=22, step=7.0):
    boxes = []
    for i in range(n):
        cx, cy = 40 + i * step, 40 + i * step
        boxes.append([cx - 8, cy - 8, cx + 8, cy + 8])  # straight, stable
    return boxes


def test_bird_is_dropped_when_drones_only():
    assert _run(_bird_path(), drones_only=True) == []


def test_bird_is_kept_when_not_drones_only():
    tracks = _run(_bird_path(), drones_only=False)
    assert any(t.kind == "bird" for t in tracks)


def test_drone_is_kept_when_drones_only():
    tracks = _run(_drone_path(), drones_only=True)
    assert len(tracks) == 1
    assert tracks[0].kind in ("drone", "uncertain", "pending")


def test_overlay_skips_birds():
    import numpy as np

    from skysentinel.overlay import draw_tracks
    from skysentinel.tracker import Track

    frame = np.zeros((80, 80, 3), np.uint8)
    t = Track(1, np.array([10, 10, 30, 30], np.float32), 0.9, "uav")
    t.kind = "bird"
    before = frame.copy()
    draw_tracks(frame, [t])
    assert np.array_equal(frame, before)   # nothing drawn for a bird
