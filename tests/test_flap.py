import numpy as np

from skysentinel.config import Config
from skysentinel.ensemble import Ensemble
from skysentinel.flap import FlapVerifier, periodicity
from skysentinel.pipeline import DetectionPipeline


def test_periodicity_high_for_sine():
    t = np.arange(40)
    assert periodicity(np.sin(2 * np.pi * t / 6)) > 0.5


def test_periodicity_low_for_constant_and_trend():
    assert periodicity(np.ones(40)) == 0.0
    assert periodicity(np.arange(40.0)) < 0.2   # pure drift, no oscillation


def _flapping_boxes(n=24):
    # width pulses every ~4 frames (wings opening/closing); drifts across frame
    boxes = []
    for i in range(n):
        cx = 60 + i * 2.0
        w = 20 + 8 * (1 if i % 2 == 0 else -1)      # alternating -> strong periodicity
        boxes.append([cx - w / 2, 50, cx + w / 2, 62])
    return boxes


def _rigid_boxes(n=24):
    # steady silhouette, smooth approach (grows slowly, no oscillation)
    boxes = []
    for i in range(n):
        cx = 60 + i * 2.0
        s = 18 + i * 0.3
        boxes.append([cx - s / 2, 50, cx + s / 2, 50 + s])
    return boxes


def test_flap_flags_bird():
    assert FlapVerifier(min_len=10).is_bird(_flapping_boxes())


def test_flap_passes_rigid_uav():
    assert not FlapVerifier(min_len=10).is_bird(_rigid_boxes())


def test_short_history_not_flagged():
    assert FlapVerifier(min_len=10).strength(_flapping_boxes(5)) == 0.0


class _Replay:
    def __init__(self, boxes):
        from skysentinel.config import ModelConfig
        self.boxes = boxes; self.cfg = ModelConfig(name="r", weights="(fake)"); self.i = -1

    def detect(self, frame):
        self.i = min(self.i + 1, len(self.boxes) - 1)
        return (np.asarray([self.boxes[self.i]], np.float32),
                np.asarray([0.9], np.float32), ["uav"])


def _run(boxes):
    cfg = Config.load(overrides={"frame_stride": 1, "min_support": 1, "min_score": 0.0})
    cfg.track_min_hits = 1
    pipe = DetectionPipeline(Ensemble([_Replay(boxes)], min_support=1, min_score=0.0), cfg)
    frame = np.zeros((200, 200, 3), np.uint8)
    last = []
    for _ in boxes:
        last = pipe.process(frame, draw=False).tracks
    return last


def test_pipeline_drops_flapping_bird():
    assert _run(_flapping_boxes()) == []


def test_pipeline_keeps_rigid_uav():
    tracks = _run(_rigid_boxes())
    assert len(tracks) == 1
    assert tracks[0].kind != "bird"
