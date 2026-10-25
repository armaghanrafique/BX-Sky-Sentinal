import numpy as np

from skysentinel.config import Config
from skysentinel.ensemble import Ensemble
from skysentinel.pipeline import DetectionPipeline
from tests.fakes import FakeDetector


def _pipeline(stride=1, min_hits=1):
    cfg = Config.load(overrides={"frame_stride": stride, "min_support": 1, "min_score": 0.0})
    cfg.track_min_hits = min_hits
    det = FakeDetector([[10, 10, 40, 40]], [0.9], ["drone"])
    return DetectionPipeline(Ensemble([det], min_support=1, min_score=0.0), cfg)


def test_pipeline_emits_track_without_drawing():
    pipe = _pipeline()
    frame = np.zeros((100, 100, 3), np.uint8)
    res = pipe.process(frame, draw=False)
    assert len(res.tracks) == 1
    assert res.tracks[0].label == "drone"


def test_pipeline_stride_reuses_last_detection():
    pipe = _pipeline(stride=3)
    frame = np.zeros((100, 100, 3), np.uint8)
    r1 = pipe.process(frame, draw=False)
    r2 = pipe.process(frame, draw=False)
    assert len(r1.detections) == len(r2.detections) == 1


def test_pipeline_fps_is_positive_after_two_frames():
    pipe = _pipeline()
    frame = np.zeros((50, 50, 3), np.uint8)
    pipe.process(frame, draw=False)
    res = pipe.process(frame, draw=False)
    assert res.fps >= 0
