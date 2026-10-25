import numpy as np

from skysentinel.ensemble import Ensemble, Detections
from tests.fakes import FakeDetector


def test_empty_ensemble_returns_empty():
    ens = Ensemble([FakeDetector([], [], [])])
    d = ens.detect(np.zeros((4, 4)))
    assert len(d) == 0


def test_two_models_agree_boost_support():
    a = FakeDetector([[0, 0, 20, 20]], [0.7], ["drone"], name="a")
    b = FakeDetector([[2, 2, 22, 22]], [0.6], ["uav"], name="b")
    ens = Ensemble([a, b], fuse_iou=0.5, min_support=2, min_score=0.0)
    d = ens.detect(None)
    assert len(d) == 1
    assert d.support[0] == 2


def test_min_support_filters_single_model_boxes():
    a = FakeDetector([[0, 0, 20, 20]], [0.9], ["drone"], name="a")
    b = FakeDetector([[200, 200, 220, 220]], [0.9], ["drone"], name="b")
    ens = Ensemble([a, b], fuse_iou=0.5, min_support=2, min_score=0.0)
    assert len(ens.detect(None)) == 0


def test_label_assigned_from_nearest_source_box():
    a = FakeDetector([[0, 0, 20, 20]], [0.9], ["quadcopter"], name="a")
    ens = Ensemble([a], min_support=1, min_score=0.0)
    d = ens.detect(None)
    assert d.labels[0] == "quadcopter"


def test_max_det_caps_results():
    boxes = [[i * 50, 0, i * 50 + 10, 10] for i in range(10)]
    a = FakeDetector(boxes, [0.9] * 10, ["drone"] * 10)
    ens = Ensemble([a], min_support=1, min_score=0.0, max_det=3)
    assert len(ens.detect(None)) == 3
