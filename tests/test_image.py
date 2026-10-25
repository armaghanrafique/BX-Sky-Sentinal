import numpy as np
import pytest

from skysentinel.ensemble import Ensemble
from skysentinel.video import process_image
from tests.fakes import FakeDetector

cv2 = pytest.importorskip("cv2")


def test_process_image_writes_annotated_file(tmp_path):
    img = np.full((120, 160, 3), 40, np.uint8)
    src = tmp_path / "in.png"
    cv2.imwrite(str(src), img)
    out = tmp_path / "out.jpg"

    det = FakeDetector([[20, 20, 60, 60]], [0.9], ["drone"])
    ens = Ensemble([det], min_support=1, min_score=0.0)

    stats = process_image(str(src), str(out), ens)
    assert stats["detections"] == 1
    assert out.exists()
    annotated = cv2.imread(str(out))
    assert annotated is not None and annotated.shape[:2] == (120, 160)
    # a box was drawn, so the output differs from the blank input
    assert not np.array_equal(annotated, img)


def test_process_image_no_detections(tmp_path):
    src = tmp_path / "in.png"
    cv2.imwrite(str(src), np.zeros((80, 80, 3), np.uint8))
    out = tmp_path / "out.jpg"
    ens = Ensemble([FakeDetector([], [], [])], min_support=1, min_score=0.0)
    assert process_image(str(src), str(out), ens)["detections"] == 0
    assert out.exists()
