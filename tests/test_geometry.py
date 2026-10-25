import numpy as np

from skysentinel.geometry import iou, iou_matrix, nms, weighted_boxes_fusion


def test_iou_identical_and_disjoint():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_iou_half_overlap():
    assert abs(iou([0, 0, 10, 10], [5, 0, 15, 10]) - 1 / 3) < 1e-6


def test_iou_matrix_shape():
    a = np.array([[0, 0, 10, 10], [0, 0, 5, 5]], np.float32)
    b = np.array([[0, 0, 10, 10]], np.float32)
    m = iou_matrix(a, b)
    assert m.shape == (2, 1)
    assert m[0, 0] == 1.0


def test_nms_removes_overlaps():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]], np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    keep = nms(boxes, scores, 0.5)
    assert 0 in keep and 2 in keep and 1 not in keep


def test_wbf_merges_agreeing_boxes_and_counts_support():
    boxes = [np.array([[0, 0, 10, 10]], np.float32), np.array([[1, 0, 11, 10]], np.float32)]
    scores = [np.array([0.8], np.float32), np.array([0.7], np.float32)]
    fb, fs, support = weighted_boxes_fusion(boxes, scores, [1.0, 1.0], iou_thresh=0.5)
    assert len(fb) == 1
    assert support[0] == 2
    assert 0 <= fb[0][0] <= 1


def test_wbf_keeps_separate_boxes_apart():
    boxes = [np.array([[0, 0, 10, 10]], np.float32), np.array([[100, 100, 110, 110]], np.float32)]
    scores = [np.array([0.8], np.float32), np.array([0.9], np.float32)]
    fb, fs, support = weighted_boxes_fusion(boxes, scores, [1.0, 1.0], iou_thresh=0.5)
    assert len(fb) == 2
    assert set(support.tolist()) == {1}
