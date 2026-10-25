"""Box math: IoU, NMS, and Weighted Boxes Fusion for combining model outputs."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

Box = Sequence[float]  # x1, y1, x2, y2


def area(box: Box) -> float:
    w = max(0.0, box[2] - box[0])
    h = max(0.0, box[3] - box[1])
    return w * h


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    a = np.asarray(boxes_a, dtype=np.float32)[:, None, :]
    b = np.asarray(boxes_b, dtype=np.float32)[None, :, :]
    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.5) -> List[int]:
    if len(boxes) == 0:
        return []
    order = np.argsort(scores)[::-1]
    keep: List[int] = []
    while len(order) > 0:
        i = int(order[0])
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        ious = iou_matrix(boxes[i : i + 1], boxes[rest])[0]
        order = rest[ious <= iou_thresh]
    return keep


def weighted_boxes_fusion(
    boxes_list: List[np.ndarray],
    scores_list: List[np.ndarray],
    weights: Sequence[float] | None = None,
    iou_thresh: float = 0.55,
    skip_thresh: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuse detections from several models into agreement-weighted boxes.

    Returns fused (boxes, scores, model_support_count). A box confirmed by more
    models scores higher, so an ensemble is both more precise and more robust
    than any single detector.
    """
    entries = []
    n_models = len(boxes_list)
    weights = list(weights) if weights is not None else [1.0] * n_models
    for mi in range(n_models):
        bxs, scs = boxes_list[mi], scores_list[mi]
        for bi in range(len(bxs)):
            if scs[bi] < skip_thresh:
                continue
            entries.append((float(scs[bi]) * weights[mi], np.asarray(bxs[bi], np.float32), mi, float(scs[bi])))
    if not entries:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int32)

    entries.sort(key=lambda e: e[0], reverse=True)
    clusters: List[dict] = []
    for wscore, box, mi, raw in entries:
        placed = False
        for c in clusters:
            if iou(c["box"], box) >= iou_thresh:
                c["boxes"].append(box)
                c["weights"].append(wscore)
                c["models"].add(mi)
                c["raw"].append(raw)
                w = np.asarray(c["weights"])[:, None]
                c["box"] = (np.asarray(c["boxes"]) * w).sum(0) / w.sum()
                placed = True
                break
        if not placed:
            clusters.append({"box": box.copy(), "boxes": [box], "weights": [wscore], "models": {mi}, "raw": [raw]})

    fused_boxes, fused_scores, support = [], [], []
    denom = sum(weights) if sum(weights) > 0 else 1.0
    for c in clusters:
        fused_boxes.append(c["box"])
        agree = sum(weights[m] for m in c["models"]) / denom
        fused_scores.append(float(np.mean(c["raw"]) * agree))
        support.append(len(c["models"]))
    return (
        np.asarray(fused_boxes, np.float32),
        np.asarray(fused_scores, np.float32),
        np.asarray(support, np.int32),
    )
