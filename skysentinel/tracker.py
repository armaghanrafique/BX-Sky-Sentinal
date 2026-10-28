"""Lightweight IoU tracker: stable IDs across frames, confirmed after a few hits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .geometry import iou_matrix


@dataclass
class Track:
    id: int
    box: np.ndarray
    score: float
    label: str
    hits: int = 1
    age: int = 0
    time_since_update: int = 0
    confirmed: bool = False
    history: List[tuple] = field(default_factory=list)
    kind: str = "pending"        # drone | bird | uncertain | pending (from trajectory)
    kind_score: float = 0.5
    flap: float = 0.0            # wingbeat strength (0 rigid UAV .. 1 flapping bird)


class IoUTracker:
    def __init__(self, max_age: int = 30, iou_thresh: float = 0.3, min_hits: int = 3,
                 max_dist: float = 90.0):
        self.max_age = max_age
        self.iou_thresh = iou_thresh
        self.min_hits = min_hits
        self.max_dist = max_dist     # px gate for matching by centroid when boxes don't overlap
        self._next_id = 1
        self.tracks: List[Track] = []

    @staticmethod
    def _centroid(box):
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    def update(self, boxes: np.ndarray, scores: np.ndarray, labels: List[str]) -> List[Track]:
        for t in self.tracks:
            t.age += 1
            t.time_since_update += 1

        if len(self.tracks) and len(boxes):
            track_boxes = np.array([t.box for t in self.tracks], dtype=np.float32)
            ious = iou_matrix(track_boxes, np.asarray(boxes, np.float32))
            matched_t, matched_d = self._greedy_match(ious)
            # Second pass: a small, fast drone can move farther than its own
            # size between frames (zero IoU), so match leftovers by centroid.
            self._match_by_distance(track_boxes, boxes, matched_t, matched_d)
        else:
            matched_t, matched_d = {}, set()

        for ti, di in matched_t.items():
            t = self.tracks[ti]
            t.box = np.asarray(boxes[di], np.float32)
            t.score = float(scores[di])
            t.label = labels[di]
            t.hits += 1
            t.time_since_update = 0
            t.history.append(tuple(t.box))
            if t.hits >= self.min_hits:
                t.confirmed = True

        for di in range(len(boxes)):
            if di in matched_d:
                continue
            t = Track(self._next_id, np.asarray(boxes[di], np.float32), float(scores[di]), labels[di])
            t.history.append(tuple(t.box))
            t.confirmed = self.min_hits <= 1
            self._next_id += 1
            self.tracks.append(t)

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return [t for t in self.tracks if t.confirmed and t.time_since_update == 0]

    def _match_by_distance(self, track_boxes, boxes, matched_t: "Dict[int,int]", matched_d: set) -> None:
        free_t = [i for i in range(len(track_boxes)) if i not in matched_t]
        free_d = [j for j in range(len(boxes)) if j not in matched_d]
        if not free_t or not free_d:
            return
        tc = [self._centroid(track_boxes[i]) for i in free_t]
        dc = [self._centroid(boxes[j]) for j in free_d]
        pairs = []
        for a, ti in enumerate(free_t):
            for b, di in enumerate(free_d):
                dist = ((tc[a][0] - dc[b][0]) ** 2 + (tc[a][1] - dc[b][1]) ** 2) ** 0.5
                if dist <= self.max_dist:
                    pairs.append((dist, ti, di))
        pairs.sort()
        used_t, used_d = set(), set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            matched_t[ti] = di
            matched_d.add(di)
            used_t.add(ti)
            used_d.add(di)      # one detection matches one track only (no duplicates)

    def _greedy_match(self, ious: np.ndarray):
        matched_t: Dict[int, int] = {}
        matched_d = set()
        pairs = [(ious[ti, di], ti, di) for ti in range(ious.shape[0]) for di in range(ious.shape[1])]
        pairs.sort(reverse=True)
        used_t = set()
        for score, ti, di in pairs:
            if score < self.iou_thresh:
                break
            if ti in used_t or di in matched_d:
                continue
            matched_t[ti] = di
            used_t.add(ti)
            matched_d.add(di)
        return matched_t, matched_d
