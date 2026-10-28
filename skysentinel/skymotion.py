"""Sky-object detector: finds every small dark object floating in open sky.

A drone (or any UAV) seen from the ground is a small, compact dark shape against
a much brighter sky. That holds whether it is moving or hovering, and each object
shows up separately, so a whole group - however many - is picked up as individual
targets rather than one blob. The detector works from brightness contrast (not
motion), so hovering drones are not lost, and there is no cap on how many it can
report in a single frame. Ground clutter is kept out with three guards that do
not depend on the target standing alone, so drones packed tight in a swarm are
still each detected:

* sky is defined as *bright and bluish*, which rejects bright man-made surfaces
  (white/neutral walls) that a brightness-only test would mistake for sky;
* a target must sit above the per-column skyline, which rejects vehicles,
  buildings and ground markings below the horizon;
* a target must be genuinely dark (a real silhouette) and sit in mostly-sky
  surroundings, which rejects cloud-edge texture and bright structures without
  demanding empty sky on every side - so clustered drones survive.

Stateless per frame; called once per processed frame like any detector.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


class SkyMotionDetector:
    def __init__(self, min_area: int = 2, max_area_ratio: float = 0.02,
                 contrast: int = 18, sky_brightness: int = 120, blur: int = 35,
                 ring: int = 5, ring_sky_frac: float = 0.70, blue_min: int = 6,
                 dark_max: int = 130):
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.contrast = contrast              # how much darker than local sky to count
        self.sky_brightness = sky_brightness  # a region is "sky" if this bright...
        self.blue_min = blue_min              # ...and this much bluer than it is red
        self.blur = blur | 1                  # odd kernel for the local-sky estimate
        self.ring = ring                      # px band checked around each blob
        self.ring_sky_frac = ring_sky_frac    # fraction of that band that must be sky
        self.dark_max = dark_max              # blob must contain a pixel at least this dark

    def detect(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        import cv2

        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        k = min(self.blur, (min(h, w) // 2) * 2 - 1) if min(h, w) > 4 else 3
        local = cv2.blur(gray, (k, k)).astype(np.int16)     # local sky brightness
        diff = local - gray.astype(np.int16)                # >0 where a dark object sits on lighter sky

        br = frame_bgr[:, :, 0].astype(np.int16) - frame_bgr[:, :, 2]   # B - R, >0 = bluish
        blue_local = cv2.blur(br, (k, k))
        # per-pixel sky (ring test) and local sky (candidates / skyline)
        is_sky = (gray > self.sky_brightness) & (br >= self.blue_min)
        local_sky = (local > self.sky_brightness) & (blue_local >= self.blue_min)
        skymask = is_sky.astype(np.uint8)

        # Skyline per column: scan down the local sky and mark where continuous
        # sky from the top ends. Targets must sit above it.
        horizon = np.full(w, h, np.int32)
        for cx in range(w):
            nz = np.where(~local_sky[:, cx])[0]
            if nz.size:
                horizon[cx] = nz[0]
        horizon = cv2.blur(horizon.reshape(1, -1).astype(np.float32), (1, 31)).ravel()

        cand = ((diff > self.contrast) & local_sky).astype(np.uint8) * 255
        cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        contours, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area = self.max_area_ratio * w * h
        r = self.ring
        boxes, scores = [], []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area or area > max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if max(bw, bh) > 6 * max(min(bw, bh), 1):        # drop long thin streaks (wires/edges)
                continue
            cx = x + bw // 2
            # above the skyline for its column (small tolerance)
            if y + bh > horizon[cx] - 4:
                continue
            # genuinely dark: a real silhouette, not bright cloud-edge texture
            if int(gray[y:y + bh, x:x + bw].min()) > self.dark_max:
                continue
            # surroundings mostly sky (lenient, so tightly packed drones survive)
            x0, y0 = max(0, x - r), max(0, y - r)
            x1, y1 = min(w, x + bw + r), min(h, y + bh + r)
            ring_px = (y1 - y0) * (x1 - x0) - bw * bh
            ring_sky = int(skymask[y0:y1, x0:x1].sum()) - int(skymask[y:y + bh, x:x + bw].sum())
            if ring_px <= 0 or ring_sky / ring_px < self.ring_sky_frac:
                continue
            contrast = float(diff[y:y + bh, x:x + bw].max())
            boxes.append([x, y, x + bw, y + bh])
            scores.append(float(min(0.95, 0.3 + contrast / 160.0)))

        if not boxes:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), []
        return np.asarray(boxes, np.float32), np.asarray(scores, np.float32), ["drone"] * len(boxes)
