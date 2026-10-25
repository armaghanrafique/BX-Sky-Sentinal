"""Process a video file or a single stream to an annotated output file."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from .config import Config
from .ensemble import Ensemble
from .fps import FpsController
from .pipeline import DetectionPipeline
from .sources import VideoSource


def process_image(path: str, output: str, ensemble: Ensemble) -> dict:
    """Detect aerial targets in a single image and save it with boxes drawn.

    A still frame has no motion, so the bird/drone motion checks do not apply;
    this draws every aerial-target detection the ensemble reports.
    """
    import cv2

    from .overlay import draw_detections

    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"could not read image: {path}")
    dets = ensemble.detect(img)
    draw_detections(img, dets.boxes, dets.scores, dets.labels)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output, img)
    return {"detections": int(len(dets)), "output": output}


def process_video(
    source: str,
    output: Optional[str],
    ensemble: Ensemble,
    cfg: Config,
    on_progress: Optional[Callable[[int, int, int], None]] = None,
    name: str = "",
) -> dict:
    import cv2

    src = VideoSource(source)
    if not src.open():
        raise RuntimeError(f"could not open source: {source}")

    cap = src._cap
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if src.kind == "file" else 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = src.fps or 25.0

    writer = None
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    pipe = DetectionPipeline(ensemble, cfg, name=name)
    # Only throttle a live stream; a file is processed as fast as possible.
    fps_ctl = FpsController(cfg.target_fps if src.is_live else 0.0)
    frames = detections = 0
    try:
        while True:
            pkt = src.read()
            if not pkt.ok:
                break
            result = pipe.process(pkt.frame)
            frames += 1
            detections += len(result.tracks)
            if writer is not None:
                writer.write(result.frame)
            if on_progress and total:
                on_progress(frames, total, len(result.tracks))
            fps_ctl.wait()
    finally:
        src.release()
        if writer is not None:
            writer.release()
    return {"frames": frames, "detections": detections, "output": output, "fps": round(pipe._fps, 1)}
