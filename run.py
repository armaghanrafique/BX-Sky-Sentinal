#!/usr/bin/env python
"""SkySentinel CLI.

  python run.py video --input clip.mp4 --output out.mp4
  python run.py stream --input rtsp://user:pass@cam/stream --output live.mp4
  python run.py web
"""

from __future__ import annotations

import argparse
import logging
import sys

from skysentinel.config import Config
from skysentinel.logging_setup import setup_logging
from skysentinel.models import build_ensemble
from skysentinel.video import process_video


def _overrides(a) -> dict:
    return {k: v for k, v in {
        "device": a.device, "frame_stride": a.stride, "min_support": a.min_support,
        "target_fps": getattr(a, "max_fps", None),
    }.items() if v is not None}


def cmd_video(a) -> int:
    cfg = Config.load(a.config, _overrides(a))
    if getattr(a, "turbo", False):
        # Maximum speed: a single nano model at low resolution.
        from skysentinel.config import ModelConfig

        cfg.models = [ModelConfig(name="yolo11n", weights="yolo11n.pt", weight=1.0, conf=0.20, imgsz=512)]
        cfg.apply_mode("fast")
        cfg.min_support = 1
    elif getattr(a, "engine", None):
        cfg.apply_engine(a.engine)         # fastest | balanced | accurate | longrange
    elif getattr(a, "mode", None):
        cfg.apply_mode(a.mode)
    if a.max_fps is not None:
        cfg.target_fps = a.max_fps          # explicit cap overrides the mode's cap
    ens = build_ensemble(cfg)

    def prog(done, total, n):
        pct = int(done / total * 100)
        print(f"\r  {pct:3d}%  frame {done}/{total}  targets now: {n}", end="", flush=True)

    stats = process_video(a.input, a.output, ens, cfg, on_progress=prog, name=a.name or "")
    print(f"\n  done: {stats['frames']} frames, {stats['detections']} target-frames, out: {stats['output']}")
    return 0


def cmd_web(a) -> int:
    from app import create_app

    app = create_app(a.config)
    url = f"http://localhost:{a.port}"
    print("\n  BX-Sky Sentinal is running — open:\n")
    print(f"      {url}\n")
    logging.getLogger("skysentinel.web").info("serving on %s (press Ctrl+C to stop)", url)
    app.run(host=a.host, port=a.port, threaded=True)
    return 0


def main(argv=None) -> int:
    setup_logging()
    p = argparse.ArgumentParser(prog="skysentinel", description="Multi-model drone / UAV detection")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name in ("video", "stream"):
        sp = sub.add_parser(name, help=f"process a {name}")
        sp.add_argument("--input", required=True)
        sp.add_argument("--output")
        sp.add_argument("--config")
        sp.add_argument("--device")
        sp.add_argument("--mode", choices=["fast", "balanced", "precise"], help="performance preset")
        sp.add_argument("--engine", choices=["fastest", "balanced", "accurate", "longrange"],
                        help="detection model (longrange = tiled, best for small drones)")
        sp.add_argument("--turbo", action="store_true", help="maximum speed: single nano model at 512px")
        sp.add_argument("--max-fps", type=float, dest="max_fps", help="cap processing FPS (0 = uncapped)")
        sp.add_argument("--stride", type=int)
        sp.add_argument("--min-support", type=int, dest="min_support")
        sp.add_argument("--name")
        sp.set_defaults(func=cmd_video)

    wp = sub.add_parser("web", help="run the web UI")
    wp.add_argument("--config")
    wp.add_argument("--host", default="0.0.0.0")
    wp.add_argument("--port", type=int, default=5000)
    wp.set_defaults(func=cmd_web)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
