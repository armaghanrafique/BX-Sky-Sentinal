#!/usr/bin/env python
"""SkySentinel web app: live multi-camera drone detection, video analysis,
runtime model switching, and network-camera discovery."""

from __future__ import annotations

import copy
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Optional

import logging

from flask import Flask, Response, abort, jsonify, render_template, request, send_file

from skysentinel.config import ENGINE_PRESETS, CameraConfig, Config
from skysentinel.logging_setup import setup_logging
from skysentinel.manager import CameraManager
from skysentinel.models import build_ensemble
from skysentinel.video import process_image, process_video

log = logging.getLogger("skysentinel.web")

BASE = Path(__file__).resolve().parent
UPLOADS = BASE / "data" / "uploads"
OUTPUTS = BASE / "data" / "output"
UPLOADS.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)
ALLOWED = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
ALLOWED_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

JOBS: dict = {}


def _clampf(v, default, lo, hi):
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return default


def _to_web_h264(src: Path, dst: Path) -> bool:
    """Re-encode to H.264 + faststart so the result plays in a browser <video>.
    OpenCV's mp4v output does not decode reliably in browsers."""
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return False
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", "-an", str(dst)],
            check=True, capture_output=True,
        )
        return dst.exists()
    except Exception:
        return False


def _apply_confidence(ensemble, value) -> None:
    conf = _clampf(value, ensemble.min_score, 0.05, 0.95)
    ensemble.min_score = conf
    for det in ensemble.detectors:
        if hasattr(det, "cfg"):
            det.cfg.conf = conf


def create_app(config_path: Optional[str] = None) -> Flask:
    setup_logging()
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024

    cfg = Config.load(config_path)
    cfg.apply_engine(cfg.engine)
    cfg.drones_only = True
    ensemble = build_ensemble(cfg)
    manager = CameraManager(ensemble, cfg)
    manager.start_all()
    log.info("SkySentinel ready | engine=%s | device=%s | drones-only", cfg.engine, cfg.device)

    engine_lock = threading.Lock()
    discover_state = {"scanning": False, "cameras": [], "message": ""}
    net_candidates: dict = {}        # token -> resolved stream URL (kept server-side)

    def encode(frame):
        import cv2

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else None

    # ---------------------------------------------------------------- pages
    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/live")
    def live():
        return render_template("live.html")

    # ---------------------------------------------------------------- engine (model speed)
    @app.get("/api/engine")
    def get_engine():
        return jsonify({
            "engine": cfg.engine,
            "options": [{"id": k, "label": v["label"]} for k, v in ENGINE_PRESETS.items()],
            "loaded": any(getattr(d, "ok", False) for d in manager.ensemble.detectors),
        })

    @app.post("/api/engine")
    def set_engine():
        choice = (request.get_json(force=True) or {}).get("engine", "")
        if choice not in ENGINE_PRESETS:
            return jsonify({"error": "unknown engine"}), 400
        with engine_lock:
            conf = manager.ensemble.min_score
            cfg.apply_engine(choice)
            new = build_ensemble(cfg)
            _apply_confidence(new, conf)
            manager.set_ensemble(new)
        log.info("detection model switched to '%s'", cfg.engine)
        return jsonify({"ok": True, "engine": cfg.engine,
                        "loaded": any(getattr(d, "ok", False) for d in new.detectors)})

    # ---------------------------------------------------------------- sensitivity
    @app.get("/api/controls")
    def get_controls():
        return jsonify({"confidence": round(manager.ensemble.min_score, 2),
                        "drones_only": cfg.drones_only, "engine": cfg.engine})

    @app.post("/api/controls")
    def set_controls():
        d = request.get_json(force=True) or {}
        _apply_confidence(manager.ensemble, d.get("confidence"))
        if "drones_only" in d:
            cfg.drones_only = bool(d["drones_only"])
        return jsonify({"ok": True, "confidence": round(manager.ensemble.min_score, 2)})

    # ---------------------------------------------------------------- cameras
    @app.get("/api/cameras")
    def cameras():
        return jsonify([
            {"id": s.id, "name": s.name, "online": s.online,
             "fps": round(s.fps, 1), "targets": s.targets, "error": s.error}
            for s in manager.states()
        ])

    @app.post("/api/cameras")
    def add_camera():
        d = request.get_json(force=True) or {}
        source = net_candidates.get(d.get("token")) or d.get("source")
        if not source:
            return jsonify({"error": "source required"}), 400
        cam = CameraConfig(id=d.get("id") or uuid.uuid4().hex[:8], source=str(source), name=d.get("name", ""))
        manager.add(cam)
        log.info("camera added: %s (%s)", cam.name or cam.id, cam.id)
        return jsonify({"ok": True, "id": cam.id})

    @app.delete("/api/cameras/<cam_id>")
    def del_camera(cam_id):
        manager.remove(cam_id)
        return jsonify({"ok": True})

    @app.get("/stream/<cam_id>")
    def stream(cam_id):
        import time

        worker = manager.worker(cam_id)
        if not worker:
            abort(404)

        def gen():
            boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            while True:
                frame = worker.snapshot()
                if frame is not None:
                    jpg = encode(frame)
                    if jpg:
                        yield boundary + jpg + b"\r\n"
                time.sleep(0.04)

        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    # ---------------------------------------------------------------- discovery
    @app.post("/api/discover")
    def start_discover():
        if discover_state["scanning"]:
            return jsonify({"ok": True, "scanning": True})
        d = request.get_json(force=True) or {}
        user, pw = d.get("user", ""), d.get("password", "")

        def scan():
            from skysentinel import netcam

            discover_state.update(scanning=True, cameras=[], message="Scanning the network…")
            try:
                found = netcam.discover()
            except Exception:
                found = []
            for cam in found:
                ip = cam["ip"]
                entry = {"ip": ip, "name": cam.get("name", "Camera"), "ok": False, "token": None}
                try:
                    url, capture, frame = netcam.resolve({"ip": ip, "user": user, "password": pw})
                    if capture is not None:
                        capture.release()
                        token = uuid.uuid4().hex[:10]
                        net_candidates[token] = url
                        entry.update(ok=True, token=token)
                except Exception:
                    pass
                discover_state["cameras"].append(entry)
            if not found:
                msg = "No cameras found. Check the cables and that the cameras are powered on."
            elif not any(c["ok"] for c in discover_state["cameras"]):
                msg = "Cameras found, but the username or password was not accepted."
            else:
                msg = ""
            discover_state.update(scanning=False, message=msg)

        threading.Thread(target=scan, daemon=True).start()
        return jsonify({"ok": True, "scanning": True})

    @app.get("/api/discover")
    def get_discover():
        return jsonify(discover_state)

    # ---------------------------------------------------------------- video upload
    @app.post("/api/upload")
    def upload():
        f = request.files.get("video")
        ext = Path(f.filename).suffix.lower() if f and f.filename else ""
        is_image = ext in ALLOWED_IMG
        if not f or (ext not in ALLOWED and not is_image):
            return jsonify({"error": "unsupported file"}), 400
        job_id = uuid.uuid4().hex[:12]
        src = UPLOADS / f"{job_id}{ext}"
        f.save(src)
        JOBS[job_id] = {"status": "queued", "progress": 0, "targets": 0, "image": is_image}

        job_cfg = copy.deepcopy(cfg)
        engine = request.form.get("engine")
        if engine in ENGINE_PRESETS:
            job_cfg.apply_engine(engine)
        conf = _clampf(request.form.get("confidence"), manager.ensemble.min_score, 0.05, 0.95)

        def work_image():
            JOBS[job_id]["status"] = "processing"
            try:
                job_ens = build_ensemble(job_cfg)
                _apply_confidence(job_ens, conf)
                out = OUTPUTS / f"{job_id}.jpg"
                stats = process_image(str(src), str(out), job_ens)
                JOBS[job_id].update(status="done", progress=100, output=out.name, image=True,
                                    targets=stats["detections"], stats=stats)
            except Exception as e:  # noqa: BLE001
                JOBS[job_id].update(status="error", error=str(e))

        def work_video():
            JOBS[job_id]["status"] = "processing"
            try:
                job_ens = build_ensemble(job_cfg)
                _apply_confidence(job_ens, conf)
                out_raw = OUTPUTS / f"{job_id}_raw.mp4"
                out = OUTPUTS / f"{job_id}.mp4"

                def prog(done, total, n):
                    JOBS[job_id].update(progress=min(95, int(done / total * 95)), targets=n)

                stats = process_video(str(src), str(out_raw), job_ens, job_cfg, on_progress=prog)
                JOBS[job_id].update(status="transcoding", progress=97)
                if _to_web_h264(out_raw, out):
                    out_raw.unlink(missing_ok=True)
                else:
                    out_raw.replace(out)
                JOBS[job_id].update(status="done", progress=100, output=out.name, image=False, stats=stats)
            except Exception as e:  # noqa: BLE001
                JOBS[job_id].update(status="error", error=str(e))

        threading.Thread(target=work_image if is_image else work_video, daemon=True).start()
        return jsonify({"job_id": job_id})

    @app.get("/api/job/<job_id>")
    def job(job_id):
        return jsonify(JOBS.get(job_id) or {"error": "not found"}), (200 if job_id in JOBS else 404)

    @app.get("/video/<job_id>")
    def video(job_id):
        out = OUTPUTS / f"{job_id}.mp4"
        if not out.exists():
            abort(404)
        return send_file(out, mimetype="video/mp4")

    @app.get("/image/<job_id>")
    def image(job_id):
        out = OUTPUTS / f"{job_id}.jpg"
        if not out.exists():
            abort(404)
        return send_file(out, mimetype="image/jpeg")

    app.manager = manager
    return app


if __name__ == "__main__":
    app = create_app()
    print("\n  SkySentinel is running — open:\n\n      http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, threaded=True)
