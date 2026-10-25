# SkySentinel

![Python](https://img.shields.io/badge/python-3.9%2B-3776AB)
![Models](https://img.shields.io/badge/detection-YOLO11%20%2B%20YOLOv8%20ensemble-2fd0e0)
![Interface](https://img.shields.io/badge/interface-web%20%2B%20CLI-35e08b)
![Tests](https://img.shields.io/badge/tests-54%20passing-35e08b)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Real-time drone and UAV detection for recorded video and live camera feeds. SkySentinel combines an ensemble of object detectors with motion-trajectory analysis, so it not only spots small objects in the sky but also separates **drones from birds** by the way they move. It ingests MP4 files or multiple RTSP/RTMP CCTV streams at once and overlays tracked targets on a live dashboard.

![Live cameras dashboard](docs/images/live.png)

<p align="center"><img src="docs/images/welcome.png" width="70%" alt="Welcome screen"></p>

---

## Detection in action

<p align="center"><img src="docs/images/detection-sample.jpg" width="62%" alt="SkySentinel detecting a distant quadcopter against an overcast sky"></p>

<p align="center"><sub>A distant quadcopter detected against a flat overcast sky and labelled <b>Drone/UAV (0.62)</b> — the kind of small, low-contrast, far-range target that single-frame detectors usually miss.</sub></p>

## Capabilities

- **Detector ensemble.** Multiple models run per frame and their outputs are merged with Weighted Boxes Fusion. Agreement between models raises confidence; isolated false positives are suppressed. A minimum-agreement threshold is configurable.
- **Drone-vs-bird discrimination.** A trajectory classifier scores each track on path straightness, speed uniformity, heading stability, hover behaviour, vertical oscillation and bounding-box stability, then fuses that with the detector's appearance label.
- **Any input.** Local video files (MP4/MOV/AVI/MKV/WEBM), RTSP/RTMP/HTTP network streams, and USB webcams.
- **Many cameras at once.** Each camera runs on its own worker thread with an independent tracker, sharing a single model ensemble.
- **Live overlay.** Bounding boxes, stable track IDs, per-camera frame rate and target counts, streamed to the browser.
- **Speed / precision control.** The dashboard exposes live knobs for confidence, model agreement, detection stride, input size and an FPS cap; the CLI offers **Fast / Balanced / Precise** presets. Processing every frame at high resolution — where a high-FPS input yields the best tracking and classification — trades throughput for accuracy; throttling and downscaling do the reverse.
- **Built for throughput.** Threaded capture that always serves the most recent frame, detection on every *N*-th frame with tracking in between, FP16 inference on CUDA, and a configurable input resolution.
- **Graceful degradation.** If model weights cannot be loaded, a motion-based detector keeps the system operational.

## Interface

A session opens on a brief **welcome screen**, then two working pages:

| Live monitoring | Video analysis |
|---|---|
| ![Live cameras](docs/images/live.png) | ![Upload and analyze](docs/images/upload.png) |
| Each tile plays the annotated feed live; click one to open a full-screen player. Tiles turn red on a detection and an audible siren can be toggled. | Upload a **video or an image**: a video is annotated, re-encoded to H.264 and **plays on the page**; an image comes back with the detection boxes drawn. Both can be downloaded. |

> Note: an image is a single frame, so the motion-based bird/drone checks don't apply to it — the annotated image shows every aerial-target detection the model reports.

**Connecting cameras is automatic.** On the Live Cameras page, enter the camera username and password and press **Scan network** — SkySentinel discovers ONVIF/RTSP cameras on the LAN (including ones plugged into the laptop over Ethernet), resolves each stream itself, and offers them to add with one click. A manual option (RTSP/RTMP link or webcam number) is tucked away for the rare case it's needed.

**Two simple controls**, applied instantly with no restart:

| Control | Effect |
|---|---|
| **Detection speed** | **Fastest / Balanced / Accurate** — switches the detection model live (nano → small → medium). Choose speed or accuracy to suit the hardware |
| **Sensitivity** | From *fewer alerts* to *more alerts* (detection confidence) |

Birds are filtered out automatically; only drones are shown and counted.

---

## System architecture

```mermaid
flowchart LR
    subgraph Inputs
        V[Video files<br/>MP4 / MOV / AVI]
        R[RTSP / RTMP<br/>CCTV streams]
        W[USB webcams]
    end

    subgraph Capture
        S[VideoSource<br/>threaded reader<br/>newest-frame for live]
    end

    subgraph Detection core
        E[Ensemble]
        M1[YOLO11]
        M2[YOLOv8]
        MF[Motion fallback]
        F[Weighted Boxes Fusion]
    end

    subgraph Reasoning
        T[IoU Tracker<br/>stable IDs]
        C[Trajectory classifier<br/>drone vs bird]
    end

    subgraph Outputs
        O[Overlay]
        WEB[Web dashboard<br/>live MJPEG]
        FILE[Annotated video file]
    end

    V --> S
    R --> S
    W --> S
    S --> E
    M1 --> E
    M2 --> E
    MF --> E
    E --> F --> T --> C --> O
    O --> WEB
    O --> FILE
```

| Signal | Drone | Bird |
|---|---|---|
| Path straightness | straight lines / smooth arcs | wandering |
| Speed uniformity | near-constant | variable (flap, glide) |
| Heading stability | smooth | frequent small corrections |
| Hover | can hold a steady hover | essentially never |
| Vertical oscillation | none | periodic bobbing while flapping |
| Bounding-box shape | stable | pulses with wingbeats |

### Two-stage bird rejection

Separating a drone from a bird uses **two independent checks**, and a target must pass both:

1. **Trajectory** (`classify.py`) — *how it travels*: drones fly straight, hold steady speed, can hover; birds wander and vary speed.
2. **Wingbeat** (`flap.py`) — *what its silhouette does*: a bird flaps, so its bounding box width and area rise and fall **periodically** (an FFT finds the repeating wingbeat). A drone or UAV is rigid — nothing on it oscillates, so its outline is steady apart from smooth drift toward or away from the camera. A clear wingbeat overrides everything and marks the target a bird.

Because the wingbeat test keys on *oscillation*, it works for **every UAV type** — quadcopters, multirotors and fixed-wing — since none of them flap. Only wing-flapping birds are rejected.

By default (`drones_only: true`) **birds are never drawn**: a confirmed drone is boxed in red, a not-yet-confirmed aerial target in amber, and anything judged a bird is removed from the overlay, the target counts and the alerts. Detection keeps a low confidence floor and small-cluster support so small, distant objects are still picked up — a far-off drone is detected the moment it appears and shown once it is cleared of being a bird.

> Appearance alone cannot name a "drone" with general COCO weights, so this discrimination is primarily **motion-based**. For the strongest appearance signal as well, load **drone-trained weights** (see below); the trajectory filter then refines them further.

---

## Getting started

```bash
python -m venv venv
# Windows: venv\Scripts\activate    Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

For GPU acceleration, install a CUDA build of PyTorch before the other requirements: <https://pytorch.org/get-started/locally/>.

### Model weights

The ensemble defaults to `yolo11x.pt` and `yolov8x.pt`, which Ultralytics downloads on first use. These are strong general detectors; for production airspace accuracy, point the configuration at **drone-trained weights** and pin the class if the weight has a single `drone` label:

```yaml
models:
  - name: drone-yolo11
    weights: weights/drone_yolo11.pt
    classes: [0]
```

### Run the dashboard

```mermaid
sequenceDiagram
    participant U as Operator
    participant B as Browser
    participant A as SkySentinel server
    participant M as Camera manager

    U->>B: Open dashboard
    U->>B: Add camera (RTSP/RTMP/file/webcam)
    B->>A: POST /api/cameras
    A->>M: Start worker thread
    M-->>A: Annotated frames + target counts
    B->>A: GET /stream/<id> (MJPEG)
    A-->>B: Live annotated feed
```

```bash
python run.py web                      # http://localhost:5000
python run.py web --config config.yaml

# Annotate a recorded clip
python run.py video --input clip.mp4 --output out.mp4

# Record an annotated live stream
python run.py stream --input rtsp://user:pass@192.168.1.50:554/stream1 --output live.mp4

# Faster, stricter: detect every 3rd frame, require two models to agree
python run.py video --input clip.mp4 --output out.mp4 --stride 3 --min-support 2
```

---

## Performance modes

The dashboard tunes everything with live knobs; the CLI bundles sensible combinations into three presets.

| Mode | Frame stride | Input size | Model agreement | Use for |
|---|---|---|---|---|
| **Fast** | every 5th frame | 512 | 1 | many cameras, limited hardware, quick triage |
| **Balanced** | every 3rd frame | 960 | 1 | default day-to-day monitoring |
| **Precise** | every frame | 1280 | 2 | investigation on high-FPS footage; best tracking + drone/bird accuracy |

```bash
python run.py video --input clip.mp4 --output out.mp4 --mode precise
python run.py stream --input rtsp://cam/stream1 --mode fast --max-fps 12
python run.py video --input clip.mp4 --output out.mp4 --turbo    # fastest: 1 nano model @ 512
```





## Testing

```bash
pytest
```

The suite validates box fusion and NMS, tracker identity and ageing, the trajectory classifier (straight → drone, erratic → bird, hover → drone), ensemble support and limits, configuration loading, source classification, and the full detect-track-classify pipeline. It runs on CPU without model downloads.


