"""Runtime configuration, loaded from YAML and overridable per run."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ModelConfig:
    name: str
    weights: str
    weight: float = 1.0
    conf: float = 0.25
    imgsz: int = 1280
    enabled: bool = True
    classes: Optional[List[int]] = None  # restrict to these class ids, if the weight is multi-class


@dataclass
class CameraConfig:
    id: str
    source: str              # file path, rtsp://, rtmp://, http://, or a webcam index as string
    name: str = ""
    enabled: bool = True


# Performance presets. Fast throttles and downscales for throughput; Precise
# processes every frame at high resolution and requires model agreement.
PERF_MODES = {
    "fast":     {"frame_stride": 5, "imgsz": 512,  "target_fps": 0, "min_support": 1},
    "balanced": {"frame_stride": 3, "imgsz": 960,  "target_fps": 0, "min_support": 1},
    "precise":  {"frame_stride": 1, "imgsz": 1280, "target_fps": 0, "min_support": 2},
}

# Detection-speed presets that swap the actual model weights at runtime. One
# model each, so switching is simple and predictable for the operator.
ENGINE_PRESETS = {
    "fastest":  {"label": "Fastest",  "weights": "yolo11n.pt", "imgsz": 512, "stride": 5},
    "balanced": {"label": "Balanced", "weights": "yolo11s.pt", "imgsz": 640, "stride": 3},
    "accurate": {"label": "Accurate", "weights": "yolo11m.pt", "imgsz": 960, "stride": 2},
}


@dataclass
class Config:
    mode: str = "balanced"               # fast | balanced | precise (see PERF_MODES)
    engine: str = "balanced"             # fastest | balanced | accurate (swaps weights)
    device: str = "auto"                 # auto | cpu | cuda | cuda:0 ...
    half: bool = True                    # fp16 on CUDA for speed
    fuse_iou: float = 0.55
    nms_iou: float = 0.5
    min_support: int = 1                 # min number of models that must agree
    min_score: float = 0.20              # lower -> catches smaller / far-off targets
    max_det: int = 100
    frame_stride: int = 1                # run detection every Nth frame; track in between
    target_fps: int = 0                  # 0 = as fast as possible
    motion_fallback: bool = True         # use motion detector when no model weight loads
    track_max_age: int = 30
    track_iou: float = 0.3
    track_min_hits: int = 3
    trajectory_classify: bool = True     # tell drones from birds by how they move
    trajectory_min_len: int = 8          # frames of history before judging motion
    flap_verify: bool = True             # second check: reject wing-flapping birds
    flap_min_len: int = 10               # frames of history before judging flapping
    flap_strength: float = 0.45          # periodicity above this = flapping bird
    drones_only: bool = True             # only ever show drones / UAVs; birds are never drawn
    models: List[ModelConfig] = field(default_factory=list)
    cameras: List[CameraConfig] = field(default_factory=list)

    @staticmethod
    def default_models() -> List[ModelConfig]:
        # Small models by default for real-time speed. Swap to yolo11m/x for more
        # accuracy, or to a single yolo11n for maximum speed (see README).
        return [
            ModelConfig(name="yolo11s", weights="yolo11s.pt", weight=1.0, conf=0.20, imgsz=960),
            ModelConfig(name="yolov8s", weights="yolov8s.pt", weight=0.8, conf=0.20, imgsz=960),
        ]

    @classmethod
    def load(cls, path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> "Config":
        data: Dict[str, Any] = {}
        if path and Path(path).exists():
            data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if overrides:
            data.update({k: v for k, v in overrides.items() if v is not None})
        models = [ModelConfig(**m) for m in data.pop("models", [])] or cls.default_models()
        cameras = [CameraConfig(**c) for c in data.pop("cameras", [])]
        known = {f for f in cls.__dataclass_fields__ if f not in ("models", "cameras")}
        cfg = cls(**{k: v for k, v in data.items() if k in known})
        cfg.models = models
        cfg.cameras = cameras
        return cfg

    def apply_engine(self, engine: str) -> "Config":
        """Swap to a single model of the chosen detection-speed preset."""
        preset = ENGINE_PRESETS.get(engine)
        if not preset:
            raise ValueError(f"unknown engine {engine!r}; choose from {sorted(ENGINE_PRESETS)}")
        self.engine = engine
        keep = self.models[0] if self.models else ModelConfig(name="model", weights=preset["weights"])
        self.models = [ModelConfig(name=engine, weights=preset["weights"], weight=1.0,
                                   conf=keep.conf, imgsz=preset["imgsz"], classes=keep.classes)]
        self.frame_stride = preset["stride"]
        self.min_support = 1
        return self

    def apply_mode(self, mode: str) -> "Config":
        """Apply a performance preset to stride, input size, cap and agreement."""
        preset = PERF_MODES.get(mode)
        if not preset:
            raise ValueError(f"unknown mode {mode!r}; choose from {sorted(PERF_MODES)}")
        self.mode = mode
        self.frame_stride = preset["frame_stride"]
        self.target_fps = preset["target_fps"]
        self.min_support = preset["min_support"]
        for m in self.models:
            m.imgsz = preset["imgsz"]
        return self

    def enabled_models(self) -> List[ModelConfig]:
        return [m for m in self.models if m.enabled]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
