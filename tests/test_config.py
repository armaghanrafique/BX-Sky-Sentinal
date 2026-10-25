from skysentinel.config import Config


def test_defaults_have_two_models():
    cfg = Config.load()
    assert len(cfg.models) == 2
    assert cfg.motion_fallback is True


def test_overrides_applied():
    cfg = Config.load(overrides={"frame_stride": 4, "device": "cpu"})
    assert cfg.frame_stride == 4
    assert cfg.device == "cpu"


def test_load_from_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "device: cpu\nmin_support: 2\n"
        "models:\n  - name: m1\n    weights: w.pt\n    weight: 0.5\n"
        "cameras:\n  - id: c1\n    source: '0'\n    name: Cam\n",
        encoding="utf-8",
    )
    cfg = Config.load(str(p))
    assert cfg.device == "cpu" and cfg.min_support == 2
    assert len(cfg.models) == 1 and cfg.models[0].name == "m1"
    assert len(cfg.cameras) == 1 and cfg.cameras[0].id == "c1"


def test_enabled_models_filter():
    cfg = Config.load()
    cfg.models[0].enabled = False
    assert len(cfg.enabled_models()) == 1
