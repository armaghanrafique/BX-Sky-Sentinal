from skysentinel.config import PERF_MODES, Config
from skysentinel.fps import FpsController, FpsMeter


def test_uncapped_never_delays():
    c = FpsController(0)
    assert c.delay(0.0) == 0.0
    assert c.delay(100.0) == 0.0


def test_cap_enforces_period():
    c = FpsController(10)          # 0.1s period
    assert c.delay(0.0) == 0.0     # first frame free
    # a frame arriving immediately must wait ~one period
    assert abs(c.delay(0.0) - 0.1) < 1e-6


def test_cap_no_wait_when_already_slow():
    c = FpsController(10)
    c.delay(0.0)
    assert c.delay(5.0) == 0.0     # arrived long after the period


def test_set_max_resets():
    c = FpsController(10)
    c.delay(0.0)
    c.set_max(0)
    assert c.delay(0.0) == 0.0


def test_meter_positive_after_two_ticks():
    m = FpsMeter()
    m.tick()
    assert m.tick() >= 0


def test_modes_exist():
    assert set(PERF_MODES) == {"fast", "balanced", "precise"}


def test_apply_mode_sets_fields_and_imgsz():
    cfg = Config.load()
    cfg.apply_mode("fast")
    assert cfg.frame_stride == 5
    assert all(m.imgsz == 512 for m in cfg.models)
    cfg.apply_mode("precise")
    assert cfg.frame_stride == 1 and cfg.min_support == 2
    assert all(m.imgsz == 1280 for m in cfg.models)


def test_apply_mode_rejects_unknown():
    cfg = Config.load()
    try:
        cfg.apply_mode("turbo")
        assert False
    except ValueError:
        pass
