from skysentinel.config import ENGINE_PRESETS, Config, ModelConfig
from skysentinel.detector import YoloDetector


def _det(tile_size=416, overlap=0.2):
    cfg = ModelConfig(name="t", weights="x.pt", tile=True, tile_size=tile_size, tile_overlap=overlap)
    return YoloDetector(cfg, device="cpu", half=False)


def test_tiles_cover_frame_edges():
    d = _det(tile_size=416, overlap=0.2)
    tiles = d._tiles(848, 400)                 # the test clip's resolution
    assert (0, 0) in tiles
    # a tile must reach the right edge so nothing on the border is missed
    assert any(x + 416 >= 848 for x, _ in tiles)


def test_single_tile_when_frame_small():
    d = _det(tile_size=416)
    assert d._tiles(300, 200) == [(0, 0)]


def test_overlap_creates_multiple_tiles():
    d = _det(tile_size=416, overlap=0.2)
    assert len(d._tiles(1280, 720)) > 1


def test_longrange_engine_enables_tiling():
    cfg = Config.load().apply_engine("longrange")
    assert cfg.models[0].tile is True
    assert "longrange" in ENGINE_PRESETS


def test_other_engines_do_not_tile():
    for e in ("fastest", "balanced", "accurate"):
        cfg = Config.load().apply_engine(e)
        assert cfg.models[0].tile is False
