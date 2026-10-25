from skysentinel.sources import classify_source, normalize_source


def test_classify_file():
    assert classify_source("clip.mp4") == "file"
    assert classify_source("C:/videos/a.mkv") == "file"


def test_classify_stream():
    assert classify_source("rtsp://u:p@cam/stream") == "stream"
    assert classify_source("rtmp://server/live/key") == "stream"
    assert classify_source("http://host/feed.m3u8") == "stream"


def test_classify_webcam():
    assert classify_source("0") == "webcam"
    assert classify_source("1") == "webcam"


def test_normalize_source():
    assert normalize_source("0") == 0 and isinstance(normalize_source("0"), int)
    assert normalize_source("rtsp://x") == "rtsp://x"
