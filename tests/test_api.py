import importlib
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEORAG_DATA_DIR", str(tmp_path))
    import videorag.api
    import videorag.config

    importlib.reload(videorag.config)
    api = importlib.reload(videorag.api)
    with TestClient(api.app) as c:
        yield c, api


def test_health(client):
    c, _ = client
    assert c.get("/api/health").json()["status"] == "ok"


def test_videos_never_expose_server_paths(client):
    c, api = client
    v = api.store.create_video("t", "t.mp4")
    api.store.update_video(v.id, path=os.path.join("secret", "t.mp4"), status="ready")
    body = c.get("/api/videos").json()
    assert body[0]["has_media"] is True and "path" not in body[0]


def test_rejects_bad_upload_type(client):
    c, _ = client
    r = c.post("/api/videos", files={"file": ("evil.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 400


def test_rejects_non_http_url(client):
    c, _ = client
    assert c.post("/api/videos/url", json={"url": "file:///etc/passwd"}).status_code == 400


def test_frame_path_validation(client):
    c, _ = client
    assert c.get("/api/frames/not-an-id/1.jpg").status_code == 400
    assert c.get("/api/frames/abcdef123456/x.png").status_code == 400


def test_unknown_video_404(client):
    c, _ = client
    assert c.get("/api/videos/abcdef123456").status_code == 404
    assert c.delete("/api/videos/abcdef123456").status_code == 404
