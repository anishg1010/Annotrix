"""API tests. The AI tests download small model weights on first run.

Run from backend/:  pytest -q          (skip model tests: pytest -q -m "not models")
"""
import io
import json
import os
import zipfile

import pytest
from PIL import Image as PILImage, ImageDraw


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LA_DATA_DIR", str(tmp_path))
    weights = os.environ.get("LA_WEIGHTS_DIR")
    import importlib

    from app import config, db

    importlib.reload(config)
    if weights:
        config.WEIGHTS_DIR = __import__("pathlib").Path(weights)
    db.reset_engine()
    from app import main

    importlib.reload(main)
    from fastapi.testclient import TestClient

    return TestClient(main.app)


def _img(w=320, h=240, color=(30, 120, 30)):
    im = PILImage.new("RGB", (w, h), color)
    d = ImageDraw.Draw(im)
    d.ellipse([100, 60, 220, 180], fill=(240, 240, 240))
    b = io.BytesIO()
    im.save(b, "JPEG")
    return b.getvalue()


def _project(client, mode="assisted"):
    r = client.post("/api/projects", json={"name": "Test", "labels": ["zebra", "elephant", " zebra "], "mode": mode})
    assert r.status_code == 201
    return r.json()


def test_project_and_upload(client):
    p = _project(client)
    assert p["labels"] == ["zebra", "elephant"]
    r = client.post(
        f"/api/projects/{p['id']}/images",
        files=[("files", ("a.jpg", _img(), "image/jpeg")), ("files", ("notes.txt", b"hi", "text/plain"))],
    )
    assert r.status_code == 201
    body = r.json()
    assert len(body["created"]) == 1 and body["skipped"][0]["file"] == "notes.txt"
    im = body["created"][0]
    assert (im["width"], im["height"]) == (320, 240)
    assert client.get(im["url"]).status_code == 200


def test_save_export_stats(client):
    p = _project(client)
    im = client.post(f"/api/projects/{p['id']}/images", files=[("files", ("a.jpg", _img(), "image/jpeg"))]).json()["created"][0]
    anns = [
        {"label": "zebra", "x": 10, "y": 10, "w": 50, "h": 40, "source": "ai_detect", "score": 0.9},
        {"label": "elephant", "x": 300, "y": 200, "w": 100, "h": 100,  # gets clamped
         "polygon": [[300, 200], [320, 200], [320, 240], [300, 240]], "source": "ai_segment"},
        {"label": "zebra", "x": 5, "y": 5, "w": 0.2, "h": 10},  # degenerate -> dropped
    ]
    r = client.put(
        f"/api/images/{im['id']}/annotations",
        json={"annotations": anns, "done": True, "session": {"active_seconds": 12.5, "n_actions": 3, "n_accepted": 1}},
    )
    assert r.status_code == 200 and r.json()["saved"] == 2
    saved = client.get(f"/api/images/{im['id']}/annotations").json()
    assert saved[1]["w"] == 20 and saved[1]["h"] == 40

    z = zipfile.ZipFile(io.BytesIO(client.get(f"/api/projects/{p['id']}/export?format=coco").content))
    coco = json.loads(z.read("annotations.json"))
    assert len(coco["annotations"]) == 2 and coco["images"][0]["file_name"] == "images/a.jpg"
    assert coco["annotations"][1]["segmentation"] and "images/a.jpg" in z.namelist()

    z = zipfile.ZipFile(io.BytesIO(client.get(f"/api/projects/{p['id']}/export?format=yolo").content))
    lines = z.read("labels/a.txt").decode().splitlines()
    assert lines[0].startswith("0 ") and lines[1].startswith("1 ")
    assert "names:" in z.read("data.yaml").decode()

    st = client.get(f"/api/projects/{p['id']}/stats").json()
    assert st["assisted"]["images"] == 1 and st["assisted"]["seconds_per_image"] == 12.5


def test_manual_mode_blocks_ai(client):
    p = _project(client, mode="manual")
    im = client.post(f"/api/projects/{p['id']}/images", files=[("files", ("a.jpg", _img(), "image/jpeg"))]).json()["created"][0]
    assert client.post(f"/api/images/{im['id']}/ai/detect", json={}).status_code == 409


@pytest.mark.models
def test_ai_segment_and_detect(client):
    p = _project(client)
    im = client.post(f"/api/projects/{p['id']}/images", files=[("files", ("a.jpg", _img(), "image/jpeg"))]).json()["created"][0]
    r = client.post(f"/api/images/{im['id']}/ai/segment", json={"points": [[160, 120]], "point_labels": [1]})
    assert r.status_code == 200, r.text
    seg = r.json()
    # the white disc spans x 100..220, y 60..180
    assert 90 <= seg["x"] <= 110 and 110 <= seg["w"] <= 130 and len(seg["polygon"]) >= 3
    r = client.post(f"/api/images/{im['id']}/ai/detect", json={"conf": 0.25})
    assert r.status_code == 200 and isinstance(r.json()["suggestions"], list)
