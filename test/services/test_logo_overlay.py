import shutil
import subprocess
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import asgi
from app.config import config
from app.controllers.v1 import video as controller
from app.models.schema import TaskVideoRequest
from app.services import state as sm
from app.services import video
from app.utils import utils


def test_logo_box_keeps_ratio_corner_and_margin():
    # 1080x1920 portrait, 15 % of the width, top right, 4 % margin of the shorter side
    assert video.logo_overlay_box(1080, 1920, 400, 200, "top_right", 0.15) == (1080 - 162 - 43, 43, 162, 81)
    assert video.logo_overlay_box(1920, 1080, 100, 100, "bottom_left", 0.1) == (43, 1080 - 192 - 43, 192, 192)
    # A very tall logo never exceeds a quarter of the height
    x, y, w, h = video.logo_overlay_box(1080, 1080, 100, 1000, "top_left", 0.4)
    assert h == 270 and w == 27 and (x, y) == (43, 43)


def test_logo_must_be_an_image_inside_the_materials_directory(tmp_path):
    (tmp_path / "logo.png").write_bytes(b"x")
    (tmp_path / "clip.mp4").write_bytes(b"x")
    with patch.object(utils, "storage_dir", return_value=str(tmp_path)):
        assert video.resolve_logo_path("logo.png") == str((tmp_path / "logo.png").resolve())
        with pytest.raises(ValueError):
            video.resolve_logo_path("clip.mp4")
        with pytest.raises(ValueError):
            video.resolve_logo_path("../../etc/passwd")
        with pytest.raises(ValueError):
            video.resolve_logo_path("missing.png")


def test_http_rejects_unknown_logo_before_queue(tmp_path):
    with patch.dict(config.app, {"api_key": ""}), patch.object(sm, "state", sm.MemoryState()), \
            patch.object(controller.task_manager, "add_task") as queued, patch.object(utils, "storage_dir", return_value=str(tmp_path)):
        with TestClient(asgi.app) as client:
            response = client.post("/api/v1/videos", json={"video_subject": "Coffee", "subtitle_enabled": False, "logo_material": "nope.png"})
        assert response.status_code == 400
        assert "logo" in response.json()["message"]
        queued.assert_not_called()
        Image.new("RGB", (10, 10), "red").save(tmp_path / "brand.png")
        with TestClient(asgi.app) as client:
            response = client.post("/api/v1/videos", json={"video_subject": "Coffee", "subtitle_enabled": False, "logo_material": "brand.png", "logo_position": "bottom_left"})
        assert response.status_code == 200
        queued.assert_called_once()


def test_logo_is_burned_into_every_frame(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("native FFmpeg required")
    footage, speech, output = (tmp_path / name for name in ("input.mp4", "speech.wav", "final.mp4"))
    base = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-threads", "1"]
    subprocess.run([*base, "-f", "lavfi", "-i", "color=c=blue:s=200x200:r=10:d=2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(footage)], check=True)
    subprocess.run([*base, "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "2", str(speech)], check=True)
    Image.new("RGB", (100, 100), (255, 0, 0)).save(tmp_path / "logo.png")
    request = TaskVideoRequest(video_subject="Logo", subtitle_enabled=False, bgm_type="", n_threads=1,
                               logo_material="logo.png", logo_position="top_right", logo_width_ratio=0.25, logo_opacity=1.0)
    with patch.object(utils, "storage_dir", return_value=str(tmp_path)):
        video.generate_video(str(footage), str(speech), "", str(output), request)
    frames = subprocess.run([*base, "-i", str(output), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    frame_size = 200 * 200 * 3
    count = len(frames) // frame_size
    assert count >= 15
    for index in (0, count // 2, count - 1):
        frame = np.frombuffer(frames[index * frame_size:(index + 1) * frame_size], dtype=np.uint8).reshape(200, 200, 3)
        # Logo: 50x50 at x = 200 - 50 - 8, y = 8 → centre (167, 33) is red; the rest stays blue
        r, g, b = frame[33, 167].tolist()
        assert r > 180 and b < 80
        r, g, b = frame[150, 50].tolist()
        assert b > 180 and r < 80
