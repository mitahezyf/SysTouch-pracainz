import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.gesture_engine.utils.landmarks import INDEX_MCP, MIDDLE_TIP, RING_MCP, WRIST
from app.gesture_trainer import calibrator as cal


def make_point(x, y, z=0.0):
    return SimpleNamespace(x=x, y=y, z=z)


def make_landmarks():
    # przygotuj 21 punktow z atrybutami x,y,z
    pts = [make_point(0.0, 0.0, 0.0) for _ in range(21)]
    pts[WRIST] = make_point(0.0, 0.0, 0.0)
    pts[MIDDLE_TIP] = make_point(0.0, 10.0, 0.0)  # odleglosc od wrist = 10
    pts[INDEX_MCP] = make_point(5.0, 0.0, 0.0)
    pts[RING_MCP] = make_point(9.0, 0.0, 0.0)  # szerokosc = 4
    return pts


def test_calibrate_and_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    tmp_file = tmp_path / "calibration.json"
    monkeypatch.setattr(cal, "CALIBRATION_PATH", tmp_file)

    landmarks = make_landmarks()
    data = cal.calibrate(landmarks)

    assert tmp_file.exists()
    # sprawdz wartosci w pliku
    on_disk = json.loads(tmp_file.read_text())
    assert pytest.approx(on_disk["hand_size"], rel=1e-6) == 10.0
    assert pytest.approx(on_disk["hand_width"], rel=1e-6) == 4.0

    # zwrocone dane musza byc zgodne z zapisanymi
    assert data == on_disk

    # load_calibration zwraca to samo
    loaded = cal.load_calibration()
    assert loaded == on_disk
