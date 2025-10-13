import json
from pathlib import Path

import pytest

from app.gesture_trainer import recorder as rec


def test_record_sample_writes_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_file = tmp_path / "raw_landmarks.json"
    monkeypatch.setattr(rec, "DATA_PATH", data_file)
    monkeypatch.setattr(rec, "normalize_landmarks", lambda lm: [1, 2, 3])

    # dwie rozne etykiety gestow
    rec.record_sample("click", [(0.0, 0.0, 0.0) for _ in range(21)])
    rec.record_sample("scroll", [(0.0, 0.0, 0.0) for _ in range(21)])

    assert data_file.exists()
    data = json.loads(data_file.read_text())
    assert data["click"] == [[1, 2, 3]]
    assert data["scroll"] == [[1, 2, 3]]


def test_load_all_samples_empty_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    data_file = tmp_path / "raw_landmarks.json"
    monkeypatch.setattr(rec, "DATA_PATH", data_file)

    assert rec.load_all_samples() == {}
