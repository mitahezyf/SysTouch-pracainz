import pytest

from app.gesture_trainer import normalizer as norm


def test_normalize_raises_without_calibration(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(norm, "load_calibration", lambda: None)
    with pytest.raises(ValueError):
        norm.normalize_landmarks([(0.0, 0.0, 0.0) for _ in range(21)])


def test_normalize_ok(monkeypatch: pytest.MonkeyPatch):
    # kalibracja: hand_size = 10
    monkeypatch.setattr(norm, "load_calibration", lambda: {"hand_size": 10.0})

    wrist = (1.0, 1.0, 1.0)
    p2 = (11.0, 1.0, 1.0)  # dx = 1.0 po normalizacji
    landmarks = [wrist] + [p2] + [(1.0, 1.0, 1.0) for _ in range(19)]

    vec = norm.normalize_landmarks(landmarks)

    assert len(vec) == 63
    # pierwsze 3 wartosci dla wrist to zera
    assert vec[0:3] == [0.0, 0.0, 0.0]
    # kolejne 3 dla p2: dx=1, dy=0, dz=0 (podzielone przez hand_size=10)
    assert vec[3:6] == [pytest.approx(1.0), 0.0, 0.0]
