import json
from pathlib import Path

import pytest

from app.gesture_trainer import manager as mgr


def test_assign_and_get_action(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    map_file = tmp_path / "gesture_action_map.json"
    monkeypatch.setattr(mgr, "MAP_PATH", map_file)

    # przypisz dwie akcje
    mgr.assign_action("click", "left_click")
    mgr.assign_action("scroll", "scroll_down")

    assert map_file.exists()
    data = json.loads(map_file.read_text())
    assert data["click"] == "left_click"
    assert data["scroll"] == "scroll_down"

    # odczyt pojedynczej akcji
    assert mgr.get_action_for_gesture("click") == "left_click"
    assert mgr.get_action_for_gesture("unknown") is None


def test_train_and_save_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # podmien load_all_samples aby dac maly zbior uczacy
    monkeypatch.setattr(
        mgr,
        "load_all_samples",
        lambda: {
            "click": [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]],
            "scroll": [[1.0, 1.0, 1.0], [0.95, 1.0, 1.0]],
        },
    )

    # podmien MODEL_PATH w module classifier aby zapis poszedl do tmp
    import app.gesture_trainer.classifier as clf_mod

    model_path = tmp_path / "model.pkl"
    monkeypatch.setattr(clf_mod, "MODEL_PATH", str(model_path))

    clf = mgr.train_and_save_model()

    assert model_path.exists()
    # szybka sanity predykcja
    assert clf.predict([0.0, 0.01, 0.0]) == "click"
