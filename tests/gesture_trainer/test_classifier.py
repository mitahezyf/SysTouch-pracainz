from pathlib import Path

import pytest

from app.gesture_trainer.classifier import GestureClassifier as RealClassifier


def test_classifier_train_predict_save_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # przekieruj sciezke modelu do katalogu tymczasowego
    model_path = tmp_path / "model.pkl"

    # nadpisz MODEL_PATH w module classifier
    import app.gesture_trainer.classifier as clf_mod

    monkeypatch.setattr(clf_mod, "MODEL_PATH", str(model_path))

    # prosty zbior uczacy (4 probki, 2 klasy)
    data = {
        "click": [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
        ],
        "scroll": [
            [1.0, 1.0, 1.0],
            [0.9, 1.0, 1.0],
        ],
    }

    clf = RealClassifier()
    clf.train(data)

    # predykcja przyklad blisko klasy "click"
    pred1 = clf.predict([0.0, 0.05, 0.0])
    assert pred1 in ("click", "scroll")
    # wynik powinien preferowac "click"
    assert pred1 == "click"

    # zapis i odczyt
    clf.save()
    assert model_path.exists()

    clf2 = RealClassifier()
    clf2.load()
    pred2 = clf2.predict([1.0, 1.0, 0.95])
    assert pred2 == "scroll"
