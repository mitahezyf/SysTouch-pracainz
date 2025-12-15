import csv
import importlib
import os
import tempfile
import types

import numpy as np

# testuje sign language recorder (unikalna nazwa funkcji aby uniknac konfliktu z gesture_trainer)
# uzywa sztucznego modułu cv2 wstrzykiwanego bezposrednio do recorder_mod.cv2 (bo recorder importuje cv2 zanim test patchuje sys.modules)
# stateful waitKey pozwala zapisac jedna probke potem zwraca ESC aby wyjsc z petli


def test_sign_language_recorder_csv(monkeypatch):
    recorder_mod = importlib.import_module("app.sign_language.recorder")
    with tempfile.TemporaryDirectory() as td:
        data_path = os.path.join(td, "dataset.csv")
        monkeypatch.setattr(recorder_mod, "DATA_FILE", data_path)
        monkeypatch.setattr(
            recorder_mod, "CLASSES", ["A"]
        )  # tylko jedna litera dla szybkiego testu

        fake_cv2 = types.SimpleNamespace()

        class DummyCap:
            def __init__(self, *_a, **_k):
                self.frames = 0

            def isOpened(self):
                return True

            def read(self):
                self.frames += 1
                # zwraca ramke z losowymi pikselami i True
                return True, np.zeros((100, 100, 3), dtype=np.uint8)

            def release(self):
                pass

        def VideoCapture(idx, *args):  # noqa: D401
            return DummyCap()

        def cvtColor(frame, code):  # noqa: D401
            return frame

        def putText(*_a, **_k):  # noqa: D401
            return None

        def imshow(*_a, **_k):  # noqa: D401
            return None

        state = {"samples": 0}

        def waitKey(ms):  # noqa: D401
            # ms==1000 w odliczaniu ignoruje; ms==1 w petli zapisu
            if ms == 1:
                # po pierwszym zapisie (samples>0) zwraca ESC
                return 27 if state["samples"] > 0 else ord("0")
            return ord("0")

        # normalizuje punkty dodajac je do plaskiej listy i zwieksza licznik probek
        class DummyNormalizer:
            def normalize(self, points):
                flat = []
                for x, y, z in points:
                    flat.extend([x, y, z])
                state["samples"] += 1
                return flat

        class DummyTracker:
            def process(self, *_a, **_k):
                return None

            def get_landmarks(self):
                return [[(0.1, 0.2, 0.0) for _ in range(21)]]

        # wstrzykuje sztuczne cv2 do modułu recorder
        fake_cv2.VideoCapture = VideoCapture
        fake_cv2.cvtColor = cvtColor
        fake_cv2.putText = putText
        fake_cv2.imshow = imshow
        fake_cv2.waitKey = waitKey
        monkeypatch.setattr(recorder_mod, "cv2", fake_cv2)
        monkeypatch.setattr(recorder_mod, "HandTracker", DummyTracker)
        monkeypatch.setattr(recorder_mod, "HandNormalizer", DummyNormalizer)

        recorder_mod.record_data()

        assert os.path.exists(data_path)
        with open(data_path, newline="") as f:
            rows = list(csv.reader(f))
        assert rows[0][0] == "label" and len(rows[0]) == 64
        # co najmniej jedna probka dla litery A
        assert any(r and r[0] == "A" for r in rows[1:])
        # upewnia sie ze stan licznika odnotowal probke
        assert state["samples"] >= 1
