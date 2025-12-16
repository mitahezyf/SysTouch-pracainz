from types import SimpleNamespace

import numpy as np

import app.gui.processing as processing


class FakeLm:
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


class FakeHand:
    def __init__(self):
        # 21 landmarkow jak w MediaPipe; wartosci nieistotne dla testu
        self.landmark = [FakeLm(0.1, 0.2, 0.0) for _ in range(21)]


class FakeTracker:
    def process(self, _frame_rgb):
        # zwraca obiekt z multi_hand_landmarks i multi_handedness jak mediapipe
        return SimpleNamespace(
            multi_hand_landmarks=[FakeHand()],
            multi_handedness=[
                SimpleNamespace(classification=[SimpleNamespace(label="Right")])
            ],
        )


class FakeVisualizer:
    def __init__(self):
        self.drawn = {"landmarks": 0, "boxes": 0}

    def draw_landmarks(self, _frame, _hand):
        self.drawn["landmarks"] += 1

    def draw_hand_box(self, _frame, _hand, label: str = ""):
        self.drawn["boxes"] += 1


def test_detect_and_draw_with_preview(monkeypatch):
    # stubuje detektor gestow, aby zawsze zwracal click z pewnoscia 0.95
    monkeypatch.setattr(processing, "detect_gesture", lambda _lms: ("click", 0.95))

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    tracker = FakeTracker()
    visualizer = FakeVisualizer()

    display_frame, gesture_res, per_hand = processing.detect_and_draw(
        frame_bgr=frame,
        tracker=tracker,
        json_runtime=None,
        visualizer=visualizer,
        preview_enabled=True,
    )

    assert gesture_res.name == "click"
    assert gesture_res.confidence == 0.95
    assert len(per_hand) == 1
    # rysowanie wlaczone
    assert visualizer.drawn["landmarks"] == 1
    assert visualizer.drawn["boxes"] == 1
    # display_frame to kopia wejsciowej ramki rozmiarowo zgodna
    assert display_frame.shape == frame.shape


def test_detect_and_draw_without_preview(monkeypatch):
    monkeypatch.setattr(processing, "detect_gesture", lambda _lms: ("click", 0.9))

    frame = np.zeros((50, 60, 3), dtype=np.uint8)
    tracker = FakeTracker()
    visualizer = FakeVisualizer()

    _, gesture_res, per_hand = processing.detect_and_draw(
        frame_bgr=frame,
        tracker=tracker,
        json_runtime=None,
        visualizer=visualizer,
        preview_enabled=False,
    )

    assert gesture_res.name == "click"
    assert len(per_hand) == 1
    # sprawdza czy rysowanie bylo wylaczone
    assert visualizer.drawn["landmarks"] == 0
    assert visualizer.drawn["boxes"] == 0
