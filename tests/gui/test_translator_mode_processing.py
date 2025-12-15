from types import SimpleNamespace

import numpy as np

import app.gui.processing as processing


# zgodny stub z interfejsem visualizer (minimalny)
class StubVisualizer:
    def draw_landmarks(self, frame, hand):  # noqa: D401
        return None

    def draw_hand_box(self, frame, hand, label=""):  # noqa: D401
        return None


class FakeLm:
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


class FakeHand:
    def __init__(self):
        self.landmark = [FakeLm(0.1, 0.2, 0.0) for _ in range(21)]


class FakeTracker:
    def process(self, _frame_rgb):
        return SimpleNamespace(
            multi_hand_landmarks=[FakeHand()],
            multi_handedness=[
                SimpleNamespace(classification=[SimpleNamespace(label="Left")])
            ],
        )


class DummyTranslator:
    def __init__(self):
        self.calls = 0

    def predict(self, normalized_landmarks):  # noqa: D401
        self.calls += 1
        # symuluje litery na przemian
        return "A" if self.calls % 2 else "B"


class DummyNormalizer:
    def normalize(self, landmarks):
        # landmarks lista krotek (x,y,z) lub obiektow z .x .y .z
        flat = []
        for lm in landmarks:
            if isinstance(lm, tuple):
                flat.extend(list(lm)[:3])
            else:
                flat.extend(
                    [
                        getattr(lm, "x", 0.0),
                        getattr(lm, "y", 0.0),
                        getattr(lm, "z", 0.0),
                    ]
                )
        # zwraca 63 elementy (21*3)
        return flat[:63]


def test_detect_and_draw_translator_mode():
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    tracker = FakeTracker()
    visualizer = StubVisualizer()
    translator = DummyTranslator()
    normalizer = DummyNormalizer()

    display_frame, gesture_res, per_hand = processing.detect_and_draw(
        frame_bgr=frame,
        tracker=tracker,
        json_runtime=None,
        visualizer=visualizer,
        preview_enabled=False,
        mode="translator",
        translator=translator,
        normalizer=normalizer,
    )

    assert gesture_res.name in {"A", "B"}
    assert gesture_res.confidence == 1.0
    assert len(per_hand) == 1
    # rzutuje na numpy jezeli nie jest
    df_arr = np.asarray(display_frame)
    assert df_arr.shape == frame.shape
    assert translator.calls == 1
