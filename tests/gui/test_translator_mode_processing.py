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

    def process_landmarks(self, landmarks):  # noqa: D401
        # landmarks to np.ndarray shape (21, 3)
        self.calls += 1
        # symuluje litery na przemian
        return "A" if self.calls % 2 else "B"

    def process_frame(self, normalized_landmarks):  # noqa: D401
        # deprecated, ale zachowane dla kompatybilnosci
        self.calls += 1
        return "A" if self.calls % 2 else "B"

    def get_state(self):  # noqa: D401
        return {
            "current_letter": "A" if self.calls % 2 else "B",
            "confidence": 0.95,
            "buffer_fill": 7,
            "buffer_size": 7,
            "time_held_ms": 100.0,
            "total_detections": self.calls,
            "session_duration_s": 10.0,
            "detections_per_minute": 6.0,
            "unique_letters": 2,
        }


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
    assert gesture_res.confidence == 0.95  # confidence ze stanu translatora
    assert len(per_hand) == 1


def test_translator_mode_does_not_call_detect_gesture(monkeypatch):
    # sprawdza ze w trybie translator nie wywoluje sie detect_gesture (tylko litery PJM)
    detect_gesture_called = []

    def mock_detect_gesture(landmarks):
        detect_gesture_called.append(True)
        return ("click", 0.9)

    monkeypatch.setattr(processing, "detect_gesture", mock_detect_gesture)

    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    tracker = FakeTracker()
    visualizer = StubVisualizer()
    translator = DummyTranslator()
    normalizer = DummyNormalizer()

    processing.detect_and_draw(
        frame_bgr=frame,
        tracker=tracker,
        json_runtime=None,
        visualizer=visualizer,
        preview_enabled=False,
        mode="translator",
        translator=translator,
        normalizer=normalizer,
    )

    # w trybie translator detect_gesture NIE powinien byc wywolany
    assert (
        len(detect_gesture_called) == 0
    ), "detect_gesture nie powinien byc wywolany w trybie translator"


def test_translator_mode_does_not_call_json_runtime(monkeypatch):
    # sprawdza czy w trybie translator nie wywoluje sie json_runtime.update
    class FakeJsonRuntime:
        def __init__(self):
            self.update_calls = 0

        def update(self, points):
            self.update_calls += 1
            return {"action": {"type": "move_mouse"}, "confidence": 0.9}

    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    tracker = FakeTracker()
    visualizer = StubVisualizer()
    translator = DummyTranslator()
    normalizer = DummyNormalizer()
    json_runtime = FakeJsonRuntime()

    display_frame, gesture_res, per_hand = processing.detect_and_draw(
        frame_bgr=frame,
        tracker=tracker,
        json_runtime=json_runtime,
        visualizer=visualizer,
        preview_enabled=False,
        mode="translator",
        translator=translator,
        normalizer=normalizer,
    )

    # w trybie translator json_runtime.update NIE powinien byc wywolany
    assert (
        json_runtime.update_calls == 0
    ), "json_runtime.update nie powinien byc wywolany w trybie translator"
    # konwertuje na numpy array
    df_arr = np.asarray(display_frame)
    assert df_arr.shape == frame.shape
    assert translator.calls == 1
