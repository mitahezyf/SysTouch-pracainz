from app.gesture_engine.logger import logger

# odporny na brak mediapipe: tworzymy stub, aby import modulu nie padal w CI
try:  # pragma: no cover
    import mediapipe as mp
except Exception:  # pragma: no cover

    class _HandsStub:
        def __init__(self, *_, **__):
            pass

        def process(self, frame_rgb):  # zgodnie z API tests
            return None

    # budujemy strukture mp.solutions.*
    class _Solutions:  # prosty holder
        pass

    solutions = _Solutions()
    hands_ns = type("hands", (), {"Hands": _HandsStub})
    setattr(solutions, "hands", hands_ns)
    setattr(solutions, "drawing_utils", object())
    setattr(solutions, "drawing_styles", object())
    mp = type("mp_stub", (), {"solutions": solutions})()
    logger.warning("mediapipe niedostepne - uzywam no-op stuba (hand_tracker)")


class HandTracker:
    def __init__(
        self, max_num_hands=2, detection_confidence=0.7, tracking_confidence=0.7
    ):
        self.max_num_hands = max_num_hands

        # inicjalizacja mediapipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            model_complexity=1,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles

        logger.info(
            f"HandTracker initialized (max_hands={max_num_hands}, detect_conf={detection_confidence}, track_conf={tracking_confidence})"
        )

    def process(self, frame_rgb):
        return self.hands.process(frame_rgb)
