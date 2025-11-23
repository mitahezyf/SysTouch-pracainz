from typing import Any, Protocol, cast

from app.gesture_engine.config import HAND_MODEL_COMPLEXITY
from app.gesture_engine.logger import logger as base_logger


# definiujemy Protocol loggera aby mozna bylo bezpiecznie nadpisac proxy
class LoggerProtocol(Protocol):
    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None: ...
    def info(self, msg: str, *args: Any, **kwargs: Any) -> None: ...
    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None: ...
    def error(self, msg: str, *args: Any, **kwargs: Any) -> None: ...
    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None: ...


logger: LoggerProtocol = base_logger

try:  # pragma: no cover
    import app.detector.hand_tracker as _alt_ht

    if hasattr(_alt_ht, "logger"):

        class _LoggerProxy:
            def __getattr__(self, name: str):
                return getattr(_alt_ht.logger, name)

        # cast na LoggerProtocol, bo proxy ma dynamiczne atrybuty
        logger = cast(LoggerProtocol, _LoggerProxy())
except Exception:
    pass

# obsluguje brak mediapipe przez tworzenie lekkiego stuba kompatybilnego z API
try:  # pragma: no cover
    import mediapipe as mp
except Exception:  # pragma: no cover

    class _HandsStub:
        def __init__(self, *_, **__):
            pass

        def process(self, frame_rgb):  # zachowuje podpis jak w prawdziwym API
            return None

    class _Solutions:  # prosty kontener przestrzeni nazw
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
        self,
        max_num_hands: int = 2,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.7,
    ) -> None:
        self.max_num_hands = max_num_hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            model_complexity=HAND_MODEL_COMPLEXITY,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles
        logger.info(
            f"HandTracker initialized (max_hands={max_num_hands}, detect_conf={detection_confidence}, track_conf={tracking_confidence}, model_complexity={HAND_MODEL_COMPLEXITY})"
        )

    def process(self, frame_rgb: Any):
        # przetwarza ramke RGB i zwraca wyniki detekcji mediapipe
        return self.hands.process(frame_rgb)
