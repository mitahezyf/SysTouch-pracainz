from typing import Any, Protocol, cast

from app.gesture_engine.config import HAND_MODEL_COMPLEXITY
from app.gesture_engine.logger import logger

# aliasy typow (biblioteka mediapipe nie dostarcza oficjalnych stubow typow)
MultiHandLandmarks = list[Any]


class _ProcessResultProtocol(Protocol):
    # wynik pojedynczego wywolania mediapipe.Hands.process: lista landmarkow lub None
    multi_hand_landmarks: MultiHandLandmarks | None
    multi_handedness: Any | None


# obsluguje brak mediapipe przez tworzenie lekkiego stuba kompatybilnego z API
try:  # pragma: no cover
    import mediapipe as mp

    # sprawdz czy mediapipe ma poprawne API (solutions)
    if not hasattr(mp, "solutions"):
        raise AttributeError("MediaPipe zainstalowany ale brak mp.solutions")
except Exception as e:  # pragma: no cover
    logger.warning(f"mediapipe niedostepne lub uszkodzony ({e}) - uzywam no-op stuba")

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
        self._last_results: _ProcessResultProtocol | None = None
        logger.info(
            f"HandTracker initialized (max_hands={max_num_hands}, detect_conf={detection_confidence}, track_conf={tracking_confidence}, model_complexity={HAND_MODEL_COMPLEXITY})"
        )

    def process(self, frame_rgb: Any) -> _ProcessResultProtocol | None:
        # przetwarza klatke rgb i zapisuje wynik wewnetrznie
        try:
            # brak typow mediapipe - rzutuje wynik na wlasny protocol
            self._last_results = cast(
                _ProcessResultProtocol, self.hands.process(frame_rgb)
            )
        except Exception as e:
            logger.error(f"Blad przetwarzania mediapipe: {e}")
            self._last_results = None
        return self._last_results

    def get_results(self) -> _ProcessResultProtocol | None:
        # zwraca pelny wynik ostatniego przetwarzania albo None
        return self._last_results

    def get_landmarks(self) -> MultiHandLandmarks | None:
        # zwraca liste landmarkow wszystkich wykrytych dloni albo None jesli brak wynikow
        if self._last_results and self._last_results.multi_hand_landmarks:
            return cast(MultiHandLandmarks, self._last_results.multi_hand_landmarks)
        return None
