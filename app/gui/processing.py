from __future__ import annotations

from typing import List, Optional, Protocol, Tuple

from app.gesture_engine.core.handlers import gesture_handlers
from app.gesture_engine.detector.gesture_detector import detect_gesture
from app.gesture_engine.logger import logger
from app.gesture_engine.utils.visualizer import Visualizer
from app.gui.models import GestureResult, SingleHandResult


class TranslatorLike(Protocol):
    def predict(self, normalized_landmarks: list[float]) -> str: ...


class NormalizerLike(Protocol):
    def normalize(self, landmarks) -> list[float]: ...


def detect_and_draw(
    frame_bgr,
    tracker,
    json_runtime,
    visualizer: Visualizer,
    preview_enabled: bool,
    mode: str = "gestures",
    translator: Optional[TranslatorLike] = None,
    normalizer: Optional[NormalizerLike] = None,
) -> Tuple[object, GestureResult, List[SingleHandResult]]:
    # wykrywa gesty dla wielu rak i rysuje wizualizacje na klatce
    # zwraca tuple: (display_frame, wynik globalny, lista wynikow per reka)
    import cv2  # lokalny import

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # tworzy kopie tylko gdy rysuje podglad inaczej zwraca oryginal
    display_frame = frame_bgr.copy() if preview_enabled else frame_bgr

    results = tracker.process(frame_rgb)

    # zbiera wyniki per reka
    per_hand: List[SingleHandResult] = []

    # wybiera najlepszy gest globalny po confidence
    best_name: Optional[str] = None
    best_conf: float = 0.0

    handed_list = getattr(results, "multi_handedness", None) if results else None

    if results and getattr(results, "multi_hand_landmarks", None):
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            gesture_name: Optional[str] = None
            confidence: float = 0.0
            points = [
                (lm.x, lm.y, getattr(lm, "z", 0.0)) for lm in hand_landmarks.landmark
            ]
            if mode == "translator":
                if translator and normalizer:
                    try:
                        norm_coords = normalizer.normalize(points)
                        letter = translator.predict(norm_coords)
                        gesture_name = letter
                        confidence = 1.0
                    except Exception as exc:
                        logger.debug(f"[translator] wyjatek: {exc}")
                        gesture_name = None
                        confidence = 0.0
                else:
                    # brak zasobow translatora w trybie translator nie fallbackuje do gestow
                    logger.debug("[translator] brak modelu/normalizera - pomijam gesty")
                    gesture_name = None
                    confidence = 0.0
            else:
                if json_runtime is not None:
                    try:
                        res = json_runtime.update(points)
                    except Exception as exc:
                        logger.debug(f"[json] blad matchera: {exc}")
                        res = None
                    if res:
                        gesture_name = res.get("action", {}).get("type")
                        confidence = float(res.get("confidence", 1.0))

                if gesture_name is None:
                    gesture = detect_gesture(hand_landmarks.landmark)
                    if gesture:
                        gesture_name, confidence = gesture
                elif gesture_name not in gesture_handlers:
                    alt = detect_gesture(hand_landmarks.landmark)
                    if alt:
                        gesture_name, confidence = alt

            handed = None
            try:
                if handed_list and idx < len(handed_list):
                    handed = handed_list[idx].classification[0].label  # Left lub Right
            except Exception:
                handed = None

            per_hand.append(
                SingleHandResult(
                    index=idx,
                    name=gesture_name,
                    confidence=confidence,
                    landmarks=hand_landmarks.landmark,
                    handedness=handed,
                )
            )

            if gesture_name is not None and confidence >= best_conf:
                best_name = gesture_name
                best_conf = confidence

            # rysuje wizualizacje kazdej reki gdy podglad wlaczony
            if preview_enabled:
                label = gesture_name or ""
                label_text = f"{label}: ({confidence * 100:.1f})" if label else ""
                visualizer.draw_landmarks(display_frame, hand_landmarks)
                visualizer.draw_hand_box(
                    display_frame, hand_landmarks, label=label_text
                )

    return display_frame, GestureResult(best_name, best_conf), per_hand
