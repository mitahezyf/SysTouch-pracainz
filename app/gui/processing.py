from __future__ import annotations

from typing import Optional, Tuple

from app.gesture_engine.detector.gesture_detector import detect_gesture
from app.gesture_engine.logger import logger
from app.gesture_engine.utils.visualizer import Visualizer
from app.gui.models import GestureResult


def detect_and_draw(
    frame_bgr, tracker, json_runtime, visualizer: Visualizer, preview_enabled: bool
) -> Tuple[object, GestureResult, object | None]:
    """Wykrywa gest i rysuje wizualizacje na klatce.

    zwraca: (display_frame, GestureResult, first_landmarks | None)
    """
    import cv2  # lokalny import

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    display_frame = frame_bgr.copy()

    results = tracker.process(frame_rgb)
    gesture_name: Optional[str] = None
    confidence: float = 0.0
    first_landmarks = None

    if (
        json_runtime is not None
        and results
        and getattr(results, "multi_hand_landmarks", None)
    ):
        try:
            lm = [
                (lm.x, lm.y, getattr(lm, "z", 0.0))
                for lm in results.multi_hand_landmarks[0].landmark
            ]
            res = json_runtime.update(lm)
        except Exception as e:
            logger.debug(f"[json] blad matchera: {e}")
            res = None
        if res:
            gesture_name = res.get("action", {}).get("type")
            confidence = float(res.get("confidence", 1.0))

    if results and results.multi_hand_landmarks:
        first_landmarks = results.multi_hand_landmarks[0].landmark

    if gesture_name is None and results and results.multi_hand_landmarks:
        gesture = detect_gesture(results.multi_hand_landmarks[0].landmark)
        if gesture:
            gesture_name, confidence = gesture

    if results and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            label_text = (
                f"{gesture_name}: ({confidence * 100:.1f})" if gesture_name else ""
            )
            if preview_enabled:
                visualizer.draw_landmarks(display_frame, hand_landmarks)
                visualizer.draw_hand_box(
                    display_frame, hand_landmarks, label=label_text
                )

    return display_frame, GestureResult(gesture_name, confidence), first_landmarks
