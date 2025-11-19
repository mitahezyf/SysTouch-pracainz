from __future__ import annotations

from typing import List, Optional, Tuple

from app.gesture_engine.core.handlers import gesture_handlers
from app.gesture_engine.detector.gesture_detector import detect_gesture
from app.gesture_engine.logger import logger
from app.gesture_engine.utils.visualizer import Visualizer
from app.gui.models import GestureResult, SingleHandResult


def detect_and_draw(
    frame_bgr, tracker, json_runtime, visualizer: Visualizer, preview_enabled: bool
) -> Tuple[object, GestureResult, List[SingleHandResult]]:
    """Wykrywa gesty (dla wielu rak) i rysuje wizualizacje na klatce.

    zwraca: (display_frame, GestureResult dla UI, lista wynikow per reka)
    """
    import cv2  # lokalny import

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # tworzy kopie tylko gdy rysuje podglad, inaczej zwraca oryginalna ramke
    display_frame = frame_bgr.copy() if preview_enabled else frame_bgr

    results = tracker.process(frame_rgb)

    # wyniki per reka
    per_hand: List[SingleHandResult] = []

    # globalny (do UI): wybieramy najlepszy po confidence
    best_name: Optional[str] = None
    best_conf: float = 0.0

    handed_list = getattr(results, "multi_handedness", None) if results else None

    if results and getattr(results, "multi_hand_landmarks", None):
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            gesture_name: Optional[str] = None
            confidence: float = 0.0

            # preferuje json_runtime, jesli wlaczony
            if json_runtime is not None:
                try:
                    lm = [
                        (lm.x, lm.y, getattr(lm, "z", 0.0))
                        for lm in hand_landmarks.landmark
                    ]
                    res = json_runtime.update(lm)
                except Exception as e:
                    logger.debug(f"[json] blad matchera: {e}")
                    res = None
                if res:
                    gesture_name = res.get("action", {}).get("type")
                    confidence = float(res.get("confidence", 1.0))

            if gesture_name is None:
                gesture = detect_gesture(hand_landmarks.landmark)
                if gesture:
                    gesture_name, confidence = gesture
            else:
                # jesli json zwrocil typ nieobslugiwany, probuje klasyczny wykrywacz
                if gesture_name not in gesture_handlers:
                    alt = detect_gesture(hand_landmarks.landmark)
                    if alt:
                        gesture_name, confidence = alt

            handed = None
            try:
                if handed_list and idx < len(handed_list):
                    handed = (
                        handed_list[idx].classification[0].label
                    )  # "Left" lub "Right"
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

            # wizualizacja kazdej reki tylko gdy podglad wlaczony
            if preview_enabled:
                # dla gestu volume rysuje wskaznik bezposrednio przy opuszkach; dla innych gestow pozostaje etykieta
                if gesture_name == "volume":
                    try:
                        # wylicza pct lokalnie (bez ustawiania systemowej glosnosci), aby overlay mial swieza wartosc
                        from app.gesture_engine.gestures.volume_gesture import (
                            PINCH_RATIO as _PINCH_RATIO,
                        )
                        from app.gesture_engine.gestures.volume_gesture import (
                            volume_state as _vs,
                        )
                        from app.gesture_engine.utils.geometry import distance as _dist
                        from app.gesture_engine.utils.landmarks import (
                            FINGER_MCPS as _MCPS,
                        )
                        from app.gesture_engine.utils.landmarks import (
                            FINGER_TIPS as _TIPS,
                        )
                        from app.gesture_engine.utils.landmarks import (
                            WRIST as _WR,
                        )

                        lm = hand_landmarks.landmark
                        hand_size = _dist(lm[_WR], lm[_MCPS["pinky"]])
                        pinch_th_cfg = (
                            _vs.get("pinch_th") if isinstance(_vs, dict) else None
                        )
                        if isinstance(pinch_th_cfg, (int, float)):
                            pinch_th = float(pinch_th_cfg)
                        else:
                            pinch_th = hand_size * _PINCH_RATIO
                        d = _dist(lm[_TIPS["thumb"]], lm[_TIPS["ring"]])
                        ref_max_cfg = (
                            _vs.get("ref_max") if isinstance(_vs, dict) else None
                        )
                        if isinstance(ref_max_cfg, (int, float)):
                            denom = float(ref_max_cfg)
                        else:
                            denom = hand_size
                        pct_calc: Optional[int] = None
                        if denom > pinch_th:
                            raw_pct = (d - pinch_th) / (denom - pinch_th) * 100.0
                            pct_calc = int(max(0, min(100, int(round(raw_pct)))))
                            # kwantyzacja 5%
                            pct_calc = int(round(pct_calc / 5.0) * 5)
                        # nie nadpisuje globalnego stanu pct w GUI

                        def _coerce_pct(val: object) -> int | None:
                            if isinstance(val, (int, float)):
                                try:
                                    return int(val)
                                except Exception:
                                    return None
                            return None

                        pct_state_val: object = (
                            _vs.get("pct") if isinstance(_vs, dict) else None
                        )
                        state_pct: Optional[int] = _coerce_pct(pct_state_val)
                        pct: int | None = (
                            state_pct
                            if state_pct is not None
                            else (int(pct_calc) if pct_calc is not None else None)
                        )
                        phase = _vs.get("phase") if isinstance(_vs, dict) else None
                        phase_str = str(phase) if phase else None

                        # proba odczytu kata i delty do etykiety
                        angle = _vs.get("angle_deg") if isinstance(_vs, dict) else None
                        delta = (
                            _vs.get("angle_delta_deg")
                            if isinstance(_vs, dict)
                            else None
                        )
                        try:
                            # nieuzywane angle_txt usuniete; od razu renderujemy tekst rogu nizej
                            pass
                        except Exception:
                            pass
                        try:
                            delta_txt = (
                                f" (Δ{float(delta):.0f}°)"
                                if isinstance(delta, (int, float))
                                else ""
                            )
                        except Exception:
                            delta_txt = ""
                        if hasattr(visualizer, "draw_volume_at_tips"):
                            visualizer.draw_volume_at_tips(
                                display_frame, hand_landmarks, pct, phase_str
                            )
                            text = (
                                f"Angle: {float(angle):.0f}°{delta_txt}"
                                if isinstance(angle, (int, float))
                                else None
                            )
                            if text:
                                cv2.putText(
                                    display_frame,
                                    text,
                                    (10, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (180, 220, 255),
                                    1,
                                )
                        else:
                            # fallback: ramka z etykieta
                            visualizer.draw_landmarks(display_frame, hand_landmarks)
                            visualizer.draw_hand_box(
                                display_frame, hand_landmarks, label="volume"
                            )
                    except Exception as e:
                        logger.debug(f"volume at tips draw error: {e}")
                else:
                    label_text = (
                        f"{gesture_name}: ({confidence * 100:.1f})"
                        if gesture_name
                        else ""
                    )
                    visualizer.draw_landmarks(display_frame, hand_landmarks)
                    visualizer.draw_hand_box(
                        display_frame, hand_landmarks, label=label_text
                    )

    # usunieto pasek w rogu (draw_volume_overlay), aby wskaznik byl tylko przy opuszkach

    return display_frame, GestureResult(best_name, best_conf), per_hand
