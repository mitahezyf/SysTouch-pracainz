from __future__ import annotations

import time
from typing import Any, Tuple

import cv2
import numpy as np

# importuje konfiguracje i narzedzia z projektu
from app.gesture_engine.config import (
    CAPTURE_HEIGHT,
    CAPTURE_WIDTH,
    DEBUG_MODE,
    DISPLAY_HEIGHT,
    DISPLAY_WIDTH,
    JSON_GESTURE_PATHS,
    SHOW_WINDOW,
    USE_JSON_GESTURES,
)
from app.gesture_engine.core.hooks import handle_gesture_start_hook
from app.gesture_engine.detector.gesture_detector import detect_gesture
from app.gesture_engine.detector.hand_tracker import HandTracker
from app.gesture_engine.logger import logger
from app.gesture_engine.utils.performance import PerformanceTracker
from app.gesture_engine.utils.video_capture import ThreadedCapture
from app.gesture_engine.utils.visualizer import Visualizer


def _create_normalizer() -> Any | None:
    # tworzy instancje handNormalizer jesli modul dostepny w przeciwnym razie zwraca None
    try:  # pragma: no cover
        from app.gesture_trainer.normalizer import HandNormalizer
    except ImportError:
        logger.warning("normalizator niedostepny (brak modulu gesture_trainer)")
        return None
    try:
        return HandNormalizer()
    except Exception as e:  # pragma: no cover
        logger.warning(f"blad inicjalizacji normalizera: {e}")
        return None


def _create_translator() -> Tuple[Any | None, bool]:
    # tworzy translator pjm jesli modul i model dostepny zwraca krotke (translator, flaga)
    try:  # pragma: no cover
        from app.sign_language.translator import SignTranslator
    except ImportError:
        logger.warning("tlumacz pjm niedostepny (brak modulu translator)")
        return None, False
    try:
        translator = SignTranslator()
        logger.info("tlumacz pjm zaladowany poprawnie")
        return translator, True
    except Exception as e:  # pragma: no cover
        logger.error(f"nie udalo sie zainicjowac translatora: {e}")
        return None, False


def _load_json_gestures() -> Any | None:
    # laduje gesty z json i tworzy runtime gestureRuntime gdy wlaczono USE_JSON_GESTURES
    if not USE_JSON_GESTURES:
        return None
    try:  # pragma: no cover
        from app.gesture_engine.core.gesture_runtime import GestureRuntime
    except ImportError:
        logger.warning("runtime gestow json niedostepny (brak modulow)")
        return None
    try:
        runtime = GestureRuntime(JSON_GESTURE_PATHS)
        logger.info(f"zaladowano {len(runtime.defs)} gestow json")
        return runtime
    except Exception as e:  # pragma: no cover
        logger.error(f"blad ladowania gestow json: {e}")
        return None


def main() -> None:
    # inicjalizuje komponenty
    cap = ThreadedCapture()
    tracker = HandTracker()
    performance = PerformanceTracker()
    visualizer = Visualizer(
        capture_size=(CAPTURE_WIDTH, CAPTURE_HEIGHT),
        display_size=(DISPLAY_WIDTH, DISPLAY_HEIGHT),
    )

    _create_normalizer()
    translator, translator_available = _create_translator()
    _json_runtime = _load_json_gestures()  # runtime gestow json (obecnie nieuzywany)

    display_enabled = bool(SHOW_WINDOW)
    translator_mode = False
    last_gestures: dict[int, str] = {}

    logger.info(
        "start petli glownej nacisnij 'ESC' aby wyjsc 't' aby przelaczyc tryb tlumacza"
    )

    # glowna petla
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            logger.info("zamkniecie aplikacji przez ESC")
            break
        elif key == ord("t"):
            if translator_available and translator is not None:
                translator_mode = not translator_mode
                mode_name = "TLUMACZ (PJM)" if translator_mode else "MYSZKA (SYSTEM)"
                logger.info(f"zmiana trybu: {mode_name}")
            else:
                logger.warning("nie mozna wlaczyc trybu tlumacza (brak modulu/modelu)")

        frame_mp = cap.read()
        if frame_mp is None:
            break

        rgb_frame = cv2.cvtColor(frame_mp, cv2.COLOR_BGR2RGB)
        display_frame = frame_mp.copy()
        preview_mirror = True
        if display_enabled and preview_mirror:
            display_frame = cv2.flip(display_frame, 1)

        tracker.process(rgb_frame)
        results = tracker.get_results() if hasattr(tracker, "get_results") else None
        current_hands_ids: set[int] = set()

        status_color = (0, 255, 255) if translator_mode else (0, 255, 0)
        status_text = "TRYB: TLUMACZ (PJM)" if translator_mode else "TRYB: STEROWANIE"
        cv2.putText(
            display_frame,
            status_text,
            (10, DISPLAY_HEIGHT - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2,
        )

        if results and getattr(results, "multi_hand_landmarks", None):
            best_hand = None
            best_area = -1.0
            handed_list = getattr(results, "multi_handedness", None)

            if translator_mode:
                if results and results.multi_hand_landmarks:
                    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        xs = [lm.x for lm in hand_landmarks.landmark]
                        ys = [lm.y for lm in hand_landmarks.landmark]
                        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                        if area > best_area:
                            best_area = area
                            handed = None
                            try:
                                if handed_list and idx < len(handed_list):
                                    handed = handed_list[idx].classification[0].label
                            except Exception:
                                handed = None
                            best_hand = (idx, hand_landmarks, handed)

                iterable = [best_hand] if best_hand else []
            else:
                iterable = []
                if results and results.multi_hand_landmarks:
                    iterable = [
                        (idx, hl, None)
                        for idx, hl in enumerate(results.multi_hand_landmarks)
                    ]

            for hand_idx, hand_landmarks, handed in iterable:
                hand_id = 0 if translator_mode else hand_idx
                current_hands_ids.add(hand_id)

                if translator_mode:
                    if translator is not None:
                        try:
                            # Konwersja landmarkow do numpy (jesli to obiekty MediaPipe)
                            if hasattr(hand_landmarks, "landmark"):
                                # MediaPipe NormalizedLandmarkList
                                lms_np = np.array(
                                    [
                                        [lm.x, lm.y, lm.z]
                                        for lm in hand_landmarks.landmark
                                    ]
                                )
                            else:
                                # Lista obiektow lub krotek
                                lms_np = np.array(
                                    [[lm.x, lm.y, lm.z] for lm in hand_landmarks]
                                )

                            predicted_letter = translator.process_landmarks(
                                lms_np, handedness=handed
                            )
                        except Exception as e:
                            predicted_letter = None
                            logger.debug(f"blad predykcji translatora: {e}")
                        if predicted_letter:
                            cv2.putText(
                                display_frame,
                                f"Litera: {predicted_letter}",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9,
                                (255, 0, 0),
                                2,
                            )
                            continue  # pomija akcje systemowe
                else:
                    gesture_res = detect_gesture(hand_landmarks)
                    best_name = gesture_res.name
                    if best_name:
                        prev_gesture = last_gestures.get(hand_id)
                        if prev_gesture != best_name:
                            handle_gesture_start_hook(
                                best_name, hand_landmarks, frame_mp.shape
                            )
                        last_gestures[hand_id] = best_name
                        if preview_mirror:
                            visualizer.draw_gesture_label(display_frame, best_name)
                        else:
                            visualizer.draw_gesture_label(display_frame, best_name)

                # rysowanie landmarkow na zmirrorowanym podgladzie gdy preview wlaczony
                if display_enabled:
                    if preview_mirror:
                        visualizer.draw_landmarks_mirrored(
                            display_frame, hand_landmarks
                        )
                    else:
                        visualizer.draw_landmarks(display_frame, hand_landmarks)

        # usuwa wpisy rak ktore zniknely
        for missing_id in list(last_gestures.keys()):
            if missing_id not in current_hands_ids:
                last_gestures.pop(missing_id, None)

        performance.update()
        visualizer.draw_fps(display_frame, performance.fps)
        visualizer.draw_frametime(display_frame, performance.frametime_ms)

        if display_enabled:
            try:
                resized = cv2.resize(display_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                cv2.imshow("SysTouch", resized)
            except Exception as e:
                if DEBUG_MODE:
                    logger.debug(f"blad wyswietlania okna: {e}")
                display_enabled = False
        else:
            time.sleep(0.01)

    cap.stop()
    if display_enabled:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":  # pragma: no cover
    main()
