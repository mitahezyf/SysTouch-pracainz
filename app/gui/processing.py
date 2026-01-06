from __future__ import annotations

from typing import List, Optional, Protocol, Tuple

import numpy as np

from app.gesture_engine.core.handlers import gesture_handlers
from app.gesture_engine.detector.gesture_detector import detect_gesture
from app.gesture_engine.logger import logger
from app.gesture_engine.utils.visualizer import Visualizer
from app.gui.models import GestureResult, SingleHandResult


class TranslatorLike(Protocol):
    _last_logged_letter: Optional[str]

    def process_frame(self, normalized_landmarks: list[float]) -> Optional[str]: ...
    def process_landmarks(
        self, landmarks: np.ndarray, handedness: str | None = None
    ) -> Optional[str]: ...
    def get_state(self) -> dict: ...


class NormalizerLike(Protocol):
    def normalize(self, landmarks) -> list[float]: ...


def log_landmark_stats(points, normalized, letter, confidence):
    """loguje statystyki landmarkow dla debugowania wykrywania liter"""
    # raw landmarks
    points_arr = np.array(points)
    logger.debug(
        "[landmarks] RAW: shape=%s, mean=(%.3f, %.3f, %.3f), std=(%.3f, %.3f, %.3f)",
        points_arr.shape,
        np.mean(points_arr[:, 0]),
        np.mean(points_arr[:, 1]),
        np.mean(points_arr[:, 2]),
        np.std(points_arr[:, 0]),
        np.std(points_arr[:, 1]),
        np.std(points_arr[:, 2]),
    )

    # normalized
    norm_arr = np.array(normalized)
    logger.debug(
        "[landmarks] NORMALIZED: shape=%s, mean=%.3f, std=%.3f, min=%.3f, max=%.3f",
        norm_arr.shape,
        np.mean(norm_arr),
        np.std(norm_arr),
        np.min(norm_arr),
        np.max(norm_arr),
    )

    # sprawdz czy cechy 3,4,5 faktycznie sa zerem
    if len(norm_arr) >= 6:
        logger.debug(
            "[landmarks] Cechy [3,4,5] (powinny byc ~0): [%.6f, %.6f, %.6f]",
            norm_arr[3],
            norm_arr[4],
            norm_arr[5],
        )

    # pokaz pierwsze 10 cech dla analizy
    if len(norm_arr) >= 10:
        logger.debug(
            "[landmarks] Pierwsze 10 cech: %s",
            ", ".join([f"{norm_arr[i]:.3f}" for i in range(10)]),
        )

    logger.info(
        "[landmarks] DETECTED: letter=%s, confidence=%.2f%%",
        letter,
        confidence * 100,
    )


def detect_and_draw(
    frame_bgr,
    tracker,
    json_runtime,
    visualizer: Visualizer,
    preview_enabled: bool,
    preview_mirror: bool = True,
    mode: str = "gestures",
    translator: Optional[TranslatorLike] = None,
    normalizer: Optional[NormalizerLike] = None,
) -> Tuple[object, GestureResult, List[SingleHandResult]]:
    # wykrywa gesty dla wielu rak i rysuje wizualizacje na klatce
    # zwraca tuple: (display_frame, wynik globalny, lista wynikow per reka)
    import cv2  # lokalny import

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # display_frame moze byc zmirrorowane dla podgladu, ale tracker dostaje nie-flip
    if preview_enabled and preview_mirror:
        display_frame = cv2.flip(frame_bgr, 1)
    elif preview_enabled:
        display_frame = frame_bgr.copy()
    else:
        display_frame = frame_bgr

    results = tracker.process(frame_rgb)

    # zbiera wyniki per reka
    per_hand: List[SingleHandResult] = []

    # wybiera najlepszy gest globalny po confidence
    best_name: Optional[str] = None
    best_conf: float = 0.0

    handed_list = getattr(results, "multi_handedness", None) if results else None

    if results and getattr(results, "multi_hand_landmarks", None):
        hand_entries = []
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handed = None
            try:
                if handed_list and idx < len(handed_list):
                    handed = handed_list[idx].classification[0].label  # Left lub Right
            except Exception:
                handed = None
            xs = [lm.x for lm in hand_landmarks.landmark]
            ys = [lm.y for lm in hand_landmarks.landmark]
            area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            hand_entries.append((idx, hand_landmarks, handed, area))

        if mode == "translator" and hand_entries:
            hand_entries = [max(hand_entries, key=lambda h: h[3])]

        for idx, hand_landmarks, handed, _area in hand_entries:
            gesture_name: Optional[str] = None
            confidence: float = 0.0
            points = [
                (lm.x, lm.y, getattr(lm, "z", 0.0)) for lm in hand_landmarks.landmark
            ]
            if mode == "translator":
                # tryb translator - wykrywa tylko litery PJM, nie gesty sterowania
                if translator:
                    try:
                        # konwersja punktow do numpy array (21, 3)
                        landmarks_array = np.array(points, dtype=np.float32)

                        # translator.process_landmarks robi automatycznie ekstrakcje cech 88D
                        letter = translator.process_landmarks(
                            landmarks_array, handedness=handed
                        )

                        if letter:
                            gesture_name = letter
                            state = translator.get_state()
                            confidence = state["confidence"]

                            # loguj landmarks tylko przy pierwszym wykryciu litery lub zmianie
                            # aby nie zasmiecac logow
                            if not hasattr(translator, "_last_logged_letter"):
                                translator._last_logged_letter = None

                            if translator._last_logged_letter != letter:
                                # dla logow stworz dummy norm_coords (translator juz ma 88D wewnetrznie)
                                log_landmark_stats(
                                    points,
                                    landmarks_array.flatten()[:63],
                                    letter,
                                    confidence,
                                )
                                translator._last_logged_letter = letter
                        else:
                            gesture_name = None
                            confidence = 0.0
                    except Exception as exc:
                        logger.debug(f"[translator] wyjatek: {exc}")
                        gesture_name = None
                        confidence = 0.0
                else:
                    # brak zasobow translatora w trybie translator nie fallbackuje do gestow
                    logger.debug("[translator] brak modelu - pomijam wykrywanie")
                    gesture_name = None
                    confidence = 0.0
            else:
                # tryb gestures - wykrywa gesty sterowania (json + detect_gesture)
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

            result_idx = 0 if mode == "translator" else idx
            per_hand.append(
                SingleHandResult(
                    index=result_idx,
                    name=gesture_name,
                    confidence=confidence,
                    landmarks=hand_landmarks.landmark,
                    handedness=handed,
                )
            )

            if gesture_name is not None and confidence >= best_conf:
                best_name = gesture_name
                best_conf = confidence

            # rysuje landmarki i ramke dla kazdej wykrytej reki
            if preview_enabled:
                label = gesture_name or ""
                label_text = f"{label}: ({confidence * 100:.1f})" if label else ""
                draw_landmarks = hand_landmarks
                if preview_mirror:
                    try:
                        # utworz kopie z x->1-x do rysowania na zmirrorowanej klatce
                        draw_landmarks = type(hand_landmarks)()
                        draw_landmarks.landmark.extend(
                            [
                                type(lm)(x=1.0 - lm.x, y=lm.y, z=lm.z)
                                for lm in hand_landmarks.landmark
                            ]
                        )
                    except Exception:
                        draw_landmarks = hand_landmarks
                visualizer.draw_landmarks(display_frame, draw_landmarks)
                visualizer.draw_hand_box(
                    display_frame, draw_landmarks, label=label_text
                )

    # rysuje duza litere PJM w prawym gornym rogu dla trybu translator
    if mode == "translator" and preview_enabled and best_name and translator:
        try:
            state = translator.get_state()
            visualizer.draw_pjm_letter(
                display_frame, best_name, state["confidence"], state["time_held_ms"]
            )
        except Exception as exc:
            logger.debug(f"[translator] blad rysowania litery: {exc}")

    return display_frame, GestureResult(best_name, best_conf), per_hand
