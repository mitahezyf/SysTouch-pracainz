from typing import Any, cast

from app.gesture_engine.config import (
    CONNECTION_COLOR,
    LABEL_FONT_SCALE,
    LABEL_THICKNESS,
    LANDMARK_CIRCLE_RADIUS,
    LANDMARK_COLOR,
    LANDMARK_LINE_THICKNESS,
)
from app.gesture_engine.logger import logger
from app.gesture_engine.utils.landmarks import (
    FINGER_TIPS,  # import indeksow koncowek palcow
)

# Bezpieczne importy: cv2 i mediapipe moga nie byc dostepne w srodowisku CI.
try:  # pragma: no cover
    import cv2 as _cv2

    cv2: Any = cast(Any, _cv2)
except Exception:  # pragma: no cover

    class _CV2Stub:
        FONT_HERSHEY_SIMPLEX = 0

        @staticmethod
        def putText(*_, **__):
            raise ImportError(
                "cv2.putText niedostepne - zainstaluj opencv-python(-headless)."
            )

        @staticmethod
        def rectangle(*_, **__):
            raise ImportError(
                "cv2.rectangle niedostepne - zainstaluj opencv-python(-headless)."
            )

    cv2 = cast(Any, _CV2Stub())

try:  # pragma: no cover
    import mediapipe as mp

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
except Exception:  # pragma: no cover

    class _MPDrawingStub:
        class DrawingSpec:
            def __init__(self, color=None, thickness=None, circle_radius=None):
                self.color = color
                self.thickness = thickness
                self.circle_radius = circle_radius

        @staticmethod
        def draw_landmarks(*_, **__):
            raise ImportError(
                "mediapipe.draw_landmarks niedostepne - zainstaluj mediapipe."
            )

    class _MPHandsStub:
        HAND_CONNECTIONS = None

    class _MPSolutionsStub:
        drawing_utils = _MPDrawingStub()
        hands = _MPHandsStub()

    class _MPStub:
        solutions = _MPSolutionsStub()

    mp = _MPStub()
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands


class Visualizer:
    def __init__(self, capture_size, display_size):
        self.capture_size = capture_size
        self.display_size = display_size
        self.scale_x = display_size[0] / capture_size[0]
        self.scale_y = display_size[1] / capture_size[1]

    # wypisuje nazwe gestu
    def draw_label(self, frame, gesture_name, confidence, position=(10, 60)):
        label = f"{gesture_name}: {int(confidence * 100)}%"
        cv2.putText(
            frame,
            label,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            LABEL_FONT_SCALE,
            (255, 255, 255),
            LABEL_THICKNESS,
        )

    # wypisuje ilosc FPS
    def draw_fps(self, frame, fps):
        text = f"FPS: {int(fps)}"
        cv2.putText(
            frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )

    # wypisuje frametime
    def draw_frametime(self, frame, frametime_ms):
        text = f"FrameTime: {int(frametime_ms)} ms"
        cv2.putText(
            frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
        )

    # aktualny gest i pewnosc
    def draw_current_gesture(self, frame, gesture_name, confidence):
        if gesture_name:
            text = f"Gesture: {gesture_name} ({int(confidence * 100)}%)"
        else:
            text = "Gesture: None"
        cv2.putText(
            frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1
        )

    def draw_gesture_label(
        self, frame, gesture_name
    ):  # kompatybilnosc z kodem wywolujacym
        if gesture_name:
            cv2.putText(
                frame,
                f"Gesture: {gesture_name}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )

    # rysuje polaczenia i punkty na dloni
    def draw_landmarks(self, frame, hand_landmarks):
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(
                color=LANDMARK_COLOR,
                thickness=LANDMARK_LINE_THICKNESS,
                circle_radius=LANDMARK_CIRCLE_RADIUS,
            ),
            mp_drawing.DrawingSpec(
                color=CONNECTION_COLOR, circle_radius=LANDMARK_LINE_THICKNESS
            ),
        )

    # rysuje ramke wokol dloni
    def draw_hand_box(self, frame, hand_landmarks, label=None):
        xs = [int(lm.x * frame.shape[1]) for lm in hand_landmarks.landmark]
        ys = [int(lm.y * frame.shape[0]) for lm in hand_landmarks.landmark]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        cv2.rectangle(
            frame, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), (255, 0, 0), 2
        )

        if label:
            cv2.putText(
                frame,
                label,
                (x_min, y_min - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

    # rysuje overlay glosnosci: pasek procentowy i etykiete fazy
    def draw_volume_overlay(self, frame, pct: int | None, phase: str | None) -> None:
        # ustawia parametry paska
        h, w = frame.shape[:2]
        bar_w = max(180, int(w * 0.25))
        bar_h = 18
        margin = 12
        x0 = margin
        y0 = h - margin - bar_h - 20  # zostawia miejsce na tekst
        x1 = x0 + bar_w
        y1 = y0 + bar_h
        # obramowanie paska
        cv2.rectangle(frame, (x0, y0), (x1, y1), (200, 200, 200), 1)
        # wypelnienie wg pct
        if pct is not None:
            pct_clamped = max(0, min(100, int(pct)))
            fill_w = int(bar_w * (pct_clamped / 100.0))
            if fill_w > 0:
                cv2.rectangle(
                    frame,
                    (x0 + 1, y0 + 1),
                    (x0 + fill_w - 1, y1 - 1),
                    (50, 180, 70),
                    -1,
                )
        # etykieta
        phase_txt = f" ({phase})" if phase else ""
        label = f"Volume: {pct if pct is not None else '-'}%{phase_txt}"
        cv2.putText(
            frame,
            label,
            (x0, y0 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    def draw_pjm_letter(
        self, frame, letter: str, confidence: float, time_held_ms: float
    ) -> None:
        """
        Rysuje duza litere PJM w prawym gornym rogu ekranu z tlem.

        Args:
            frame: klatka wideo do rysowania
            letter: litera PJM do wyswietlenia
            confidence: pewnosc rozpoznania (0.0-1.0)
            time_held_ms: czas trzymania litery w milisekundach
        """
        try:
            h, w = frame.shape[:2]

            # formatuj tekst
            main_text = str(letter)
            conf_text = f"{int(confidence * 100)}%"
            time_text = f"{int(time_held_ms)}ms"

            # parametry pozycji (prawy gorny rog z marginesem)
            margin = 20
            font_scale_main = 2.5
            font_scale_info = 0.7
            thickness_main = 4
            thickness_info = 2

            # oblicz rozmiary tekstow
            (tw_main, th_main), _ = cv2.getTextSize(
                main_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale_main, thickness_main
            )
            (tw_conf, th_conf), _ = cv2.getTextSize(
                conf_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale_info, thickness_info
            )
            (tw_time, th_time), _ = cv2.getTextSize(
                time_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale_info, thickness_info
            )

            # pozycja x (prawy rog)
            x_main = w - tw_main - margin
            x_conf = w - tw_conf - margin
            x_time = w - tw_time - margin

            # pozycje y
            y_main = margin + th_main
            y_conf = y_main + 10 + th_conf
            y_time = y_conf + 5 + th_time

            # rysuj ciemne tlo (prostokat)
            padding = 10
            bg_x0 = x_main - padding
            bg_y0 = margin - padding
            bg_x1 = w - margin + padding
            bg_y1 = y_time + padding

            overlay = frame.copy()
            cv2.rectangle(overlay, (bg_x0, bg_y0), (bg_x1, bg_y1), (0, 0, 0), -1)
            # polprzezroczyste tlo
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # rysuj duza litere (zielony kolor jak w panelu)
            cv2.putText(
                frame,
                main_text,
                (x_main, y_main),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale_main,
                (76, 175, 80),  # zielony #4CAF50
                thickness_main,
            )

            # rysuj confidence (bialy)
            cv2.putText(
                frame,
                conf_text,
                (x_conf, y_conf),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale_info,
                (255, 255, 255),
                thickness_info,
            )

            # rysuj czas (szary)
            cv2.putText(
                frame,
                time_text,
                (x_time, y_time),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale_info,
                (136, 136, 136),
                thickness_info,
            )

        except Exception as e:
            logger.debug("draw_pjm_letter error: %s", e)

    def draw_volume_at_tips(
        self, frame, hand_landmarks, pct: int | None, phase: str | None
    ) -> None:
        """Rysuje wskaznik glosnosci przy opuszkach kciuka i palca serdecznego.

        - laczy kciuk i serdeczny linia tylko w fazie 'adjusting'
        - zaznacza kropki na opuszkach
        - wyswietla tekst z procentem przy srodku odcinka
        """
        try:
            h, w = frame.shape[:2]
            t_idx = FINGER_TIPS["thumb"]
            r_idx = FINGER_TIPS["ring"]
            t = hand_landmarks.landmark[t_idx]
            r = hand_landmarks.landmark[r_idx]
            x1, y1 = int(t.x * w), int(t.y * h)
            x2, y2 = int(r.x * w), int(r.y * h)
            mx, my = (x1 + x2) // 2, (y1 + y2) // 2
            # kropki na opuszkach
            cv2.circle(frame, (x1, y1), 5, (0, 255, 0), -1)
            cv2.circle(frame, (x2, y2), 5, (0, 255, 0), -1)
            # linia miedzy opuszkami tylko w fazie adjusting
            if phase == "adjusting":
                cv2.line(frame, (x1, y1), (x2, y2), (50, 200, 255), 2)
            # etykieta nad srodkiem
            pct_txt = "-" if pct is None else str(int(pct))
            phase_txt = f" ({phase})" if phase else ""
            label = f"vol: {pct_txt}%{phase_txt}"
            # tlo pod tekstem dla lepszej czytelnosci
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            tx = mx - tw // 2
            ty = max(16, my - 14)
            # rysuje ciemne tlo (prostokat)
            x0, y0 = max(0, tx - 4), max(0, ty - th - 4)
            x1b, y1b = min(w - 1, tx + tw + 4), min(h - 1, ty + 4)
            overlay = frame.copy()
            cv2.rectangle(overlay, (x0, y0), (x1b, y1b), (0, 0, 0), -1)
            # miesza overlay, uzyskujac polprzezroczyste tlo
            alpha = 0.35
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            # bialy tekst z cieniem/obrysem
            cv2.putText(
                frame,
                label,
                (tx + 1, ty + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )
            cv2.putText(
                frame,
                label,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
        except Exception as e:
            # nie przerywa renderu w razie braku pol
            logger.debug("draw_volume_at_tips error: %s", e)

    def draw_landmarks_mirrored(self, frame, hand_landmarks):
        # rysuje landmarki na zmirrorowanej klatce, odbijajac wspolrzedna x
        try:
            mirrored = type(hand_landmarks)()
            mirrored.landmark.extend(
                [
                    type(lm)(x=1.0 - lm.x, y=lm.y, z=lm.z)
                    for lm in hand_landmarks.landmark
                ]
            )
        except Exception:
            mirrored = hand_landmarks
        self.draw_landmarks(frame, mirrored)
