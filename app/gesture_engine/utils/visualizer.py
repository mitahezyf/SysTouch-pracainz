# Bezpieczne importy: cv2 i mediapipe mogą nie być dostępne w środowisku CI.
try:  # pragma: no cover
    import cv2  # type: ignore
except Exception:  # pragma: no cover

    class _CV2Stub:
        FONT_HERSHEY_SIMPLEX = 0

        @staticmethod
        def putText(*_, **__):
            raise ImportError(
                "cv2.putText niedostępne – zainstaluj opencv-python(-headless)."
            )

        @staticmethod
        def rectangle(*_, **__):
            raise ImportError(
                "cv2.rectangle niedostępne – zainstaluj opencv-python(-headless)."
            )

    cv2 = _CV2Stub()  # type: ignore

try:  # pragma: no cover
    import mediapipe as mp  # type: ignore

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
                "mediapipe.draw_landmarks niedostępne – zainstaluj mediapipe."
            )

    class _MPHandsStub:
        HAND_CONNECTIONS = None

    class _MPSolutionsStub:
        drawing_utils = _MPDrawingStub()
        hands = _MPHandsStub()

    class _MPStub:
        solutions = _MPSolutionsStub()

    mp = _MPStub()  # type: ignore
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

from app.gesture_engine.config import (
    CONNECTION_COLOR,
    LABEL_FONT_SCALE,
    LABEL_THICKNESS,
    LANDMARK_CIRCLE_RADIUS,
    LANDMARK_COLOR,
    LANDMARK_LINE_THICKNESS,
)


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
