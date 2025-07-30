import cv2
import mediapipe as mp

from app.config import CONNECTION_COLOR
from app.config import LABEL_FONT_SCALE
from app.config import LABEL_THICKNESS
from app.config import LANDMARK_CIRCLE_RADIUS
from app.config import LANDMARK_COLOR
from app.config import LANDMARK_LINE_THICKNESS

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
