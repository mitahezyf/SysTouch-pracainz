import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

class Visualizer:

    #etykieta z nazwa, pewnoscia
    def draw_label(self, frame, gesture_name, confidence, position=(10, 60)):
        label = f"{gesture_name}: {int(confidence * 100)}%"
        cv2.putText(frame, label, position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    #rysowanie fps w rogu
    def draw_fps(self, frame, fps):
        text = f"FPS: {int(fps)}"
        cv2.putText(frame, text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    #rysowanie frametime pod fps
    def draw_frametime(self, frame, frametime_ms):
        text = f"FrameTime: {int(frametime_ms)} ms"
        cv2.putText(frame, text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    #rysowanie landmarkow na dloni
    def draw_landmarks(self, frame, hand_landmarks):
        mp_drawing.draw_landmarks(
            frame, hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )

    #kwadrat do oznaczania obszaru gestu
    def draw_hand_box(self, frame, hand_landmarks, label=None):
        xs = [int(lm.x * frame.shape[1]) for lm in hand_landmarks.landmark]
        ys = [int(lm.y * frame.shape[0]) for lm in hand_landmarks.landmark]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        cv2.rectangle(frame, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), (255, 0, 0), 2)

        if label:
            cv2.putText(frame, label, (x_min, y_min - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


