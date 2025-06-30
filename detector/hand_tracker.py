import cv2
import mediapipe as mp

#klasa wykrywajaca dlonie
class HandTracker:
    def __init__(self):
        #inicjalizacja kamery i mediapipe
        self.cap = cv2.VideoCapture(0)
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.9,
            min_tracking_confidence=0.9
        )
        self.drawer = mp.solutions.drawing_utils
        self.connections = mp.solutions.hands.HAND_CONNECTIONS

    def get_hand_landmarks(self):
        success, frame = self.cap.read()
        if not success:
            return frame, []

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        landmarks_with_handedness = []
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                self.drawer.draw_landmarks(frame, hand_landmarks, self.connections)
                label = handedness.classification[0].label  # "Left" albo "Right"
                landmarks_with_handedness.append((hand_landmarks, label))

        return frame, landmarks_with_handedness
    #zwolnienie kamery
    def release(self):
        self.cap.release()
