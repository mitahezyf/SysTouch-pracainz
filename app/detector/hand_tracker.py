import mediapipe as mp


class HandTracker:
    def __init__(
        self, max_num_hands=2, detection_confidence=0.7, tracking_confidence=0.7
    ):
        self.max_num_hands = max_num_hands

        # inicjalizacja mediapipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            model_complexity=1,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles

    def process(self, frame_rgb):
        return self.hands.process(frame_rgb)
