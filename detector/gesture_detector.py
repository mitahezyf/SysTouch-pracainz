from utils.geometry import distance
from config import CLICK_TRESHOLD

class Gesture:
    def __init__(self, name, confidence, landmarks):
        self.name = name
        self.confidence = confidence
        self.landmarks = landmarks

def is_mouse_control_pose(landmarks):
    def is_extended(tip_id, pip_id):
        return landmarks.landmark[tip_id].y < landmarks.landmark[pip_id].y

    def is_folded(tip_id, pip_id):
        return landmarks.landmark[tip_id].y > landmarks.landmark[pip_id].y + 0.03

    if (
        is_extended(8, 6) and           # wskazujący
        is_extended(12, 10) and         # środkowy
        is_folded(16, 14) and           # serdeczny
        is_folded(20, 18)               # mały
    ):
        return True
    return False

def detect_gesture(landmarks):
    thumb_tip = landmarks.landmark[4]
    index_tip = landmarks.landmark[8]
    dist = distance(thumb_tip, index_tip)
    click_confidence = max(0.0, 1.0 - (dist / CLICK_TRESHOLD))

    if is_mouse_control_pose(landmarks):
        return Gesture("move_mouse", 1.0, landmarks)

    if click_confidence > 0.9:
        return Gesture("click", click_confidence, landmarks)
    return None
