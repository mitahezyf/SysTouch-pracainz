from utils.landmarks import FINGER_TIPS
from utils.geometry import distance
from config import CLICK_THRESHOLD

def detect_click_gesture(landmarks):
    thumb_tip = landmarks[FINGER_TIPS["thumb"]]
    index_tip = landmarks[FINGER_TIPS["index"]]

    dist = distance(thumb_tip, index_tip)
    confidence = max(0.0, 1.0 - (dist / CLICK_THRESHOLD))

    if confidence > 0.9:
        return "click", confidence
    return None
