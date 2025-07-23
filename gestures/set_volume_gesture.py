#todo dokonczyc gest
from utils.landmarks import FINGER_TIPS
from utils.geometry import distance
from config import VOLUME_THRESHOLD
import time

def detect_volume_gesture(landmarks):
    thumb_tip = landmarks[FINGER_TIPS["thumb"]]
    index_tip = landmarks[FINGER_TIPS["ring"]]


    dist = distance(thumb_tip, index_tip)
    confidence = max(0.0, 1.0 - (dist / VOLUME_THRESHOLD))

    if confidence > 0.9:
        return "volume_adjust", confidence
    return None

