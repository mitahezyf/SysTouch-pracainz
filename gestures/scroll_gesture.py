#todo optymalizacja
from utils.landmarks import FINGER_TIPS, FINGER_PIPS, FINGER_MCPS
from config import FLEX_THRESHOLD


def detect_scroll_gesture(landmarks):

    #sprawdza czy maly wyprostowany
    pinky_straight = landmarks[FINGER_TIPS["pinky"]].y < landmarks[FINGER_PIPS["pinky"]].y
    #sprawdza czy reszta zgieta
    index_bent = (landmarks[FINGER_TIPS["index"]].y - landmarks[FINGER_MCPS["index"]].y) > FLEX_THRESHOLD
    middle_bent = (landmarks[FINGER_TIPS["middle"]].y - landmarks[FINGER_MCPS["middle"]].y) > FLEX_THRESHOLD
    ring_bent = (landmarks[FINGER_TIPS["ring"]].y - landmarks[FINGER_MCPS["ring"]].y) > FLEX_THRESHOLD

    if pinky_straight and index_bent and middle_bent and ring_bent:
        return "scroll", 1.0
    return None
