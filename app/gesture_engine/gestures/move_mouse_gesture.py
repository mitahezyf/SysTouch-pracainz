from app.gesture_engine.config import FLEX_THRESHOLD
from app.gesture_engine.utils.landmarks import FINGER_MCPS
from app.gesture_engine.utils.landmarks import FINGER_PIPS
from app.gesture_engine.utils.landmarks import FINGER_TIPS


def detect_move_mouse_gesture(landmarks):

    # sprawdza czy wyprostowane
    index_straight = (
        landmarks[FINGER_TIPS["index"]].y < landmarks[FINGER_PIPS["index"]].y
    )
    middle_straight = (
        landmarks[FINGER_TIPS["middle"]].y < landmarks[FINGER_PIPS["middle"]].y
    )

    # czy zgiety
    ring_bent = (
        landmarks[FINGER_TIPS["ring"]].y - landmarks[FINGER_MCPS["ring"]].y
    ) > FLEX_THRESHOLD
    pinky_bent = (
        landmarks[FINGER_TIPS["pinky"]].y - landmarks[FINGER_MCPS["pinky"]].y
    ) > FLEX_THRESHOLD

    if index_straight and middle_straight and ring_bent and pinky_bent:
        return "move_mouse", 1.0
    return None
