from app.gesture_engine.config import CLICK_THRESHOLD
from app.gesture_engine.logger import logger
from app.gesture_engine.utils.geometry import distance
from app.gesture_engine.utils.landmarks import FINGER_DIPS
from app.gesture_engine.utils.landmarks import FINGER_TIPS

_log_counter = 0


def detect_click_gesture(landmarks):
    global _log_counter
    _log_counter += 1

    thumb_tip = landmarks[FINGER_TIPS["thumb"]]
    index_tip = landmarks[FINGER_TIPS["index"]]

    pinky_tip = landmarks[FINGER_TIPS["pinky"]]
    pinky_dip = landmarks[FINGER_DIPS["pinky"]]

    ring_tip = landmarks[FINGER_TIPS["ring"]]
    ring_dip = landmarks[FINGER_DIPS["ring"]]

    middle_tip = landmarks[FINGER_TIPS["middle"]]
    middle_dip = landmarks[FINGER_DIPS["middle"]]

    dist = distance(thumb_tip, index_tip)
    confidence = max(0.0, 1.0 - (dist / CLICK_THRESHOLD))

    pinky_straight = pinky_tip.y < pinky_dip.y
    ring_straight = ring_tip.y < ring_dip.y
    middle_straight = middle_tip.y < middle_dip.y

    # loguje co 10 wywolan
    if _log_counter % 10 == 0:
        logger.debug(f"[click] dist={dist:.4f}, confidence={confidence:.2f}")
        logger.debug(
            f"[click] fingers: pinky={pinky_straight}, ring={ring_straight}, middle={middle_straight}"
        )

    if confidence > 0.9 and pinky_straight and ring_straight and middle_straight:
        logger.debug("[click] MATCH: click gesture detected")
        return "click", confidence

    return None
