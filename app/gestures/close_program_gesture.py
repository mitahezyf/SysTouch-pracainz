from app.logger import logger
from app.utils.geometry import angle_between
from app.utils.landmarks import FINGER_MCPS
from app.utils.landmarks import FINGER_PIPS
from app.utils.landmarks import FINGER_TIPS
from app.utils.landmarks import THUMB_IP

_log_counter = 0


def detect_close_program_gesture(landmarks):
    global _log_counter
    _log_counter += 1

    def is_finger_bent(finger: str) -> bool:
        tip = landmarks[FINGER_TIPS[finger]]
        pip = landmarks[FINGER_PIPS[finger]]
        mcp = landmarks[FINGER_MCPS[finger]]
        a = angle_between(mcp, pip, tip)

        if _log_counter % 10 == 0:
            logger.debug(f"[gesture] {finger} angle = {a:.2f}")

        return a < 120

    bent_results = {f: is_finger_bent(f) for f in ["index", "middle", "ring", "pinky"]}
    if not all(bent_results.values()):
        if _log_counter % 10 == 0:
            logger.debug(f"[gesture] not all fingers bent: {bent_results}")
        return None

    thumb_tip = landmarks[FINGER_TIPS["thumb"]]
    thumb_ip = landmarks[THUMB_IP]

    dx = abs(thumb_tip.x - thumb_ip.x)
    dy = abs(thumb_tip.y - thumb_ip.y)

    is_horizontal = dx > 0.025
    is_level = dy < 0.04

    if _log_counter % 10 == 0:
        logger.debug(f"[gesture] thumb dx = {dx:.4f}, dy = {dy:.4f}")
        logger.debug(
            f"[gesture] thumb conditions: horizontal={is_horizontal}, level={is_level}"
        )

    if is_horizontal and is_level:
        logger.debug("[gesture] MATCH: close_program")
        return "close_program", 1.0

    return None
