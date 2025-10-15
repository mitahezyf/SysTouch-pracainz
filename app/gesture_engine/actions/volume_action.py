# todo: do przerobienia od nowa, obecna wersja jest zbyt niestabilna
from app.gesture_engine.gestures.volume_gesture import PINCH_RATIO, volume_state
from app.gesture_engine.logger import logger
from app.gesture_engine.utils.geometry import distance
from app.gesture_engine.utils.landmarks import FINGER_MCPS, FINGER_TIPS, WRIST
from app.gesture_engine.utils.pycaw_controller import set_system_volume


def handle_volume(landmarks, frame_shape):
    phase = volume_state.get("phase")
    logger.debug(f"[volume_action] phase = {phase}")
    if phase != "adjusting":
        return

    hand_size = distance(landmarks[WRIST], landmarks[FINGER_MCPS["pinky"]])
    pinch_th = hand_size * PINCH_RATIO
    d = distance(
        landmarks[FINGER_TIPS["thumb"]],
        landmarks[FINGER_TIPS["index"]],
    )

    raw_pct = (d - pinch_th) / (hand_size - pinch_th) * 100
    pct = int(max(0, min(100, raw_pct)))
    logger.debug(
        f"[volume_action] hand_size={hand_size:.4f} | pinch_th={pinch_th:.4f} | d={d:.4f} | pct={pct}"
    )
    set_system_volume(pct)
