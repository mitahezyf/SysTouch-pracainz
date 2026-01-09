from app.gesture_engine.logger import logger
from app.gesture_engine.utils.geometry import angle_between
from app.gesture_engine.utils.landmarks import (
    FINGER_MCPS,
    FINGER_PIPS,
    FINGER_TIPS,
    THUMB_IP,
)

_log_counter = 0

# Stan blokady powtorzen - zapobiega ciąglemu zamykaniu okien
# Kluczem jest handedness (Left/Right) - kazda reka ma swoj stan
_execution_state: dict[str, bool] = {
    "Left": False,  # czy lewa reka wykonala juz close_program
    "Right": False,  # czy prawa reka wykonala juz close_program
}


def detect_close_program_gesture(landmarks, handedness: str = "Right"):
    """
    Wykrywa gest zamkniecia programu (piesc z kciukiem na bok).

    Mechanizm zapobiegania powtorzeniom:
    - Gdy gest jest wykryty pierwszy raz -> wykonaj akcje
    - Gdy gest jest trzymany -> blokuj (nie powtarzaj)
    - Gdy gest zniknie (None) -> reset, gotowy do kolejnego wykonania
    """
    global _log_counter, _execution_state
    _log_counter += 1

    # Ustal ktora reka
    hand_key = handedness if handedness in ("Left", "Right") else "Right"

    def is_finger_bent(finger: str) -> bool:
        tip = landmarks[FINGER_TIPS[finger]]
        pip = landmarks[FINGER_PIPS[finger]]
        mcp = landmarks[FINGER_MCPS[finger]]
        a = float(angle_between(mcp, pip, tip))

        if _log_counter % 30 == 0:
            logger.debug(f"[gesture] {finger} angle = {a:.2f}")

        return a < 120  # palec zgiety gdy kat < 120 stopni

    bent_results = {f: is_finger_bent(f) for f in ["index", "middle", "ring", "pinky"]}
    if not all(bent_results.values()):
        # Gest nie wykryty - resetuj blokade dla tej reki
        if _execution_state.get(hand_key, False):
            _execution_state[hand_key] = False
            logger.debug(f"[gesture] close_program: reset blokady dla {hand_key}")
        return None

    thumb_tip = landmarks[FINGER_TIPS["thumb"]]
    thumb_ip = landmarks[THUMB_IP]

    dx = abs(thumb_tip.x - thumb_ip.x)
    dy = abs(thumb_tip.y - thumb_ip.y)

    is_horizontal = dx > 0.02  # obnizony prog
    is_level = dy < 0.05  # podniesiony prog

    if _log_counter % 30 == 0:
        logger.debug(f"[gesture] thumb dx = {dx:.4f}, dy = {dy:.4f}")
        logger.debug(
            f"[gesture] thumb conditions: horizontal={is_horizontal}, level={is_level}"
        )

    if is_horizontal and is_level:
        # Gest wykryty!
        if _execution_state.get(hand_key, False):
            # Juz wykonany - blokuj powtorzenie
            if _log_counter % 30 == 0:
                logger.debug(
                    f"[gesture] close_program wykryty ale juz wykonany dla {hand_key}"
                )
            return None

        # Pierwsza detekcja - wykonaj!
        logger.info(f"[gesture] MATCH: close_program (reka: {hand_key})")
        _execution_state[hand_key] = True
        return "close_program", 1.0

    # Palce zgięte ale kciuk nie w pozycji - nie resetuj stanu
    return None


def reset_close_program_state(handedness: str | None = None):
    """Resetuje stan blokady dla danej reki lub wszystkich."""
    global _execution_state
    if handedness:
        _execution_state[handedness] = False
    else:
        _execution_state["Left"] = False
        _execution_state["Right"] = False
    logger.debug(
        f"[gesture] close_program: reset stanu dla {handedness or 'wszystkich'}"
    )
