# todo do zrobienia od nowa, pokonalo mnie to
from time import monotonic

from app.gesture_engine.utils.geometry import angle_between, distance
from app.gesture_engine.utils.landmarks import (
    FINGER_MCPS,
    FINGER_PIPS,
    FINGER_TIPS,
    WRIST,
)

# globalny stan gestu
timezone_offset = 0
volume_state = {
    "phase": "idle",
    "_extend_start": None,
}

# progi i czasy
PINCH_RATIO = 0.5  # prog zacisku (ok. 50% dloni)
MIN_REF_RATIO = 0.50  # prog rozwarcia (ok. 50% dloni)
ANGLE_THRESHOLD = 160.0  # kat [deg] uznawany za wyprostowany palec
STABLE_DURATION = 0.3  # czas [s], przez ktory kat musi byc stabilny


def log_state(phase, hand_size, pinch_th, min_ref, d):
    print(
        f"[volume_gesture] phase={phase:<12} | hand_size={hand_size:.4f} | pinch_th={pinch_th:.4f} | min_ref={min_ref:.4f} | d={d:.4f}"
    )


def is_fingers_extended(landmarks):
    """
    sprawdza, czy index, middle, ring, pinky sa wyprostowane,
    korzystajac z pomiaru kata mcp–pip–tip (angle_between)
    """
    for name in ["index", "middle", "ring", "pinky"]:
        mcp = landmarks[FINGER_MCPS[name]]
        pip = landmarks[FINGER_PIPS[name]]
        tip = landmarks[FINGER_TIPS[name]]
        ang = angle_between(mcp, pip, tip)
        if ang < ANGLE_THRESHOLD:
            return False
    return True


def detect_volume_gesture(landmarks):
    """
    prosty fsm (przykladowy):
      idle -> init          : d < pinch_th
      init -> reference_set : palce wyprostowane stabilnie przez STABLE_DURATION
      reference_set -> adjusting
      adjusting -> adjusting
    reset do idle, gdy d > 1.2 * min_ref
    """
    prev = volume_state["phase"]
    now = monotonic()

    # metryki
    hand_size = distance(landmarks[WRIST], landmarks[FINGER_MCPS["pinky"]])
    pinch_th = hand_size * PINCH_RATIO
    min_ref = hand_size * MIN_REF_RATIO
    d = distance(landmarks[FINGER_TIPS["thumb"]], landmarks[FINGER_TIPS["index"]])

    # reset
    if prev in ("init", "reference_set", "adjusting") and d > min_ref * 1.2:
        volume_state["phase"] = "idle"
        volume_state["_extend_start"] = None
        print(
            f"[volume_gesture] reset -> d ({d:.4f}) > 1.2*min_ref ({min_ref*1.2:.4f})"
        )
        log_state("idle", hand_size, pinch_th, min_ref, d)
        return None

        # idle -> init
        if prev == "idle":
            if d < pinch_th:
                volume_state["phase"] = "init"
                volume_state["_extend_start"] = None
                print(
                    f"[volume_gesture] idle -> init (d < pinch_th: {d:.4f} < {pinch_th:.4f})"
                )
                log_state("init", hand_size, pinch_th, min_ref, d)
                return "volume", 1.0
            return None

    # init -> reference_set
    if prev == "init":
        if is_fingers_extended(landmarks):
            if volume_state["_extend_start"] is None:
                volume_state["_extend_start"] = now
            elif now - volume_state["_extend_start"] >= STABLE_DURATION:
                volume_state["phase"] = "reference_set"
                volume_state["_extend_start"] = None
                print(
                    "[volume_gesture] init -> reference_set (palce stabilnie wyprostowane)"
                )
                log_state("reference_set", hand_size, pinch_th, min_ref, d)
                return "volume", 1.0
        else:
            volume_state["_extend_start"] = None
        return None

    # reference_set -> adjusting
    if prev == "reference_set":
        volume_state["phase"] = "adjusting"
        volume_state["_start_time"] = now
        print("[volume_gesture] reference_set -> adjusting")
        log_state("adjusting", hand_size, pinch_th, min_ref, d)
        return "volume", 1.0

    # adjusting -> adjusting
    if prev == "adjusting":
        log_state("adjusting", hand_size, pinch_th, min_ref, d)
        return "volume", 1.0

    return None
