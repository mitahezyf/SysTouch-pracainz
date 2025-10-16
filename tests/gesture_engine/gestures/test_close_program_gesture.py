from typing import List

import app.gesture_engine.gestures.close_program_gesture as cg
from app.gesture_engine.utils.landmarks import (
    FINGER_MCPS,
    FINGER_PIPS,
    FINGER_TIPS,
    THUMB_IP,
)


class P:
    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z


def make_landmarks() -> List[P]:
    return [P(0.0, 0.0, 0.0) for _ in range(21)]


def set_bent(pts: List[P], finger: str) -> None:
    # ustawia kat ~90 deg w stawie PIP: mcp(0,0), pip(0,1), tip(0.5,1.0)
    pts[FINGER_MCPS[finger]] = P(0.0, 0.0, 0.0)
    pts[FINGER_PIPS[finger]] = P(0.0, 1.0, 0.0)
    pts[FINGER_TIPS[finger]] = P(0.5, 1.0, 0.0)


def set_straight(pts: List[P], finger: str) -> None:
    # ustawienie prawie proste (kat blisko 180 deg): mcp(0,0), pip(0,1), tip(0,2)
    pts[FINGER_MCPS[finger]] = P(0.0, 0.0, 0.0)
    pts[FINGER_PIPS[finger]] = P(0.0, 1.0, 0.0)
    pts[FINGER_TIPS[finger]] = P(0.0, 2.0, 0.0)


def set_thumb_level_horizontal(pts: List[P], dx: float = 0.2, dy: float = 0.0) -> None:
    # kciuk: tip oddalony w poziomie od IP, z minimalnym odchyleniem w pionie
    pts[FINGER_TIPS["thumb"]] = P(dx, dy, 0.0)
    pts[THUMB_IP] = P(0.0, 0.0, 0.0)


def test_detect_close_program_positive():
    pts = make_landmarks()
    for f in ["index", "middle", "ring", "pinky"]:
        set_bent(pts, f)
    set_thumb_level_horizontal(pts, dx=0.2, dy=0.0)

    res = cg.detect_close_program_gesture(pts)
    assert res is not None and res[0] == "close_program"


def test_detect_close_program_negative_when_not_all_bent():
    pts = make_landmarks()
    # trzy zgiete, jeden prosty
    for f in ["middle", "ring", "pinky"]:
        set_bent(pts, f)
    set_straight(pts, "index")
    set_thumb_level_horizontal(pts)

    assert cg.detect_close_program_gesture(pts) is None
