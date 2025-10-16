from typing import List

import app.gesture_engine.gestures.move_mouse_gesture as mg
from app.gesture_engine.config import FLEX_THRESHOLD
from app.gesture_engine.utils.landmarks import FINGER_MCPS, FINGER_PIPS, FINGER_TIPS


class P:
    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z


def make_landmarks() -> List[P]:
    return [P(0.0, 0.0, 0.0) for _ in range(21)]


def set_straight(pts: List[P], finger: str) -> None:
    # tip nad pip -> prosty
    pts[FINGER_TIPS[finger]] = P(0.0, 0.0, 0.0)
    pts[FINGER_PIPS[finger]] = P(0.0, 0.5, 0.0)


def set_bent_over_threshold(pts: List[P], finger: str) -> None:
    # tip zdecydowanie nizej niz mcp -> zgiety > FLEX_THRESHOLD
    pts[FINGER_TIPS[finger]] = P(0.0, FLEX_THRESHOLD + 1.0, 0.0)
    pts[FINGER_MCPS[finger]] = P(0.0, 0.0, 0.0)


def test_detect_move_mouse_positive():
    pts = make_landmarks()
    set_straight(pts, "index")
    set_straight(pts, "middle")
    set_bent_over_threshold(pts, "ring")
    set_bent_over_threshold(pts, "pinky")

    res = mg.detect_move_mouse_gesture(pts)
    assert res == ("move_mouse", 1.0)


def test_detect_move_mouse_negative_when_not_all_conditions():
    pts = make_landmarks()
    set_straight(pts, "index")
    set_straight(pts, "middle")
    set_bent_over_threshold(pts, "ring")
    # brak zgiecia pinky: roznica y <= FLEX_THRESHOLD
    pts[FINGER_TIPS["pinky"]] = P(0.0, 0.0, 0.0)
    pts[FINGER_MCPS["pinky"]] = P(0.0, 0.0, 0.0)

    assert mg.detect_move_mouse_gesture(pts) is None
