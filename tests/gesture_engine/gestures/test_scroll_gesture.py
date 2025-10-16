from typing import List

import app.gesture_engine.gestures.scroll_gesture as sg
from app.gesture_engine.config import FLEX_THRESHOLD
from app.gesture_engine.utils.landmarks import FINGER_MCPS, FINGER_PIPS, FINGER_TIPS


class P:
    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z


def make_landmarks() -> List[P]:
    return [P(0.0, 0.0, 0.0) for _ in range(21)]


def set_pinky_straight(pts: List[P]) -> None:
    # pinky prosty: tip nad pip
    pts[FINGER_TIPS["pinky"]] = P(0.0, 0.0, 0.0)
    pts[FINGER_PIPS["pinky"]] = P(0.0, 0.5, 0.0)


def set_bent_over_threshold(pts: List[P], finger: str) -> None:
    # palec zgiety: tip znaczaco ponizej MCP -> > FLEX_THRESHOLD
    pts[FINGER_TIPS[finger]] = P(0.0, FLEX_THRESHOLD + 1.0, 0.0)
    pts[FINGER_MCPS[finger]] = P(0.0, 0.0, 0.0)


def test_detect_scroll_positive():
    pts = make_landmarks()
    set_pinky_straight(pts)
    for f in ["index", "middle", "ring"]:
        set_bent_over_threshold(pts, f)

    res = sg.detect_scroll_gesture(pts)
    assert res == ("scroll", 1.0)


def test_detect_scroll_negative_when_pinky_not_straight():
    pts = make_landmarks()
    # pinky nieprosty: tip nizej niz pip
    pts[FINGER_TIPS["pinky"]] = P(0.0, 1.0, 0.0)
    pts[FINGER_PIPS["pinky"]] = P(0.0, 0.2, 0.0)
    for f in ["index", "middle", "ring"]:
        set_bent_over_threshold(pts, f)

    assert sg.detect_scroll_gesture(pts) is None
