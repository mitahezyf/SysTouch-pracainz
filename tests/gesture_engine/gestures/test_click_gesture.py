from typing import List

import app.gesture_engine.gestures.click_gesture as cg
from app.gesture_engine.config import CLICK_THRESHOLD
from app.gesture_engine.utils.landmarks import FINGER_DIPS, FINGER_TIPS


class P:
    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z


def make_landmarks() -> List[P]:
    return [P(0.0, 0.0, 0.0) for _ in range(21)]


def set_tip_and_dip(pts: List[P], finger: str, tip_y: float, dip_y: float) -> None:
    pts[FINGER_TIPS[finger]] = P(0.0, tip_y, 0.0)
    pts[FINGER_DIPS[finger]] = P(0.0, dip_y, 0.0)


def test_detect_click_positive():
    pts = make_landmarks()
    # kciuk i wskazujacy bardzo blisko siebie (wysoka confidence)
    pts[FINGER_TIPS["thumb"]] = P(0.0, 0.0, 0.0)
    pts[FINGER_TIPS["index"]] = P(0.05 * CLICK_THRESHOLD, 0.0, 0.0)
    # pozostale proste: tip.y < dip.y
    set_tip_and_dip(pts, "pinky", tip_y=0.1, dip_y=0.2)
    set_tip_and_dip(pts, "ring", tip_y=0.1, dip_y=0.2)
    set_tip_and_dip(pts, "middle", tip_y=0.1, dip_y=0.2)

    res = cg.detect_click_gesture(pts)
    assert res is not None and res[0] == "click" and res[1] > 0.9


def test_detect_click_negative():
    pts = make_landmarks()
    # kciuk i wskazujacy daleko -> niska confidence
    pts[FINGER_TIPS["thumb"]] = P(0.0, 0.0, 0.0)
    pts[FINGER_TIPS["index"]] = P(2 * CLICK_THRESHOLD, 0.0, 0.0)
    # palce proste
    set_tip_and_dip(pts, "pinky", tip_y=0.1, dip_y=0.2)
    set_tip_and_dip(pts, "ring", tip_y=0.1, dip_y=0.2)
    set_tip_and_dip(pts, "middle", tip_y=0.1, dip_y=0.2)

    assert cg.detect_click_gesture(pts) is None

    # lub gdy jeden z palcow nie jest prosty
    pts2 = make_landmarks()
    pts2[FINGER_TIPS["thumb"]] = P(0.0, 0.0, 0.0)
    pts2[FINGER_TIPS["index"]] = P(0.05 * CLICK_THRESHOLD, 0.0, 0.0)
    set_tip_and_dip(pts2, "pinky", tip_y=0.3, dip_y=0.2)  # nieprosty
    set_tip_and_dip(pts2, "ring", tip_y=0.1, dip_y=0.2)
    set_tip_and_dip(pts2, "middle", tip_y=0.1, dip_y=0.2)
    assert cg.detect_click_gesture(pts2) is None
