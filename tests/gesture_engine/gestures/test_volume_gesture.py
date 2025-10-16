from typing import List

import pytest

import app.gesture_engine.gestures.volume_gesture as vg
from app.gesture_engine.utils.landmarks import (
    FINGER_MCPS,
    FINGER_PIPS,
    FINGER_TIPS,
    WRIST,
)


class P:
    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z


def make_landmarks() -> List[P]:
    # tworzy 21 punktow; domyslnie w (0,0,0)
    return [P(0.0, 0.0, 0.0) for _ in range(21)]


@pytest.fixture(autouse=True)
def reset_state():
    vg.volume_state["phase"] = "idle"
    vg.volume_state["_extend_start"] = None


def set_hand_geometry(pts: List[P], hand_size: float = 1.0) -> None:
    # wrist przy (0,0), pinky_mcp w (hand_size, 0)
    pts[WRIST] = P(0.0, 0.0, 0.0)
    pts[FINGER_MCPS["pinky"]] = P(hand_size, 0.0, 0.0)


def set_pinch_distance(pts: List[P], d: float) -> None:
    # ustawia odleglosc miedzy thumb_tip a index_tip na d w osi X
    tt = FINGER_TIPS["thumb"]
    it = FINGER_TIPS["index"]
    pts[tt] = P(0.0, 0.0, 0.0)
    pts[it] = P(d, 0.0, 0.0)


def extend_fingers(pts: List[P]) -> None:
    # ustawia MCP->PIP->TIP na linii dla index/middle/ring/pinky, aby angle_between ~ 180deg
    for name, x in zip(["index", "middle", "ring", "pinky"], [0.0, 0.2, 0.4, 0.6]):
        mcp = FINGER_MCPS[name]
        pip = FINGER_PIPS[name]
        tip = FINGER_TIPS[name]
        pts[mcp] = P(x, 0.0, 0.0)
        pts[pip] = P(x + 0.1, 0.0, 0.0)
        pts[tip] = P(x + 0.2, 0.0, 0.0)


def test_idle_to_init_pinch_below_threshold():
    pts = make_landmarks()
    set_hand_geometry(pts, hand_size=1.0)
    # pinch_th = 0.5, ustaw d=0.2
    set_pinch_distance(pts, 0.2)

    res = vg.detect_volume_gesture(pts)
    assert res == ("volume", 1.0)
    assert vg.volume_state["phase"] == "init"


def test_init_to_reference_set_after_stable_extension(monkeypatch):
    pts = make_landmarks()
    set_hand_geometry(pts, hand_size=1.0)
    set_pinch_distance(pts, 0.2)
    extend_fingers(pts)

    # przejscie do init
    _ = vg.detect_volume_gesture(pts)
    assert vg.volume_state["phase"] == "init"

    # stabilna ekst. przez STABLE_DURATION: dwa wywolania z roznymi czasami
    t0 = 100.0
    monkeypatch.setattr(
        "app.gesture_engine.gestures.volume_gesture.monotonic", lambda: t0
    )
    _ = vg.detect_volume_gesture(pts)
    monkeypatch.setattr(
        "app.gesture_engine.gestures.volume_gesture.monotonic",
        lambda: t0 + vg.STABLE_DURATION + 0.01,
    )
    res = vg.detect_volume_gesture(pts)

    assert res == ("volume", 1.0)
    assert vg.volume_state["phase"] == "reference_set"


def test_reference_set_to_adjusting_then_keep():
    pts = make_landmarks()
    set_hand_geometry(pts, hand_size=1.0)
    set_pinch_distance(pts, 0.2)
    extend_fingers(pts)

    # init
    _ = vg.detect_volume_gesture(pts)
    # sztucznie ustawiamy stan na reference_set, aby przeskoczyc stabilizacje czasu
    vg.volume_state["phase"] = "reference_set"

    # przejscie do adjusting
    res1 = vg.detect_volume_gesture(pts)
    assert res1 == ("volume", 1.0)
    assert vg.volume_state["phase"] == "adjusting"

    # pozostaje w adjusting
    res2 = vg.detect_volume_gesture(pts)
    assert res2 == ("volume", 1.0)


def test_reset_to_idle_when_pinched_far_apart():
    pts = make_landmarks()
    set_hand_geometry(pts, hand_size=1.0)  # min_ref = 0.5, prog resetu 0.6
    vg.volume_state["phase"] = "adjusting"
    set_pinch_distance(pts, 0.7)  # > 0.6

    res = vg.detect_volume_gesture(pts)
    assert res is None
    assert vg.volume_state["phase"] == "idle"
