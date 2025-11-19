from typing import List

import pytest

import app.gesture_engine.gestures.volume_gesture as vg
from app.gesture_engine.utils.landmarks import (
    FINGER_MCPS,
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
    vg.volume_state["pinch_since"] = None
    vg.volume_state["pinch_th"] = None
    vg.volume_state["ref_max"] = None
    vg.volume_state["max_since"] = None
    vg.volume_state["last_peak_ts"] = None


def set_hand_geometry(pts: List[P], hand_size: float = 1.0) -> None:
    # wrist przy (0,0), pinky_mcp w (hand_size, 0)
    pts[WRIST] = P(0.0, 0.0, 0.0)
    pts[FINGER_MCPS["pinky"]] = P(hand_size, 0.0, 0.0)


def set_thumb_ring_distance(pts: List[P], d: float) -> None:
    # ustawia odleglosc miedzy thumb_tip a ring_tip na d w osi X
    tt = FINGER_TIPS["thumb"]
    rt = FINGER_TIPS["ring"]
    pts[tt] = P(0.0, 0.0, 0.0)
    pts[rt] = P(d, 0.0, 0.0)


def test_minimal_constants_and_state_defaults():
    # sprawdza istnienie stalej PINCH_RATIO
    assert isinstance(vg.PINCH_RATIO, float)
    assert 0.0 < vg.PINCH_RATIO < 1.0

    # sprawdza minimalny stan domyslny
    assert vg.volume_state["phase"] == "idle"
    assert vg.volume_state["_extend_start"] is None
    assert vg.volume_state.get("pct") is None
    assert vg.volume_state.get("pinch_th") is None
    assert vg.volume_state.get("ref_max") is None


def test_state_is_mutable_for_runtime_overrides():
    # pozwala na nadpisywanie progu pincha i ref_max w runtime
    vg.volume_state["pinch_th"] = 0.123
    vg.volume_state["ref_max"] = 0.987

    assert vg.volume_state["pinch_th"] == 0.123
    assert vg.volume_state["ref_max"] == 0.987

    # sprzatanie
    vg.volume_state["pinch_th"] = None
    vg.volume_state["ref_max"] = None
