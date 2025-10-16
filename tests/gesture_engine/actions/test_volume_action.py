from typing import List
from unittest.mock import patch

import app.gesture_engine.actions.volume_action as va
from app.gesture_engine.gestures.volume_gesture import volume_state
from app.gesture_engine.utils.landmarks import FINGER_MCPS, FINGER_TIPS, WRIST


class P:
    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z


def make_landmarks() -> List[P]:
    return [P(0.0, 0.0, 0.0) for _ in range(21)]


def set_hand_size(pts: List[P], size: float) -> None:
    pts[WRIST] = P(0.0, 0.0, 0.0)
    pts[FINGER_MCPS["pinky"]] = P(size, 0.0, 0.0)


def set_pinch(pts: List[P], d: float) -> None:
    pts[FINGER_TIPS["thumb"]] = P(0.0, 0.0, 0.0)
    pts[FINGER_TIPS["index"]] = P(d, 0.0, 0.0)


@patch("app.gesture_engine.actions.volume_action.set_system_volume")
def test_handle_volume_adjusting_maps_to_percent(mock_set):
    # phase != adjusting -> nic nie robi
    volume_state["phase"] = "idle"
    va.handle_volume([], (480, 640, 3))
    mock_set.assert_not_called()

    # phase == adjusting: mapowanie pincha do 0..100
    volume_state["phase"] = "adjusting"
    pts = make_landmarks()
    set_hand_size(pts, size=1.0)  # pinch_th = 0.5

    set_pinch(pts, d=0.5)  # tuz na progu -> 0%
    va.handle_volume(pts, (480, 640, 3))
    mock_set.assert_called_with(0)

    set_pinch(pts, d=1.0)  # na rozmiarze dloni -> 100%
    va.handle_volume(pts, (480, 640, 3))
    mock_set.assert_called_with(100)

    set_pinch(pts, d=0.75)  # posredni -> ~50%
    va.handle_volume(pts, (480, 640, 3))
    # nie wymagamy dokladnosci, ale powinno byc w poblizu 50
    assert 45 <= mock_set.call_args[0][0] <= 55
