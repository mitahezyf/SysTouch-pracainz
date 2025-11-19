from typing import List

import app.gesture_engine.actions.volume_action as va
from app.gesture_engine.gestures.volume_gesture import volume_state
from app.gesture_engine.utils.landmarks import FINGER_MCPS


class P:
    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z


def make_landmarks() -> List[P]:
    return [P(0.0, 0.0, 0.0) for _ in range(21)]


def set_mcp_positions(
    pts: List[P], index_xy: tuple[float, float], pinky_xy: tuple[float, float]
) -> None:
    ix, iy = index_xy
    px, py = pinky_xy
    pts[FINGER_MCPS["index"]] = P(ix, iy, 0.0)
    pts[FINGER_MCPS["pinky"]] = P(px, py, 0.0)


def test_handle_volume_knob_angle_mapping():
    # start: phase idle -> brak pct
    volume_state.clear()
    volume_state.update({"phase": "idle"})
    va.handle_volume([], (480, 640, 3))
    assert volume_state.get("pct") is None

    # przejscie do adjusting (trigger z hooka) i baseline
    volume_state["phase"] = "adjusting"
    pts = make_landmarks()
    # baseline: wektor index->pinky poziomo w prawo (kat 0 deg) -> pct=50
    set_mcp_positions(pts, index_xy=(0.0, 0.0), pinky_xy=(1.0, 0.0))
    va.handle_volume(pts, (480, 640, 3))
    assert volume_state.get("pct") == 50

    # obrot +90 deg (w gore): oczekiwane ~100%
    set_mcp_positions(pts, index_xy=(0.0, 0.0), pinky_xy=(0.0, 1.0))
    va.handle_volume(pts, (480, 640, 3))
    assert volume_state.get("pct") == 100

    # obrot -90 deg (w dol): oczekiwane ~0%
    set_mcp_positions(pts, index_xy=(0.0, 0.0), pinky_xy=(0.0, -1.0))
    va.handle_volume(pts, (480, 640, 3))
    assert volume_state.get("pct") == 0
