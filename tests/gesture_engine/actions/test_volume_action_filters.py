# testy filtrow akcji glosnosci
import app.gesture_engine.actions.volume_action as va
from app.gesture_engine.gestures.volume_gesture import volume_state
from app.gesture_engine.utils.landmarks import FINGER_MCPS, WRIST


class P:
    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z


def make_landmarks():
    return [P(0.0, 0.0, 0.0) for _ in range(21)]


def set_hand_roll(pts, wrist_xy, middle_mcp_xy):
    """ustawia pozycje wrist i middle_mcp dla testu roll"""
    wx, wy = wrist_xy
    mx, my = middle_mcp_xy
    pts[WRIST] = P(wx, wy, 0.0)
    pts[FINGER_MCPS["middle"]] = P(mx, my, 0.0)


def test_accept_after_stable_delay_knob():
    volume_state.clear()
    volume_state.update({"phase": "adjusting"})

    pts = make_landmarks()
    set_hand_roll(pts, (0.0, 0.0), (1.0, 0.0))  # baseline 0 deg -> 50%
    va.handle_volume(pts, (480, 640, 3))

    # kilka wywolan dla tego samego kata -> pct stabilne, faza nie zmienia sie
    for _ in range(3):
        va.handle_volume(pts, (480, 640, 3))
        assert volume_state.get("pct") == 50
    assert volume_state.get("phase") == "adjusting"
