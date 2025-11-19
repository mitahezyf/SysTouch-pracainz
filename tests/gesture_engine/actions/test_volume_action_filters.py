import app.gesture_engine.actions.volume_action as va
from app.gesture_engine.gestures.volume_gesture import volume_state
from app.gesture_engine.utils.landmarks import FINGER_MCPS


class P:
    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z


def make_landmarks():
    return [P(0.0, 0.0, 0.0) for _ in range(21)]


def set_mcp_positions(pts, index_xy, pinky_xy):
    ix, iy = index_xy
    px, py = pinky_xy
    pts[FINGER_MCPS["index"]] = P(ix, iy, 0.0)
    pts[FINGER_MCPS["pinky"]] = P(px, py, 0.0)


def test_quantization_to_5_percent_knob():
    volume_state.clear()
    volume_state.update({"phase": "adjusting"})
    pts = make_landmarks()

    # baseline -> 50%
    set_mcp_positions(pts, (0.0, 0.0), (1.0, 0.0))  # 0 deg
    va.handle_volume(pts, (480, 640, 3))

    # kat okolo +18 deg -> okolo 60% -> po kwantyzacji 60%
    set_mcp_positions(pts, (0.0, 0.0), (0.95, 0.3))
    va.handle_volume(pts, (480, 640, 3))
    assert volume_state.get("pct") in (55, 60, 65)

    # kat okolo -18 deg -> okolo 40% -> po kwantyzacji 40%
    set_mcp_positions(pts, (0.0, 0.0), (0.95, -0.3))
    va.handle_volume(pts, (480, 640, 3))
    assert volume_state.get("pct") in (35, 40, 45)


def test_accept_after_stable_delay_knob():
    volume_state.clear()
    volume_state.update({"phase": "adjusting"})

    pts = make_landmarks()
    set_mcp_positions(pts, (0.0, 0.0), (1.0, 0.0))  # baseline 0 deg -> 50%
    va.handle_volume(pts, (480, 640, 3))

    # kilka wywolan dla tego samego kata -> pct stabilne, faza nie zmienia sie
    for _ in range(3):
        va.handle_volume(pts, (480, 640, 3))
        assert volume_state.get("pct") == 50
    assert volume_state.get("phase") == "adjusting"
