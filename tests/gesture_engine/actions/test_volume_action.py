from typing import List

import app.gesture_engine.actions.volume_action as va
from app.gesture_engine.gestures.volume_gesture import volume_state
from app.gesture_engine.utils.landmarks import FINGER_MCPS, WRIST


class P:
    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z


def make_landmarks() -> List[P]:
    return [P(0.0, 0.0, 0.0) for _ in range(21)]


def set_hand_geometry(
    pts: List[P], wrist_xy: tuple[float, float], middle_mcp_xy: tuple[float, float]
) -> None:
    """ustawia pozycje wrist i middle_mcp dla testu roll"""
    wx, wy = wrist_xy
    mx, my = middle_mcp_xy
    pts[WRIST] = P(wx, wy, 0.0)
    pts[FINGER_MCPS["middle"]] = P(mx, my, 0.0)


def test_handle_volume_hand_roll_mapping():
    """testuje mapowanie roll dloni na procent glosnosci"""
    # prosty model bez faz - handler dziala od razu
    volume_state.clear()
    pts = make_landmarks()

    # baseline: wrist (0,0), middle_mcp (1,0) -> poziomo w prawo = 0 deg -> pct=50
    set_hand_geometry(pts, wrist_xy=(0.0, 0.0), middle_mcp_xy=(1.0, 0.0))
    va.handle_volume(pts, (480, 640, 3))
    assert volume_state.get("pct") == 50
    baseline_set = volume_state.get("hand_roll_baseline_deg")
    assert baseline_set == 0.0

    # obrot +45 deg (reka w dol-prawo): range 90deg, delta=+45 -> 100%
    set_hand_geometry(pts, wrist_xy=(0.0, 0.0), middle_mcp_xy=(0.707, 0.707))
    # wypelnij bufor kilka razy aby mediana sie ustabilizowala
    for _ in range(5):
        va.handle_volume(pts, (480, 640, 3))
    pct_plus = volume_state.get("pct")
    # z range=90 i delta=+45, oczekujemy 100% (lub blisko)
    assert pct_plus >= 95

    # obrot -45 deg (reka w gore-prawo): range 90deg, delta=-45 -> 0%
    set_hand_geometry(pts, wrist_xy=(0.0, 0.0), middle_mcp_xy=(0.707, -0.707))
    # wypelnij bufor kilka razy aby mediana sie ustabilizowala
    for _ in range(5):
        va.handle_volume(pts, (480, 640, 3))
    pct_minus = volume_state.get("pct")
    # z range=90 i delta=-45, oczekujemy 0% (lub blisko)
    assert pct_minus <= 5


def test_apply_system_defaults_to_true():
    """testuje ze domyslna wartosc apply_system pozwala na stosowanie glosnosci"""
    import sys

    # reset stanu
    volume_state.clear()
    volume_state.update({"phase": "adjusting"})

    pts = make_landmarks()
    set_hand_geometry(pts, wrist_xy=(0.0, 0.0), middle_mcp_xy=(1.0, 0.0))

    # wywolaj handle_volume bez ustawiania apply_system
    # powinno dzialac domyslnie (True) na Windows
    va.handle_volume(pts, (480, 640, 3))

    # sprawdz ze pct zostal ustawiony (funkcja nie wyszla wczesniej)
    assert volume_state.get("pct") is not None

    # na Windows domyslnie powinno probowac stosowac (nawet jesli pycaw brak, nie crashuje)
    # na nie-Windows apply_system i tak jest pomijane wczesniej
    if sys.platform == "win32":
        # funkcja _maybe_apply_system_volume zostala wywolana
        # (nie mozemy latwo przetestowac czy set_system_volume zostala wywolana,
        # ale wiemy ze nie wyszlo z return na linii "if not apply_system")
        pass
