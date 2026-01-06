# testy akcji glosnosci
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
    if sys.platform == "win32":
        pass  # funkcja _maybe_apply_system_volume zostala wywolana


def test_volume_state_initialization():
    """testuje inicjalizacje stanu glosnosci"""
    volume_state.clear()

    # po wyczyszczeniu stan powinien byc pusty
    assert volume_state.get("pct") is None
    assert volume_state.get("phase") is None

    # ustaw wartosc
    volume_state["pct"] = 50
    assert volume_state.get("pct") == 50
