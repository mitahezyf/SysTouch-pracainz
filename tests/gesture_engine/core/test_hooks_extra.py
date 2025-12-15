from unittest.mock import patch

import app.gesture_engine.core.hooks as hooks
from app.gesture_engine.gestures.volume_gesture import volume_state


@patch("app.core.hooks.logger")
def test_volume_hook_registers_and_sets_adjusting(mock_logger):
    # resetuje hook 'volume' aby miec czyste warunki testu
    hooks.gesture_start_hooks.pop("volume", None)
    volume_state["phase"] = "idle"
    volume_state["_extend_start"] = None
    hooks.last_gesture_name = None

    # pierwsze wywolanie rejestruje hook (na koncu funkcji), nie wykonuje startu
    hooks.handle_gesture_start_hook(
        "volume", landmarks=object(), frame_shape=(480, 640, 3)
    )
    assert hooks.gesture_start_hooks.get("volume") is not None

    # wywolanie innego gestu przygotowuje warunki do startu hooka volume
    hooks.handle_gesture_start_hook("scroll", landmarks=None, frame_shape=None)

    # ponowne wywolanie ustawia faze 'adjusting'
    hooks.handle_gesture_start_hook(
        "volume", landmarks=object(), frame_shape=(480, 640, 3)
    )

    assert volume_state["phase"] == "adjusting"
    assert volume_state["_extend_start"] is None


@patch("app.core.hooks.logger")
def test_volume_reset_on_exit(mock_logger):
    # ustawia stan trwajacego 'volume'
    volume_state["phase"] = "adjusting"
    volume_state["_extend_start"] = 1.23
    hooks.last_gesture_name = "volume"

    # przejscie na inny gest resetuje volume_state
    hooks.handle_gesture_start_hook("scroll", landmarks=None, frame_shape=None)

    assert volume_state["phase"] == "idle"
    assert volume_state["_extend_start"] is None
