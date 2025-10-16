from unittest.mock import patch

import app.gesture_engine.core.hooks as hooks
from app.gesture_engine.gestures.volume_gesture import volume_state


@patch("app.core.hooks.logger")
def test_volume_hook_registers_and_sets_adjusting(mock_logger):
    # upewnia sie, ze hook 'volume' nie jest zarejestrowany
    hooks.gesture_start_hooks.pop("volume", None)
    volume_state["phase"] = "idle"
    volume_state["_extend_start"] = None
    hooks.last_gesture_name = None

    # pierwsze wywolanie: rejestruje hook (na koncu funkcji), ale nie wykona go
    hooks.handle_gesture_start_hook(
        "volume", landmarks=object(), frame_shape=(480, 640, 3)
    )
    assert hooks.gesture_start_hooks.get("volume") is not None

    # zmiana na inny gest, aby przy kolejnym 'volume' wywolac start hook
    hooks.handle_gesture_start_hook("scroll", landmarks=None, frame_shape=None)

    # teraz 'volume' powinien wykonac zarejestrowany start hook i ustawic faze 'adjusting'
    hooks.handle_gesture_start_hook(
        "volume", landmarks=object(), frame_shape=(480, 640, 3)
    )

    assert volume_state["phase"] == "adjusting"
    assert volume_state["_extend_start"] is None


@patch("app.core.hooks.logger")
def test_volume_reset_on_exit(mock_logger):
    # ustaw stan jakby w trakcie 'volume'
    volume_state["phase"] = "adjusting"
    volume_state["_extend_start"] = 1.23
    hooks.last_gesture_name = "volume"

    # przejscie na inny gest powinno zresetowac volume_state
    hooks.handle_gesture_start_hook("scroll", landmarks=None, frame_shape=None)

    assert volume_state["phase"] == "idle"
    assert volume_state["_extend_start"] is None
