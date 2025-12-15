import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from unittest.mock import MagicMock, patch

import app.gesture_engine.core.hooks as hooks
from app.gesture_engine.gestures.volume_gesture import volume_state


# weryfikuje rejestracje hooka i jego wywolanie przy zmianie gestu
@patch("app.core.hooks.logger")
def test_register_and_trigger_hook(mock_logger):
    mock_func = MagicMock()
    hooks.register_gesture_start_hook("test_gest", mock_func)

    hooks.handle_gesture_start_hook("test_gest", "land", "shape")

    mock_func.assert_called_once_with("land", "shape")
    mock_logger.debug.assert_any_call("[hook] zmiana gestu: None -> test_gest")
    mock_logger.debug.assert_any_call("[hook] wywołanie hooka dla: test_gest")


# weryfikuje reset stanu click przy przejsciu na inny gest
@patch("app.core.hooks.logger")
@patch("app.core.hooks.update_click_state")
def test_handle_click_reset_on_change(mock_update, mock_logger):
    hooks.handle_click.active = True
    hooks.handle_gesture_start_hook("scroll", None, None)

    assert hooks.handle_click.active is False
    mock_update.assert_called_once_with(False)
    mock_logger.debug.assert_any_call("[hook] click released (gest zmienił się)")


# weryfikuje reset stanu click przy zakonczeniu gestu (None)
@patch("app.core.hooks.logger")
@patch("app.core.hooks.update_click_state")
def test_handle_click_reset_on_none(mock_update, mock_logger):
    hooks.last_gesture_name = "click"
    hooks.handle_click.active = True

    hooks.handle_gesture_start_hook(None, None, None)

    assert hooks.handle_click.active is False
    mock_update.assert_called_once_with(False)
    mock_logger.debug.assert_any_call("[hook] click released (gest się zakończył)")


# testuje rejestracje hooka volume i faze adjusting po uruchomieniu
@patch("app.core.hooks.logger")
def test_volume_hook_registers_and_sets_adjusting(mock_logger):
    hooks.gesture_start_hooks.pop("volume", None)
    volume_state["phase"] = "idle"
    volume_state["_extend_start"] = None
    hooks.last_gesture_name = None

    hooks.handle_gesture_start_hook(
        "volume", landmarks=object(), frame_shape=(480, 640, 3)
    )
    assert hooks.gesture_start_hooks.get("volume") is not None

    hooks.handle_gesture_start_hook("scroll", landmarks=None, frame_shape=None)

    hooks.handle_gesture_start_hook(
        "volume", landmarks=object(), frame_shape=(480, 640, 3)
    )

    assert volume_state["phase"] == "adjusting"
    assert volume_state["_extend_start"] is None


@patch("app.core.hooks.logger")
def test_volume_reset_on_exit(mock_logger):
    volume_state["phase"] = "adjusting"
    volume_state["_extend_start"] = 1.23
    hooks.last_gesture_name = "volume"

    hooks.handle_gesture_start_hook("scroll", landmarks=None, frame_shape=None)

    assert volume_state["phase"] == "idle"
    assert volume_state["_extend_start"] is None


@patch("app.core.hooks.update_click_state")
@patch("app.core.hooks.logger")
def test_reset_hooks_state_clears_click_and_volume(mock_logger, mock_update):
    from app.gesture_engine.actions import click_action

    setattr(click_action.handle_click, "active", True)
    click_action.click_state["start_time"] = 123.0
    click_action.click_state["holding"] = True
    click_action.click_state["mouse_down"] = True
    click_action.click_state["click_sent"] = True
    click_action.click_state["was_active"] = True

    hooks.last_gesture_name = "volume"
    if hooks.volume_state is not None:
        hooks.volume_state["phase"] = "adjusting"
        hooks.volume_state["_extend_start"] = 1.0

    hooks.reset_hooks_state()

    assert getattr(click_action.handle_click, "active") is False
    assert click_action.click_state["start_time"] is None
    assert click_action.click_state["holding"] is False
    assert click_action.click_state["mouse_down"] is False
    assert click_action.click_state["click_sent"] is False
    assert click_action.click_state["was_active"] is False

    assert hooks.last_gesture_name is None

    if hooks.volume_state is not None:
        assert hooks.volume_state["phase"] == "idle"
        assert hooks.volume_state["_extend_start"] is None

    mock_update.assert_called()
