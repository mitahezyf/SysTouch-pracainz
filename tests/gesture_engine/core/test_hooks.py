import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from unittest.mock import MagicMock, patch

import app.gesture_engine.core.hooks as hooks


# weryfikuje rejestracje hooka i jego wywolanie przy zmianie gestu
@patch("app.gesture_engine.core.hooks.logger")
def test_register_and_trigger_hook(mock_logger):
    mock_func = MagicMock()
    hooks.register_gesture_start_hook("test_gest", mock_func)

    hooks.handle_gesture_start_hook("test_gest", "land", "shape")

    mock_func.assert_called_once_with("land", "shape")
    mock_logger.debug.assert_any_call("[hook] zmiana gestu: None -> test_gest")
    mock_logger.debug.assert_any_call("[hook] wywoĹ‚anie hooka dla: test_gest")


# weryfikuje reset stanu click przy przejsciu na inny gest (nie move_mouse)
@patch("app.gesture_engine.actions.click_action.release_click")
@patch("app.gesture_engine.core.hooks.logger")
def test_handle_click_reset_on_change(mock_logger, mock_release):
    hooks.last_gesture_name = "click"
    hooks.handle_gesture_start_hook("scroll", None, None)

    mock_release.assert_called_once()
    mock_logger.debug.assert_any_call("[hook] click released (gest zmieniĹ‚ siÄ™)")


# weryfikuje ze click NIE jest zwalniany przy przejsciu na move_mouse
@patch("app.gesture_engine.actions.click_action.release_click")
def test_handle_click_not_released_on_move_mouse(mock_release):
    hooks.last_gesture_name = "click"
    hooks.handle_gesture_start_hook("move_mouse", None, None)

    # release_click NIE powinno byc wywolane
    mock_release.assert_not_called()


# weryfikuje reset stanu click przy zakonczeniu gestu (None)
@patch("app.gesture_engine.actions.click_action.release_click")
@patch("app.gesture_engine.core.hooks.logger")
def test_handle_click_reset_on_none(mock_logger, mock_release):
    hooks.last_gesture_name = "click"

    hooks.handle_gesture_start_hook(None, None, None)

    mock_release.assert_called_once()
    mock_logger.debug.assert_any_call("[hook] click released (gest siÄ™ zakoĹ„czyĹ‚)")


# weryfikuje reset click gdy move_mouse sie konczy podczas click-hold
@patch("app.gesture_engine.actions.click_action.is_click_holding", return_value=True)
@patch("app.gesture_engine.actions.click_action.release_click")
@patch("app.gesture_engine.core.hooks.logger")
def test_handle_move_mouse_end_releases_click_hold(
    mock_logger, mock_release, mock_is_holding
):
    hooks.last_gesture_name = "move_mouse"

    hooks.handle_gesture_start_hook(None, None, None)

    mock_is_holding.assert_called_once()
    mock_release.assert_called_once()
    mock_logger.debug.assert_any_call(
        "[hook] click released (move_mouse zakoĹ„czony podczas rysowania)"
    )


@patch("app.gesture_engine.actions.click_action.release_click")
@patch("app.gesture_engine.core.hooks.logger")
def test_reset_hooks_state_clears_click(mock_logger, mock_release):
    from app.gesture_engine.actions import click_action

    # Ustaw stan click jako aktywny
    click_action.click_state["gesture_start"] = 123.0
    click_action.click_state["mouse_down_active"] = True
    click_action.click_state["click_executed"] = True

    hooks.last_gesture_name = "click"

    hooks.reset_hooks_state()

    # release_click powinno byc wywolane
    mock_release.assert_called_once()
    assert hooks.last_gesture_name is None
