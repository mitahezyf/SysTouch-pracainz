import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from unittest.mock import MagicMock, patch

import app.gesture_engine.core.hooks as hooks


# czy rejestracja hooka dziala i jest wykonywana przy zmianie gestu
@patch("app.core.hooks.logger")
def test_register_and_trigger_hook(mock_logger):
    mock_func = MagicMock()
    hooks.register_gesture_start_hook("test_gest", mock_func)

    hooks.handle_gesture_start_hook("test_gest", "land", "shape")

    mock_func.assert_called_once_with("land", "shape")
    mock_logger.debug.assert_any_call("[hook] zmiana gestu: None -> test_gest")
    mock_logger.debug.assert_any_call("[hook] wywołanie hooka dla: test_gest")


# czy zmiana gestu z click na cos innego resetuje click_state
@patch("app.core.hooks.logger")
@patch("app.core.hooks.update_click_state")
def test_handle_click_reset_on_change(mock_update, mock_logger):
    hooks.handle_click.active = True
    hooks.handle_gesture_start_hook("scroll", None, None)

    assert hooks.handle_click.active is False
    mock_update.assert_called_once_with(False)
    mock_logger.debug.assert_any_call("[hook] click released (gest zmienił się)")


# czy zakonczenie clicka (None) resetuje stan klikniecia
@patch("app.core.hooks.logger")
@patch("app.core.hooks.update_click_state")
def test_handle_click_reset_on_none(mock_update, mock_logger):
    hooks.last_gesture_name = "click"
    hooks.handle_click.active = True

    hooks.handle_gesture_start_hook(None, None, None)

    assert hooks.handle_click.active is False
    mock_update.assert_called_once_with(False)
    mock_logger.debug.assert_any_call("[hook] click released (gest się zakończył)")
