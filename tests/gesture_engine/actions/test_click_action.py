import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from unittest.mock import patch

import pytest

import app.gesture_engine.actions.click_action as click_action


# resetuje stan click_state przed kazdym testem
@pytest.fixture(autouse=True)
def reset_click_state():
    click_action.click_state.update(
        {
            "start_time": None,
            "last_seen": None,
            "holding": False,
            "mouse_down": False,
            "click_sent": False,
        }
    )


# handle_click przy pierwszym wywolaniu zapisuje czas startu
@patch("app.gesture_engine.actions.click_action.logger")
@patch("app.gesture_engine.actions.click_action.time.monotonic", return_value=1234.567)
def test_handle_click_first_call(mock_time, _):
    click_action.handle_click(None, None)
    state = click_action.click_state
    assert state["start_time"] == 1234.567
    assert state["mouse_down"] is False
    assert state["click_sent"] is False


# handle_click aktywuje mouseDown pop przekroczeniu HOLD_MS
@patch("app.gesture_engine.actions.click_action.logger")
@patch("app.gesture_engine.actions.click_action.pyautogui.mouseDown")
@patch("app.gesture_engine.actions.click_action.time.monotonic")
def test_handle_click_triggers_hold(mock_time, mock_mouse_down, _):
    # Pierwsze wywolanie - ustaw czas startu
    mock_time.return_value = 1000
    click_action.handle_click(None, None)

    # Drugie wywolanie - po przekroczeniu progu (HOLD_MS = 1500ms = 1.5s)
    mock_time.return_value = 1000 + 1.5 + 0.1
    click_action.handle_click(None, None)

    assert click_action.click_state["mouse_down"] is True
    assert click_action.click_state["holding"] is True
    mock_mouse_down.assert_called_once()


# handle_click nie wywoluje mouseDown ponownie jesli juz aktywne
@patch("app.gesture_engine.actions.click_action.pyautogui.mouseDown")
@patch("app.gesture_engine.actions.click_action.time.monotonic")
def test_handle_click_hold_not_repeated(mock_time, mock_mouse_down):
    # Ustaw stan jako juz w trybie hold
    click_action.click_state["start_time"] = 1000
    click_action.click_state["mouse_down"] = True
    click_action.click_state["holding"] = True

    mock_time.return_value = 1002
    click_action.handle_click(None, None)

    # mouseDown nie powinno byc wywolane ponownie
    mock_mouse_down.assert_not_called()


# release_click wykonuje click() przy krotkim tapie
@patch("app.gesture_engine.actions.click_action.logger")
@patch("app.gesture_engine.actions.click_action.pyautogui.click")
@patch("app.gesture_engine.actions.click_action.time.monotonic")
def test_release_click_short_tap(mock_time, mock_click, _):
    click_action.click_state.update(
        {
            "start_time": 1000,
            "mouse_down": False,
            "click_sent": False,
        }
    )
    # Krotki czas - ponizej TAP_MAX_MS (500ms = 0.5s)
    mock_time.return_value = 1000 + 0.3

    click_action.release_click()

    mock_click.assert_called_once()
    # Stan powinien byc zresetowany
    assert click_action.click_state["start_time"] is None
    assert click_action.click_state["mouse_down"] is False


# release_click wykonuje mouseUp przy holdzie
@patch("app.gesture_engine.actions.click_action.logger")
@patch("app.gesture_engine.actions.click_action.pyautogui.mouseUp")
@patch("app.gesture_engine.actions.click_action.time.monotonic")
def test_release_click_hold(mock_time, mock_mouse_up, _):
    click_action.click_state.update(
        {
            "start_time": 1000,
            "mouse_down": True,
            "holding": True,
        }
    )
    mock_time.return_value = 1002

    click_action.release_click()

    mock_mouse_up.assert_called_once()
    # Stan powinien byc zresetowany
    assert click_action.click_state["start_time"] is None
    assert click_action.click_state["mouse_down"] is False


# release_click ignoruje jesli brak start_time
def test_release_click_no_start_time():
    click_action.click_state["start_time"] = None
    # Nie powinno rzucic wyjatku
    click_action.release_click()
    # Stan powinien pozostac None
    assert click_action.click_state["start_time"] is None


# is_mouse_down zwraca True gdy mouse_down aktywny
def test_is_click_holding_true():
    click_action.click_state["mouse_down"] = True
    assert click_action.is_mouse_down() is True


# is_mouse_down zwraca False gdy mouse_down jest False
def test_is_click_holding_false():
    click_action.click_state["mouse_down"] = False
    assert click_action.is_mouse_down() is False


# get_click_state_name nie istnieje - usuwam test
# Funkcja get_click_state_name byla usunieta z API
def test_get_click_state_name():
    # Ten test jest nieaktualny - funkcja nie istnieje w nowym API
    pytest.skip("get_click_state_name() nie istnieje w nowym API")
