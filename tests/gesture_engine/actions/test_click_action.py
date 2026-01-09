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
            "gesture_start": None,
            "mouse_down_active": False,
            "click_executed": False,
        }
    )


# handle_click przy pierwszym wywolaniu zapisuje czas startu
@patch("app.gesture_engine.actions.click_action.logger")
@patch("app.gesture_engine.actions.click_action.time.time", return_value=1234.567)
def test_handle_click_first_call(mock_time, _):
    click_action.handle_click(None, None)
    state = click_action.click_state
    assert state["gesture_start"] == 1234.567
    assert state["mouse_down_active"] is False
    assert state["click_executed"] is False


# handle_click aktywuje mouseDown po przekroczeniu HOLD_TIME_THRESHOLD
@patch("app.gesture_engine.actions.click_action.logger")
@patch("app.gesture_engine.actions.click_action.pyautogui.mouseDown")
@patch("app.gesture_engine.actions.click_action.time.time")
def test_handle_click_triggers_hold(mock_time, mock_mouse_down, _):
    # Pierwsze wywolanie - ustaw czas startu
    mock_time.return_value = 1000
    click_action.handle_click(None, None)

    # Drugie wywolanie - po przekroczeniu progu
    mock_time.return_value = 1000 + click_action.HOLD_TIME_THRESHOLD + 0.1
    click_action.handle_click(None, None)

    assert click_action.click_state["mouse_down_active"] is True
    assert click_action.click_state["click_executed"] is True
    mock_mouse_down.assert_called_once()


# handle_click nie wywoluje mouseDown ponownie jesli juz aktywne
@patch("app.gesture_engine.actions.click_action.pyautogui.mouseDown")
@patch("app.gesture_engine.actions.click_action.time.time")
def test_handle_click_hold_not_repeated(mock_time, mock_mouse_down):
    # Ustaw stan jako juz w trybie hold
    click_action.click_state["gesture_start"] = 1000
    click_action.click_state["mouse_down_active"] = True
    click_action.click_state["click_executed"] = True

    mock_time.return_value = 1002
    click_action.handle_click(None, None)

    # mouseDown nie powinno byc wywolane ponownie
    mock_mouse_down.assert_not_called()


# release_click wykonuje click() przy krotkim tapie
@patch("app.gesture_engine.actions.click_action.logger")
@patch("app.gesture_engine.actions.click_action.pyautogui.click")
@patch("app.gesture_engine.actions.click_action.time.time")
def test_release_click_short_tap(mock_time, mock_click, _):
    click_action.click_state.update(
        {
            "gesture_start": 1000,
            "mouse_down_active": False,
            "click_executed": False,
        }
    )
    mock_time.return_value = 1000 + click_action.HOLD_TIME_THRESHOLD - 0.1

    click_action.release_click()

    mock_click.assert_called_once()
    # Stan powinien byc zresetowany
    assert click_action.click_state["gesture_start"] is None
    assert click_action.click_state["mouse_down_active"] is False


# release_click wykonuje mouseUp przy holdzie
@patch("app.gesture_engine.actions.click_action.logger")
@patch("app.gesture_engine.actions.click_action.pyautogui.mouseUp")
@patch("app.gesture_engine.actions.click_action.time.time")
def test_release_click_hold(mock_time, mock_mouse_up, _):
    click_action.click_state.update(
        {
            "gesture_start": 1000,
            "mouse_down_active": True,
            "click_executed": True,
        }
    )
    mock_time.return_value = 1002

    click_action.release_click()

    mock_mouse_up.assert_called_once()
    # Stan powinien byc zresetowany
    assert click_action.click_state["gesture_start"] is None
    assert click_action.click_state["mouse_down_active"] is False


# release_click ignoruje jesli brak gesture_start
def test_release_click_no_start_time():
    click_action.click_state["gesture_start"] = None
    # Nie powinno rzucic wyjatku
    click_action.release_click()
    # Stan powinien pozostac None
    assert click_action.click_state["gesture_start"] is None


# is_click_holding zwraca True gdy mouse_down_active
def test_is_click_holding_true():
    click_action.click_state["mouse_down_active"] = True
    assert click_action.is_click_holding() is True


# is_click_holding zwraca False gdy mouse_down_active jest False
def test_is_click_holding_false():
    click_action.click_state["mouse_down_active"] = False
    assert click_action.is_click_holding() is False


# get_click_state_name zwraca nazwe w zaleznosci od stanu
def test_get_click_state_name():
    # Stan: mouse_down_active
    click_action.click_state.update({"mouse_down_active": True, "gesture_start": 123})
    assert click_action.get_click_state_name() == "click-hold"

    # Stan: gesture_start ale nie mouse_down
    click_action.click_state.update({"mouse_down_active": False, "gesture_start": 123})
    assert click_action.get_click_state_name() == "click"

    # Stan: brak gestu
    click_action.click_state.update({"gesture_start": None, "mouse_down_active": False})
    assert click_action.get_click_state_name() is None
