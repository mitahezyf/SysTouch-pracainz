import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from unittest.mock import patch

import pytest

import app.gesture_engine.actions.click_action as click_action


# resetuje stan click_state i handle_click.active przed kazdym testem
@pytest.fixture(autouse=True)
def reset_click_state():
    click_action.click_state.update(
        {
            "start_time": None,
            "holding": False,
            "mouse_down": False,
            "click_sent": False,
            "was_active": False,
        }
    )
    click_action.handle_click.active = False


# start_click ustawia wartosci poczatkowe
@patch("app.actions.click_action.logger")
@patch("app.actions.click_action.time.time", return_value=1234.567)
def test_start_click(mock_time, _):
    click_action.start_click()
    state = click_action.click_state
    assert state["start_time"] == 1234.567
    assert state["holding"] is False
    assert state["mouse_down"] is False
    assert state["click_sent"] is False


# handle_active aktywuje hold i mouseDown po przekroczeniu progu
@patch("app.actions.click_action.logger")
@patch("app.actions.click_action.pyautogui.mouseDown")
@patch("app.actions.click_action.time.time")
def test_handle_active_triggers_hold(mock_time, mock_mouse_down, _):
    click_action.click_state["start_time"] = 1000
    mock_time.return_value = 1000 + click_action.HOLD_THRESHOLD + 0.1

    click_action.handle_active()

    assert click_action.click_state["holding"] is True
    assert click_action.click_state["mouse_down"] is True
    mock_mouse_down.assert_called_once()


# release_click wykonuje click() przy krotkim tapie
@patch("app.actions.click_action.logger")
@patch("app.actions.click_action.pyautogui.click")
@patch("app.actions.click_action.time.time")
def test_release_click_short_tap(mock_time, mock_click, _):
    click_action.click_state.update(
        {
            "start_time": 1000,
            "holding": False,
            "mouse_down": False,
            "click_sent": False,
            "was_active": False,
        }
    )
    mock_time.return_value = 1000 + click_action.HOLD_THRESHOLD - 0.1

    click_action.release_click()

    assert click_action.click_state["click_sent"] is True
    mock_click.assert_called_once()


# release_click wykonuje mouseUp przy holdzie
@patch("app.actions.click_action.logger")
@patch("app.actions.click_action.pyautogui.mouseUp")
@patch("app.actions.click_action.time.time")
def test_release_click_hold(mock_time, mock_mouse_up, _):
    click_action.click_state.update(
        {
            "start_time": 1000,
            "holding": True,
            "mouse_down": True,
        }
    )
    mock_time.return_value = 1002

    click_action.release_click()

    mock_mouse_up.assert_called_once()
    assert click_action.click_state["holding"] is False


# release_click ignoruje jesli brak start_time
@patch("app.actions.click_action.logger")
def test_release_click_no_start_time(mock_logger):
    click_action.click_state["start_time"] = None
    click_action.release_click()
    mock_logger.debug.assert_called()


# update_click_state uruchamia start_click i handle_active
@patch("app.actions.click_action.handle_active")
@patch("app.actions.click_action.start_click")
def test_update_click_state_active(mock_start, mock_handle):
    click_action.click_state["was_active"] = False
    click_action.update_click_state(True)

    mock_start.assert_called_once()
    mock_handle.assert_called_once()
    assert click_action.click_state["was_active"] is True


# update_click_state nic nie zmienia jesli juz bylo aktywne
@patch("app.actions.click_action.handle_active")
def test_update_click_state_repeat_active(mock_handle):
    click_action.click_state["was_active"] = True
    click_action.update_click_state(True)
    mock_handle.assert_called_once()


# update_click_state wywoluje release_click przy dezaktywacji
@patch("app.actions.click_action.release_click")
def test_update_click_state_deactivates(mock_release):
    click_action.click_state["was_active"] = True
    click_action.update_click_state(False)
    mock_release.assert_called_once()
    assert click_action.click_state["was_active"] is False


# get_click_state_name zwraca nazwe w zaleznosci od stanu
def test_get_click_state_name():
    click_action.click_state.update(
        {"holding": True, "start_time": 123, "click_sent": False}
    )
    assert click_action.get_click_state_name() == "click-hold"

    click_action.click_state.update(
        {"holding": False, "start_time": 123, "click_sent": False}
    )
    assert click_action.get_click_state_name() == "click"

    click_action.click_state.update({"start_time": None, "click_sent": True})
    assert click_action.get_click_state_name() is None


# handle_click ustawia aktywnosc i wywoluje update_click_state
@patch("app.actions.click_action.update_click_state")
def test_handle_click_sets_active(mock_update):
    click_action.handle_click(None, None)
    assert click_action.handle_click.active is True
    mock_update.assert_called_once_with(True)
