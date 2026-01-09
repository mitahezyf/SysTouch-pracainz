# -*- coding: utf-8 -*-
"""Integration tests for click-hold with move_mouse (drawing functionality)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from unittest.mock import patch

import pytest

import app.gesture_engine.actions.click_action as click_action
from app.gesture_engine.core import hooks


@pytest.fixture(autouse=True)
def reset_state():
    """Reset click state and hooks before each test."""
    click_action.click_state.update(
        {
            "gesture_start": None,
            "mouse_down_active": False,
            "click_executed": False,
        }
    )
    hooks.last_gesture_name = None


@patch("app.gesture_engine.actions.click_action.pyautogui.mouseDown")
@patch("app.gesture_engine.actions.click_action.pyautogui.mouseUp")
@patch("app.gesture_engine.actions.click_action.time.time")
def test_click_hold_then_move_mouse_drawing(mock_time, mock_mouse_up, mock_mouse_down):
    """Test complete drawing workflow: click-hold -> move_mouse -> release."""
    # 1. Start click gesture
    mock_time.return_value = 1000
    click_action.handle_click(None, None)

    # 2. Hold for 1.5s+ to activate mouseDown
    mock_time.return_value = 1000 + click_action.HOLD_TIME_THRESHOLD + 0.1
    click_action.handle_click(None, None)

    assert click_action.click_state["mouse_down_active"] is True
    mock_mouse_down.assert_called_once()

    # 3. Switch to move_mouse (via hooks)
    hooks.last_gesture_name = "click"
    hooks.handle_gesture_start_hook("move_mouse", None, None)

    # mouseDown should still be active (NOT released)
    assert click_action.click_state["mouse_down_active"] is True
    mock_mouse_up.assert_not_called()

    # 4. End move_mouse (user lifts hand)
    hooks.last_gesture_name = "move_mouse"
    with patch(
        "app.gesture_engine.actions.click_action.is_click_holding", return_value=True
    ):
        hooks.handle_gesture_start_hook(None, None, None)

    # Now mouseUp should be called
    mock_mouse_up.assert_called_once()
    assert click_action.click_state["mouse_down_active"] is False


@patch("app.gesture_engine.actions.click_action.pyautogui.click")
@patch("app.gesture_engine.actions.click_action.time.time")
def test_click_short_tap_no_move_mouse(mock_time, mock_click):
    """Test short tap (< 1.5s) performs single click, not drawing."""
    # Start click
    mock_time.return_value = 1000
    click_action.handle_click(None, None)

    # Release before threshold
    mock_time.return_value = 1000 + 0.5
    click_action.release_click()

    # Should perform single click
    mock_click.assert_called_once()
    assert click_action.click_state["mouse_down_active"] is False


@patch("app.gesture_engine.actions.click_action.release_click")
def test_click_to_scroll_releases_click(mock_release):
    """Test that switching from click to scroll releases click."""
    hooks.last_gesture_name = "click"
    hooks.handle_gesture_start_hook("scroll", None, None)

    # Should release click when switching to scroll
    mock_release.assert_called_once()


@patch("app.gesture_engine.actions.click_action.release_click")
def test_click_to_volume_releases_click(mock_release):
    """Test that switching from click to volume releases click."""
    hooks.last_gesture_name = "click"
    hooks.handle_gesture_start_hook("volume", None, None)

    # Should release click when switching to volume
    mock_release.assert_called_once()
