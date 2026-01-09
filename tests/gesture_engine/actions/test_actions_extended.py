# -*- coding: utf-8 -*-
"""Additional tests for gesture actions to increase coverage."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from unittest.mock import patch


def test_handle_close_program_success():
    """Test close_program action with valid window."""
    from app.gesture_engine.actions.close_program_action import handle_close_program

    with patch(
        "app.gesture_engine.actions.close_program_action.win32gui"
    ) as mock_win32:
        mock_win32.GetForegroundWindow.return_value = 12345
        mock_win32.GetWindowText.return_value = "Test Window"

        # Should not raise exception
        handle_close_program(None, None)

        mock_win32.PostMessage.assert_called_once()


def test_handle_move_mouse_basic():
    """Test move_mouse action sets position."""
    from app.gesture_engine.actions.move_mouse_action import handle_move_mouse

    # Create fake landmarks with index finger tip at specific position
    class FakeLandmark:
        def __init__(self, x, y, z=0):
            self.x = x
            self.y = y
            self.z = z

    landmarks = [FakeLandmark(0, 0)] * 21
    landmarks[8] = FakeLandmark(0.5, 0.5)  # INDEX_FINGER_TIP

    frame_shape = (480, 640, 3)

    # Should not raise exception
    handle_move_mouse(landmarks, frame_shape)
