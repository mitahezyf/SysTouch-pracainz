import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from unittest.mock import patch, MagicMock
import app.actions.move_mouse_action as move_mouse


# handle_move_mouse przelicza pozycje z landmarkow na wspolrzedne ekranu
@patch("app.actions.move_mouse_action.pyautogui.size", return_value=(1920, 1080))
def test_handle_move_mouse_sets_position(mock_size):
    mock_landmark = MagicMock(x=0.5, y=0.25)
    landmarks = [None] * 21
    landmarks[move_mouse.FINGER_TIPS["index"]] = mock_landmark

    move_mouse.handle_move_mouse(landmarks, frame_shape=(480, 640))

    with move_mouse.lock:
        assert move_mouse.latest_position == (960, 270)


# stop_mouse_thread ustawia running = False i czeka na watek
@patch("app.actions.move_mouse_action.worker_thread.join")
def test_stop_mouse_thread(mock_join):
    move_mouse.running = True
    move_mouse.stop_mouse_thread()

    assert move_mouse.running is False
    mock_join.assert_called_once()
