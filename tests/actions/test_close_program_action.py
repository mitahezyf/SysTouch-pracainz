import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from unittest.mock import patch

from app.gesture_engine.actions.close_program_action import handle_close_program


# jezeli aktywne okno istnieje powinno zostac zamkniete
@patch("app.actions.close_program_action.logger")
@patch("app.actions.close_program_action.win32gui.PostMessage")
@patch("app.actions.close_program_action.win32gui.GetForegroundWindow", return_value=123)
def test_handle_close_program_ok(mock_get_hwnd, mock_post, mock_logger):
    handle_close_program(None, None)
    mock_get_hwnd.assert_called_once()
    mock_post.assert_called_once_with(123, 0x0010, 0, 0)
    mock_logger.info.assert_called_once_with("[close] Zamknięto aktywne okno")


# jezeli brak aktywnego okna powinien byc warning
@patch("app.actions.close_program_action.logger")
@patch("app.actions.close_program_action.win32gui.GetForegroundWindow", return_value=None)
def test_handle_close_program_no_window(mock_get_hwnd, mock_logger):
    handle_close_program(None, None)
    mock_get_hwnd.assert_called_once()
    mock_logger.warning.assert_called_once_with("[close] Nie znaleziono aktywnego okna do zamknięcia")
