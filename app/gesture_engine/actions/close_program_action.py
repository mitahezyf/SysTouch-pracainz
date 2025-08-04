import win32con
import win32gui

from app.gesture_engine.logger import logger


def handle_close_program(landmarks, frame_shape):
    hwnd = win32gui.GetForegroundWindow()
    if hwnd:
        win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
        logger.info("[close] Zamknięto aktywne okno")
    else:
        logger.warning("[close] Nie znaleziono aktywnego okna do zamknięcia")
