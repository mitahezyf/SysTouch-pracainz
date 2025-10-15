from app.gesture_engine.logger import logger

# leniwy import win32* z no-op stubem, aby testy i CI nie padaly bez pywin32
try:  # pragma: no cover
    import win32con as _win32con
    import win32gui as _win32gui
except Exception:  # pragma: no cover

    class _Win32ConStub:
        WM_CLOSE = 0x0010

    class _Win32GuiStub:
        @staticmethod
        def GetForegroundWindow():
            return None

        @staticmethod
        def PostMessage(hwnd, msg, wparam, lparam):
            pass

    logger.warning("pywin32 niedostepne – uzywam no-op stuba (close_program)")
    win32con = _Win32ConStub()
    win32gui = _Win32GuiStub()
else:
    win32con = _win32con
    win32gui = _win32gui


def handle_close_program(landmarks, frame_shape):
    hwnd = win32gui.GetForegroundWindow()
    if hwnd:
        win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
        logger.info("[close] Zamknięto aktywne okno")
    else:
        logger.warning("[close] Nie znaleziono aktywnego okna do zamknięcia")
