# todo ogarnac logike do konca, dopiac do volume gest
import sys
from ctypes import POINTER, cast

from app.gesture_engine.logger import logger

# proba importu comtypes/pycaw; jesli brak - ustawiamy na None i dzialamy no-op (tylko Windows)
if sys.platform == "win32":  # pragma: no cover
    try:
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    except Exception:  # pragma: no cover
        CLSCTX_ALL = None
        AudioUtilities = None
        IAudioEndpointVolume = None
else:
    CLSCTX_ALL = None
    AudioUtilities = None
    IAudioEndpointVolume = None


def _get_volume_interface():
    if AudioUtilities is None or IAudioEndpointVolume is None or CLSCTX_ALL is None:
        raise RuntimeError("pycaw/comtypes niedostepne w tym srodowisku")
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(interface, POINTER(IAudioEndpointVolume))


def set_system_volume(volume_percent: int) -> None:
    """ustawia systemowa glosnosc (0-100%); jesli brak pycaw lub nie-Windows - no-op"""
    if sys.platform != "win32":  # pragma: no cover
        logger.warning("set_system_volume: nie-Windows - pomijam")
        return
    try:
        volume = _get_volume_interface()
    except Exception as e:  # pragma: no cover
        logger.warning("pycaw niedostepne - pomijam set_system_volume: {}".format(e))
        return
    volume_level = max(0.0, min(1.0, float(volume_percent) / 100.0))
    volume.SetMasterVolumeLevelScalar(volume_level, None)


def get_system_volume() -> int:
    if sys.platform != "win32":  # pragma: no cover
        logger.warning("get_system_volume: nie-Windows - zwracam 0%")
        return 0
    try:
        volume = _get_volume_interface()
    except Exception as e:  # pragma: no cover
        logger.warning("pycaw niedostepne - zwracam 0%: {}".format(e))
        return 0
    scalar = volume.GetMasterVolumeLevelScalar()
    return int(max(0.0, min(1.0, float(scalar or 0.0))) * 100)


def poke_volume_osd() -> None:
    """Probuje wywolac systemowe OSD glosnosci na Windows bez zmiany stanu.

    Realizuje to przez wyslanie komunikatow WM_APPCOMMAND (UP i DOWN), aby efekt netto byl zerowy.
    W razie braku win32gui uzywa ctypes. Ciche no-op jesli cokolwiek sie nie powiedzie lub nie-Windows.
    """
    if sys.platform != "win32":  # pragma: no cover
        return
    try:
        try:
            import win32gui
        except Exception:
            win32gui = None
        WM_APPCOMMAND = 0x0319
        APPCOMMAND_VOLUME_UP = 0x0A
        APPCOMMAND_VOLUME_DOWN = 0x09
        lparam_up = APPCOMMAND_VOLUME_UP << 16
        lparam_down = APPCOMMAND_VOLUME_DOWN << 16
        hwnd = 0
        if win32gui is not None:  # pragma: no cover
            try:
                hwnd = int(win32gui.GetForegroundWindow())
                win32gui.PostMessage(hwnd, WM_APPCOMMAND, 0, lparam_up)
                win32gui.PostMessage(hwnd, WM_APPCOMMAND, 0, lparam_down)
                return
            except Exception as e:
                logger.debug("poke_volume_osd win32gui error: %s", e)
        # fallback: ctypes
        try:  # pragma: no cover
            import ctypes

            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            user32.PostMessageW(hwnd, WM_APPCOMMAND, 0, lparam_up)
            user32.PostMessageW(hwnd, WM_APPCOMMAND, 0, lparam_down)
        except Exception:
            # cichy no-op; nie logujemy na ERROR by nie spamowac
            logger.debug("poke_volume_osd: PostMessageW failed")
    except Exception as e:  # pragma: no cover
        logger.debug("poke_volume_osd error: %s", e)
