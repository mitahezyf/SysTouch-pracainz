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
