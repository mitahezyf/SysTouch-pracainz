# todo ogarnac logike do konca, dopiac do volume gest
from ctypes import POINTER, cast

from app.gesture_engine.logger import logger

# proba importu comtypes/pycaw; jesli brak – ustawiamy na None i dzialamy no-op
try:  # pragma: no cover
    from comtypes import CLSCTX_ALL  # type: ignore
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  # type: ignore
except Exception:  # pragma: no cover
    CLSCTX_ALL = None  # type: ignore
    AudioUtilities = None  # type: ignore
    IAudioEndpointVolume = None  # type: ignore


def _get_volume_interface():
    if AudioUtilities is None or IAudioEndpointVolume is None or CLSCTX_ALL is None:
        raise RuntimeError("pycaw/comtypes niedostepne w tym srodowisku")
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(interface, POINTER(IAudioEndpointVolume))


def set_system_volume(volume_percent: int):
    """ustawia systemowa glosnosc (0–100%); jesli brak pycaw – no-op"""
    try:
        volume = _get_volume_interface()
    except Exception as e:  # pragma: no cover
        logger.warning(f"pycaw niedostepne – pomijam set_system_volume: {e}")
        return
    volume_level = max(0.0, min(1.0, volume_percent / 100.0))
    volume.SetMasterVolumeLevelScalar(volume_level, None)


def get_system_volume() -> int:
    try:
        volume = _get_volume_interface()
    except Exception as e:  # pragma: no cover
        logger.warning(f"pycaw niedostepne – zwracam 0%: {e}")
        return 0
    scalar = volume.GetMasterVolumeLevelScalar()
    return int(max(0.0, min(1.0, scalar)) * 100)
