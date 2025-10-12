# todo ogarnac logike do konca, dopiac do volume gest
from ctypes import cast
from ctypes import POINTER

from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities
from pycaw.pycaw import IAudioEndpointVolume


def _get_volume_interface():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(interface, POINTER(IAudioEndpointVolume))


def set_system_volume(volume_percent: int):
    """Ustawia systemową głośność (0–100%)"""
    volume = _get_volume_interface()
    volume_level = volume_percent / 100.0
    volume.SetMasterVolumeLevelScalar(volume_level, None)


def get_system_volume() -> int:
    volume = _get_volume_interface()
    scalar = volume.GetMasterVolumeLevelScalar()
    return int(scalar * 100)
