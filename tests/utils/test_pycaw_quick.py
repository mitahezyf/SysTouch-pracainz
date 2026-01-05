#!/usr/bin/env python
"""Szybki test czy pycaw dziala poprawnie."""

import sys

import pytest

# plik pomijany na nie-Windows platformach
pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="pycaw wymaga Windows")


def test_pycaw_imports() -> None:
    """sprawdza czy importy pycaw dzialaja"""
    if sys.platform != "win32":
        pytest.skip("pycaw wymaga Windows")

    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

    assert CLSCTX_ALL is not None
    assert AudioUtilities is not None
    assert IAudioEndpointVolume is not None


def test_pycaw_get_speakers() -> None:
    """sprawdza czy mozna pobrac urzadzenie audio"""
    if sys.platform != "win32":
        pytest.skip("pycaw wymaga Windows")

    from pycaw.pycaw import AudioUtilities

    devices = AudioUtilities.GetSpeakers()
    assert devices is not None


def test_pycaw_volume_interface() -> None:
    """sprawdza czy mozna uzyskac interfejs volume"""
    if sys.platform != "win32":
        pytest.skip("pycaw wymaga Windows")

    from ctypes import POINTER, cast

    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    assert volume is not None


def test_pycaw_read_volume() -> None:
    """sprawdza czy mozna odczytac glosnosc"""
    if sys.platform != "win32":
        pytest.skip("pycaw wymaga Windows")

    from ctypes import POINTER, cast

    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    current = volume.GetMasterVolumeLevelScalar()  # type: ignore[attr-defined]
    assert 0.0 <= current <= 1.0


def test_pycaw_controller() -> None:
    """sprawdza czy dziala modul pycaw_controller"""
    if sys.platform != "win32":
        pytest.skip("pycaw wymaga Windows")

    from app.gesture_engine.utils.pycaw_controller import (
        get_system_volume,
        set_system_volume,
    )

    vol_before = get_system_volume()
    assert 0 <= vol_before <= 100

    # zmien glosnosc o 1% (bezpieczny test)
    test_vol = max(0, min(100, vol_before + 1))
    set_system_volume(test_vol)

    vol_after = get_system_volume()
    assert 0 <= vol_after <= 100

    # przywroc
    set_system_volume(vol_before)


def main() -> int:
    """glowna funkcja diagnostyczna (dla manualnego uruchomienia)"""
    print(f"Platform: {sys.platform}")

    if sys.platform != "win32":
        print("Nie-Windows, test pomijany")
        return 0

    print("\n1. Test importow...")
    try:
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

        print("   OK Importy OK")
    except Exception as e:
        print(f"   X Blad importu: {e}")
        return 1

    print("\n2. Test pobrania urzadzenia audio...")
    try:
        devices = AudioUtilities.GetSpeakers()
        print(f"   OK GetSpeakers: {devices}")
    except Exception as e:
        print(f"   X Blad GetSpeakers: {e}")
        return 1

    print("\n3. Test interfejsu volume...")
    try:
        from ctypes import POINTER, cast

        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        print("   OK Interface OK")
    except Exception as e:
        print(f"   X Blad interface: {e}")
        return 1

    print("\n4. Test odczytu glosnosci...")
    try:
        current = volume.GetMasterVolumeLevelScalar()  # type: ignore[attr-defined]
        print(f"   OK Aktualna glosnosc: {int(current * 100)}%")
    except Exception as e:
        print(f"   X Blad odczytu: {e}")
        return 1

    print("\n5. Test z modulu pycaw_controller...")
    try:
        from app.gesture_engine.utils.pycaw_controller import (
            get_system_volume,
            set_system_volume,
        )

        vol_before = get_system_volume()
        print(f"   Glosnosc przed: {vol_before}%")

        # zmien glosnosc o 1% (bezpieczny test)
        test_vol = max(0, min(100, vol_before + 1))
        set_system_volume(test_vol)

        vol_after = get_system_volume()
        print(f"   Glosnosc po:    {vol_after}%")

        # przywroc
        set_system_volume(vol_before)
        print("   OK Kontroler dziala poprawnie!")

    except Exception as e:
        print(f"   X Blad kontrolera: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print("\nOK Wszystkie testy pycaw OK!")
    print("\nGest volume powinien teraz dzialac w GUI.")
    return 0


if __name__ == "__main__":
    result = main()
    # nie uzywaj sys.exit() w pliku testowym - przerywa pytest podczas importu
    print(f"\nExit code: {result}")
