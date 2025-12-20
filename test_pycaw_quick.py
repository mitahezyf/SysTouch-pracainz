#!/usr/bin/env python
"""Szybki test czy pycaw dziala poprawnie."""

import sys

print(f"Platform: {sys.platform}")

if sys.platform != "win32":
    print("Nie-Windows, test pomijany")
    sys.exit(0)

print("\n1. Test importow...")
try:
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

    print("   ✅ Importy OK")
except Exception as e:
    print(f"   ❌ Blad importu: {e}")
    sys.exit(1)

print("\n2. Test pobrania urzadzenia audio...")
try:
    devices = AudioUtilities.GetSpeakers()
    print(f"   ✅ GetSpeakers: {devices}")
except Exception as e:
    print(f"   ❌ Blad GetSpeakers: {e}")
    sys.exit(1)

print("\n3. Test interfejsu volume...")
try:
    from ctypes import POINTER, cast

    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    print("   ✅ Interface OK")
except Exception as e:
    print(f"   ❌ Blad interface: {e}")
    sys.exit(1)

print("\n4. Test odczytu glosnosci...")
try:
    current = volume.GetMasterVolumeLevelScalar()  # type: ignore[attr-defined]
    print(f"   ✅ Aktualna glosnosc: {int(current * 100)}%")
except Exception as e:
    print(f"   ❌ Blad odczytu: {e}")
    sys.exit(1)

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
    print("   ✅ Kontroler dziala poprawnie!")

except Exception as e:
    print(f"   ❌ Blad kontrolera: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\n✅ Wszystkie testy pycaw OK!")
print("\nGest volume powinien teraz dzialac w GUI.")
