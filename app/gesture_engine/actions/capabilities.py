from __future__ import annotations

import sys
from types import ModuleType
from typing import Dict, Tuple

# uwaga: importuje moduly akcji, ktore juz implementuja fallbacki na stuby
from app.gesture_engine.actions import (
    click_action,
    close_program_action,
    move_mouse_action,
)
from app.gesture_engine.utils import pycaw_controller


def _is_module(obj: object) -> bool:
    # sprawdza czy obiekt to realny modul, a nie stub (klasa/instancja)
    return isinstance(obj, ModuleType)


def detect_action_capabilities() -> Dict[str, Tuple[bool, str]]:
    """Wykrywa dostepnosc realnych zaleznosci dla akcji systemowych.

    Zwraca slownik: nazwa -> (available, message)
    - pyautogui: wymagane dla click/move_mouse
    - pycaw: wymagane dla volume
    - pywin32: wymagane dla close_program
    """
    caps: Dict[str, Tuple[bool, str]] = {}

    # pyautogui (click/move_mouse)
    try:
        pya_click = getattr(click_action, "pyautogui", None)
        pya_move = getattr(move_mouse_action, "pyautogui", None)
        is_ok = _is_module(pya_click) and _is_module(pya_move)
    except Exception:
        is_ok = False
    if is_ok:
        caps["pyautogui"] = (True, "PyAutoGUI OK")
    else:
        caps["pyautogui"] = (
            False,
            "PyAutoGUI brak lub stub – zainstaluj pakiet 'PyAutoGUI' i uruchom ponownie",
        )

    # pycaw (volume) – tylko Windows
    if sys.platform == "win32":
        audio_utils = getattr(pycaw_controller, "AudioUtilities", None)
        endpoint_iface = getattr(pycaw_controller, "IAudioEndpointVolume", None)
        clsctx = getattr(pycaw_controller, "CLSCTX_ALL", None)
        is_ok = (
            audio_utils is not None
            and endpoint_iface is not None
            and clsctx is not None
        )
        if is_ok:
            caps["pycaw"] = (True, "pycaw OK")
        else:
            caps["pycaw"] = (
                False,
                "pycaw/comtypes brak – zainstaluj 'pycaw' i 'comtypes' (tylko Windows)",
            )
    else:
        caps["pycaw"] = (
            False,
            "Nie-Windows – regulacja glosnosci niedostepna (pomijam)",
        )

    # pywin32 (close_program) – tylko Windows
    if sys.platform == "win32":
        win32gui = getattr(close_program_action, "win32gui", None)
        win32con = getattr(close_program_action, "win32con", None)
        is_ok = _is_module(win32gui) and _is_module(win32con)
        if is_ok:
            caps["pywin32"] = (True, "pywin32 OK")
        else:
            caps["pywin32"] = (
                False,
                "pywin32 brak – zainstaluj 'pywin32' (tylko Windows)",
            )
    else:
        caps["pywin32"] = (
            False,
            "Nie-Windows – zamykanie okna niedostepne (pomijam)",
        )

    return caps
