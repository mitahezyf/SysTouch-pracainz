from types import ModuleType

import app.gesture_engine.actions.capabilities as capabilities


def test_capabilities_all_present(monkeypatch):
    # wymusza Windows
    monkeypatch.setattr(capabilities, "sys", ModuleType("sys"))
    capabilities.sys.platform = "win32"

    # pyautogui obecne: podstawiamy atrapy modulow
    from app.gesture_engine.actions import (
        click_action,
        close_program_action,
        move_mouse_action,
    )
    from app.gesture_engine.utils import pycaw_controller

    monkeypatch.setattr(
        click_action, "pyautogui", ModuleType("pyautogui"), raising=False
    )
    monkeypatch.setattr(
        move_mouse_action, "pyautogui", ModuleType("pyautogui"), raising=False
    )

    # pycaw/comtypes obecne
    monkeypatch.setattr(pycaw_controller, "AudioUtilities", object(), raising=False)
    monkeypatch.setattr(
        pycaw_controller, "IAudioEndpointVolume", object(), raising=False
    )
    monkeypatch.setattr(pycaw_controller, "CLSCTX_ALL", object(), raising=False)

    # pywin32 obecne
    monkeypatch.setattr(
        close_program_action, "win32gui", ModuleType("win32gui"), raising=False
    )
    monkeypatch.setattr(
        close_program_action, "win32con", ModuleType("win32con"), raising=False
    )

    caps = capabilities.detect_action_capabilities()
    assert caps["pyautogui"][0] is True
    assert caps["pycaw"][0] is True
    assert caps["pywin32"][0] is True


def test_capabilities_missing_on_non_windows(monkeypatch):
    # nie-Windows: pycaw i pywin32 powinny byc niedostepne z komunikatem
    monkeypatch.setattr(capabilities, "sys", ModuleType("sys"))
    capabilities.sys.platform = "linux"

    from app.gesture_engine.actions import click_action

    # pyautogui brak: podstawiamy nie-modul (np. None lub klasa stuba)
    monkeypatch.setattr(click_action, "pyautogui", object(), raising=False)
    # ale move_mouse_action ma niezaleznie swoj atrybut; wynik pyautogui i tak bedzie False

    caps = capabilities.detect_action_capabilities()

    assert caps["pyautogui"][0] is False
    assert "PyAutoGUI" in caps["pyautogui"][1]
    assert caps["pycaw"][0] is False
    assert "Nie-Windows" in caps["pycaw"][1]
    assert caps["pywin32"][0] is False
    assert "Nie-Windows" in caps["pywin32"][1]
