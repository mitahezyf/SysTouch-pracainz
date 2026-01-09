# Click action - obsługa kliknięć i ciągłego przytrzymania
# Logika:
# - Szybkie dotknięcie (< 1.5s) = pojedynczy click()
# - Przytrzymanie >= 1.5s = mouseDown() i trzymanie (do rysowania z move_mouse)

import time

from app.gesture_engine.logger import logger

# Próg czasu do aktywacji ciągłego kliknięcia (mouseDown)
HOLD_TIME_THRESHOLD = 1.5  # sekundy

# leniwy import pyautogui
try:
    import pyautogui as _pyautogui
except Exception:

    class _PyAutoGuiStub:
        def click(self, *_, **__):
            pass

        def mouseDown(self, *_, **__):
            pass

        def mouseUp(self, *_, **__):
            pass

        def size(self):
            return (1920, 1080)

    logger.warning("pyautogui niedostepne - uzywam no-op stubu")
    _pyautogui = _PyAutoGuiStub()

pyautogui = _pyautogui


# Stan kliknięcia
click_state = {
    "gesture_start": None,  # kiedy zaczął się gest (dotknięcie kciuk+wskazujący)
    "mouse_down_active": False,  # czy mouseDown jest aktywne (tryb rysowania)
    "click_executed": False,  # czy wykonano już akcję w tym cyklu gestu
}


def handle_click(_landmarks, _frame_shape):
    """
    Wywoływane w każdej klatce gdy wykryto gest click (kciuk dotyka wskazującego).

    Logika:
    1. Pierwsze wykrycie -> zapisz czas startu
    2. Jeśli trzymasz < 1.5s -> nic (czekamy)
    3. Jeśli trzymasz >= 1.5s -> mouseDown() (tryb rysowania)
    4. Gdy puścisz (release_click):
       - Jeśli < 1.5s -> click()
       - Jeśli >= 1.5s -> mouseUp()
    """
    current_time = time.time()

    # Pierwsze wykrycie gestu w tym cyklu
    if click_state["gesture_start"] is None:
        click_state["gesture_start"] = current_time
        click_state["mouse_down_active"] = False
        click_state["click_executed"] = False
        logger.debug("[click] Gest rozpoczęty - czekam na przytrzymanie lub puszczenie")
        return

    # Sprawdź czy przekroczono próg czasowy dla ciągłego kliknięcia
    duration = current_time - click_state["gesture_start"]

    if duration >= HOLD_TIME_THRESHOLD and not click_state["mouse_down_active"]:
        # Przekroczono próg - włącz tryb ciągłego kliknięcia (rysowanie)
        pyautogui.mouseDown()
        click_state["mouse_down_active"] = True
        click_state["click_executed"] = True
        logger.info(
            f"[click] mouseDown() - tryb rysowania aktywny (po {duration:.1f}s)"
        )


def release_click():
    """
    Wywoływane gdy gest click się kończy (palce się rozłączyły).
    """
    if click_state["gesture_start"] is None:
        # Brak aktywnego gestu - ignoruj
        return

    duration = time.time() - click_state["gesture_start"]

    if click_state["mouse_down_active"]:
        # Był w trybie ciągłego kliknięcia - zwolnij przycisk
        pyautogui.mouseUp()
        logger.info(f"[click] mouseUp() - koniec rysowania (czas: {duration:.1f}s)")
    elif not click_state["click_executed"]:
        # Krótkie dotknięcie - wykonaj pojedyncze kliknięcie
        pyautogui.click()
        logger.info(f"[click] click() - pojedyncze kliknięcie (czas: {duration:.2f}s)")

    # Reset stanu
    click_state["gesture_start"] = None
    click_state["mouse_down_active"] = False
    click_state["click_executed"] = False


def get_click_state_name():
    """Zwraca nazwę stanu dla UI."""
    if click_state["mouse_down_active"]:
        return "click-hold"
    elif click_state["gesture_start"] is not None:
        return "click"
    else:
        return None


def is_click_holding():
    """Zwraca True jeśli w trybie ciągłego kliknięcia (rysowanie)."""
    return click_state["mouse_down_active"]


# Atrybut "active" dla kompatybilności z hooks
setattr(handle_click, "active", False)
