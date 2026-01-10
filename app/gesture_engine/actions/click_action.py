from __future__ import annotations

import threading
import time
from typing import Any

from app.gesture_engine.logger import logger

try:
    import pyautogui

    # krytyczne: wywal failsafe (rzuca wyjatek jak kursor w rogu)
    try:
        pyautogui.FAILSAFE = False
    except Exception:
        pass
except Exception:  # pragma: no cover
    pyautogui = None


# --- TUNING ---
#
# Cel: zachowanie jak w fizycznej myszy:
# - szybki "tap" -> pojedynczy klik
# - przytrzymanie -> mouseDown i trzymanie do puszczenia gestu
#
# HOLD_MS = 1500ms (1.5s) daje czas na ustawienie drugiej reki w geście move_mouse
# przed aktywacja mouseDown, co umozliwia dwureczne rysowanie w Paint.

HOLD_MS = 1500
TAP_MAX_MS = 500

# ile czasu tolerujemy brak detekcji "click" zanim uznamy, ze puszczono (anty-jitter)
RELEASE_GRACE_MS = 200


_lock = threading.Lock()
click_state: dict[str, Any] = {
    "start_time": None,  # monotonic
    "last_seen": None,  # monotonic
    "holding": False,
    "mouse_down": False,
    "click_sent": False,
}


def _now() -> float:
    return time.monotonic()


def is_mouse_down() -> bool:
    with _lock:
        return bool(click_state["mouse_down"])


def _reset_state() -> None:
    click_state["start_time"] = None
    click_state["last_seen"] = None
    click_state["holding"] = False
    click_state["mouse_down"] = False
    # click_sent celowo nie resetuje tutaj w trybie debug (pomaga diagnozowac)


def handle_click(landmarks: Any, frame_shape: Any) -> None:
    """Wywoluj co klatke, gdy wykrywasz gest 'click'.

    NIE robi pojedynczego kliku od razu.
    - Jesli gest trwa >= HOLD_MS -> robi mouseDown (hold).
    - Pojedynczy klik (tap) jest wysylany dopiero w release_click().
    """
    if pyautogui is None:
        return

    now = _now()
    with _lock:
        if click_state["start_time"] is None:
            click_state["start_time"] = now
            click_state["last_seen"] = now
            click_state["holding"] = False
            click_state["mouse_down"] = False
            click_state["click_sent"] = False
            setattr(handle_click, "active", True)
            logger.info("[click] start")
        else:
            click_state["last_seen"] = now

        start_time = click_state["start_time"]
        dur_ms = int((now - float(start_time)) * 1000) if start_time is not None else 0

        # wejscie w hold
        if (not click_state["mouse_down"]) and dur_ms >= HOLD_MS:
            try:
                pyautogui.mouseDown()
                click_state["mouse_down"] = True
                click_state["holding"] = True
                logger.info("[click] mouseDown() hold dur=%dms", dur_ms)
            except Exception as e:
                logger.warning("[click] mouseDown EXC: %r", e)


def maybe_release_click(*, grace_ms: int = RELEASE_GRACE_MS) -> None:
    """Wywoluj co klatke, gdy NIE ma gestu 'click'.

    Zwalnia stan dopiero jak brak "click" utrzymuje sie >= grace_ms.
    To usuwa szarpanie (jitter) klasyfikatora.
    """
    now = _now()
    with _lock:
        start = click_state["start_time"]
        last_seen = click_state["last_seen"]

    if start is None or last_seen is None:
        return

    missing_ms = int((now - float(last_seen)) * 1000)
    if missing_ms >= int(grace_ms):
        release_click(reason=f"not_seen_{missing_ms}ms")


def release_click(*, reason: str = "gesture_end") -> None:
    """Konczy klik:
    - jezeli byl hold -> mouseUp
    - jezeli nie bylo hold i gest byl krotki -> click()
    """
    if pyautogui is None:
        _reset_state()
        setattr(handle_click, "active", False)
        return

    now = _now()
    with _lock:
        start = click_state["start_time"]
        mouse_down = bool(click_state["mouse_down"])
        click_sent = bool(click_state["click_sent"])

    if start is None:
        return

    dur_ms = int((now - float(start)) * 1000)

    # 1) hold -> puszczamy
    if mouse_down:
        try:
            pyautogui.mouseUp()
            logger.info("[click] mouseUp() dur=%dms reason=%s", dur_ms, reason)
        except Exception as e:
            logger.warning("[click] mouseUp EXC: %r", e)
        finally:
            with _lock:
                click_state["mouse_down"] = False
                click_state["holding"] = False

    # 2) tap -> wysylamy pojedynczy klik
    elif (not click_sent) and dur_ms <= TAP_MAX_MS:
        try:
            pyautogui.click()
            with _lock:
                click_state["click_sent"] = True
            logger.info("[click] tap click() dur=%dms reason=%s", dur_ms, reason)
        except Exception as e:
            logger.warning("[click] click() EXC: %r", e)

    # reset
    with _lock:
        _reset_state()
    setattr(handle_click, "active", False)
