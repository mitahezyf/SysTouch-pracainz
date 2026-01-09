# todo usunac powielone wywolanie update_click_state(False) przy zmianie gestu z click

from typing import Dict, Optional  # noqa: F401

from app.gesture_engine.logger import logger

# deklaracja typu na poziomie modulu
volume_state = None  # type: Optional[Dict[str, object]]

# volume: import stanu do resetu/przejsc
try:
    from app.gesture_engine.gestures.volume_gesture import volume_state as _volume_state

    volume_state = _volume_state
except Exception:
    # gdy modul gestu nie jest dostepny (np. w minimalnym trybie testowym)
    volume_state = None

last_gesture_name = None
gesture_start_hooks = {}


def register_gesture_start_hook(gesture_name, func):
    gesture_start_hooks[gesture_name] = func


def handle_gesture_start_hook(gesture_name, landmarks, frame_shape):
    global last_gesture_name

    # zwalnia click przy zmianie na inny gest
    # WYJĄTEK: NIE zwalniaj gdy przełączamy na move_mouse (potrzebne do rysowania!)
    if (
        last_gesture_name in ("click", "click-hold")
        and gesture_name is not None
        and gesture_name not in ("click", "click-hold", "move_mouse")
    ):
        from app.gesture_engine.actions.click_action import release_click

        release_click()
        logger.debug("[hook] click released (gest zmienił się)")

    if gesture_name != last_gesture_name:
        logger.debug(
            "[hook] zmiana gestu: {} -> {}".format(last_gesture_name, gesture_name)
        )
        hook = gesture_start_hooks.get(gesture_name)
        if hook:
            logger.debug("[hook] wywołanie hooka dla: {}".format(gesture_name))
            if landmarks is not None and frame_shape is not None:
                hook(landmarks, frame_shape)
            else:
                logger.debug("[hook] pomijam wywolanie hooka (brak danych)")

    # specjalny przypadek: koniec gestu (None) po clicku
    if gesture_name is None and last_gesture_name in ("click", "click-hold"):
        from app.gesture_engine.actions.click_action import release_click

        release_click()
        logger.debug("[hook] click released (gest się zakończył)")

    # specjalny przypadek: koniec move_mouse podczas aktywnego click-hold
    # (użytkownik kończył rysowanie przez podniesienie wszystkich palców)
    if gesture_name is None and last_gesture_name == "move_mouse":
        from app.gesture_engine.actions.click_action import (
            is_click_holding,
            release_click,
        )

        if is_click_holding():
            release_click()
            logger.debug(
                "[hook] click released (move_mouse zakończony podczas rysowania)"
            )

    # Uwaga: close_program cooldown jest teraz obslugiwany w gesture detector

    last_gesture_name = gesture_name

    def test_scroll_hook(landmarks, frame_shape):
        logger.debug("[hook] scroll hook wywolany")

    if "scroll" not in gesture_start_hooks:
        register_gesture_start_hook("scroll", test_scroll_hook)


def reset_hooks_state() -> None:
    """Resetuje globalne stany hookow/gestow przed nowym uruchomieniem przetwarzania.

    - zwalnia ewentualny aktywny click
    - zeruje last_gesture_name
    - ustawia volume w stan idle
    - czysci wewnetrzne flagi click_state
    """
    global last_gesture_name
    try:
        # zwalnia click, jesli byl aktywny
        from app.gesture_engine.actions.click_action import release_click

        release_click()
    except Exception as e:
        logger.debug("reset_hooks_state: click reset error: %s", e)

    # resetuje volume
    try:
        if volume_state is not None:
            volume_state["phase"] = "idle"
            volume_state["_extend_start"] = None
            try:
                volume_state["pct"] = None
                volume_state["ref_max"] = None
                volume_state["pinch_since"] = None
                volume_state["pinch_th"] = None
                volume_state["control_wrist"] = None
                volume_state["last_seen_ts"] = None
                volume_state["_exit_pinched_since"] = None
                volume_state["_last_pct"] = None
                volume_state["_stable_since"] = None
                # resetuje pola nowego trybu galki
                volume_state["knob_baseline_angle_deg"] = None
                volume_state["knob_range_deg"] = None
                volume_state["knob_invert"] = None
            except Exception as e:
                logger.debug("reset_hooks_state: volume nested reset error: %s", e)
    except Exception as e:
        logger.debug("reset_hooks_state: volume reset error: %s", e)

    # Uwaga: close_program cooldown jest teraz obslugiwany w worker.py (single-shot actions)

    last_gesture_name = None
    logger.debug("reset_hooks_state: state cleared")
